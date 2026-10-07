"""Independent patient-generation and task-evaluation scheduling."""

import asyncio
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Iterable, TypeVar


Plan = TypeVar("Plan")
TaskValue = TypeVar("TaskValue")
Evaluation = TypeVar("Evaluation")


@dataclass
class PatientResult:
    """Generated tasks and their evaluations, retaining the input plan."""

    plan: Any
    tasks: list[Any]
    evaluations: dict[int, Any] = field(default_factory=dict)


async def run_generation_evaluation(
    plans: Iterable[Plan],
    generation_limit: int,
    evaluation_limit: int,
    generate: Callable[[Plan], Awaitable[list[TaskValue]]],
    evaluate: Callable[[Plan, int, TaskValue], Awaitable[Evaluation]],
) -> list[PatientResult]:
    """Generate patients under one limit and evaluate each task under another.

    Generation permits are released before generated tasks enter the evaluation
    queue. Results are returned in input and task order regardless of completion
    order.
    """
    if generation_limit <= 0 or evaluation_limit <= 0:
        raise ValueError("Concurrency limits must be positive")

    indexed_plans = list(enumerate(plans))
    evaluation_workers = min(evaluation_limit, max(1, len(indexed_plans)))
    generated: dict[int, PatientResult] = {}
    queue: asyncio.Queue[tuple[int, Plan, int, TaskValue] | None] = asyncio.Queue()
    generation_slots = asyncio.Semaphore(generation_limit)

    async def produce_patient(plan_index: int, plan: Plan) -> None:
        async with generation_slots:
            tasks = await generate(plan)
        outcome = PatientResult(plan=plan, tasks=list(tasks))
        generated[plan_index] = outcome
        for task_index, task in enumerate(outcome.tasks):
            queue.put_nowait((plan_index, plan, task_index, task))

    async def produce() -> None:
        generators = [asyncio.create_task(produce_patient(index, plan))
                      for index, plan in indexed_plans]
        try:
            await asyncio.gather(*generators)
        except BaseException:
            for worker in generators:
                if not worker.done():
                    worker.cancel()
            await asyncio.gather(*generators, return_exceptions=True)
            raise
        for _ in range(evaluation_workers):
            queue.put_nowait(None)

    async def consume() -> None:
        while True:
            item = await queue.get()
            try:
                if item is None:
                    return
                plan_index, plan, task_index, task = item
                evaluation = await evaluate(plan, task_index, task)
                generated[plan_index].evaluations[task_index] = evaluation
            finally:
                queue.task_done()

    producer = asyncio.create_task(produce())
    consumers = [asyncio.create_task(consume()) for _ in range(evaluation_workers)]
    workers = [producer, *consumers]
    try:
        await asyncio.gather(*workers)
    except BaseException:
        for worker in workers:
            if not worker.done():
                worker.cancel()
        await asyncio.gather(*workers, return_exceptions=True)
        raise

    return [generated[index] for index, _ in indexed_plans]
