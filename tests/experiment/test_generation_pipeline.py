import asyncio

import pytest

from experiments.scheduling import run_generation_evaluation


def test_generation_continues_while_evaluation_is_blocked_and_keeps_limits_independent():
    async def scenario():
        evaluation_started = asyncio.Event()
        release_evaluation = asyncio.Event()
        third_generated = asyncio.Event()
        generation_active = evaluation_active = 0
        max_generation_active = max_evaluation_active = 0

        async def generate(patient):
            nonlocal generation_active, max_generation_active
            generation_active += 1
            max_generation_active = max(max_generation_active, generation_active)
            await asyncio.sleep(0)
            generation_active -= 1
            if patient == "p3":
                third_generated.set()
            return [f"{patient}-q1", f"{patient}-q2"]

        async def evaluate(patient, index, task):
            nonlocal evaluation_active, max_evaluation_active
            evaluation_active += 1
            max_evaluation_active = max(max_evaluation_active, evaluation_active)
            evaluation_started.set()
            if task == "p1-q1":
                await release_evaluation.wait()
            evaluation_active -= 1
            return (patient, index, task)

        pending = asyncio.create_task(run_generation_evaluation(
            ["p1", "p2", "p3"], 1, 2, generate, evaluate,
        ))
        await asyncio.wait_for(evaluation_started.wait(), timeout=1)
        await asyncio.wait_for(third_generated.wait(), timeout=1)
        assert not pending.done()
        assert max_generation_active == 1
        assert max_evaluation_active <= 2
        release_evaluation.set()
        outcomes = await pending
        assert [outcome.plan for outcome in outcomes] == ["p1", "p2", "p3"]
        assert [outcome.evaluations[i][2] for outcome in outcomes for i in sorted(outcome.evaluations)] == [
            f"p{patient}-q{question}" for patient in range(1, 4) for question in (1, 2)
        ]

    asyncio.run(scenario())


def test_empty_generation_is_not_evaluated_and_task_failure_can_be_recorded_individually():
    async def scenario():
        evaluated = []

        async def generate(patient):
            return [] if patient == "empty" else ["bad", "good"]

        async def evaluate(patient, index, task):
            evaluated.append(task)
            if task == "bad":
                return {"status": "failed"}
            return {"status": "completed"}

        outcomes = await run_generation_evaluation(
            ["empty", "patient"], 2, 1, generate, evaluate,
        )
        assert evaluated == ["bad", "good"]
        assert outcomes[0].evaluations == {}
        assert [outcomes[1].evaluations[i]["status"] for i in (0, 1)] == ["failed", "completed"]

    asyncio.run(scenario())


def test_evaluation_limit_allows_two_tasks_from_one_patient_to_start_together():
    async def scenario():
        started = set()
        both_started = asyncio.Event()
        release = asyncio.Event()

        async def generate(patient):
            return ["q1", "q2"]

        async def evaluate(patient, index, task):
            started.add(task)
            if len(started) == 2:
                both_started.set()
            await release.wait()
            return task

        pending = asyncio.create_task(run_generation_evaluation(
            ["patient"], 1, 2, generate, evaluate,
        ))
        await asyncio.wait_for(both_started.wait(), timeout=1)
        assert started == {"q1", "q2"}
        release.set()
        outcomes = await pending
        assert [outcomes[0].evaluations[index] for index in (0, 1)] == ["q1", "q2"]

    asyncio.run(scenario())


def test_cancellation_cancels_and_awaits_generation_and_evaluation_workers():
    async def scenario():
        started = asyncio.Event()
        cleaned = asyncio.Event()

        async def generate(patient):
            return [patient]

        async def evaluate(patient, index, task):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cleaned.set()

        running = asyncio.create_task(run_generation_evaluation(
            ["p1", "p2"], 2, 1, generate, evaluate,
        ))
        await asyncio.wait_for(started.wait(), timeout=1)
        running.cancel()
        with pytest.raises(asyncio.CancelledError):
            await running
        assert cleaned.is_set()

    asyncio.run(scenario())


def test_generation_failure_cleans_up_evaluation_workers():
    async def scenario():
        started = asyncio.Event()
        cleaned = asyncio.Event()

        async def generate(patient):
            if patient == "p2":
                raise RuntimeError("generation failed")
            return [patient]

        async def evaluate(patient, index, task):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cleaned.set()

        with pytest.raises(RuntimeError, match="generation failed"):
            await run_generation_evaluation(["p1", "p2"], 2, 1, generate, evaluate)
        assert started.is_set()
        assert cleaned.is_set()

    asyncio.run(scenario())
