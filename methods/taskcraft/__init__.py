"""TaskCraft 医疗适配：作者源码的阶段算法接入公共对比实验入口。

源码基线：/Users/guagua/code/TaskCraft 提交 1e405e0b02c4738654e2d885b16677b46c9fe969，
MIT 许可，声明见 LICENSE-TaskCraft。阶段对应关系与适配差异见 stages.py、prompts.py。
"""

from contracts import (
    EnvironmentConfig, GenerationDirection, PatientTrajectory, RunResult, RuntimeConfig,
    TaskBatch,
)
from environments import create_environment, get_tools, release_environment
from methods.common import GenerationBudget

from .environment import READ_ONLY_TOOLS, bind_generation_tools, read_only_tools
from .ledger import Ledger
from .prompts import AUTHOR_COMMIT
from .stages import Pipeline, TaskCraftParameters

__all__ = ["READ_ONLY_TOOLS", "TaskCraftParameters", "generate_tasks", "read_only_tools"]


def _with_pipeline(result: RunResult, pipeline: Pipeline, parameters: TaskCraftParameters,
                   directions: list[GenerationDirection]) -> RunResult:
    entry = {
        "stage": "taskcraft/pipeline",
        "author_commit": AUTHOR_COMMIT,
        "parameters": parameters.model_dump(mode="json"),
        "directions": [direction.model_dump(mode="json") for direction in directions],
        "pool": [{"origin": candidate.origin, "task_type": candidate.question.task_type,
                  "instruction": candidate.question.instruction, "parents": candidate.parents,
                  "locator": candidate.locator,
                  "evidence": [item.source_id for item in candidate.question.evidence]}
                 for candidate in pipeline.pool],
        "records": pipeline.records,
    }
    return result.model_copy(update={"trace": [*result.trace, entry]})


async def generate_tasks(
    trajectory: PatientTrajectory,
    environment: EnvironmentConfig,
    directions: list[GenerationDirection],
    runtime: RuntimeConfig,
    *,
    parameters: TaskCraftParameters | None = None,
) -> RunResult:
    """按 common/SPEC.md 的 generate_tasks 契约生成任务；阶段规模由 parameters 显式配置。"""
    if len(directions) != 2:
        raise ValueError("directions must contain exactly two generation positions")
    settings = parameters or TaskCraftParameters()
    budget = GenerationBudget(runtime)
    ledger = Ledger(budget)
    instance = None
    pipeline = Pipeline(trajectory, [], directions, settings, ledger)
    try:
        instance = await create_environment(environment)
        tools = bind_generation_tools(await get_tools(instance, [trajectory.patient_id]))
        pipeline = Pipeline(trajectory, tools, directions, settings, ledger)
        await pipeline.run()
        if ledger.exhausted:
            result = budget.finish(directions, status="budget_exhausted",
                                   error="Cumulative generator budget exhausted in TaskCraft stages")
        elif not pipeline.pool:
            result = budget.finish(directions, status="failed",
                                   error=pipeline.failure or "TaskCraft produced no task")
        else:
            selected = pipeline.select()
            # 阶段失败已在 trace 中记录；有候选即按公共约束校验后返回，末次调用失败不改写整批状态。
            result = budget.finish(directions, status="completed", output=TaskBatch(
                questions=[candidate.question for candidate in selected]))
            if result.status == "completed":
                # 完成批次的失败调用保留在 trace，不在运行结果上重复报错。
                result = result.model_copy(update={"error": None})
    except Exception as exc:
        result = budget.finish(directions, status="failed", error=f"{type(exc).__name__}: {exc}")
    finally:
        if instance is not None:
            await release_environment(environment, instance)
    return _with_pipeline(result, pipeline, settings, directions)
