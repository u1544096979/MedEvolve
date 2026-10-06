"""TaskCraft 医疗适配：作者源码的阶段算法接入公共对比实验入口。

源码基线：/Users/guagua/code/TaskCraft 提交 1e405e0b02c4738654e2d885b16677b46c9fe969，
许可与作者归属见 LICENSE-TaskCraft。阶段对应关系与适配差异见 stages.py、prompts.py。
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
from .scheduling import DEFAULT_SEED
from .stages import Pipeline, TaskCraftParameters

__all__ = ["READ_ONLY_TOOLS", "TaskCraftParameters", "bind_generator", "generate_tasks",
           "read_only_tools"]


def _with_pipeline(result: RunResult, pipeline: Pipeline, parameters: TaskCraftParameters,
                   directions: list[GenerationDirection]) -> RunResult:
    entry = {
        "stage": "taskcraft/pipeline",
        "author_commit": AUTHOR_COMMIT,
        "parameters": parameters.model_dump(mode="json"),
        "directions": [direction.model_dump(mode="json") for direction in directions],
        "experiment_seed": pipeline.random_seed,
        "derived_seeds": pipeline.derived_seeds,
        "branch_order": pipeline.branch_order,
        "pool": [{"origin": candidate.origin, "parents": candidate.parents,
                  "locator": candidate.locator,
                  "question": candidate.question.model_dump(mode="json"),
                  "detail": candidate.detail}
                 for candidate in pipeline.pool],
        "selection": {"filtering": pipeline.selection_records,
                      "selected": pipeline.selected},
        "records": pipeline.records,
    }
    return result.model_copy(update={"trace": [*result.trace, entry]})


def bind_generator(random_seed: int):
    """把实验种子绑定到公共四参数入口；``RuntimeConfig`` 不承载方法私有字段。"""
    async def bound(trajectory: PatientTrajectory, environment: EnvironmentConfig,
                    directions: list[GenerationDirection], runtime: RuntimeConfig) -> RunResult:
        return await generate_tasks(trajectory, environment, directions, runtime,
                                    random_seed=random_seed)
    return bound


async def generate_tasks(
    trajectory: PatientTrajectory,
    environment: EnvironmentConfig,
    directions: list[GenerationDirection],
    runtime: RuntimeConfig,
    *,
    parameters: TaskCraftParameters | None = None,
    random_seed: int = DEFAULT_SEED,
) -> RunResult:
    """按 common/SPEC.md 的 generate_tasks 契约生成任务；阶段规模由 parameters 显式配置。

    公共调用保持四个位置参数；``random_seed`` 是实验入口绑定的可选关键字。
    """
    if len(directions) != 2:
        raise ValueError("directions must contain exactly two generation positions")
    settings = parameters or TaskCraftParameters()
    budget = GenerationBudget(runtime)
    ledger = Ledger(budget)
    pipeline = Pipeline(trajectory, [], directions, settings, ledger, random_seed)
    if trajectory.environment != environment.name:
        result = budget.finish(
            directions, status="failed",
            error=(f"Trajectory environment {trajectory.environment!r} does not match "
                   f"the experiment environment {environment.name!r}"))
        return _with_pipeline(result, pipeline, settings, directions)
    instance = None
    try:
        instance = await create_environment(environment)
        tools = bind_generation_tools(await get_tools(instance, [trajectory.patient_id]))
        pipeline = Pipeline(trajectory, tools, directions, settings, ledger, random_seed)
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
