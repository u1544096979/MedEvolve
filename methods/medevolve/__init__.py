"""Initial exploration and single-call task evolution."""

import json

from agents import function_tool

from contracts import (
    DataExploration,
    EnvironmentConfig,
    EnvironmentExploration,
    EvaluationResult,
    ExplorationGuidance,
    Feedback,
    GenerationDirection,
    PatientTrajectory,
    RunResult,
    RuntimeConfig,
    Strategy,
    Task,
    TaskBatch,
)
from environments import create_environment, get_tools, release_environment
from runtime import call_llm, run_agent
from methods.common import output_constraints, validate_task_batch

from .prompts import (
    DATA_EXPLORATION,
    ENVIRONMENT_EXPLORATION,
    FEEDBACK,
    GENERATION,
    STRATEGY_UPDATE,
)

__all__ = [
    "explore_data",
    "explore_environment",
    "generate_tasks",
    "analyze_feedback",
    "update_strategy",
]


def _trajectory_input(trajectory: PatientTrajectory) -> dict:
    return {
        "patient_id": trajectory.patient_id,
        "environment": trajectory.environment,
        "text": trajectory.text,
        "sources": [source.model_dump(mode="json") for source in trajectory.sources],
    }


def _evaluation_input(evaluation: EvaluationResult) -> dict:
    payload = evaluation.model_dump(mode="json", exclude={"runs", "solver_run"})
    if evaluation.solver_run is not None:
        payload["solver_run"] = evaluation.solver_run.model_dump(mode="json", exclude={"trace"})
        payload["solver_run"]["trace"] = [
            item for item in evaluation.solver_run.trace
            if item.get("type") not in {"model_request", "model_response"}
        ]
    return payload


async def explore_data(
    trajectories: list[PatientTrajectory], runtime: RuntimeConfig
) -> RunResult:
    cache = {trajectory.patient_id: trajectory for trajectory in trajectories}

    @function_tool(failure_error_function=None)
    async def list_patients() -> list[str]:
        """列出本次数据探索开放的缓存患者 ID。"""
        return list(cache)

    @function_tool(failure_error_function=None)
    async def read_trajectory(patient_id: str) -> str:
        """读取本次开放患者的完整缓存轨迹与原始证据引用。"""
        if patient_id not in cache:
            raise ValueError("patient_id is outside the supplied trajectory cache")
        return json.dumps(_trajectory_input(cache[patient_id]), ensure_ascii=False)

    return await run_agent(
        instructions=DATA_EXPLORATION,
        input_text="请先列出开放患者，再读取缓存轨迹，归纳抽象数据指导。",
        tools=[list_patients, read_trajectory],
        output_type=DataExploration,
        model=runtime.models["data_explorer"],
        budget=runtime.budgets["data_explorer"],
    )


async def explore_environment(
    config: EnvironmentConfig, patient_ids: list[str], runtime: RuntimeConfig
) -> RunResult:
    instance = await create_environment(config)
    try:
        tools = await get_tools(instance, patient_ids)
        return await run_agent(
            instructions=ENVIRONMENT_EXPLORATION,
            input_text=json.dumps(
                {"environment": instance.environment, "patient_ids": patient_ids},
                ensure_ascii=False,
            ),
            tools=tools,
            output_type=EnvironmentExploration,
            model=runtime.models["environment_explorer"],
            budget=runtime.budgets["environment_explorer"],
        )
    finally:
        await release_environment(config, instance)


async def generate_tasks(
    trajectory: PatientTrajectory,
    guidance: ExplorationGuidance,
    strategy: Strategy,
    directions: list[GenerationDirection],
    runtime: RuntimeConfig,
) -> RunResult:
    if len(directions) != 2:
        raise ValueError("directions must contain exactly two generation positions")
    result = await call_llm(
        instructions=GENERATION + "\n" + output_constraints(directions),
        input_text=json.dumps(
            {
                "trajectory": _trajectory_input(trajectory),
                "guidance": guidance.model_dump(mode="json"),
                "strategy": {"generation_guidance": strategy.generation_guidance},
                "directions": [direction.model_dump(mode="json") for direction in directions],
            },
            ensure_ascii=False,
        ),
        output_type=TaskBatch,
        model=runtime.models["generator"],
    )
    return validate_task_batch(result, directions)


async def analyze_feedback(
    tasks: list[Task],
    trajectories: list[PatientTrajectory],
    evaluations: list[EvaluationResult],
    runtime: RuntimeConfig,
) -> RunResult:
    return await call_llm(
        instructions=FEEDBACK,
        input_text=json.dumps(
            {
                "tasks": [task.model_dump(mode="json") for task in tasks],
                "trajectories": [_trajectory_input(trajectory) for trajectory in trajectories],
                "evaluations": [_evaluation_input(evaluation) for evaluation in evaluations],
            },
            ensure_ascii=False,
        ),
        output_type=Feedback,
        model=runtime.models["feedback"],
    )


async def update_strategy(
    strategy: Strategy, feedback: Feedback, runtime: RuntimeConfig
) -> RunResult:
    return await call_llm(
        instructions=STRATEGY_UPDATE,
        input_text=json.dumps(
            {
                "strategy": strategy.model_dump(mode="json"),
                "feedback": feedback.model_dump(mode="json"),
            },
            ensure_ascii=False,
        ),
        output_type=Strategy,
        model=runtime.models["updater"],
    )
