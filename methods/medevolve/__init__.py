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
from methods.common import USAGE_FIELDS, normalize_usage, output_constraints, validate_task_batch
from methods.generation import trajectory_payload
from .compression import COMPRESSION, CompressedTrajectory, compression_input

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
    return trajectory_payload(trajectory)


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
    compression_runs = []

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

    @function_tool
    async def compress_trajectory(patient_id: str) -> str:
        """压缩本次开放患者的缓存轨迹；必须指定患者 ID，执行一次模型调用。"""
        if patient_id not in cache:
            return json.dumps({"error": "patient_id is outside the supplied trajectory cache"})
        model = runtime.models.get("trajectory_compressor")
        if model is None:
            return json.dumps({"error": "runtime.models.trajectory_compressor is not configured"})
        try:
            input_text = compression_input(cache[patient_id], model)
        except ValueError as exc:
            return json.dumps({"error": str(exc)})
        run = await call_llm(instructions=COMPRESSION, input_text=input_text,
                             output_type=CompressedTrajectory, model=model)
        compression_runs.append({"patient_id": patient_id, "result": run.model_dump(mode="json")})
        if run.status != "completed" or run.output is None:
            return json.dumps({"error": run.error or run.status})
        return json.dumps({"patient_id": patient_id,
                           "input_truncated": json.loads(input_text)["input_truncated"],
                           "summary": CompressedTrajectory.model_validate(run.output).summary},
                          ensure_ascii=False)

    result = await run_agent(
        instructions=DATA_EXPLORATION,
        input_text="请先列出开放患者，再读取缓存轨迹，归纳抽象数据指导。",
        tools=[list_patients, read_trajectory, compress_trajectory],
        output_type=DataExploration,
        model=runtime.models["data_explorer"],
        budget=runtime.budgets["data_explorer"],
    )
    if not compression_runs:
        return result
    usages = [normalize_usage(result.usage),
              *(normalize_usage(run["result"]["usage"]) for run in compression_runs)]
    usage = {**result.usage, **{field: (sum(u[field] or 0 for u in usages)
                                    if any(u[field] is not None for u in usages) else None)
                              for field in USAGE_FIELDS}}
    usage["runs_without_known_tokens"] = sum(
        u.get("runs_without_known_tokens", int(u["total_tokens"] is None)) for u in usages
    )
    return result.model_copy(update={"usage": usage,
        "trace": [*result.trace, *({"type": "trajectory_compression", **run} for run in compression_runs)]})


async def explore_environment(
    config: EnvironmentConfig, patient_ids: list[str], runtime: RuntimeConfig,
) -> RunResult:
    instance = None
    result = None
    release_error = None
    try:
        instance = await create_environment(config)
        tools = await get_tools(instance, patient_ids,
                                tool_result_limit=config.tool_result_limit)
        result = await run_agent(
            instructions=ENVIRONMENT_EXPLORATION,
            input_text=json.dumps({"environment": instance.environment, "patient_ids": patient_ids},
                                  ensure_ascii=False),
            tools=tools, output_type=EnvironmentExploration,
            model=runtime.models["environment_explorer"],
            budget=runtime.budgets["environment_explorer"],
        )
    except Exception as exc:
        result = RunResult(trace=[{"type": "run_error", "failure_category": "environment_error",
                                  "error": str(exc)}], usage={"model_turns": 0, "tool_calls": 0},
                           status="failed", error=f"{type(exc).__name__}: {exc}")
    finally:
        if instance is not None:
            try:
                await release_environment(config, instance)
            except Exception as exc:
                release_error = f"{type(exc).__name__}: {exc}"
    if release_error is not None:
        result = result.model_copy(update={
            "output": None, "status": "failed", "error": "release_environment failed: " + release_error,
            "trace": [*result.trace, {"type": "environment_release", "released": False,
                "instance_id": instance.instance_id, "error": release_error,
                "rejected_output": result.output}],
        })
    return result


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
