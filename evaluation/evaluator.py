"""Evidence verification, quality review, isolated solving and state review."""

import json
import statistics
from datetime import datetime, timezone
from importlib import import_module

from pydantic import BaseModel

from contracts import (
    CompletionReview, ComplexityMetrics, DifficultyJudgment, DifficultyScore,
    EnvironmentConfig, EvaluationConfig, EvaluationResult, PatientTrajectory,
    QualityDimension, QualityJudgment, QualityReview, RunResult, SolverAnswer,
    SolverInput, Task, Verification,
)
from environments.capabilities import read_only_scope_instructions
from . import prompts
from .session import EvaluationEnvironments

JUDGE_COUNT = 3
TIME_FIELDS = (
    "effectiveDateTime", "onsetDateTime", "performedDateTime", "occurrenceDateTime",
    "authoredOn", "sent", "start", "date", "recordedDate", "issued",
)
PERIOD_FIELDS = ("effectivePeriod", "onsetPeriod", "performedPeriod", "period")


def _review_material(task: Task, trajectory: PatientTrajectory) -> dict:
    # Generation provenance is deliberately excluded from reviewer inputs.
    return {
        "task": task.model_dump(include={
            "instruction", "task_type", "reference_answer", "evidence",
            "decision_point", "patient_id", "environment",
        }),
        "trajectory": {
            "patient_id": trajectory.patient_id,
            "environment": trajectory.environment,
            "text": trajectory.text,
            "sources": [{"source_id": source.source_id} for source in trajectory.sources],
        },
    }


def _json(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False)


def _output(run: RunResult, schema: type[BaseModel]) -> BaseModel:
    if run.status != "completed" or run.output is None:
        raise ValueError(run.error or f"model run {run.status}")
    return schema.model_validate(run.output)


def _judge_outputs(runs: list[RunResult], schema: type[BaseModel]) -> list[BaseModel]:
    if len(runs) != JUDGE_COUNT:
        raise ValueError(f"judging requires exactly {JUDGE_COUNT} runs")
    return [_output(run, schema) for run in runs]


def _aggregate_quality_scores(runs: list[RunResult]) -> QualityReview:
    judges = _judge_outputs(runs, QualityJudgment)

    def aggregate(dimension: str) -> QualityDimension:
        judgments = [getattr(judge, dimension) for judge in judges]
        scores = [judgment.score for judgment in judgments]
        for judgment in judgments:
            if not judgment.reason.strip():
                raise ValueError(f"{dimension}.reason must not be empty")
        for score in scores:
            if not 1 <= score <= 5:
                raise ValueError(f"{dimension}.score must be 1-5")
        return QualityDimension(
            score=int(statistics.median(scores)), scores=scores,
            reason=" | ".join(f"J{index}: {judgment.reason}"
                              for index, judgment in enumerate(judgments, start=1)),
        )

    return QualityReview(
        factual_accuracy=aggregate("factual_accuracy"),
        clinical_soundness=aggregate("clinical_soundness"),
        evidence_sufficiency=aggregate("evidence_sufficiency"),
        task_clarity=aggregate("task_clarity"),
    )


def _aggregate_difficulty_scores(runs: list[RunResult],
                                 complexity: ComplexityMetrics) -> DifficultyScore:
    judges = _judge_outputs(runs, DifficultyJudgment)
    depths = [judge.reasoning_depth for judge in judges]
    for judge in judges:
        if not judge.reason.strip():
            raise ValueError("reason must not be empty")
    for depth in depths:
        if not 1 <= depth <= 4:
            raise ValueError("reasoning_depth must be 1-4")
    return DifficultyScore(
        reasoning_depth=int(statistics.median(depths)), reasoning_depths=depths,
        reason=" | ".join(f"J{index}: {judge.reason}"
                          for index, judge in enumerate(judges, start=1)),
        complexity_metrics=complexity,
    )


def _parse_moment(value) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)


def _statement_time(statement: str) -> datetime | None:
    try:
        resource = json.loads(statement)
    except (TypeError, ValueError):
        return None
    if not isinstance(resource, dict):
        return None
    for field in TIME_FIELDS:
        parsed = _parse_moment(resource.get(field))
        if parsed is not None:
            return parsed
    for field in PERIOD_FIELDS:
        period = resource.get(field)
        if isinstance(period, dict):
            parsed = _parse_moment(period.get("start"))
            if parsed is not None:
                return parsed
    context = resource.get("context")
    if isinstance(context, dict):
        period = context.get("period")
        if isinstance(period, dict):
            return _parse_moment(period.get("start"))
    return None


def _compute_complexity(task: Task, trajectory: PatientTrajectory) -> ComplexityMetrics:
    source_ids = {evidence.source_id for evidence in task.evidence}
    resource_types = {source_id.split("/", 1)[0]
                      for source_id in source_ids if "/" in source_id}
    moments = []
    for source in trajectory.sources:
        if source.source_id not in source_ids:
            continue
        moment = _statement_time(source.statement)
        if moment is not None:
            moments.append(moment)
    span = None
    if moments:
        span = (max(moments) - min(moments)).total_seconds() / 86400
    return ComplexityMetrics(
        evidence_count=len(source_ids), resource_type_count=len(resource_types),
        time_span_days=span,
    )


async def evaluate_task(task: Task, trajectory: PatientTrajectory,
                        environment: EnvironmentConfig,
                        config: EvaluationConfig,
                        environments: EvaluationEnvironments | None = None) -> EvaluationResult:
    """Evaluate one task.

    ``environments`` carries reusable instances and the container budget of the run.
    Without it the call stays self-contained and releases its shared instance on return.
    """
    session = environments if environments is not None else EvaluationEnvironments()
    try:
        return await _evaluate_task(task, trajectory, environment, config, session)
    finally:
        if environments is None:
            await session.close()


async def _evaluate_task(task: Task, trajectory: PatientTrajectory,
                         environment: EnvironmentConfig,
                         config: EvaluationConfig,
                         session: EvaluationEnvironments) -> EvaluationResult:
    """Evidence verification, quality review, isolated solving and state review."""
    result = EvaluationResult(
        task_id=task.id, observed_state={}, status="failed", runs=[]
    )
    phase = "environment_error"
    try:
        if (task.patient_id != trajectory.patient_id
                or task.environment != trajectory.environment
                or task.environment != environment.name):
            raise ValueError("task, trajectory and environment must match")
        source_ids = {source.source_id for source in trajectory.sources}
        if not task.evidence:
            result.verification = Verification(
                valid=False, reason="Task has no evidence references"
            )
            result.status = "invalid"
            result.failure_category = "insufficient_evidence"
            return result
        missing_sources = sorted(
            {evidence.source_id for evidence in task.evidence} - source_ids
        )
        if missing_sources:
            result.verification = Verification(
                valid=False,
                reason=f"Evidence sources are absent from trajectory: {', '.join(missing_sources)}",
            )
            result.status = "invalid"
            result.failure_category = "insufficient_evidence"
            return result
        env = import_module("environments")
        phase = "model_error"
        runtime = import_module("runtime")
        for role in ("verifier", "quality", "difficulty", "solver", "completion"):
            if role not in config.runtime.models:
                raise ValueError(f"missing evaluation model: {role}")
        for role in ("verifier", "solver"):
            if role not in config.runtime.budgets:
                raise ValueError(f"missing evaluation budget: {role}")
        material = _review_material(task, trajectory)

        async def model_run(entry, *, role, schema, instructions, payload, tools=None):
            kwargs = dict(
                instructions=instructions, input_text=_json(payload),
                output_type=schema, model=config.runtime.models[role],
            )
            if tools is not None:
                kwargs.update(tools=tools, budget=config.runtime.budgets[role])
            try:
                run = await entry(**kwargs)
            except Exception as exc:
                run = RunResult(
                    trace=[], usage={}, status="failed", error=str(exc)
                )
            result.runs.append(run)
            return run

        phase = "environment_error"
        # 实例获取与释放失败归为环境故障；正文故障保留正文设置的阶段。
        body_failed = False
        try:
            async with session.instance(environment, phase="verify") as instance:
                try:
                    # Verification may inspect the full timeline, including future evidence.
                    evidence_records = []
                    for evidence in task.evidence:
                        phase = "environment_error"
                        resource = await env.read_evidence(
                            instance, task.patient_id, evidence.source_id
                        )
                        evidence_records.append({"source_id": evidence.source_id, "resource": resource})
                        phase = "model_error"
                        run = await model_run(
                            runtime.call_llm, role="verifier", schema=Verification,
                            instructions=prompts.EVIDENCE_VERIFIER,
                            payload={"evidence": evidence.model_dump(), "resource": resource},
                        )
                        checked = _output(run, Verification)
                        if not checked.valid:
                            result.verification = checked
                            break
                    else:
                        run = await model_run(
                            runtime.call_llm, role="verifier", schema=Verification,
                            instructions=prompts.VERIFIER,
                            payload={"task": material["task"], "evidence_records": evidence_records},
                        )
                        result.verification = _output(run, Verification)
                except BaseException:
                    body_failed = True
                    raise
        except BaseException:
            if not body_failed:
                phase = "environment_error"
            raise
        if not result.verification.valid:
            result.status = "invalid"
            return result

        phase = "model_error"
        material["verification"] = result.verification.model_dump()
        result.quality = _aggregate_quality_scores([
            await model_run(runtime.call_llm, role="quality", schema=QualityJudgment,
                            instructions=prompts.QUALITY, payload=material)
            for _ in range(JUDGE_COUNT)
        ])
        complexity = _compute_complexity(task, trajectory)
        result.difficulty = _aggregate_difficulty_scores([
            await model_run(runtime.call_llm, role="difficulty", schema=DifficultyJudgment,
                            instructions=prompts.DIFFICULTY,
                            payload={"task": material["task"],
                                     "complexity_metrics": complexity.model_dump()})
            for _ in range(JUDGE_COUNT)
        ], complexity)

        phase = "environment_error"
        body_failed = False
        try:
            async with session.instance(environment, phase="solve") as instance:
                try:
                    tools = await env.get_tools(
                        instance, [task.patient_id], decision_point=task.decision_point,
                        tool_result_limit=environment.tool_result_limit,
                        read_only=True,
                    )
                    phase = "model_error"
                    result.solver_run = await model_run(
                        runtime.run_agent, role="solver", schema=SolverAnswer,
                        instructions=read_only_scope_instructions() + "\n\n" + prompts.SOLVER,
                        payload=SolverInput(
                            instruction=task.instruction, patient_id=task.patient_id,
                            decision_point=task.decision_point,
                        ).model_dump(), tools=tools,
                    )
                    phase = "environment_error"
                    result.observed_state = await env.read_state(instance, task)
                except BaseException:
                    body_failed = True
                    raise
        except BaseException:
            if not body_failed:
                phase = "environment_error"
            raise
        phase = "model_error"
        answer = _output(result.solver_run, SolverAnswer)
        # Review execution evidence without replaying the solver's prompts and model requests.
        completion_material = {
            "task": material["task"],
            "evidence_records": [
                source.model_dump() for source in trajectory.sources
                if source.source_id in {evidence.source_id for evidence in task.evidence}
            ],
            "solver_answer": answer.model_dump(),
            "solver_trace": [
                item for item in result.solver_run.trace
                if item.get("type") in {"tool_start", "tool_output", "run_error", "tool_budget"}
            ],
            "observed_state": result.observed_state,
        }
        run = await model_run(
            runtime.call_llm, role="completion", schema=CompletionReview,
            instructions=prompts.COMPLETION, payload=completion_material,
        )
        result.completion = _output(run, CompletionReview)
        result.status = "completed"
        if not result.completion.completed:
            result.failure_category = "clinical"
        return result
    except Exception as exc:
        result.status = "failed"
        failed_run = result.runs[-1] if result.runs else None
        run_error = next(
            (item for item in reversed(failed_run.trace)
             if item.get("type") == "run_error"),
            None,
        ) if failed_run is not None else None
        supported_run_categories = {"tool_error", "environment_error", "model_error"}
        run_category = run_error.get("failure_category") if run_error else None
        result.failure_category = (
            run_category if run_category in supported_run_categories else phase
        )
        # Preserve environment/configuration errors alongside model run records.
        result.observed_state = {
            **result.observed_state, "evaluation_error": str(exc)
        }
        return result
