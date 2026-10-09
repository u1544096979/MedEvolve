"""Evidence verification, quality review, isolated solving and state review."""

import asyncio
import json
import statistics
from datetime import datetime, timezone
from importlib import import_module

from pydantic import BaseModel

from contracts import (
    CompletionReview, ComplexityMetrics, DifficultyJudgment, DifficultyScore,
    EnvironmentConfig, EvaluationConfig, EvaluationResult, PatientTrajectory,
    JointJudgment, QualityDimension, QualityJudgment, QualityReview, RunResult, SolverAnswer,
    SolverInput, Task, Verification,
)
from environments.capabilities import capability_instructions, read_only_scope_instructions
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


def _review_instructions(environment: str, instructions: str) -> str:
    """Prefix reviewer instructions with the read-only scope and environment facts.

    A reviewer may run for a harness that names its own environment outside the
    fixed capability table; the read-only restriction still binds every role, so
    only the environment facts are dropped there.
    """
    try:
        capability = capability_instructions(environment)
    except ValueError:
        capability = read_only_scope_instructions()
    return f"{capability}\n\n{instructions}"


def _completion_material(task: Task, trajectory: PatientTrajectory,
                        solver_run: RunResult, observed_state: dict) -> dict:
    material = _review_material(task, trajectory)
    answer = _output(solver_run, SolverAnswer)
    reviewed = {
        "task": material["task"],
        "evidence_records": [
            source.model_dump() for source in trajectory.sources
            if source.source_id in {evidence.source_id for evidence in task.evidence}
        ],
        "solver_answer": answer.model_dump(),
        # Review execution evidence without replaying the solver's prompts and model requests.
        "solver_trace": [
            item for item in solver_run.trace
            if item.get("type") in {"tool_start", "tool_output", "run_error", "tool_budget"}
        ],
    }
    # The evaluator solves without reading database state; only a caller that
    # observed one adds it, so absent state never reaches the reviewer.
    if observed_state:
        reviewed["observed_state"] = observed_state
    return reviewed


async def review_completion(
    task: Task, trajectory: PatientTrajectory, solver_run: RunResult,
    observed_state: dict, config: EvaluationConfig, *, record=None,
) -> tuple[CompletionReview, RunResult]:
    """Judge one executed run against the frozen task; never re-solve the task.

    The reviewed material is the real harness answer, its filtered execution
    trace and the observed state. Verification and quality results stay with the
    frozen dataset, so changing harness does not regenerate the question or
    replay the fixed solver. The reference answer is held here by the review
    process; `record` receives the model run so callers keep their run history.
    """
    runtime = import_module("runtime")
    if "completion" not in config.runtime.models:
        raise ValueError("missing evaluation model: completion")
    material = _completion_material(task, trajectory, solver_run, observed_state)
    model = config.runtime.models["completion"]
    try:
        run = await runtime.call_llm(
            instructions=_review_instructions(task.environment, prompts.COMPLETION),
            input_text=_json(material), output_type=CompletionReview, model=model,
        )
    except Exception as exc:
        run = RunResult(trace=[], usage={}, status="failed", error=str(exc))
    run = run.model_copy(update={"trace": [
        *run.trace,
        {"type": "model_identity",
         "model_identity": {"role": "completion", "model": model.model}},
    ]})
    if record is not None:
        record(run)
    return _output(run, CompletionReview), run


def _judge_outputs(runs: list[RunResult], schema: type[BaseModel]) -> list[BaseModel]:
    if len(runs) not in {2, JUDGE_COUNT}:
        raise ValueError("judging requires exactly two or three runs")
    return [_output(run, schema) for run in runs]


def _aggregate_quality_judgments(judges: list[QualityJudgment]) -> QualityReview:

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


def _aggregate_difficulty_judgments(judges: list[DifficultyJudgment],
                                 complexity: ComplexityMetrics) -> DifficultyScore:
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


class ScoringError(ValueError):
    """聚合失败时保留本轮所有真实评分运行记录。"""

    def __init__(self, message: str, runs: list[RunResult],
                 quality: QualityReview | None = None,
                 difficulty: DifficultyScore | None = None):
        super().__init__(message)
        self.runs = runs
        self.quality = quality
        self.difficulty = difficulty


async def review_scores(
    task: Task, trajectory: PatientTrajectory, verification: Verification,
    config: EvaluationConfig, *, request_semaphore: asyncio.Semaphore | None = None,
    record=None,
) -> tuple[QualityReview, DifficultyScore, list[RunResult]]:
    """独立审核质量与难度；record 在聚合前收到全部实际模型 runs。"""
    runtime = import_module("runtime")
    material = _review_material(task, trajectory)
    material["verification"] = verification.model_dump()
    complexity = _compute_complexity(task, trajectory)
    joint = config.scoring_mode == "joint"
    count = len(config.judges) or JUDGE_COUNT
    if joint and count != JUDGE_COUNT:
        raise ValueError("joint scoring requires exactly three judges")
    if config.judges:
        judges = config.judges
    elif joint:
        if "quality" not in config.runtime.models:
            raise ValueError("missing evaluation model: quality")
        judges = [config.runtime.models["quality"]] * JUDGE_COUNT
    else:
        judges = [config.runtime.models["quality"] for _ in range(count)]

    async def request(index: int, role: str, schema, instructions: str, payload: dict,
                      model) -> RunResult:
        kwargs = dict(instructions=_review_instructions(task.environment, instructions),
                      input_text=_json(payload), output_type=schema, model=model)
        try:
            if request_semaphore is None:
                run = await runtime.call_llm(**kwargs)
            else:
                async with request_semaphore:
                    run = await runtime.call_llm(**kwargs)
        except Exception as exc:
            run = RunResult(trace=[], usage={}, status="failed", error=str(exc))
        return run.model_copy(update={"trace": [*run.trace, {
            "type": "model_identity", "model_identity": {
                "role": role, "model": model.model, "judge_index": index,
            },
        }]})

    runs: list[RunResult] = []
    quality = None
    difficulty = None
    try:
        if joint:
            runs = await asyncio.gather(*[
                request(index, "quality", JointJudgment, prompts.JOINT_SCORING,
                        {**material, "complexity_metrics": complexity.model_dump(),
                         "judge_index": index}, judges[index])
                for index in range(JUDGE_COUNT)
            ])
            if record is not None:
                for run in runs:
                    record(run)
            judgments = [_output(run, JointJudgment) for run in runs]
            quality = _aggregate_quality_judgments([item.quality for item in judgments])
            difficulty = _aggregate_difficulty_judgments(
                [item.difficulty for item in judgments], complexity)
        else:
            quality_runs = await asyncio.gather(*[
                request(index, "quality", QualityJudgment, prompts.QUALITY, material,
                        judges[index]) for index in range(count)
            ])
            runs.extend(quality_runs)
            if record is not None:
                for run in quality_runs:
                    record(run)
            quality = _aggregate_quality_judgments([
                _output(run, QualityJudgment) for run in quality_runs
            ])
            difficulty_judges = config.judges or [
                config.runtime.models["difficulty"] for _ in range(count)
            ]
            difficulty_runs = await asyncio.gather(*[
                request(index, "difficulty", DifficultyJudgment, prompts.DIFFICULTY,
                        {"task": material["task"],
                         "complexity_metrics": complexity.model_dump()},
                        difficulty_judges[index]) for index in range(count)
            ])
            runs.extend(difficulty_runs)
            if record is not None:
                for run in difficulty_runs:
                    record(run)
            difficulty = _aggregate_difficulty_judgments([
                _output(run, DifficultyJudgment) for run in difficulty_runs
            ], complexity)
        return quality, difficulty, runs
    except Exception as exc:
        if isinstance(exc, ScoringError):
            raise
        raise ScoringError(str(exc), runs, quality, difficulty) from exc


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
        required_roles = ["verifier", "solver", "completion"]
        if not config.judges and config.scoring_mode == "separate":
            required_roles.extend(("quality", "difficulty"))
        for role in required_roles:
            if role not in config.runtime.models:
                raise ValueError(f"missing evaluation model: {role}")
        for role in ("verifier", "solver"):
            if role not in config.runtime.budgets:
                raise ValueError(f"missing evaluation budget: {role}")
        material = _review_material(task, trajectory)

        def role_instructions(instructions: str) -> str:
            return f"{capability_instructions(environment.name)}\n\n{instructions}"

        async def model_run(entry, *, role, schema, instructions, payload, tools=None,
                            model=None, judge_index=None, record=True):
            selected_model = model or config.runtime.models[role]
            kwargs = dict(
                instructions=instructions, input_text=_json(payload),
                output_type=schema, model=selected_model,
            )
            if tools is not None:
                kwargs.update(tools=tools, budget=config.runtime.budgets[role])
            try:
                run = await entry(**kwargs)
            except Exception as exc:
                run = RunResult(
                    trace=[], usage={}, status="failed", error=str(exc)
                )
            identity = {"role": role, "model": selected_model.model}
            if judge_index is not None:
                identity["judge_index"] = judge_index
            run = run.model_copy(update={"trace": [
                *run.trace, {"type": "model_identity", "model_identity": identity}
            ]})
            if record:
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
                            instructions=role_instructions(prompts.EVIDENCE_VERIFIER),
                            payload={"evidence": evidence.model_dump(), "resource": resource},
                        )
                        checked = _output(run, Verification)
                        if not checked.valid:
                            result.verification = checked
                            break
                    else:
                        run = await model_run(
                            runtime.call_llm, role="verifier", schema=Verification,
                            instructions=role_instructions(prompts.VERIFIER),
                            payload={"task": material["task"], "evidence_records": evidence_records},
                        )
                        result.verification = _output(run, Verification)
                        if result.verification.read_only_task is None:
                            raise ValueError("overall verification missing read_only_task")
                        if not result.verification.read_only_task:
                            result.verification = result.verification.model_copy(update={
                                "valid": False,
                                "reason": result.verification.reason +
                                "；任务要求超出只读查询范围",
                            })
                except BaseException:
                    body_failed = True
                    raise
        except BaseException:
            if not body_failed:
                phase = "environment_error"
            raise
        if not result.verification.valid:
            result.status = "invalid"
            if result.verification.read_only_task is False:
                result.failure_category = "unsupported_operation"
            return result

        phase = "model_error"
        material["verification"] = result.verification.model_dump()
        try:
            result.quality, result.difficulty, _ = await review_scores(
                task, trajectory, result.verification, config, record=result.runs.append,
            )
        except ScoringError as exc:
            result.quality = exc.quality
            result.difficulty = exc.difficulty
            raise

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
                        instructions=role_instructions(prompts.SOLVER),
                        payload=SolverInput(
                            instruction=task.instruction, patient_id=task.patient_id,
                            decision_point=task.decision_point,
                        ).model_dump(), tools=tools,
                    )
                except BaseException:
                    body_failed = True
                    raise
        except BaseException:
            if not body_failed:
                phase = "environment_error"
            raise
        phase = "model_error"
        result.completion, _ = await review_completion(
            task, trajectory, result.solver_run, result.observed_state, config,
            record=result.runs.append,
        )
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
