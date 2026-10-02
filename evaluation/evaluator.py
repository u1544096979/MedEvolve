"""Evidence verification, quality review, isolated solving and state review."""

import json
from importlib import import_module

from pydantic import BaseModel

from contracts import (
    CompletionReview, EnvironmentConfig, EvaluationConfig, EvaluationResult,
    PatientTrajectory, QualityReview, RunResult, SolverAnswer, SolverInput, Task,
    Verification,
)
from . import prompts

QUALITY_DIMENSIONS = {
    "clinical_value", "correctness", "longitudinal_integration", "clarity"
}


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


async def evaluate_task(task: Task, trajectory: PatientTrajectory,
                        environment: EnvironmentConfig,
                        config: EvaluationConfig) -> EvaluationResult:
    """Evaluate one task using fixed models and separate seeded instances."""
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
        for role in ("verifier", "quality", "solver", "completion"):
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
        instance = await env.create_environment(environment)
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
        finally:
            phase_before_release = phase
            phase = "environment_error"
            await env.release_environment(environment, instance)
            phase = phase_before_release
        if not result.verification.valid:
            result.status = "invalid"
            return result

        phase = "model_error"
        material["verification"] = result.verification.model_dump()
        run = await model_run(
            runtime.call_llm, role="quality", schema=QualityReview,
            instructions=prompts.QUALITY, payload=material,
        )
        quality = _output(run, QualityReview)
        if set(quality.dimensions) != QUALITY_DIMENSIONS:
            raise ValueError("quality review must contain the four required dimensions")
        for dimension in quality.dimensions.values():
            if not isinstance(dimension, dict) or any(
                not isinstance(dimension.get(key), str) or not dimension[key].strip()
                for key in ("judgment", "reason")
            ):
                raise ValueError("each quality dimension requires judgment and reason")
        result.quality = quality

        phase = "environment_error"
        instance = await env.create_environment(environment)
        try:
            tools = await env.get_tools(
                instance, [task.patient_id], decision_point=task.decision_point
            )
            phase = "model_error"
            result.solver_run = await model_run(
                runtime.run_agent, role="solver", schema=SolverAnswer,
                instructions=prompts.SOLVER,
                payload=SolverInput(
                    instruction=task.instruction, patient_id=task.patient_id,
                    decision_point=task.decision_point,
                ).model_dump(), tools=tools,
            )
            phase = "environment_error"
            result.observed_state = await env.read_state(instance, task)
        finally:
            phase_before_release = phase
            phase = "environment_error"
            await env.release_environment(environment, instance)
            phase = phase_before_release
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
        result.failure_category = (
            run_error.get("failure_category")
            if run_error and run_error.get("failure_category") == "tool_error"
            else phase
        )
        # Preserve environment/configuration errors alongside model run records.
        result.observed_state = {
            **result.observed_state, "evaluation_error": str(exc)
        }
        return result
