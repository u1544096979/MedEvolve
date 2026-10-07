import json

from contracts import (
    EnvironmentConfig,
    EnvironmentExploration,
    GenerationDirection,
    PatientTrajectory,
    RunResult,
    RuntimeConfig,
    Strategy,
    TaskBatch,
)
from methods.common import GenerationBudget, output_constraints, validate_task_batch
from methods.generation import trajectory_payload

from .prompts import GENERATION

__all__ = ["generate_tasks"]


async def generate_tasks(
    trajectory: PatientTrajectory,
    environment: EnvironmentConfig,
    directions: list[GenerationDirection],
    runtime: RuntimeConfig,
    *,
    environment_guidance: EnvironmentExploration,
    strategy: Strategy,
) -> RunResult:
    output_constraints(directions)
    if trajectory.environment != environment.name:
        raise ValueError("Patient trajectory and environment do not match")
    if not environment_guidance.guidance.strip():
        raise ValueError("environment guidance must not be empty")
    budget = GenerationBudget(runtime, environment.name)
    run = await budget.call_llm(
        "generate",
        instructions=GENERATION + "\n" + output_constraints(directions),
        input_text=json.dumps(
            {
                "trajectory": trajectory_payload(trajectory),
                "guidance": {"environment_guidance": environment_guidance.guidance},
                "strategy": {"generation_guidance": strategy.generation_guidance},
                "directions": [direction.model_dump(mode="json") for direction in directions],
            },
            ensure_ascii=False,
        ),
        output_type=TaskBatch,
    )
    checked = validate_task_batch(run, directions)
    if checked.status != "completed":
        return budget.finish(directions, status=checked.status, error=checked.error).model_copy(
            update={"trace": [{"stage": "generate", "result": checked.model_dump(mode="json")}]}
        )
    known = {evidence.source_id for evidence in trajectory.sources}
    kept, rejected = [], []
    for question in TaskBatch.model_validate(checked.output).questions:
        problems = []
        if not question.evidence:
            problems.append("Task has no evidence references")
        missing = sorted({evidence.source_id for evidence in question.evidence} - known)
        if missing:
            problems.append("Evidence sources are absent from trajectory: " + ", ".join(missing))
        if problems:
            rejected.append({"question": question.model_dump(mode="json"), "problems": problems})
        else:
            kept.append(question)
    result = budget.finish(directions, output=TaskBatch(questions=kept))
    if rejected:
        result = result.model_copy(
            update={"trace": [*result.trace, {"type": "evidence_references_rejected", "questions": rejected}]}
        )
    return result
