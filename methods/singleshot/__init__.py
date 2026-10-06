"""One-shot task generation with free task selection, without exploration or repair."""

import json

from contracts import (
    EnvironmentConfig,
    PatientTrajectory,
    RunResult,
    RuntimeConfig,
    TaskBatch,
    TaskType,
)
from methods.common import GenerationBudget, output_constraints

from .prompts import GENERATION

__all__ = ["generate_tasks"]


def _trajectory_input(trajectory: PatientTrajectory) -> dict:
    return trajectory.model_dump(mode="json")


async def generate_tasks(
    trajectory: PatientTrajectory,
    environment: EnvironmentConfig,
    task_types: list[TaskType],
    runtime: RuntimeConfig,
) -> RunResult:
    """Select suitable task types for the patient and generate at most two tasks in one call."""
    if environment.name != trajectory.environment:
        raise ValueError("Environment configuration does not match the trajectory")
    constraints = output_constraints(task_types=task_types)
    budget = GenerationBudget(runtime)
    await budget.call_llm(
        "generate",
        instructions=GENERATION + "\n" + constraints,
        input_text=json.dumps(
            {"trajectory": _trajectory_input(trajectory), "task_types": task_types},
            ensure_ascii=False,
        ),
        output_type=TaskBatch,
    )
    return budget.finish(task_types=task_types)
