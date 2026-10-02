"""Single-call task generation without exploration, verification or repair."""

import json

from contracts import (
    EnvironmentConfig,
    GenerationDirection,
    PatientTrajectory,
    RunResult,
    RuntimeConfig,
    TaskBatch,
)
from methods.common import GenerationBudget, output_constraints

from .prompts import GENERATION

__all__ = ["generate_tasks"]


def _trajectory_input(trajectory: PatientTrajectory) -> dict:
    return {
        "patient_id": trajectory.patient_id,
        "environment": trajectory.environment,
        "text": trajectory.text,
        "sources": [source.source_id for source in trajectory.sources],
    }


async def generate_tasks(
    trajectory: PatientTrajectory,
    environment: EnvironmentConfig,
    directions: list[GenerationDirection],
    runtime: RuntimeConfig,
) -> RunResult:
    """Generate the two requested positions in one model call."""
    if len(directions) != 2:
        raise ValueError("directions must contain exactly two generation positions")
    if environment.name != trajectory.environment:
        raise ValueError("Environment configuration does not match the trajectory")
    budget = GenerationBudget(runtime)
    await budget.call_llm(
        "generate",
        instructions=GENERATION + "\n" + output_constraints(directions),
        input_text=json.dumps(
            {
                "trajectory": _trajectory_input(trajectory),
                "directions": [direction.model_dump(mode="json") for direction in directions],
            },
            ensure_ascii=False,
        ),
        output_type=TaskBatch,
    )
    return budget.finish(directions)
