"""Fixed-stage task generation: locate, generate and internally check."""

import json
import re

from pydantic import BaseModel, Field

from contracts import (
    EnvironmentConfig,
    GeneratedQuestion,
    GenerationDirection,
    PatientTrajectory,
    RunResult,
    RuntimeConfig,
    TaskBatch,
)
from methods.common import GenerationBudget, output_constraints

from .prompts import GENERATE, INTERNAL_CHECK, LOCATE

__all__ = ["generate_tasks", "FocusSelection", "BatchReview"]


class ClinicalFocus(BaseModel):
    direction_index: int = Field(ge=0, le=1)
    clinical_focus: str
    source_ids: list[str]


class FocusSelection(BaseModel):
    focuses: list[ClinicalFocus]


class QuestionReview(BaseModel):
    index: int = Field(ge=0)
    clarity: bool
    answer_alignment: bool
    citations_supported: bool
    reason: str


class BatchReview(BaseModel):
    reviews: list[QuestionReview]


def _trajectory_input(trajectory: PatientTrajectory) -> dict:
    return {
        "patient_id": trajectory.patient_id,
        "environment": trajectory.environment,
        "text": trajectory.text,
        "sources": [{"source_id": source.source_id} for source in trajectory.sources],
    }


def _records(trajectory: PatientTrajectory) -> dict[str, str]:
    """Map source_id to its record body using the trajectory text produced for methods."""
    parts = re.split(r"(?m)^\[[^\]]*\]\s*(\S+)\s*$", trajectory.text)
    return {parts[index]: parts[index + 1].strip()
            for index in range(1, len(parts) - 1, 2)}


def _focuses(selection: FocusSelection) -> list[ClinicalFocus]:
    by_index = {focus.direction_index: focus for focus in selection.focuses}
    if len(by_index) != 2 or set(by_index) != {0, 1}:
        raise ValueError("locate must return exactly one clinical focus per direction index")
    return [by_index[0], by_index[1]]


def _decisions(review: BatchReview, questions: list[GeneratedQuestion]) -> list[dict]:
    by_index = {item.index: item for item in review.reviews}
    if set(by_index) != set(range(len(questions))):
        raise ValueError("internal check must judge every candidate exactly once")
    return [
        {
            "index": index,
            "keep": by_index[index].clarity
            and by_index[index].answer_alignment
            and by_index[index].citations_supported,
            "clarity": by_index[index].clarity,
            "answer_alignment": by_index[index].answer_alignment,
            "citations_supported": by_index[index].citations_supported,
            "reason": by_index[index].reason,
        }
        for index in range(len(questions))
    ]


async def generate_tasks(
    trajectory: PatientTrajectory,
    environment: EnvironmentConfig,
    directions: list[GenerationDirection],
    runtime: RuntimeConfig,
) -> RunResult:
    """Run the fixed locate, generate and internal check stages for one patient."""
    constraints = output_constraints(directions)
    budget = GenerationBudget(runtime)

    located = await budget.call_llm(
        "locate",
        instructions=LOCATE,
        input_text=json.dumps(
            {
                "trajectory": _trajectory_input(trajectory),
                "directions": [direction.model_dump(mode="json") for direction in directions],
            },
            ensure_ascii=False,
        ),
        output_type=FocusSelection,
    )
    if located.status != "completed":
        return budget.finish(directions, status=located.status, error=f"locate:{located.error}")
    try:
        focuses = _focuses(FocusSelection.model_validate(located.output))
    except Exception as exc:
        return budget.finish(directions, status="failed", error=f"locate: {type(exc).__name__}: {exc}")

    generated = await budget.call_llm(
        "generate",
        instructions=f"{GENERATE}\n{constraints}",
        input_text=json.dumps(
            {
                "trajectory": _trajectory_input(trajectory),
                "focuses": [focus.model_dump(mode="json") for focus in focuses],
                "directions": [direction.model_dump(mode="json") for direction in directions],
            },
            ensure_ascii=False,
        ),
        output_type=TaskBatch,
    )
    if generated.status != "completed":
        return budget.finish(directions, status=generated.status, error=f"generate:{generated.error}")
    try:
        candidates = TaskBatch.model_validate(generated.output).questions
    except Exception as exc:
        return budget.finish(directions, status="failed", error=f"generate: {type(exc).__name__}: {exc}")
    if not candidates:
        return budget.finish(directions, output=TaskBatch(questions=[]))
    records = _records(trajectory)

    checked = await budget.call_llm(
        "check",
        instructions=INTERNAL_CHECK,
        input_text=json.dumps(
            {
                "candidates": [
                    {
                        "candidate": question.model_dump(mode="json"),
                        "evidence_records": [
                            {"source_id": evidence.source_id,
                             "record": records.get(evidence.source_id)}
                            for evidence in question.evidence
                        ],
                    }
                    for question in candidates
                ]
            },
            ensure_ascii=False,
        ),
        output_type=BatchReview,
    )
    if checked.status != "completed":
        return budget.finish(directions, status=checked.status, error=f"check:{checked.error}")
    try:
        decisions = _decisions(BatchReview.model_validate(checked.output), candidates)
    except Exception as exc:
        return budget.finish(directions, status="failed", error=f"check: {type(exc).__name__}: {exc}")

    kept = [question for question, decision in zip(candidates, decisions) if decision["keep"]]
    result = budget.finish(directions, output=TaskBatch(questions=kept))
    return result.model_copy(
        update={
            "trace": [
                *result.trace,
                {
                    "type": "internal_check",
                    "decisions": decisions,
                    "kept": [d["index"] for d in decisions if d["keep"]],
                    "dropped": [d["index"] for d in decisions if not d["keep"]],
                },
            ]
        }
    )
