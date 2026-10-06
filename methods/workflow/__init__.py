"""Fixed-stage task generation: locate, generate and internally check."""

import json

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
from methods.common import (GenerationBudget, output_constraints, summarize_results,
                            validate_task_batch)

from .prompts import GENERATE_BY_TYPE, INTERNAL_CHECK, LOCATE

__all__ = ["generate_tasks", "FocusSelection", "BatchReview"]


class ClinicalFocus(BaseModel):
    direction_index: int = Field(ge=0, le=4)
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


def _selected_focuses(selection: FocusSelection, directions: list[GenerationDirection],
                      trajectory: PatientTrajectory) -> list[ClinicalFocus]:
    """Validate the two located task types against the candidate pool and trajectory."""
    output_constraints(directions, candidate_pool=True)
    focuses = selection.focuses
    if len(focuses) != 2 or len({focus.direction_index for focus in focuses}) != 2:
        raise ValueError("locate must select two distinct task types")
    known = {source.source_id for source in trajectory.sources}
    for focus in focuses:
        if not 0 <= focus.direction_index < len(directions):
            raise ValueError("focus index is outside candidate pool")
        if not focus.clinical_focus.strip() or not focus.source_ids:
            raise ValueError("focus requires a topic and evidence sources")
        if set(focus.source_ids) - known:
            raise ValueError("focus sources are absent from trajectory")
    return focuses


def _selection_entry(focuses: list[ClinicalFocus]) -> dict:
    """Record the located task types and focuses alongside the raw stage records."""
    return {
        "type": "selection",
        "selected_directions": [focus.direction_index for focus in focuses],
        "focuses": [focus.model_dump(mode="json") for focus in focuses],
    }


def _decisions(review: BatchReview, questions: list[GeneratedQuestion]) -> list[dict]:
    if len(review.reviews) != len(questions):
        raise ValueError("internal check must judge every candidate exactly once")
    by_index = {item.index: item for item in review.reviews}
    if len(by_index) != len(review.reviews) or set(by_index) != set(range(len(questions))):
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


def _finish(budget: GenerationBudget, selected: list[GenerationDirection],
            focuses: list[ClinicalFocus], *, output: TaskBatch | None = None,
            status: str | None = None, error: str | None = None) -> RunResult:
    """Close the run with the selected directions and keep the located selection in the trace."""
    result = budget.finish(selected, output=output, status=status, error=error)
    return result.model_copy(update={"trace": [*result.trace, _selection_entry(focuses)]})


async def generate_tasks(
    trajectory: PatientTrajectory,
    environment: EnvironmentConfig,
    directions: list[GenerationDirection],
    runtime: RuntimeConfig,
) -> RunResult:
    """Run the fixed locate, per-type generate and internal check stages for one patient."""
    output_constraints(directions, candidate_pool=True)
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
        return summarize_results(budget.records, status=located.status, error=f"locate: {located.error}")
    try:
        focuses = _selected_focuses(FocusSelection.model_validate(located.output), directions, trajectory)
    except Exception as exc:
        return summarize_results(budget.records, status="failed",
                                 error=f"locate: {type(exc).__name__}: {exc}")

    selected = [directions[focus.direction_index] for focus in focuses]
    known = {source.source_id for source in trajectory.sources}
    candidates: list[GeneratedQuestion] = []
    for index, (focus, direction) in enumerate(zip(focuses, selected)):
        stage = f"generate_{index}"
        generated = await budget.call_llm(
            stage,
            instructions=f"{GENERATE_BY_TYPE[direction.task_type]}\n"
                         + output_constraints([direction], max_questions=1),
            input_text=json.dumps(
                {
                    "trajectory": _trajectory_input(trajectory),
                    "focus": focus.model_dump(mode="json"),
                    "direction": direction.model_dump(mode="json"),
                },
                ensure_ascii=False,
            ),
            output_type=TaskBatch,
        )
        validated = validate_task_batch(generated, [direction], max_questions=1)
        if validated.status != "completed":
            return _finish(budget, selected, focuses, status=validated.status,
                           error=f"{stage}: {validated.error}")
        batch = TaskBatch.model_validate(validated.output)
        if any(evidence.source_id not in known
               for question in batch.questions for evidence in question.evidence):
            return _finish(budget, selected, focuses, status="failed",
                           error=f"{stage}: unknown evidence source")
        candidates.extend(batch.questions)

    if not candidates:
        return _finish(budget, selected, focuses, output=TaskBatch(questions=[]))

    records = {source.source_id: source.statement for source in trajectory.sources}
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
        return _finish(budget, selected, focuses, status=checked.status, error=f"check: {checked.error}")
    try:
        decisions = _decisions(BatchReview.model_validate(checked.output), candidates)
    except Exception as exc:
        return _finish(budget, selected, focuses, status="failed",
                       error=f"check: {type(exc).__name__}: {exc}")

    kept = [question for question, decision in zip(candidates, decisions) if decision["keep"]]
    result = _finish(budget, selected, focuses, output=TaskBatch(questions=kept))
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
