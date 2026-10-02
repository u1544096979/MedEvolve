"""Self-generated tasks with independent self-answering and consistency filtering."""

import json

from pydantic import BaseModel

from contracts import (
    EnvironmentConfig,
    GenerationDirection,
    PatientTrajectory,
    RunResult,
    RuntimeConfig,
    TaskBatch,
)
from methods.common import GenerationBudget, output_constraints

from .prompts import CANDIDATE_GENERATION, CONSISTENCY_CHECK, SELF_ANSWER

__all__ = ["generate_tasks"]

SELF_ANSWER_POSITIONS = 2


class SelfAnswer(BaseModel):
    answer: str


class AnswerConsistency(BaseModel):
    consistent: bool
    reason: str


def _patient_material(trajectory: PatientTrajectory) -> dict:
    """One fixed material layout, shared by candidate generation and every self-answer."""
    return {
        "patient_id": trajectory.patient_id,
        "environment": trajectory.environment,
        "text": trajectory.text,
        "source_ids": [source.source_id for source in trajectory.sources],
    }


def _question_material(question) -> dict:
    return {"instruction": question.instruction, "task_type": question.task_type}


def _self_answer_input(question, trajectory: PatientTrajectory) -> str:
    """Self-answer input holds the question and the shared patient material only."""
    return json.dumps(
        {"question": _question_material(question), "patient": _patient_material(trajectory)},
        ensure_ascii=False,
    )


def _record_decision(budget: GenerationBudget, stage: str, decision: dict) -> None:
    # Filtering judgments are method records rather than model calls, so their cost stays zero.
    budget.records.append(
        (stage, RunResult(
            trace=[{"type": "consistency_filter", **decision}],
            usage={"model_turns": 0, "tool_calls": 0, "input_tokens": 0, "output_tokens": 0,
                   "total_tokens": 0},
            status="completed"))
    )


async def _check_candidate(budget: GenerationBudget, index: int, question,
                           trajectory: PatientTrajectory):
    """Returns the kept question, or (None, blocking run) when the candidate cannot be kept."""
    answers = [question.reference_answer]
    for position in range(SELF_ANSWER_POSITIONS):
        run = await budget.call_llm(
            f"self_answer/{index}/{position}", instructions=SELF_ANSWER,
            input_text=_self_answer_input(question, trajectory), output_type=SelfAnswer,
        )
        if run.status != "completed":
            return None, run
        answers.append(SelfAnswer.model_validate(run.output).answer)
    run = await budget.call_llm(
        f"consistency/{index}", instructions=CONSISTENCY_CHECK,
        input_text=json.dumps(
            {"question": _question_material(question),
             "answers": [{"position": position + 1, "answer": answer}
                         for position, answer in enumerate(answers)]},
            ensure_ascii=False,
        ),
        output_type=AnswerConsistency,
    )
    if run.status != "completed":
        return None, run
    verdict = AnswerConsistency.model_validate(run.output)
    _record_decision(budget, f"filter/{index}", {
        "candidate_index": index, "kept": verdict.consistent, "reason": verdict.reason,
        "candidate_answer": answers[0], "self_answers": answers[1:],
        "evidence_verification": "pending public verification",
    })
    return (question if verdict.consistent else None), None


async def generate_tasks(
    trajectory: PatientTrajectory,
    environment: EnvironmentConfig,
    directions: list[GenerationDirection],
    runtime: RuntimeConfig,
) -> RunResult:
    """Generate candidates once, self-answer twice per candidate, keep the consistent ones.

    The method has no environment tools and no cross-round state; the environment
    argument is part of the shared generation entry.
    """
    budget = GenerationBudget(runtime)
    generation = await budget.call_llm(
        "candidate_generation",
        instructions=CANDIDATE_GENERATION + "\n" + output_constraints(directions),
        input_text=json.dumps(
            {"patient": _patient_material(trajectory),
             "directions": [direction.model_dump(mode="json") for direction in directions]},
            ensure_ascii=False,
        ),
        output_type=TaskBatch,
    )
    if generation.status != "completed":
        return budget.finish(directions, status=generation.status, error=generation.error)
    kept = []
    for index, question in enumerate(TaskBatch.model_validate(generation.output).questions):
        selected, failure = await _check_candidate(budget, index, question, trajectory)
        if failure is None:
            if selected is not None:
                kept.append(selected)
            continue
        if failure.status == "budget_exhausted":
            reason = f"self-answer or consistency stage did not complete: {failure.error}"
        else:
            reason = f"model run failed, candidate dropped without resampling: {failure.error}"
        _record_decision(budget, f"filter/{index}", {
            "candidate_index": index, "kept": False, "reason": reason,
            "candidate_answer": question.reference_answer,
            "evidence_verification": "pending public verification",
        })
        if failure.status == "budget_exhausted":
            # The required per-question stages were cut short, so no batch is claimed.
            return budget.finish(directions, status="budget_exhausted", error=failure.error)
    return budget.finish(directions, output=TaskBatch(questions=kept))
