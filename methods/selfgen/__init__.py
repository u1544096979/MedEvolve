import json
from typing import Any

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
from methods.common import GenerationBudget, output_constraints, validate_task_batch

from .prompts import DRAFT, REFINE, REVIEW


class DirectedBatch(BaseModel):
    batch: TaskBatch
    direction_indices: list[int]


class _DraftBatch(BaseModel):
    # 保留正常题目的字段 schema，坏字段留到逐题校验时丢弃。
    questions: list[GeneratedQuestion | dict[str, Any]]


class _DirectedDraft(BaseModel):
    batch: _DraftBatch
    direction_indices: list[int]


class QuestionReview(BaseModel):
    candidate_index: int = Field(ge=0)
    passed: bool
    issues: list[str]


class BatchReview(BaseModel):
    questions: list[QuestionReview]


def validate_draft(value, trajectory, directions):
    """Keep every candidate whose own fields hold; drop the ones that do not."""
    current = _DirectedDraft.model_validate(value)
    slots = current.direction_indices
    questions = current.batch.questions
    allowed = {source.source_id for source in trajectory.sources}
    kept_slots: list[int] = []
    kept_questions: list[GeneratedQuestion] = []
    drops: list[dict] = []
    used: set[int] = set()
    for position, question in enumerate(questions):
        slot = slots[position] if position < len(slots) else None
        if slot is None or slot not in (0, 1):
            drops.append({"index": position,
                          "reason": "direction index is missing or outside the two positions"})
            continue
        if slot in used:
            drops.append({"index": position, "reason": "direction index is duplicated"})
            continue
        payload = question.model_dump(mode="json") if isinstance(question, GeneratedQuestion) else question
        probe = validate_task_batch(
            RunResult(output={"questions": [payload]},
                      status="completed", trace=[], usage={}),
            [directions[slot]], max_questions=1)
        if probe.status != "completed":
            drops.append({"index": position, "reason": probe.error or "output rejected"})
            continue
        question = TaskBatch.model_validate(probe.output).questions[0]
        if not question.evidence or any(e.source_id not in allowed for e in question.evidence):
            drops.append({"index": position,
                          "reason": "candidate evidence is absent or outside patient sources"})
            continue
        used.add(slot)
        kept_slots.append(slot)
        kept_questions.append(question)
    return (DirectedBatch(batch=TaskBatch(questions=kept_questions),
                          direction_indices=kept_slots), drops)


def validate_review(value):
    """Reject self-contradicting verdicts; leave coverage gaps to the caller."""
    review = BatchReview.model_validate(value)
    for item in review.questions:
        if any(not issue.strip() for issue in item.issues):
            raise ValueError("Review issue must be nonempty")
        if item.passed != (not item.issues):
            raise ValueError("Review verdict conflicts with issues")
    return review


def passed_indices(review, size):
    """Indices the review passed; every candidate it did not judge counts as failed.

    A candidate judged twice with disagreeing verdicts has no verdict to trust, and the
    later verdicts for it cannot restore that trust.
    """
    judged: dict[int, QuestionReview] = {}
    conflicting: set[int] = set()
    drops: list[dict] = []
    for item in review.questions:
        if not 0 <= item.candidate_index < size:
            drops.append({"index": item.candidate_index, "reason": "review index is outside the candidates"})
            continue
        if item.candidate_index in conflicting:
            continue
        previous = judged.get(item.candidate_index)
        if previous is None:
            judged[item.candidate_index] = item
        elif previous.passed != item.passed:
            judged.pop(item.candidate_index)
            conflicting.add(item.candidate_index)
    passed = sorted(index for index, item in judged.items() if item.passed)
    for index in range(size):
        if index not in judged:
            reason = ("candidate was reviewed with conflicting verdicts" if index in conflicting
                      else "candidate was not reviewed")
            drops.append({"index": index, "reason": reason})
    return passed, drops


def validate_revision(previous, revised):
    """A revision may drop or keep directions, never introduce a new one."""
    before = set(previous.direction_indices)
    if not set(revised.direction_indices) <= before:
        raise ValueError("Revision added a direction")


def merge_passed(previous, passed, revised):
    """Restore every passed direction from the archived batch, whatever the revision returned."""
    before = dict(zip(previous.direction_indices, previous.batch.questions, strict=False))
    after = dict(zip(revised.direction_indices, revised.batch.questions, strict=False))
    restored: list[int] = []
    for index in passed:
        if index >= len(previous.direction_indices):
            continue
        slot = previous.direction_indices[index]
        if slot not in after or after[slot] != before[slot]:
            restored.append(slot)
        after[slot] = before[slot]
    order = [slot for slot in previous.direction_indices if slot in after]
    order += [slot for slot in revised.direction_indices if slot in after and slot not in order]
    return (DirectedBatch(batch=TaskBatch(questions=[after[slot] for slot in order]),
                          direction_indices=order), restored)


def record(budget, stage, payload, *, status="completed", error=None):
    budget.records.append((stage, RunResult(trace=[payload], usage={
        "model_turns": 0, "tool_calls": 0, "input_tokens": 0,
        "output_tokens": 0, "total_tokens": 0}, status=status, error=error)))


async def generate_tasks(trajectory: PatientTrajectory, environment: EnvironmentConfig,
                         directions: list[GenerationDirection], runtime: RuntimeConfig) -> RunResult:
    if trajectory.environment != environment.name:
        raise ValueError("Trajectory environment mismatch")
    constraints = output_constraints(directions)
    if environment.name not in {"PhysicianBench", "MedAgentBench"}:
        raise ValueError("Unsupported environment")
    budget = GenerationBudget(runtime, environment.name)
    notes: list[dict] = []

    def note(kind: str, **payload) -> None:
        notes.append({"type": kind, **payload})

    def closing(result: RunResult) -> RunResult:
        if not notes:
            return result
        return result.model_copy(update={"trace": [*result.trace, *notes]})

    context = {
        "trajectory": trajectory.model_dump(mode="json"),
        "directions": [d.model_dump(mode="json") for d in directions],
    }

    async def call(stage, instructions, payload, schema):
        run = await budget.call_llm(stage, instructions=instructions,
            input_text=json.dumps(payload, ensure_ascii=False), output_type=schema)
        # 不变量：common.GenerationBudget._execute 每次调用都恰好追加一条记录，
        # 因此这里回填的一定是本次调用的记录。
        budget.records[-1] = (stage, run.model_copy(update={"trace": [
            {"type": "method_input", "input": payload}, *run.trace]}))
        return run

    def failed_run(run):
        return closing(budget.finish(directions, status=run.status, error=run.error))

    try:
        run = await call("candidate_generation", DRAFT + constraints, context, _DirectedDraft)
        if run.status != "completed":
            return failed_run(run)
        current, draft_drops = validate_draft(run.output, trajectory, directions)
        if draft_drops:
            note("drops", stage="candidate_generation", drops=draft_drops)
        if not current.batch.questions:
            return closing(budget.finish(directions, output=current.batch))
        for number in range(1, 4):
            body = {**context, "current": current.model_dump(mode="json")}
            run = await call(f"self_review/{number}", REVIEW, body, BatchReview)
            if run.status != "completed":
                return failed_run(run)
            review = validate_review(run.output)
            passed, review_drops = passed_indices(review, len(current.batch.questions))
            if review_drops:
                note("drops", stage=f"self_review/{number}", drops=review_drops)
            if len(passed) == len(current.batch.questions) or number == 3:
                final = TaskBatch(questions=[current.batch.questions[i] for i in passed])
                record(budget, "final_filter", {"review_number": number,
                    "kept_indices": passed, "feedback": review.model_dump(mode="json"),
                    "output": final.model_dump(mode="json")})
                return closing(budget.finish(directions, output=final))
            run = await call(f"refine/{number}", REFINE + constraints,
                {**body, "feedback": review.model_dump(mode="json")}, _DirectedDraft)
            if run.status != "completed":
                return failed_run(run)
            revised, refine_drops = validate_draft(run.output, trajectory, directions)
            validate_revision(current, revised)
            revised, restored = merge_passed(current, passed, revised)
            if refine_drops:
                note("drops", stage=f"refine/{number}", drops=refine_drops)
            if restored:
                note("revision_merge", round=number, restored=restored)
            current = revised
            if not current.batch.questions:
                return closing(budget.finish(directions, output=current.batch))
    except (ValueError, TypeError) as exc:
        record(budget, "validation_failed", {"error": str(exc)}, status="failed", error=str(exc))
        return closing(budget.finish(directions, status="failed", error=str(exc)))
