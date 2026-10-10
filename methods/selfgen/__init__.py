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
from methods.common import GenerationBudget, output_constraints, validate_task_batch

from .prompts import DRAFT, REFINE, REVIEW


class DirectedBatch(BaseModel):
    batch: TaskBatch
    direction_indices: list[int]


class QuestionReview(BaseModel):
    candidate_index: int = Field(ge=0)
    passed: bool
    issues: list[str]


class BatchReview(BaseModel):
    questions: list[QuestionReview]


def validate_draft(value, trajectory, directions):
    """Keep every candidate whose own fields hold; drop the ones that do not."""
    current = DirectedBatch.model_validate(value)
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
        probe = validate_task_batch(
            RunResult(output={"questions": [question.model_dump(mode="json")]},
                      status="completed", trace=[], usage={}),
            [directions[slot]], max_questions=1)
        if probe.status != "completed":
            drops.append({"index": position, "reason": probe.error or "output rejected"})
            continue
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
    """Indices the review passed; every candidate it did not judge counts as failed."""
    judged: dict[int, QuestionReview] = {}
    drops: list[dict] = []
    for item in review.questions:
        if not 0 <= item.candidate_index < size:
            drops.append({"index": item.candidate_index, "reason": "review index is outside the candidates"})
            continue
        if item.candidate_index in judged:
            drops.append({"index": item.candidate_index, "reason": "candidate was reviewed twice"})
            continue
        judged[item.candidate_index] = item
    passed = sorted(index for index, item in judged.items() if item.passed)
    for index in range(size):
        if index not in judged:
            drops.append({"index": index, "reason": "candidate was not reviewed"})
    return passed, drops


def validate_revision(previous, revised):
    """A revision may drop or keep directions, never introduce a new one."""
    before = set(previous.direction_indices)
    if not set(revised.direction_indices) <= before:
        raise ValueError("Revision added a direction")


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
        run = await call("candidate_generation", DRAFT + constraints, context, DirectedBatch)
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
                {**body, "feedback": review.model_dump(mode="json")}, DirectedBatch)
            if run.status != "completed":
                return failed_run(run)
            revised, refine_drops = validate_draft(run.output, trajectory, directions)
            validate_revision(current, revised)
            if refine_drops:
                note("drops", stage=f"refine/{number}", drops=refine_drops)
            current = revised
            if not current.batch.questions:
                return closing(budget.finish(directions, output=current.batch))
    except (ValueError, TypeError) as exc:
        record(budget, "validation_failed", {"error": str(exc)}, status="failed", error=str(exc))
        return closing(budget.finish(directions, status="failed", error=str(exc)))
