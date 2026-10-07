import json

from pydantic import BaseModel, Field

from contracts import (
    EnvironmentConfig,
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
    current = DirectedBatch.model_validate(value)
    slots = current.direction_indices
    questions = current.batch.questions
    if len(slots) != len(questions) or len(set(slots)) != len(slots):
        raise ValueError("Direction mapping must be unique and complete")
    probe = validate_task_batch(RunResult(output=current.batch.model_dump(mode="json"),
        status="completed", trace=[], usage={}), directions)
    if probe.status != "completed":
        raise ValueError(probe.error)
    allowed = {s.source_id for s in trajectory.sources}
    for slot, q in zip(slots, questions):
        if slot not in (0, 1) or q.task_type != directions[slot].task_type:
            raise ValueError("Candidate changed its direction")
        if not q.evidence or any(e.source_id not in allowed for e in q.evidence):
            raise ValueError("Candidate evidence is absent or outside patient sources")
    return current


def validate_review(value, size):
    review = BatchReview.model_validate(value)
    indices = [q.candidate_index for q in review.questions]
    if sorted(indices) != list(range(size)):
        raise ValueError("Review must cover each candidate exactly once")
    for item in review.questions:
        if any(not issue.strip() for issue in item.issues):
            raise ValueError("Review issue must be nonempty")
        if item.passed != (not item.issues):
            raise ValueError("Review verdict conflicts with issues")
    return review


def validate_revision(previous, review, revised):
    before = dict(zip(previous.direction_indices, previous.batch.questions))
    after = dict(zip(revised.direction_indices, revised.batch.questions))
    if not after.keys() <= before.keys():
        raise ValueError("Revision added a direction")
    for item in review.questions:
        slot = previous.direction_indices[item.candidate_index]
        if item.passed and (slot not in after or
                after[slot].model_dump() != before[slot].model_dump()):
            raise ValueError("Revision changed or removed a passed candidate")


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
        return budget.finish(directions, status=run.status, error=run.error)

    try:
        run = await call("candidate_generation", DRAFT + constraints, context, DirectedBatch)
        if run.status != "completed":
            return failed_run(run)
        current = validate_draft(run.output, trajectory, directions)
        if not current.batch.questions:
            return budget.finish(directions, output=current.batch)
        for number in range(1, 4):
            body = {**context, "current": current.model_dump(mode="json")}
            run = await call(f"self_review/{number}", REVIEW, body, BatchReview)
            if run.status != "completed":
                return failed_run(run)
            review = validate_review(run.output, len(current.batch.questions))
            passed = sorted(item.candidate_index for item in review.questions if item.passed)
            if len(passed) == len(current.batch.questions) or number == 3:
                final = TaskBatch(questions=[current.batch.questions[i] for i in passed])
                record(budget, "final_filter", {"review_number": number,
                    "kept_indices": passed, "feedback": review.model_dump(mode="json"),
                    "output": final.model_dump(mode="json")})
                return budget.finish(directions, output=final)
            run = await call(f"refine/{number}", REFINE + constraints,
                {**body, "feedback": review.model_dump(mode="json")}, DirectedBatch)
            if run.status != "completed":
                return failed_run(run)
            revised = validate_draft(run.output, trajectory, directions)
            validate_revision(current, review, revised)
            current = revised
            if not current.batch.questions:
                return budget.finish(directions, output=current.batch)
    except (ValueError, TypeError) as exc:
        record(budget, "validation_failed", {"error": str(exc)}, status="failed", error=str(exc))
        return budget.finish(directions, status="failed", error=str(exc))
