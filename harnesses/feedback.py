"""Training feedback for one harness attempt.

Harness evolution must not learn clinical lessons from infrastructure faults. A
failed run, an exhausted budget or a completion review that never produced a
conclusion is reported with `completed=None` and an infrastructure failure
category, so the learning stage can record the fault without updating clinical
experience.

The reference answer and the frozen evidence travel with the feedback because the
learning role reflects on them. They are never injected into the actor's next
attempt: `make_training_feedback` only reads the executed run and never rewrites
the solving material.
"""

from contracts import EvaluationConfig, PatientTrajectory, RunResult, Task
from evaluation.evaluator import review_completion
from harnesses.contracts import TrainingFeedback

INFRASTRUCTURE_CATEGORIES = ("tool_error", "environment_error", "model_error")


def _infrastructure_category(run: RunResult | None) -> str:
    if run is not None:
        for item in reversed(run.trace or []):
            if not isinstance(item, dict) or item.get("type") != "run_error":
                continue
            category = item.get("failure_category")
            if category in INFRASTRUCTURE_CATEGORIES:
                return category
    return "model_error"


def _evidence(task: Task) -> list[dict]:
    return [item.model_dump() for item in task.evidence]


def _infrastructure_feedback(task: Task, reason: str, run: RunResult | None) -> TrainingFeedback:
    return TrainingFeedback(
        task_id=task.id,
        completed=None,
        failure_category=_infrastructure_category(run),
        reason=reason,
        reference_answer=task.reference_answer,
        evidence=_evidence(task),
    )


async def make_training_feedback(
    task: Task,
    trajectory: PatientTrajectory,
    run: RunResult,
    observed_state: dict,
    config: EvaluationConfig,
) -> TrainingFeedback:
    """Build the learning feedback for one executed run.

    Infrastructure faults never call the completion review: there is nothing
    clinical to judge, and the review budget is not spent on a broken attempt.
    """
    if run.status != "completed" or run.output is None:
        return _infrastructure_feedback(
            task,
            f"基础设施故障，不构成临床失败经验：{run.error or run.status}",
            run,
        )
    try:
        review, _ = await review_completion(task, trajectory, run, observed_state, config)
    except Exception as exc:
        return _infrastructure_feedback(
            task,
            "完成评价未产出结论，不构成临床失败经验："
            f"{type(exc).__name__}: {exc}",
            run,
        )
    return TrainingFeedback(
        task_id=task.id,
        completed=review.completed,
        failure_category=None if review.completed else "clinical",
        reason=review.reason,
        reference_answer=task.reference_answer,
        evidence=_evidence(task),
    )
