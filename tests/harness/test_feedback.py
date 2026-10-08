"""Training feedback for one harness attempt.

Infrastructure faults must stay infrastructure faults: a failed run or a failed
completion review never turns into a clinical failure experience. Reference
answers live in the review process and are handed to the learning stage only,
never to the solving context.
"""

import json
import sys
from types import SimpleNamespace

import pytest

from contracts import (
    EvaluationConfig, ModelConfig, PatientTrajectory, RunResult, RuntimeConfig, Task,
)
from harnesses.feedback import make_training_feedback

REFERENCE_ANSWER = "参考答案只应出现在评审与学习材料中"
EVIDENCE = [{"source_id": "source-1", "statement": "该记录支持任务要求"}]


@pytest.fixture
def feedback_rig(monkeypatch):
    task = Task(
        id="task-1", patient_id="p1", environment="fixture", round_index=0,
        source_method="medevolve", instruction="为患者开具指定处方",
        task_type="药物处方", reference_answer=REFERENCE_ANSWER, evidence=EVIDENCE,
    )
    trajectory = PatientTrajectory(
        patient_id="p1", environment="fixture", text="患者轨迹正文",
        sources=[{"source_id": "source-1", "statement": "轨迹证据"}],
    )
    config = EvaluationConfig(runtime=RuntimeConfig(
        models={"completion": ModelConfig(model="completion", api_key_env="UNUSED")},
        budgets={},
    ))
    state = {"completed": True, "reason": "回答覆盖了核心要求", "fail": None, "calls": []}

    async def call_llm(**kwargs):
        state["calls"].append(kwargs)
        if state["fail"]:
            raise RuntimeError("model unavailable")
        return RunResult(
            output={"completed": state["completed"], "reason": state["reason"]},
            trace=[{"role": "completion"}], usage={}, status="completed",
        )

    monkeypatch.setitem(sys.modules, "runtime", SimpleNamespace(call_llm=call_llm))
    return SimpleNamespace(
        task=task, trajectory=trajectory, config=config, state=state,
        run=RunResult(
            output={"answer": "处方已创建"},
            trace=[
                {"type": "model_request", "instructions": "历史提示词"},
                {"type": "tool_start", "name": "fhir_medication_request_create"},
                {"type": "tool_output", "name": "fhir_medication_request_create",
                 "output": {"resourceType": "MedicationRequest", "id": "rx-1"}},
            ],
            usage={}, status="completed",
        ),
    )


async def feedback(rig, run=None, observed_state=None):
    return await make_training_feedback(
        rig.task, rig.trajectory, run or rig.run,
        observed_state if observed_state is not None else {"prescriptions": [{"id": "rx-1"}]},
        rig.config,
    )


@pytest.mark.asyncio
async def test_completed_task_reports_no_clinical_failure(feedback_rig):
    result = await feedback(feedback_rig)
    assert result.completed is True
    assert result.failure_category is None
    assert result.reason == "回答覆盖了核心要求"
    assert result.task_id == "task-1"
    assert result.reference_answer == REFERENCE_ANSWER
    assert result.evidence == EVIDENCE
    assert len(feedback_rig.state["calls"]) == 1


@pytest.mark.asyncio
async def test_clinical_failure_carries_the_review_reason(feedback_rig):
    feedback_rig.state["completed"] = False
    feedback_rig.state["reason"] = "答案只给出风险，没有回答题目要求的处方"
    result = await feedback(feedback_rig)
    assert result.completed is False
    assert result.failure_category == "clinical"
    assert "没有回答题目要求的处方" in result.reason


@pytest.mark.asyncio
@pytest.mark.parametrize("category", ["tool_error", "environment_error", "model_error"])
async def test_solver_infrastructure_failure_is_not_a_clinical_failure(feedback_rig, category):
    failed = RunResult(
        trace=[{"type": "run_error", "failure_category": category, "error": "boom"}],
        usage={}, status="failed", error="boom",
    )
    result = await feedback(feedback_rig, run=failed)
    assert result.completed is None
    assert result.failure_category == category
    assert "不构成临床失败经验" in result.reason
    assert feedback_rig.state["calls"] == []


@pytest.mark.asyncio
async def test_budget_exhausted_run_is_infrastructure_failure(feedback_rig):
    exhausted = RunResult(
        trace=[{"type": "run_error", "failure_category": "model_error",
                "error": "Model turn budget exhausted"}],
        usage={}, status="budget_exhausted", error="budget exhausted",
    )
    result = await feedback(feedback_rig, run=exhausted)
    assert result.completed is None
    assert result.failure_category == "model_error"


@pytest.mark.asyncio
async def test_failed_completion_review_is_infrastructure_failure(feedback_rig):
    feedback_rig.state["fail"] = True
    result = await feedback(feedback_rig)
    assert result.completed is None
    assert result.failure_category == "model_error"
    assert "不构成临床失败经验" in result.reason
    assert len(feedback_rig.state["calls"]) == 1


@pytest.mark.asyncio
async def test_reference_answer_stays_out_of_the_solving_material(feedback_rig):
    run_before = feedback_rig.run.model_dump(mode="json")
    result = await feedback(feedback_rig)
    assert result.reference_answer == REFERENCE_ANSWER
    assert json.dumps(run_before, ensure_ascii=False).find(REFERENCE_ANSWER) == -1
    assert feedback_rig.run.model_dump(mode="json") == run_before
    solver_trace = feedback_rig.run.trace
    assert all(REFERENCE_ANSWER not in json.dumps(item, ensure_ascii=False)
               for item in solver_trace)
    # The review payload holds the reference answer, but the reviewed answer is the
    # harness answer and the trace is the filtered execution evidence.
    payload = json.loads(feedback_rig.state["calls"][0]["input_text"])
    assert payload["solver_answer"] == {"answer": "处方已创建"}
    assert [item["type"] for item in payload["solver_trace"]] == ["tool_start", "tool_output"]
    assert "历史提示词" not in feedback_rig.state["calls"][0]["input_text"]
