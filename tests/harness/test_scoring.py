"""Completion scoring for harness runs, with all external edges faked locally."""

import json
import sys
from types import SimpleNamespace

import pytest

from contracts import (
    EnvironmentConfig,
    EvaluationConfig,
    ModelConfig,
    PatientTrajectory,
    RunResult,
    RuntimeConfig,
    SolverInput,
    Task,
)
from harnesses.feedback import make_training_feedback
from harnesses.medical_bridge import ToolSessions

REFERENCE_ANSWER = "参考答案：按当前证据调整治疗并安排随访"


class FakeEnvironment:
    def __init__(self, events):
        self.events = events
        self.instance = SimpleNamespace(instance_id="instance-1")

    async def create_environment(self, config):
        self.events.append("open-instance")
        return self.instance

    async def get_tools(self, instance, patient_ids, **kwargs):
        assert instance is self.instance
        assert patient_ids == ["p1"]
        assert kwargs["read_only"] is True
        return []

    async def read_state(self, instance, task):
        assert instance is self.instance
        self.events.append("read-state")
        return {"patient_id": task.patient_id, "observed": "stable"}

    async def release_environment(self, config, instance):
        assert instance is self.instance
        self.events.append("release-instance")


class FakeRunner:
    def __init__(self, events, answer):
        self.events = events
        self.answer = answer

    async def execute(self, solver_input, session_id):
        self.events.append("execute")
        return RunResult(
            output={"answer": self.answer},
            trace=[
                {"type": "tool_start", "name": "fhir_condition_search_problems",
                 "arguments": {"patient": "p1"}},
                {"type": "tool_output", "name": "fhir_condition_search_problems",
                 "output": {"conditions": ["稳定"]}},
            ],
            usage={"input_tokens": 9, "output_tokens": 3},
            status="completed",
        )


@pytest.fixture
def scoring_rig(monkeypatch):
    events = []
    environment = EnvironmentConfig(
        name="fixture", manager_url="http://unused", seed_data="seed",
    )
    task = Task(
        id="task-1", patient_id="p1", environment="fixture", round_index=0,
        source_method="medevolve", instruction="判断当前病情并说明处理方案",
        task_type="诊断与解释", reference_answer=REFERENCE_ANSWER,
        evidence=[{"source_id": "Condition/c1", "statement": "病情稳定"}],
        execution_requirement="read_only",
    )
    solver_input = SolverInput(instruction=task.instruction, patient_id="p1")
    trajectory = PatientTrajectory(
        patient_id="p1", environment="fixture", text="患者当前病情稳定",
        sources=[{"source_id": "Condition/c1", "statement": '{"clinicalStatus":"active"}'}],
    )
    config = EvaluationConfig(runtime=RuntimeConfig(
        models={"completion": ModelConfig(model="completion-review", api_key_env="UNUSED")},
        budgets={},
    ))
    state = {"completed": True, "fail": False, "payload": None}
    fake_environment = FakeEnvironment(events)

    async def call_llm(**kwargs):
        assert "release-instance" not in events
        events.append("review")
        state["payload"] = json.loads(kwargs["input_text"])
        if state["fail"]:
            raise RuntimeError("review unavailable")
        return RunResult(
            output={"completed": state["completed"], "reason": "依据执行答案完成判定"},
            trace=[{"type": "model_response"}],
            usage={"input_tokens": 20, "output_tokens": 4},
            status="completed",
        )

    monkeypatch.setitem(sys.modules, "runtime", SimpleNamespace(call_llm=call_llm))
    return SimpleNamespace(
        events=events, environment=environment, task=task,
        solver_input=solver_input, trajectory=trajectory, config=config,
        state=state, registry=ToolSessions(environments=fake_environment),
        runner=None,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(("completed", "answer", "failure_category"), [
    (True, "病情稳定，继续当前治疗并随访。", None),
    (False, "立即停用全部治疗，无需复查。", "clinical"),
])
async def test_scores_real_runner_answer_while_read_only_instance_is_open(
    scoring_rig, completed, answer, failure_category,
):
    rig = scoring_rig
    rig.state["completed"] = completed
    client_id = "harness-runner"
    session_id = await rig.registry.open(
        rig.solver_input, rig.environment, client_id=client_id, read_only=True,
    )
    recorded = []
    try:
        context = rig.registry.inspect(session_id, client_id=client_id)
        assert context.read_only is True
        run = await FakeRunner(rig.events, answer).execute(rig.solver_input, session_id)
        observed_state = {"patient_id": "p1", "observed": "stable"}
        feedback = await make_training_feedback(
            rig.task, rig.trajectory, run, observed_state, rig.config,
            record=recorded.append,
        )
        assert "release-instance" not in rig.events
        assert feedback.completed is completed
        assert feedback.failure_category == failure_category
        assert len(recorded) == 1
        assert recorded[0].usage == {"input_tokens": 20, "output_tokens": 4}
        assert recorded[0].trace[-1]["type"] == "model_identity"
        assert rig.events.index("execute") < rig.events.index("review")
        assert rig.state["payload"]["solver_answer"] == {"answer": answer}
        assert [item["type"] for item in rig.state["payload"]["solver_trace"]] == [
            "tool_start", "tool_output",
        ]
        assert rig.state["payload"]["task"]["reference_answer"] == REFERENCE_ANSWER
        assert rig.state["payload"]["observed_state"] == observed_state
    finally:
        await rig.registry.close(session_id, client_id=client_id)
    assert rig.events.index("review") < rig.events.index("release-instance")


@pytest.mark.asyncio
async def test_failed_completion_review_records_failure_without_fabricating_score(scoring_rig):
    rig = scoring_rig
    rig.state["fail"] = True
    session_id = await rig.registry.open(
        rig.solver_input, rig.environment, client_id="runner", read_only=True,
    )
    recorded = []
    try:
        run = await FakeRunner(rig.events, "给出真实执行答案").execute(
            rig.solver_input, session_id,
        )
        feedback = await make_training_feedback(
            rig.task, rig.trajectory, run, {"observed": "stable"}, rig.config,
            record=recorded.append,
        )
        assert "release-instance" not in rig.events
        assert feedback.completed is None
        assert feedback.failure_category == "model_error"
        assert "ValueError" in feedback.reason
        assert len(recorded) == 1
        assert recorded[0].status == "failed"
        assert "review unavailable" in (recorded[0].error or "")
        assert recorded[0].usage == {}
    finally:
        await rig.registry.close(session_id, client_id="runner")
