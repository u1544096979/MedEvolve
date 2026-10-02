"""Critical evaluation boundaries with in-memory runtime and environments."""

import asyncio
import json
import sys
from types import SimpleNamespace

import pytest

from contracts import (
    Budget, CompletionReview, EnvironmentConfig, EnvironmentInstance,
    EvaluationConfig, ModelConfig, PatientTrajectory, QualityReview, RunResult,
    RuntimeConfig, SolverAnswer, Task, Verification,
)
from evaluation import evaluate_task


@pytest.fixture
def harness(monkeypatch):
    task = Task(
        id="task-1", patient_id="patient-1", environment="fixture", round_index=0,
        source_method="private-method", instruction="为患者开具指定处方",
        task_type="药物处方", reference_answer="private-reference", evidence=[],
        decision_point="2026-01-01T00:00:00+08:00",
    )
    trajectory = PatientTrajectory(
        patient_id=task.patient_id, environment=task.environment,
        text="private-trajectory", sources=[],
    )
    roles = ["verifier", "quality", "solver", "completion"]
    config = EvaluationConfig(runtime=RuntimeConfig(
        models={r: ModelConfig(model=r, api_key_env="UNUSED") for r in roles},
        budgets={r: Budget(max_model_turns=3, max_tool_calls=5) for r in roles},
    ))
    h = SimpleNamespace(
        task=task, trajectory=trajectory, config=config,
        environment=EnvironmentConfig(name="fixture", manager_url="unused", seed_data="seed"),
        created=[], released=[], tool_calls=[], model_calls=[], state_reads=[],
        state={"prescriptions": []}, fail=None, valid=True,
    )

    async def create_environment(config):
        instance = EnvironmentInstance(
            instance_id=str(len(h.created)), endpoint="unused", environment=config.name
        )
        h.created.append(instance)
        return instance

    async def release_environment(config, instance):
        h.released.append(instance)

    async def get_tools(instance, patient_ids, decision_point=None):
        h.tool_calls.append((instance, patient_ids, decision_point))
        if h.fail == "tools":
            raise RuntimeError("tool setup failed")
        return [instance.instance_id]

    async def read_state(instance, task):
        h.state_reads.append(instance)
        if h.fail == "state":
            raise RuntimeError("state unavailable")
        return h.state

    async def run(**kwargs):
        h.model_calls.append(kwargs)
        role = kwargs["model"].model
        if h.fail == role:
            raise RuntimeError("model unavailable")
        if h.fail == role + "_budget":
            return RunResult(trace=[{"role": role}], usage={}, status="budget_exhausted")
        if role == "verifier":
            output = {"valid": h.valid, "reason": "checked original records"}
        elif role == "quality":
            output = {"dimensions": {key: {"judgment": "suitable", "reason": "supported"}
                      for key in ("clinical_value", "correctness", "longitudinal_integration", "clarity")}}
        elif role == "solver":
            output = {"answer": "处方已创建"}
        else:
            payload = json.loads(kwargs["input_text"])
            output = {
                "completed": bool(payload["observed_state"].get("prescriptions")),
                "reason": "checked actual prescriptions",
            }
        return RunResult(output=output, trace=[{"role": role}], usage={}, status="completed")

    monkeypatch.setitem(sys.modules, "environments", SimpleNamespace(
        create_environment=create_environment, get_tools=get_tools,
        read_state=read_state, release_environment=release_environment,
    ))
    monkeypatch.setitem(sys.modules, "runtime", SimpleNamespace(run_agent=run, call_llm=run))
    return h


def evaluate(h):
    return asyncio.run(evaluate_task(h.task, h.trajectory, h.environment, h.config))


def test_solver_receives_only_patient_question_and_decision_point(harness):
    h = harness
    result = evaluate(h)
    solver = next(c for c in h.model_calls if c["output_type"] is SolverAnswer)
    assert json.loads(solver["input_text"]) == {
        "instruction": h.task.instruction, "patient_id": h.task.patient_id,
        "decision_point": h.task.decision_point,
    }
    for call in h.model_calls:
        assert "private-method" not in call["input_text"]
    assert h.tool_calls[0][2] is None
    assert h.tool_calls[1][1:] == ([h.task.patient_id], h.task.decision_point)
    assert h.created[0] != h.created[1]
    assert len(result.runs) == 4


@pytest.mark.parametrize("state, completed", [
    ({"prescriptions": []}, False),
    ({"prescriptions": [{"id": "rx-1"}]}, True),
])
def test_write_completion_uses_observed_state(harness, state, completed):
    h = harness
    h.state = state
    result = evaluate(h)
    assert result.status == "completed"
    assert result.observed_state == state
    assert result.completion.completed is completed
    assert result.failure_category == (None if completed else "clinical")
    call = next(c for c in h.model_calls if c["output_type"] is CompletionReview)
    assert json.loads(call["input_text"])["observed_state"] == state
    assert h.state_reads == [h.created[1]]
    assert h.released == h.created


@pytest.mark.parametrize("failure, category, instances, runs", [
    ("tools", "environment_error", 1, 0),
    ("verifier", "model_error", 1, 1),
    ("verifier_budget", "model_error", 1, 1),
    ("quality", "model_error", 1, 2),
    ("solver", "model_error", 2, 3),
    ("solver_budget", "model_error", 2, 3),
    ("state", "environment_error", 2, 3),
    ("completion", "model_error", 2, 4),
])
def test_faults_release_instances_without_clinical_conclusions(
    harness, failure, category, instances, runs
):
    h = harness
    h.fail = failure
    result = evaluate(h)
    assert result.status == "failed"
    assert result.failure_category == category
    assert result.completion is None
    assert len(h.created) == instances
    assert h.released == h.created
    assert len(result.runs) == runs
    if failure.startswith("verifier"):
        assert result.verification is None
    if failure.endswith("budget"):
        assert result.runs[-1].status == "budget_exhausted"


def test_invalid_task_stops_before_solver_and_releases_instance(harness):
    h = harness
    h.valid = False
    result = evaluate(h)
    assert result.status == "invalid"
    assert result.verification.valid is False
    assert result.solver_run is None
    assert result.completion is None
    assert len(h.created) == 1
    assert h.released == h.created
    assert [c["output_type"] for c in h.model_calls] == [Verification]


def test_cancellation_releases_solver_instance(harness, monkeypatch):
    h = harness
    original = sys.modules["runtime"].run_agent

    async def cancelled_run(**kwargs):
        if kwargs["output_type"] is SolverAnswer:
            raise asyncio.CancelledError()
        return await original(**kwargs)

    monkeypatch.setattr(sys.modules["runtime"], "run_agent", cancelled_run)
    with pytest.raises(asyncio.CancelledError):
        evaluate(h)
    assert len(h.created) == 2
    assert h.released == h.created
