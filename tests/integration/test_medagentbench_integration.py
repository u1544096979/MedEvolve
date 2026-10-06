"""Public evaluation wiring for native submission semantics."""

import asyncio
import json
import sys
from types import SimpleNamespace

from contracts import (
    Budget, EnvironmentConfig, EnvironmentInstance, EvaluationConfig, ModelConfig,
    PatientTrajectory, RunResult, RuntimeConfig, SolverAnswer, Task,
    CompletionReview, Verification, QualityReview,
)
from evaluation import evaluate_task


def test_completion_receives_native_submission_and_unmodified_solver_input(monkeypatch):
    task = Task(
        id="submission", patient_id="patient-1", environment="fixture",
        round_index=0, source_method="method", instruction="提交指定处方请求",
        task_type="药物处方", reference_answer="期望的处方请求",
        evidence=[{"source_id": "Observation/1", "statement": "处方依据"}],
        decision_point=None,
    )
    trajectory = PatientTrajectory(
        patient_id="patient-1", environment="fixture", text="fixture",
        sources=[{"source_id": "Observation/1", "statement": "证据"}],
    )
    roles = ("verifier", "quality", "solver", "completion")
    config = EvaluationConfig(runtime=RuntimeConfig(
        models={role: ModelConfig(model=role, api_key_env="UNUSED") for role in roles},
        budgets={role: Budget(max_model_turns=2, max_tool_calls=2) for role in roles},
    ))
    state = {"operation_semantics": "post_acceptance_without_persistence", "resources": []}
    calls = []

    async def create_environment(config):
        return EnvironmentInstance(instance_id="1", endpoint="fake", environment=config.name)

    async def release_environment(config, instance):
        return None

    async def read_evidence(instance, patient_id, source_id):
        return {"resourceType": "Observation", "id": "1"}

    async def get_tools(instance, patient_ids, decision_point=None, tool_result_limit=None):
        return []

    async def read_state(instance, task):
        return state

    async def run(**kwargs):
        calls.append(kwargs)
        schema = kwargs["output_type"]
        if schema is Verification:
            output = {"valid": True, "reason": "证据存在"}
        elif schema is QualityReview:
            output = {"dimensions": {
                key: {"judgment": "合格", "reason": "满足要求"}
                for key in ("clinical_value", "correctness", "longitudinal_integration", "clarity")
            }}
        elif schema is SolverAnswer:
            output = {"answer": "已提交处方请求"}
            trace = [
                {"type": "tool_start", "name": "POST", "call_id": "post-1",
                 "arguments": {"url": "http://fake/fhir/MedicationRequest", "payload": {"status": "draft"}}},
                {"type": "tool_output", "name": "POST", "call_id": "post-1",
                 "output": {"accepted": True, "message": "Request accepted"}},
            ]
            return RunResult(output=output, trace=trace, usage={}, status="completed")
        else:
            payload = json.loads(kwargs["input_text"])
            assert payload["observed_state"] == state
            assert payload["solver_trace"][0]["arguments"]["url"].endswith("/MedicationRequest")
            assert payload["solver_trace"][0]["arguments"]["payload"] == {"status": "draft"}
            assert payload["solver_trace"][1]["output"]["accepted"] is True
            output = {"completed": True, "reason": "请求已接受"}
        return RunResult(output=output, trace=[], usage={}, status="completed")

    monkeypatch.setitem(sys.modules, "environments", SimpleNamespace(
        create_environment=create_environment, release_environment=release_environment,
        read_evidence=read_evidence, get_tools=get_tools, read_state=read_state,
    ))
    monkeypatch.setitem(sys.modules, "runtime", SimpleNamespace(run_agent=run, call_llm=run))
    result = asyncio.run(evaluate_task(
        task, trajectory,
        EnvironmentConfig(name="fixture", manager_url="unused", seed_data="seed"), config,
    ))

    assert result.status == "completed"
    assert result.completion.completed is True
    solver_call = next(call for call in calls if call["output_type"] is SolverAnswer)
    assert json.loads(solver_call["input_text"]) == {
        "instruction": task.instruction, "patient_id": task.patient_id,
        "decision_point": None,
    }
    assert "reference_answer" not in solver_call["input_text"]
    assert "evidence" not in solver_call["input_text"]
