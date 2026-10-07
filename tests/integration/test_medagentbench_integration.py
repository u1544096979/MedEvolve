"""Public evaluation wiring for native submission semantics."""

import asyncio
import json
import sys
from types import SimpleNamespace

from contracts import (
    Budget, EnvironmentConfig, EnvironmentInstance, EvaluationConfig, ModelConfig,
    PatientTrajectory, RunResult, RuntimeConfig, SolverAnswer, Task,
    CompletionReview, Verification, QualityJudgment, DifficultyJudgment,
)
from evaluation import evaluate_task


def test_public_verifier_rejects_submission_task_before_solver(monkeypatch):
    task = Task(
        id="submission", patient_id="patient-1", environment="PhysicianBench",
        round_index=0, source_method="method", instruction="提交指定处方请求",
        task_type="药物处方", reference_answer="期望的处方请求",
        evidence=[{"source_id": "Observation/1", "statement": "处方依据"}],
        decision_point=None,
    )
    trajectory = PatientTrajectory(
        patient_id="patient-1", environment="PhysicianBench", text="fixture",
        sources=[{"source_id": "Observation/1", "statement": "证据"}],
    )
    roles = ("verifier", "quality", "difficulty", "solver", "completion")
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

    async def get_tools(instance, patient_ids, decision_point=None, tool_result_limit=None,
                       read_only=False):
        return []

    async def read_state(instance, task):
        return state

    async def run(**kwargs):
        calls.append(kwargs)
        schema = kwargs["output_type"]
        if schema is Verification:
            payload = json.loads(kwargs["input_text"])
            output = {"valid": True, "reason": "证据存在"}
            if "task" in payload:
                output["read_only_task"] = False
        elif schema is QualityJudgment:
            output = {key: {"score": 4, "reason": "满足要求"}
                      for key in ("factual_accuracy", "clinical_soundness",
                                  "evidence_sufficiency", "task_clarity")}
        elif schema is DifficultyJudgment:
            output = {"reasoning_depth": 2, "reason": "跨记录整合"}
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
        EnvironmentConfig(name="PhysicianBench", manager_url="unused", seed_data="seed"), config,
    ))

    assert result.status == "invalid"
    assert result.failure_category == "unsupported_operation"
    assert result.verification.read_only_task is False
    assert result.solver_run is None
    assert result.quality is None
    assert [call["output_type"] for call in calls] == [Verification, Verification]
