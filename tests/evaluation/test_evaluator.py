"""Critical evaluation boundaries with in-memory runtime and environments."""

import asyncio
import json
import sys
from types import SimpleNamespace

import pytest

from contracts import (
    Budget, CompletionReview, ComplexityMetrics, DifficultyJudgment, EnvironmentConfig,
    EnvironmentInstance, Evidence, EvaluationConfig, ModelConfig, PatientTrajectory,
    QualityJudgment, QualityReview, RunResult, RuntimeConfig, SolverAnswer, Task,
    Verification,
)
from evaluation import evaluate_task


@pytest.fixture
def harness(monkeypatch):
    task = Task(
        id="task-1", patient_id="patient-1", environment="fixture", round_index=0,
        source_method="private-method", instruction="为患者开具指定处方",
        task_type="药物处方", reference_answer="private-reference",
        evidence=[{"source_id": "source-1", "statement": "支持该任务的证据"}],
        decision_point="2026-01-01T00:00:00+08:00",
    )
    trajectory = PatientTrajectory(
        patient_id=task.patient_id, environment=task.environment,
        text="private-trajectory", sources=[{"source_id": "source-1", "statement": "轨迹证据"}],
    )
    roles = ["verifier", "quality", "difficulty", "solver", "completion"]
    config = EvaluationConfig(runtime=RuntimeConfig(
        models={r: ModelConfig(model=r, api_key_env="UNUSED") for r in roles},
        budgets={r: Budget(max_model_turns=3, max_tool_calls=5) for r in roles},
    ))
    h = SimpleNamespace(
        task=task, trajectory=trajectory, config=config,
        environment=EnvironmentConfig(name="fixture", manager_url="unused", seed_data="seed"),
        created=[], released=[], tool_calls=[], model_calls=[], state_reads=[],
        state={"prescriptions": []}, fail=None, valid=True,
        quality_scores=[4, 4, 4], depths=[2, 2, 2], blank_reason=None,
    )

    async def create_environment(config):
        instance = EnvironmentInstance(
            instance_id=str(len(h.created)), endpoint="unused", environment=config.name
        )
        h.created.append(instance)
        return instance

    async def release_environment(config, instance):
        h.released.append(instance)

    async def get_tools(instance, patient_ids, decision_point=None, tool_result_limit=None):
        h.tool_calls.append((instance, patient_ids, decision_point))
        if h.fail == "tools":
            raise RuntimeError("tool setup failed")
        return [instance.instance_id]

    async def read_state(instance, task):
        h.state_reads.append(instance)
        if h.fail == "state":
            raise RuntimeError("state unavailable")
        return h.state

    async def read_evidence(instance, patient_id, source_id):
        if h.fail == "evidence":
            raise RuntimeError("evidence unavailable")
        return {"resourceType": "Observation", "id": source_id, "valueString": "original"}

    async def run(**kwargs):
        h.model_calls.append(kwargs)
        role = kwargs["model"].model
        if h.fail == role:
            raise RuntimeError("model unavailable")
        if h.fail == role + "_budget":
            return RunResult(trace=[{"role": role}], usage={}, status="budget_exhausted")
        if h.fail == role + "_tool_error":
            return RunResult(
                trace=[{"type": "run_error", "failure_category": "tool_error"}],
                usage={}, status="failed", error="opaque runtime error",
            )
        if h.fail == role + "_environment_error":
            return RunResult(
                trace=[{"type": "run_error", "failure_category": "environment_error"}],
                usage={}, status="failed", error="opaque runtime error",
            )
        if h.fail == role + "_model_error":
            return RunResult(
                trace=[{"type": "run_error", "failure_category": "model_error"}],
                usage={}, status="failed", error="opaque runtime error",
            )
        if h.fail == role + "_unknown_error":
            return RunResult(
                trace=[{"type": "run_error", "failure_category": "unknown_error"}],
                usage={}, status="failed", error="opaque runtime error",
            )
        if role == "verifier":
            output = {"valid": h.valid, "reason": "checked original records"}
        elif role == "quality":
            index = sum(1 for call in h.model_calls[:-1]
                        if call["output_type"] is QualityJudgment)
            reason = "" if h.blank_reason == "quality" else "supported"
            output = {key: {"score": h.quality_scores[index], "reason": reason}
                      for key in ("factual_accuracy", "clinical_soundness",
                                  "evidence_sufficiency", "task_clarity")}
        elif role == "difficulty":
            index = sum(1 for call in h.model_calls[:-1]
                        if call["output_type"] is DifficultyJudgment)
            reason = "" if h.blank_reason == "difficulty" else "跨记录整合"
            output = {"reasoning_depth": h.depths[index], "reason": reason}
        elif role == "solver":
            output = {"answer": "处方已创建"}
        else:
            payload = json.loads(kwargs["input_text"])
            output = {
                "completed": bool(payload["observed_state"].get("prescriptions")),
                "reason": "checked actual prescriptions",
            }
        trace = [{"role": role}]
        if role == "solver":
            trace = [
                {"type": "model_request", "instructions": "历史提示词"},
                {"type": "tool_start", "name": "create_prescription", "call_id": "write-1", "arguments": "{}"},
                {"type": "tool_output", "name": "create_prescription", "call_id": "write-1", "output": h.state},
            ]
        return RunResult(output=output, trace=trace, usage={}, status="completed")

    monkeypatch.setitem(sys.modules, "environments", SimpleNamespace(
        create_environment=create_environment, get_tools=get_tools,
        read_state=read_state, release_environment=release_environment,
        read_evidence=read_evidence,
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
    assert h.tool_calls[0][1:] == ([h.task.patient_id], h.task.decision_point)
    assert h.created[0] != h.created[1]
    assert len(result.runs) == 10


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
    payload = json.loads(call["input_text"])
    assert [item["type"] for item in payload["solver_trace"]] == ["tool_start", "tool_output"]
    assert "历史提示词" not in call["input_text"]
    assert payload["evidence_records"] == [{"source_id": "source-1", "statement": "轨迹证据"}]
    assert result.solver_run.trace[0]["instructions"] == "历史提示词"
    assert h.state_reads == [h.created[1]]
    assert h.released == h.created


def test_recovered_parameter_error_reaches_completion(harness, monkeypatch):
    h = harness
    h.state = {"prescriptions": [{"id": "written"}]}
    runtime_module = sys.modules["runtime"]

    async def recovered_solver(**kwargs):
        h.model_calls.append(kwargs)
        return RunResult(output={"answer": "处方已创建"}, usage={}, status="completed",
            trace=[
                {"type": "tool_output", "name": "native_action", "output": {
                    "kind": "GET", "failure_category": "tool_error",
                    "status_code": 400, "reply": "Unknown search parameter status"}},
                {"type": "tool_output", "name": "native_action", "output": {
                    "kind": "GET", "reply": "query succeeded", "data": {"entry": []}}},
            ])

    monkeypatch.setattr(runtime_module, "run_agent", recovered_solver)
    result = evaluate(h)
    assert result.status == "completed"
    assert result.completion.completed is True
    assert h.state_reads
    completion_call = next(c for c in h.model_calls
                           if c["output_type"] is CompletionReview)
    payload = json.loads(completion_call["input_text"])
    assert payload["solver_trace"][0]["output"]["failure_category"] == "tool_error"
    assert payload["solver_trace"][1]["output"]["data"] == {"entry": []}


@pytest.mark.parametrize("failure, category, instances, runs", [
    ("evidence", "environment_error", 1, 0),
    ("verifier", "model_error", 1, 1),
    ("verifier_budget", "model_error", 1, 1),
    ("quality", "model_error", 1, 5),
    ("difficulty", "model_error", 1, 8),
    ("solver", "model_error", 2, 9),
    ("solver_tool_error", "tool_error", 2, 9),
    ("solver_environment_error", "environment_error", 2, 9),
    ("solver_model_error", "model_error", 2, 9),
    ("solver_unknown_error", "model_error", 2, 9),
    ("solver_budget", "model_error", 2, 9),
    ("state", "environment_error", 2, 9),
    ("completion", "model_error", 2, 10),
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
    if failure == "difficulty":
        assert result.quality is not None
        assert result.difficulty is None


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
    payload = json.loads(h.model_calls[0]["input_text"])
    assert set(payload) == {"evidence", "resource"}
    assert payload["evidence"] == h.task.evidence[0].model_dump()
    assert "private-reference" not in h.model_calls[0]["input_text"]


@pytest.mark.parametrize("evidence, expected_reason", [
    ([], "no evidence"),
    ([Evidence(source_id="missing", statement="unsupported")], "absent from trajectory"),
])
def test_unverifiable_evidence_stops_before_models(harness, evidence, expected_reason):
    h = harness
    h.task = h.task.model_copy(update={"evidence": evidence})
    result = evaluate(h)
    assert result.status == "invalid"
    assert result.failure_category == "insufficient_evidence"
    assert expected_reason in result.verification.reason
    assert h.model_calls == []
    assert h.created == []


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


def test_three_judges_take_median_and_keep_raw_scores(harness):
    h = harness
    h.quality_scores = [5, 3, 4]
    h.depths = [4, 1, 3]
    result = evaluate(h)
    factual = result.quality.factual_accuracy
    assert factual.score == 4
    assert factual.scores == [5, 3, 4]
    assert factual.reason == "J1: supported | J2: supported | J3: supported"
    assert result.quality.task_clarity.score == 4
    assert result.difficulty.reasoning_depth == 3
    assert result.difficulty.reasoning_depths == [4, 1, 3]
    assert result.difficulty.reason == "J1: 跨记录整合 | J2: 跨记录整合 | J3: 跨记录整合"
    assert len([c for c in h.model_calls
                if c["output_type"] is QualityJudgment]) == 3
    assert len([c for c in h.model_calls
                if c["output_type"] is DifficultyJudgment]) == 3


def test_difficulty_receives_only_task_and_computed_complexity(harness):
    h = harness
    result = evaluate(h)
    quality_call = next(c for c in h.model_calls if c["output_type"] is QualityJudgment)
    assert set(json.loads(quality_call["input_text"])) == {"task", "trajectory", "verification"}
    difficulty_call = next(c for c in h.model_calls if c["output_type"] is DifficultyJudgment)
    payload = json.loads(difficulty_call["input_text"])
    assert set(payload) == {"task", "complexity_metrics"}
    assert payload["task"]["instruction"] == h.task.instruction
    assert payload["complexity_metrics"] == {"evidence_count": 1, "resource_type_count": 0,
                                            "time_span_days": None}
    assert result.difficulty.complexity_metrics == ComplexityMetrics(
        evidence_count=1, resource_type_count=0, time_span_days=None)


def test_complexity_metrics_count_sources_types_and_time_span(harness):
    h = harness
    h.task = h.task.model_copy(update={"evidence": [
        Evidence(source_id="Observation/1", statement="首次检验"),
        Evidence(source_id="Condition/2", statement="首次诊断"),
        Evidence(source_id="Encounter/4", statement="首次就诊"),
        Evidence(source_id="MedicationRequest/5", statement=json.dumps(
            {"resourceType": "MedicationRequest",
             "effectivePeriod": {"start": "2026-01-31T00:00:00+08:00"}})),
    ]})
    h.trajectory = h.trajectory.model_copy(update={"sources": [
        Evidence(source_id="Observation/1", statement=json.dumps(
            {"resourceType": "Observation", "effectiveDateTime": "2026-01-01T00:00:00+08:00"})),
        Evidence(source_id="Condition/2", statement=json.dumps(
            {"resourceType": "Condition", "onsetDateTime": "2026-01-11T00:00:00"})),
        Evidence(source_id="Encounter/4", statement=json.dumps(
            {"resourceType": "Encounter",
             "context": {"period": {"start": "2026-01-21T00:00:00+08:00"}}})),
        Evidence(source_id="MedicationRequest/5", statement=json.dumps(
            {"resourceType": "MedicationRequest",
             "effectivePeriod": {"start": "2026-01-31T00:00:00+08:00"}})),
        Evidence(source_id="Observation/9", statement=json.dumps(
            {"resourceType": "Observation", "effectiveDateTime": "2027-05-01T00:00:00+08:00"})),
        Evidence(source_id="Patient/3", statement="不是 JSON"),
    ]})
    result = evaluate(h)
    assert result.difficulty.complexity_metrics == ComplexityMetrics(
        evidence_count=4, resource_type_count=4, time_span_days=30.0)


@pytest.mark.parametrize("role", ["quality", "difficulty"])
def test_empty_judge_reason_fails_evaluation(harness, role):
    h = harness
    h.blank_reason = role
    result = evaluate(h)
    assert result.status == "failed"
    assert result.failure_category == "model_error"
    assert result.completion is None


@pytest.mark.parametrize("field, scores", [("quality_scores", [6, 4, 4]), ("depths", [5, 2, 2])])
def test_out_of_range_judgments_fail_evaluation(harness, field, scores):
    h = harness
    setattr(h, field, scores)
    result = evaluate(h)
    assert result.status == "failed"
    assert result.failure_category == "model_error"
    assert result.completion is None
