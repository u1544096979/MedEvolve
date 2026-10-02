"""SingleShot boundary: one generation call, no exploration or repair."""

import json
from unittest.mock import AsyncMock

import pytest

import runtime as runtime_module
from contracts import (
    Budget,
    EnvironmentConfig,
    Evidence,
    GeneratedQuestion,
    GenerationDirection,
    ModelConfig,
    PatientTrajectory,
    RunResult,
    RuntimeConfig,
    TaskBatch,
)
from methods import singleshot


@pytest.fixture
def runtime():
    return RuntimeConfig(
        models={"generator": ModelConfig(model="generator", api_key_env="TEST_UNUSED_KEY")},
        budgets={"generator": Budget(max_model_turns=1, max_tool_calls=0)},
    )


@pytest.fixture
def environment():
    return EnvironmentConfig(name="test", manager_url="http://unused", seed_data="seed")


def trajectory():
    return PatientTrajectory(
        patient_id="patient-1",
        environment="test",
        text="[2020-01-01] Observation/o1\n{}",
        sources=[Evidence(source_id="Observation/o1", statement="观测")],
    )


def directions():
    return [GenerationDirection(task_type="诊断与解释", difficulty="基础", weight=1),
            GenerationDirection(task_type="检查与风险分层", difficulty="基础", weight=1)]


def question(task_type="诊断与解释"):
    return GeneratedQuestion(
        instruction="解释病程变化", task_type=task_type, reference_answer="参考答案",
        evidence=[Evidence(source_id="Observation/o1", statement="观测")],
    )


def response(*, questions=(), status="completed", usage=None):
    return RunResult(
        output=TaskBatch(questions=list(questions)).model_dump(mode="json") if status == "completed" else None,
        trace=[{"stage": "raw"}], usage=usage or {"model_turns": 1, "prompt_tokens": 11, "completion_tokens": 4},
        status=status, error="call failed" if status == "failed" else None,
    )


@pytest.mark.asyncio
async def test_single_call_returns_public_constrained_batch(runtime, environment, monkeypatch):
    patient = trajectory()
    call = AsyncMock(return_value=response(questions=[question()]))
    agent = AsyncMock(side_effect=AssertionError("unexpected tool agent"))
    monkeypatch.setattr(runtime_module, "call_llm", call)
    monkeypatch.setattr(runtime_module, "run_agent", agent)

    result = await singleshot.generate_tasks(patient, environment, directions(), runtime)

    assert result.status == "completed"
    assert TaskBatch.model_validate(result.output).questions == [question()]
    assert result.usage == {"model_turns": 1, "tool_calls": 0, "input_tokens": 11, "output_tokens": 4,
                            "total_tokens": 15, "runs_without_known_tokens": 0}
    call.assert_awaited_once()
    agent.assert_not_awaited()
    request = call.await_args.kwargs
    assert request["output_type"] is TaskBatch
    assert request["model"] is runtime.models["generator"]
    payload = json.loads(request["input_text"])
    assert set(payload) == {"trajectory", "directions"}
    assert payload["trajectory"] == {"patient_id": patient.patient_id, "environment": "test",
                                     "text": patient.text, "sources": ["Observation/o1"]}
    assert payload["directions"] == [direction.model_dump(mode="json") for direction in directions()]
    assert "trajectory" not in request["instructions"] and "guidance" not in request["instructions"]


@pytest.mark.asyncio
@pytest.mark.parametrize("output,status", [([question(task_type="药物处方")], "completed"), ([], "failed")])
async def test_rejected_or_failed_generation_is_not_retried(
    runtime, environment, monkeypatch, output, status
):
    call = AsyncMock(return_value=response(questions=output, status=status))
    monkeypatch.setattr(runtime_module, "call_llm", call)

    result = await singleshot.generate_tasks(trajectory(), environment, directions(), runtime)

    assert result.status == "failed" and result.output is None
    assert result.trace[0]["stage"] == "generate"
    assert call.await_count == 1
    if status == "completed":
        assert "exceed requested directions" in result.error
        assert result.trace[-1]["type"] == "output_rejected"
        assert len(result.trace[-1]["output"]["questions"]) == 1
    else:
        assert result.error == "call failed"

    with pytest.raises(ValueError, match="exactly two"):
        await singleshot.generate_tasks(trajectory(), environment, directions()[:1], runtime)
    assert call.await_count == 1
