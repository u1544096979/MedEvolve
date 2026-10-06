"""SingleShot boundary: one generation call, free task selection, no repair."""

import json
from typing import get_args
from unittest.mock import AsyncMock

import pytest

import runtime as runtime_module
from contracts import (
    Budget,
    EnvironmentConfig,
    Evidence,
    GeneratedQuestion,
    ModelConfig,
    PatientTrajectory,
    RunResult,
    RuntimeConfig,
    TaskBatch,
    TaskType,
)
from methods import singleshot


def environment(name="PhysicianBench"):
    return EnvironmentConfig(name=name, manager_url="http://unused", seed_data="seed")


def trajectory(name="PhysicianBench"):
    return PatientTrajectory(
        patient_id="patient-1",
        environment=name,
        text="[2020-01-01] Observation/o1\n{}",
        sources=[Evidence(source_id="Observation/o1", statement="观测")],
    )


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
@pytest.mark.parametrize("name", ["PhysicianBench", "MedAgentBench"])
async def test_full_sources_and_same_type_questions(name, monkeypatch):
    types = list(get_args(TaskType))
    env = EnvironmentConfig(name=name, manager_url="http://unused", seed_data="seed")
    patient = PatientTrajectory(patient_id="p", environment=name,
        text="原始患者轨迹", sources=[Evidence(source_id="Patient/p",
            statement='{"resourceType":"Patient","id":"p","gender":"female"}')])
    cfg = RuntimeConfig(models={"generator": ModelConfig(model="mock",
        api_key_env="UNUSED")}, budgets={"generator": Budget(
        max_model_turns=8, max_tool_calls=40)})
    q = GeneratedQuestion(instruction="说明记录内容", task_type="诊断与解释",
        reference_answer="记录内容", evidence=[Evidence(
            source_id="Patient/p", statement="gender=female")])
    call = AsyncMock(return_value=RunResult(output=TaskBatch(
        questions=[q, q]).model_dump(mode="json"), trace=[],
        usage={"model_turns": 1, "tool_calls": 0}, status="completed"))
    agent = AsyncMock(side_effect=AssertionError("unexpected generation agent"))
    monkeypatch.setattr(runtime_module, "call_llm", call)
    monkeypatch.setattr(runtime_module, "run_agent", agent)

    result = await singleshot.generate_tasks(patient, env, types, cfg)

    assert result.status == "completed"
    assert len(TaskBatch.model_validate(result.output).questions) == 2
    call.assert_awaited_once()
    agent.assert_not_awaited()
    request = call.await_args.kwargs
    assert request["output_type"] is TaskBatch
    assert request["model"] is cfg.models["generator"]
    payload = json.loads(request["input_text"])
    assert set(payload) == {"trajectory", "task_types"}
    assert payload["task_types"] == types
    assert payload["trajectory"] == patient.model_dump(mode="json")
    assert payload["trajectory"]["sources"][0]["statement"] == patient.sources[0].statement
    assert "directions" not in payload and "guidance" not in payload
    assert result.usage["model_turns"] == 1
    assert result.usage["tool_calls"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("budget_turns,status,output,expected,calls", [
    (0, "completed", {"questions": []}, "budget_exhausted", 0),
    (1, "completed", {"questions": []}, "completed", 1),
    (8, "failed", None, "failed", 1),
    (8, "completed", {"questions": [{"instruction": "题干",
        "task_type": "未知题型", "reference_answer": "答案", "evidence": []}]},
        "failed", 1),
])
async def test_budget_and_failure_do_not_retry(
        budget_turns, status, output, expected, calls, monkeypatch):
    env = EnvironmentConfig(name="PhysicianBench", manager_url="http://unused", seed_data="s")
    patient = PatientTrajectory(patient_id="p", environment=env.name, text="records", sources=[])
    cfg = RuntimeConfig(models={"generator": ModelConfig(model="mock", api_key_env="UNUSED")},
        budgets={"generator": Budget(max_model_turns=budget_turns, max_tool_calls=0)})
    call = AsyncMock(return_value=RunResult(output=output, status=status,
        error="request failed" if status == "failed" else None,
        trace=[{"raw": "response"}], usage={"model_turns": 1, "tool_calls": 0}))
    monkeypatch.setattr(runtime_module, "call_llm", call)

    result = await singleshot.generate_tasks(patient, env, list(get_args(TaskType)), cfg)

    assert result.status == expected and call.await_count == calls
    assert result.usage["model_turns"] == calls
    if expected != "completed":
        assert result.output is None
    if calls and expected == "failed":
        assert result.trace
    with pytest.raises(ValueError):
        await singleshot.generate_tasks(patient, env, list(get_args(TaskType))[:-1], cfg)
    assert call.await_count == calls


@pytest.mark.asyncio
async def test_mismatched_environment_is_rejected_before_generation(monkeypatch):
    call = AsyncMock(side_effect=AssertionError("unexpected generation call"))
    monkeypatch.setattr(runtime_module, "call_llm", call)
    cfg = RuntimeConfig(models={"generator": ModelConfig(model="mock", api_key_env="UNUSED")},
        budgets={"generator": Budget(max_model_turns=1, max_tool_calls=0)})

    with pytest.raises(ValueError, match="does not match the trajectory"):
        await singleshot.generate_tasks(trajectory("PhysicianBench"), environment("MedAgentBench"),
                                        list(get_args(TaskType)), cfg)

    call.assert_not_awaited()
