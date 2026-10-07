import json
from unittest.mock import AsyncMock
import pytest
from contracts import (
    Budget, EnvironmentConfig, EnvironmentExploration, Evidence,
    GenerationDirection, ModelConfig, PatientTrajectory, RunResult,
    RuntimeConfig, Strategy,
)
from methods import envexplore

@pytest.fixture
def ctx():
    ds = [GenerationDirection(task_type=t, difficulty="基础", weight=1)
          for t in ("诊断与解释", "治疗规划")]
    return dict(
        trajectory=PatientTrajectory(patient_id="p1", environment="PhysicianBench",
            text="完整原始轨迹", sources=[Evidence(source_id="Condition/c1",
                statement='{"resourceType":"Condition","id":"c1"}')]),
        environment=EnvironmentConfig(name="PhysicianBench", manager_url="http://unused", seed_data="seed"),
        directions=ds,
        runtime=RuntimeConfig(models={"generator": ModelConfig(model="mock", api_key_env="UNUSED")},
            budgets={"generator": Budget(max_model_turns=1, max_tool_calls=0)}),
        environment_guidance=EnvironmentExploration(guidance="已验证可读取诊断"),
        strategy=Strategy(generation_guidance="固定初始策略", allocation=ds),
    )

def candidate(source="Condition/c1", evidence=True):
    return dict(instruction="解释现有诊断", task_type="诊断与解释",
        reference_answer="依据记录解释", decision_point=None, execution_requirement="read_only",
        evidence=[dict(source_id=source, statement="诊断记录")] if evidence else [])

def completed(questions):
    return RunResult(output={"questions": questions}, trace=[{"raw": "response"}],
        usage={"model_turns": 1, "tool_calls": 0, "input_tokens": 7, "output_tokens": 3},
        status="completed")

@pytest.mark.asyncio
async def test_one_call_full_material_and_fixed_context(ctx, monkeypatch):
    call = AsyncMock(return_value=completed([candidate()]))
    agent = AsyncMock(side_effect=AssertionError("生成阶段调用工具 Agent"))
    monkeypatch.setattr("runtime.call_llm", call)
    monkeypatch.setattr("runtime.run_agent", agent)
    result = await envexplore.generate_tasks(**ctx)
    call.assert_awaited_once()
    agent.assert_not_awaited()
    body = json.loads(call.await_args.kwargs["input_text"])
    assert body["trajectory"] == ctx["trajectory"].model_dump(mode="json")
    assert body["guidance"] == {"environment_guidance": ctx["environment_guidance"].guidance}
    assert body["strategy"] == {"generation_guidance": ctx["strategy"].generation_guidance}
    assert result.status == "completed" and len(result.output["questions"]) == 1
    assert result.usage["model_turns"] == 1 and result.usage["tool_calls"] == 0
    assert result.usage["total_tokens"] == 10

@pytest.mark.asyncio
@pytest.mark.parametrize("bad", [candidate(evidence=False), candidate("Condition/missing")])
async def test_bad_sources_drop_only_bad_candidate(ctx, monkeypatch, bad):
    monkeypatch.setattr("runtime.call_llm", AsyncMock(return_value=completed([bad, candidate()])))
    ctx["directions"] = [ctx["directions"][0], ctx["directions"][0]]
    result = await envexplore.generate_tasks(**ctx)
    assert result.status == "completed" and result.output["questions"] == [candidate()]
    rejection = next(t for t in result.trace if t.get("type") == "evidence_references_rejected")
    assert rejection["questions"][0]["question"] == bad
    assert result.trace[0]["result"]["output"]["questions"][0] == bad

@pytest.mark.asyncio
async def test_zero_budget_and_failed_call_keep_status(ctx, monkeypatch):
    call = AsyncMock(return_value=RunResult(trace=[{"raw": "error"}],
        usage={"model_turns": 1, "input_tokens": 7, "output_tokens": 3},
        status="failed", error="provider offline"))
    monkeypatch.setattr("runtime.call_llm", call)
    ctx["runtime"].budgets["generator"].max_model_turns = 0
    stopped = await envexplore.generate_tasks(**ctx)
    assert stopped.status == "budget_exhausted" and stopped.output is None
    call.assert_not_awaited()
    ctx["runtime"].budgets["generator"].max_model_turns = 1
    failed = await envexplore.generate_tasks(**ctx)
    assert failed.status == "failed" and failed.output is None
    assert failed.usage["total_tokens"] == 10
    call.assert_awaited_once()

@pytest.mark.asyncio
@pytest.mark.parametrize("questions", [[], [candidate(evidence=False)]])
async def test_empty_or_all_rejected_has_no_tasks(ctx, monkeypatch, questions):
    monkeypatch.setattr("runtime.call_llm", AsyncMock(return_value=completed(questions)))
    result = await envexplore.generate_tasks(**ctx)
    assert result.status == "completed" and result.output == {"questions": []}

@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["time", "type", "structure"])
async def test_invalid_output_is_not_repaired(ctx, monkeypatch, change):
    q = candidate()
    if change == "time": q["decision_point"] = "2024-01-01"
    if change == "type": q["task_type"] = "药物处方"
    response = completed([q])
    if change == "structure": response.output = {"questions": "invalid"}
    call = AsyncMock(return_value=response)
    monkeypatch.setattr("runtime.call_llm", call)
    result = await envexplore.generate_tasks(**ctx)
    assert result.status == "failed" and result.output is None
    assert result.usage["total_tokens"] == 10
    call.assert_awaited_once()

@pytest.mark.asyncio
async def test_bad_input_stops_before_model(ctx, monkeypatch):
    call = AsyncMock()
    monkeypatch.setattr("runtime.call_llm", call)
    ctx["environment"] = ctx["environment"].model_copy(update={"name": "MedAgentBench"})
    with pytest.raises(ValueError, match="environment"):
        await envexplore.generate_tasks(**ctx)
    call.assert_not_awaited()
