import json
from unittest.mock import AsyncMock
import pytest
from contracts import (
    Budget, Evidence, ExplorationGuidance, GenerationDirection,
    ModelConfig, PatientTrajectory, RunResult, RuntimeConfig, Strategy,
)
from methods import medevolve
from methods.generation import trajectory_payload

@pytest.mark.asyncio
async def test_main_generation_keeps_material_and_one_call(monkeypatch):
    trajectory = PatientTrajectory(
        patient_id="p1", environment="PhysicianBench", text="完整轨迹正文",
        sources=[Evidence(source_id="Condition/c1", statement='{"id":"c1"}')],
    )
    directions = [GenerationDirection(task_type=t, difficulty="基础", weight=1)
                  for t in ("诊断与解释", "治疗规划")]
    guidance = ExplorationGuidance(data_guidance="数据指导", environment_guidance="环境指导")
    strategy = Strategy(generation_guidance="初始策略", allocation=directions)
    runtime = RuntimeConfig(
        models={"generator": ModelConfig(model="mock", api_key_env="UNUSED")},
        budgets={"generator": Budget(max_model_turns=1, max_tool_calls=0)},
    )
    call = AsyncMock(return_value=RunResult(
        output={"questions": []}, trace=[], usage={"model_turns": 1}, status="completed",
    ))
    monkeypatch.setattr(medevolve, "call_llm", call)
    result = await medevolve.generate_tasks(trajectory, guidance, strategy, directions, runtime)
    assert result.status == "completed"
    call.assert_awaited_once()
    body = json.loads(call.await_args.kwargs["input_text"])
    assert body["trajectory"] == trajectory.model_dump(mode="json")
    assert body["guidance"] == guidance.model_dump(mode="json")
    assert body["strategy"] == {"generation_guidance": "初始策略"}
    assert body["directions"] == [d.model_dump(mode="json") for d in directions]
    assert "本轮实验只允许基于既有患者记录生成查询任务" in call.await_args.kwargs["instructions"]
    assert 'execution_requirement 必须为 "read_only"' in call.await_args.kwargs["instructions"]
    assert trajectory_payload(trajectory) == body["trajectory"]


@pytest.mark.asyncio
async def test_final_batch_requires_explicit_read_only_requirement(monkeypatch):
    trajectory = PatientTrajectory(patient_id="p", environment="PhysicianBench", text="record", sources=[])
    directions = [GenerationDirection(task_type="诊断与解释", difficulty="基础", weight=1),
                  GenerationDirection(task_type="治疗规划", difficulty="基础", weight=1)]
    guidance = ExplorationGuidance(data_guidance="数据", environment_guidance="环境")
    strategy = Strategy(generation_guidance="策略", allocation=directions)
    runtime = RuntimeConfig(models={"generator": ModelConfig(model="mock", api_key_env="UNUSED")},
        budgets={"generator": Budget(max_model_turns=1, max_tool_calls=0)})
    monkeypatch.setattr("methods.medevolve.call_llm", AsyncMock(return_value=RunResult(
        output={"questions": [{"instruction": "诊断", "task_type": "诊断与解释",
            "reference_answer": "依据记录", "evidence": [], "decision_point": None}]},
        trace=[], usage={"model_turns": 1}, status="completed")))
    result = await medevolve.generate_tasks(trajectory, guidance, strategy, directions, runtime)
    assert result.status == "failed" and result.output is None
    assert "execution_requirement must be read_only" in result.error
