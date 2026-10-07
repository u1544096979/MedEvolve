import json
from unittest.mock import AsyncMock

import pytest
from agents.tool_context import ToolContext

from contracts import Budget, ModelConfig, PatientTrajectory, RunResult, RuntimeConfig
from methods import medevolve
from methods.medevolve.compression import COMPRESSION, CompressedTrajectory, compression_input


def model(window=12000):
    return ModelConfig(model="compressor", api_key_env="UNUSED", context_window=window, max_output_tokens=512)


def trajectory():
    return PatientTrajectory(patient_id="train", environment="PhysicianBench", text='日期"病历\\记录' * 8000, sources=[])


def test_context_truncation_preserves_json_and_original_cache():
    patient = trajectory()
    original = patient.model_dump()
    payload = compression_input(patient, model())
    parsed = json.loads(payload)
    assert parsed["input_truncated"] is True
    assert patient.text.startswith(parsed["text"])
    instructions = COMPRESSION + "\nReturn only a JSON object matching this schema:\n" + json.dumps(CompressedTrajectory.model_json_schema(), ensure_ascii=False)
    assert len(payload.encode()) + len(instructions.encode()) + 4096 + 512 <= 12000
    assert patient.model_dump() == original
    assert json.loads(compression_input(patient, model(1000000)))["input_truncated"] is False
    with pytest.raises(ValueError, match="context_window"):
        compression_input(patient, ModelConfig(model="unset", api_key_env="UNUSED"))
    with pytest.raises(ValueError, match="no room"):
        compression_input(patient, model(100))


@pytest.mark.asyncio
@pytest.mark.parametrize("failed", [False, True])
async def test_compression_tool_one_call_scope_and_cost(monkeypatch, failed):
    compressed = RunResult(output=None if failed else {"summary": "仅总结可见病历"},
        trace=[{"type": "model_response"}], usage={"model_turns": 1, "tool_calls": 0,
        "prompt_tokens": 9, "completion_tokens": 3, "total_tokens": 12},
        status="failed" if failed else "completed", error="timeout" if failed else None)
    call = AsyncMock(return_value=compressed)
    monkeypatch.setattr(medevolve, "call_llm", call)
    patient = trajectory()
    runtime = RuntimeConfig(models={"data_explorer": model(), "trajectory_compressor": model()},
                            budgets={"data_explorer": Budget(max_model_turns=3, max_tool_calls=5)})

    async def agent(**kwargs):
        tool = next(t for t in kwargs["tools"] if t.name == "compress_trajectory")
        assert "patient_id" in tool.params_json_schema["required"]
        async def invoke(args):
            raw = json.dumps(args)
            ctx = ToolContext(context=None, tool_name=tool.name, tool_call_id="compress", tool_arguments=raw)
            return json.loads(await tool.on_invoke_tool(ctx, raw))
        denied = await invoke({"patient_id": "test"})
        assert "outside" in denied["error"]
        call.assert_not_awaited()
        output = await invoke({"patient_id": "train"})
        if failed:
            assert output == {"error": "timeout"}
        else:
            assert output == {"patient_id": "train", "input_truncated": True, "summary": "仅总结可见病历"}
        return RunResult(output={"guidance": "data"}, trace=[], status="completed",
                         usage={"model_turns": 2, "tool_calls": 2, "input_tokens": 20, "output_tokens": 5, "total_tokens": 25})

    monkeypatch.setattr(medevolve, "run_agent", agent)
    result = await medevolve.explore_data([patient], runtime)
    call.assert_awaited_once()
    assert call.call_args.kwargs["model"] == runtime.models["trajectory_compressor"]
    assert result.usage["model_turns"] == 3 and result.usage["total_tokens"] == 37
    assert result.trace[-1]["result"]["status"] == compressed.status
