import json
from unittest.mock import AsyncMock

import pytest
from agents.tool_context import ToolContext

from contracts import Budget, Evidence, ModelConfig, PatientTrajectory, RunResult, RuntimeConfig
from methods import medevolve
from methods.medevolve.compression import COMPRESSION, CompressedTrajectory, compression_input


def model(window=12000):
    return ModelConfig(model="compressor", api_key_env="UNUSED", context_window=window, max_output_tokens=512)


def trajectory():
    sources = [Evidence(source_id=f"Observation/o{index}",
                        statement=f'日期"病历\\记录 {index} 值 12 mg/dL')
               for index in range(300)]
    return PatientTrajectory(patient_id="train", environment="PhysicianBench",
        text="\n\n".join(f"[2025-01-{index % 28 + 1:02d}] {source.source_id}\n{source.statement}"
                          for index, source in enumerate(sources)), sources=sources)


def test_context_truncation_preserves_json_and_original_cache():
    patient = trajectory()
    original = patient.model_dump()
    payload = compression_input(patient, model())
    parsed = json.loads(payload)
    assert parsed["input_truncated"] is True
    source_map = {source["source_id"]: source["statement"] for source in parsed["sources"]}
    assert all(source_id in parsed["text"] and statement in parsed["text"]
               for source_id, statement in source_map.items())
    instructions = COMPRESSION + "\nReturn only a JSON object matching this schema:\n" + json.dumps(CompressedTrajectory.model_json_schema(), ensure_ascii=False)
    assert len(payload.encode()) + len(instructions.encode()) + 512 <= 12000 - 1024
    assert patient.model_dump() == original
    assert json.loads(compression_input(patient, model(1000000)))["input_truncated"] is False
    assert not json.loads(compression_input(
        patient, ModelConfig(model="unset", api_key_env="UNUSED")))['input_truncated']
    with pytest.raises(ValueError, match="input budget"):
        compression_input(patient, model(100))


def test_compression_input_keeps_whole_records_with_matching_sources():
    patient = trajectory()
    payload = json.loads(compression_input(patient, model()))
    source_map = {source["source_id"]: source["statement"] for source in payload["sources"]}
    blocks = payload["text"].split("\n\n")
    assert payload["input_truncated"] is True
    assert len(blocks) == len(source_map)
    assert all(any(block.startswith(f"[2025-") and source_id in block
                       and statement in block
                       for block in blocks)
               for source_id, statement in source_map.items())
    assert all(any(source_id in block for block in blocks) for source_id in source_map)


def test_unlocated_oversized_text_is_not_cut_into_an_unreferenced_fragment():
    patient = PatientTrajectory(patient_id="train", environment="PhysicianBench",
        text="未定位临床记录" * 10000, sources=[])
    with pytest.raises(ValueError, match="unlocated text"):
        compression_input(patient, model())


@pytest.mark.asyncio
async def test_unfittable_required_record_fails_before_compressor_call(monkeypatch):
    patient = PatientTrajectory(patient_id="train", environment="PhysicianBench",
        text="Patient/train\n" + "临床记录" * 10000,
        sources=[Evidence(source_id="Patient/train", statement="临床记录" * 10000)])
    call = AsyncMock(return_value=RunResult(output={"summary": "摘要"}, trace=[],
        usage={"model_turns": 1}, status="completed"))
    monkeypatch.setattr(medevolve, "call_llm", call)
    runtime = RuntimeConfig(models={"data_explorer": model(), "trajectory_compressor": model()},
                            budgets={"data_explorer": Budget(max_model_turns=3, max_tool_calls=5)})

    async def agent(**kwargs):
        tool = next(t for t in kwargs["tools"] if t.name == "compress_trajectory")
        raw = json.dumps({"patient_id": "train"})
        ctx = ToolContext(context=None, tool_name=tool.name, tool_call_id="compress", tool_arguments=raw)
        output = json.loads(await tool.on_invoke_tool(ctx, raw))
        assert "error" in output
        call.assert_not_awaited()
        return RunResult(output={"guidance": "data"}, trace=[], status="completed",
                         usage={"model_turns": 0, "tool_calls": 1})

    monkeypatch.setattr(medevolve, "run_agent", agent)
    result = await medevolve.explore_data([patient], runtime)
    assert result.status == "completed"
    call.assert_not_awaited()


def test_summary_preserves_partial_material_marker():
    patient = trajectory().model_copy(update={"input_truncated": True})
    assert json.loads(compression_input(patient, model(1000000)))["input_truncated"] is True


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
