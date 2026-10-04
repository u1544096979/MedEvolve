"""Workflow fixed-stage boundary checks with substitute runtime calls."""

import json
from unittest.mock import AsyncMock

import pytest

from contracts import (
    Budget,
    EnvironmentConfig,
    Evidence,
    GenerationDirection,
    ModelConfig,
    PatientTrajectory,
    RunResult,
    RuntimeConfig,
    TaskBatch,
)
from methods import workflow


def directions():
    return [
        GenerationDirection(task_type="诊断与解释", difficulty="整合纵向证据", weight=1),
        GenerationDirection(task_type="治疗规划", difficulty="整合纵向证据", weight=1),
    ]


def patient():
    return PatientTrajectory(
        patient_id="MRN0001",
        environment="test",
        text='[2024-01-01] Observation/o1\n{"resourceType":"Observation","id":"o1","valueString":"158/96"}'
             '\n\n[2024-03-01] MedicationRequest/m1\n{"resourceType":"MedicationRequest","id":"m1"}',
        sources=[
            Evidence(source_id="Observation/o1", statement="2024-01-01 血压 158/96 mmHg"),
            Evidence(source_id="MedicationRequest/m1", statement="2024-03-01 开始氨氯地平"),
        ],
    )


def runtime(max_model_turns=8):
    return RuntimeConfig(
        models={"generator": ModelConfig(model="generator", api_key_env="TEST_UNUSED_KEY")},
        budgets={"generator": Budget(max_model_turns=max_model_turns, max_tool_calls=40)},
    )


def completed(output, usage=None):
    return RunResult(output=output, trace=[{"raw": "response"}], usage=usage or {
        "model_turns": 1, "tool_calls": 0, "prompt_tokens": 10, "completion_tokens": 5,
    }, status="completed")


def focus_selection():
    return {
        "focuses": [
            {"direction_index": 0, "clinical_focus": "血压控制是否达标", "source_ids": ["Observation/o1"]},
            {"direction_index": 1, "clinical_focus": "降压方案的调整依据", "source_ids": ["MedicationRequest/m1"]},
        ]
    }


def candidates():
    return [
        {"instruction": "解释血压变化", "task_type": "诊断与解释",
         "reference_answer": "血压持续偏高", "evidence": [{"source_id": "Observation/o1", "statement": "158/96"}]},
        {"instruction": "制定下一步治疗", "task_type": "治疗规划",
         "reference_answer": "维持氨氯地平", "evidence": [{"source_id": "MedicationRequest/m1", "statement": "开始氨氯地平"}]},
    ]


def reviews(kept=(0,)):
    return {"reviews": [
        {"index": index, "clarity": index in kept, "answer_alignment": index in kept,
         "citations_supported": index in kept, "reason": "证据支持" if index in kept else "引用不支持声明"}
        for index in (0, 1)
    ]}


def substitute(results):
    return AsyncMock(side_effect=results)


@pytest.mark.asyncio
async def test_stages_consume_previous_products_and_accumulate_cost(monkeypatch):
    call = substitute([completed(focus_selection()), completed({"questions": candidates()}),
                       completed(reviews())])
    monkeypatch.setattr("runtime.call_llm", call)

    result = await workflow.generate_tasks(
        patient(), EnvironmentConfig(name="test", manager_url="http://unused", seed_data="seed"),
        directions(), runtime(),
    )

    assert result.status == "completed"
    questions = TaskBatch.model_validate(result.output).questions
    assert [question.instruction for question in questions] == ["解释血压变化"]
    assert questions[0].model_dump(exclude={"decision_point"}) == candidates()[0]
    assert questions[0].decision_point is None
    assert result.usage["model_turns"] == 3
    assert result.usage["total_tokens"] == 45
    assert [entry["stage"] for entry in result.trace[:3]] == ["locate", "generate", "check"]
    assert result.trace[-1] == {
        "type": "internal_check",
        "decisions": [
            {"index": 0, "keep": True, "clarity": True, "answer_alignment": True,
             "citations_supported": True, "reason": "证据支持"},
            {"index": 1, "keep": False, "clarity": False, "answer_alignment": False,
             "citations_supported": False, "reason": "引用不支持声明"},
        ],
        "kept": [0], "dropped": [1],
    }

    assert [entry.kwargs["output_type"] for entry in call.await_args_list] == [
        workflow.FocusSelection, TaskBatch, workflow.BatchReview,
    ]
    assert len({entry.kwargs["instructions"] for entry in call.await_args_list}) == 3
    payloads = [json.loads(entry.kwargs["input_text"]) for entry in call.await_args_list]
    assert payloads[0]["trajectory"]["patient_id"] == "MRN0001"
    assert payloads[0]["trajectory"]["sources"][0]["source_id"] == "Observation/o1"
    assert payloads[1]["focuses"][1]["clinical_focus"] == "降压方案的调整依据"
    assert payloads[1]["directions"][0]["task_type"] == "诊断与解释"
    assert payloads[2]["candidates"][0]["candidate"] == {**candidates()[0], "decision_point": None}
    assert payloads[2]["candidates"][0]["evidence_records"] == [
        {"source_id": "Observation/o1",
         "record": '{"resourceType":"Observation","id":"o1","valueString":"158/96"}'},
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "stage_index,expected_calls",
    [(0, 1), (1, 2)],
)
async def test_failed_stage_stops_flow_and_keeps_cost(monkeypatch, stage_index, expected_calls):
    results = [completed(focus_selection()), completed({"questions": candidates()}),
               completed(reviews())]
    results[stage_index] = RunResult(
        trace=[{"raw": "provider error"}], status="failed", error="RuntimeError: provider unavailable",
        usage={"model_turns": 1, "tool_calls": 0, "prompt_tokens": 10, "completion_tokens": 5},
    )
    call = substitute(results)
    monkeypatch.setattr("runtime.call_llm", call)

    result = await workflow.generate_tasks(
        patient(), EnvironmentConfig(name="test", manager_url="http://unused", seed_data="seed"),
        directions(), runtime(),
    )

    assert result.status == "failed"
    assert result.output is None
    assert result.usage["model_turns"] == expected_calls
    assert call.await_count == expected_calls


@pytest.mark.asyncio
async def test_incomplete_check_verdicts_are_not_accepted_as_batch(monkeypatch):
    call = substitute([completed(focus_selection()), completed({"questions": candidates()}),
                       completed({"reviews": [reviews()["reviews"][0]]})])
    monkeypatch.setattr("runtime.call_llm", call)

    result = await workflow.generate_tasks(
        patient(), EnvironmentConfig(name="test", manager_url="http://unused", seed_data="seed"),
        directions(), runtime(),
    )

    assert result.status == "failed"
    assert result.output is None
    assert "every candidate" in result.error
    assert result.usage["model_turns"] == 3


@pytest.mark.asyncio
async def test_cumulative_budget_blocks_later_stage_and_empty_generation_skips_check(monkeypatch):
    call = substitute([completed(focus_selection()), completed({"questions": candidates()})])
    monkeypatch.setattr("runtime.call_llm", call)

    result = await workflow.generate_tasks(
        patient(), EnvironmentConfig(name="test", manager_url="http://unused", seed_data="seed"),
        directions(), runtime(max_model_turns=2),
    )

    assert call.await_count == 2
    assert result.status == "budget_exhausted"
    assert result.output is None
    assert result.usage["model_turns"] == 2

    empty = substitute([completed(focus_selection()), completed({"questions": []})])
    monkeypatch.setattr("runtime.call_llm", empty)
    result = await workflow.generate_tasks(
        patient(), EnvironmentConfig(name="test", manager_url="http://unused", seed_data="seed"),
        directions(), runtime(),
    )

    assert empty.await_count == 2
    assert result.status == "completed"
    assert TaskBatch.model_validate(result.output).questions == []
    assert [entry["stage"] for entry in result.trace] == ["locate", "generate"]

    with pytest.raises(ValueError, match="exactly two"):
        await workflow.generate_tasks(
            patient(), EnvironmentConfig(name="test", manager_url="http://unused", seed_data="seed"),
            directions()[:1], runtime(),
        )
    assert empty.await_count == 2
