"""Workflow fixed-stage boundary checks with substitute runtime calls."""

import json
from typing import get_args
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
    TaskType,
)
from methods import workflow


def directions():
    return [
        GenerationDirection(task_type="诊断与解释", difficulty="整合纵向证据", weight=1),
        GenerationDirection(task_type="治疗规划", difficulty="整合纵向证据", weight=1),
    ]


def directions_pool():
    order: list[TaskType] = ["诊断与解释", "检查与风险分层", "治疗规划", "药物处方", "治疗重规划"]
    return [GenerationDirection(task_type=task_type, difficulty="整合纵向证据", weight=1)
            for task_type in order]


def environment():
    return EnvironmentConfig(name="PhysicianBench", manager_url="http://unused", seed_data="seed")


def patient():
    return PatientTrajectory(
        patient_id="MRN0001",
        environment="PhysicianBench",
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


def pool_selection():
    return workflow.FocusSelection(focuses=[
        dict(direction_index=0, clinical_focus="血压控制", source_ids=["Observation/o1"]),
        dict(direction_index=3, clinical_focus="降压用药", source_ids=["MedicationRequest/m1"]),
    ])


def located():
    return pool_selection().model_dump(mode="json")


def candidate(task_type, instruction, source_id, statement):
    return {"instruction": instruction, "task_type": task_type, "reference_answer": "依据轨迹给出结论",
            "evidence": [{"source_id": source_id, "statement": statement}],
            "execution_requirement": "read_only"}


def first_candidate():
    return candidate("诊断与解释", "解释血压变化", "Observation/o1", "158/96")


def second_candidate():
    return candidate("药物处方", "给出降压用药方案", "MedicationRequest/m1", "开始氨氯地平")


def reviews(kept=(0,), count=2):
    return {"reviews": [
        {"index": index, "clarity": index in kept, "answer_alignment": index in kept,
         "citations_supported": index in kept, "reason": "证据支持" if index in kept else "引用不支持声明"}
        for index in range(count)
    ]}


def decisions_output():
    return {
        "type": "internal_check",
        "decisions": [
            {"index": 0, "keep": True, "clarity": True, "answer_alignment": True,
             "citations_supported": True, "reason": "证据支持"},
            {"index": 1, "keep": False, "clarity": False, "answer_alignment": False,
             "citations_supported": False, "reason": "引用不支持声明"},
        ],
        "kept": [0], "dropped": [1],
    }


def selection_output():
    return {
        "type": "selection",
        "selected_directions": [0, 3],
        "focuses": [
            {"direction_index": 0, "clinical_focus": "血压控制", "source_ids": ["Observation/o1"]},
            {"direction_index": 3, "clinical_focus": "降压用药", "source_ids": ["MedicationRequest/m1"]},
        ],
    }


def substitute(results):
    return AsyncMock(side_effect=results)


def test_selected_focuses_match_pool_and_reject_duplicates():
    pool = directions_pool()
    selection = pool_selection()

    focuses, drops = workflow._selected_focuses(selection, pool, patient())

    assert [focus.direction_index for focus in focuses] == [0, 3]
    assert focuses == selection.focuses
    assert drops == []
    duplicate = selection.model_copy(update={"focuses": [selection.focuses[0]] * 2})
    with pytest.raises(ValueError):
        workflow._selected_focuses(duplicate, pool, patient())


def test_selected_focuses_reject_three_focuses_and_blank_topic():
    pool = directions_pool()
    selection = pool_selection()
    third = dict(direction_index=2, clinical_focus="检查选择", source_ids=["Observation/o1"])
    cases = [
        selection.model_copy(update={"focuses": [*selection.focuses, third]}),
        selection.model_copy(update={"focuses": [
            selection.focuses[0],
            selection.focuses[1].model_copy(update={"clinical_focus": "   "}),
        ]}),
    ]

    for case in cases:
        with pytest.raises(ValueError):
            workflow._selected_focuses(case, pool, patient())


def test_selected_focuses_drop_unknown_sources_and_keep_the_rest():
    pool = directions_pool()
    selection = pool_selection()
    mixed = selection.model_copy(update={"focuses": [
        selection.focuses[0].model_copy(
            update={"source_ids": ["Observation/o1", "Observation/missing"]}),
        selection.focuses[1],
    ]})

    focuses, drops = workflow._selected_focuses(mixed, pool, patient())

    assert [focus.source_ids for focus in focuses] == [["Observation/o1"], ["MedicationRequest/m1"]]
    assert drops == [{"focus": 0, "reason": "unknown source",
                      "source_ids": ["Observation/missing"]}]


def test_selected_focuses_reject_when_every_focus_loses_its_sources():
    pool = directions_pool()
    selection = pool_selection()
    missing = selection.model_copy(update={"focuses": [
        focus.model_copy(update={"source_ids": ["Observation/missing"]})
        for focus in selection.focuses
    ]})

    with pytest.raises(ValueError):
        workflow._selected_focuses(missing, pool, patient())


def test_selected_focuses_require_a_valid_task_pool():
    selection = pool_selection()

    with pytest.raises(ValueError):
        workflow._selected_focuses(selection, directions(), patient())


def test_locate_prompt_selects_by_fit_without_weight_sampling():
    assert "分别为 0 和 1" not in workflow.LOCATE
    assert "互不相同" in workflow.LOCATE
    assert "权重" in workflow.LOCATE


def test_internal_check_prompt_uses_trajectory_body_records():
    assert "轨迹正文" in workflow.INTERNAL_CHECK
    assert "原始记录" not in workflow.INTERNAL_CHECK


def test_generate_prompts_cover_every_task_type():
    assert set(workflow.GENERATE_BY_TYPE) == set(get_args(TaskType))


@pytest.mark.asyncio
async def test_nodes_generate_one_type_each_then_unified_check(monkeypatch):
    call = substitute([completed(located()), completed({"questions": [first_candidate()]}),
                       completed({"questions": [second_candidate()]}), completed(reviews())])
    monkeypatch.setattr("runtime.call_llm", call)

    result = await workflow.generate_tasks(patient(), environment(), directions_pool(), runtime())

    assert result.status == "completed"
    questions = TaskBatch.model_validate(result.output).questions
    assert [question.instruction for question in questions] == ["解释血压变化"]
    assert questions[0].model_dump(exclude={"decision_point"}) == first_candidate()
    assert questions[0].decision_point is None
    assert result.usage["model_turns"] == 4
    assert result.usage["total_tokens"] == 60
    assert [entry["stage"] for entry in result.trace[:4]] == [
        "locate", "generate_0", "generate_1", "check"]
    assert result.trace[4] == selection_output()
    assert result.trace[-1] == decisions_output()

    assert [entry.kwargs["output_type"] for entry in call.await_args_list] == [
        workflow.FocusSelection, TaskBatch, TaskBatch, workflow.BatchReview]
    payloads = [json.loads(entry.kwargs["input_text"]) for entry in call.await_args_list]
    assert payloads[0]["trajectory"]["patient_id"] == "MRN0001"
    assert payloads[0]["trajectory"]["sources"][0]["source_id"] == "Observation/o1"
    assert payloads[0]["directions"][0]["task_type"] == "诊断与解释"
    assert set(payloads[1]) == {"trajectory", "focus", "direction"}
    assert payloads[1]["focus"]["direction_index"] == 0
    assert payloads[1]["direction"]["task_type"] == "诊断与解释"
    assert payloads[2]["focus"]["direction_index"] == 3
    assert payloads[2]["direction"]["task_type"] == "药物处方"
    assert payloads[3]["candidates"][0]["candidate"] == {**first_candidate(), "decision_point": None}
    assert payloads[3]["candidates"][1]["candidate"] == {**second_candidate(), "decision_point": None}
    assert payloads[3]["candidates"][0]["evidence_records"] == [
        {"source_id": "Observation/o1",
         "record": '[2024-01-01] Observation/o1\n'
                   '{"resourceType":"Observation","id":"o1","valueString":"158/96"}'},
    ]
    assert payloads[3]["candidates"][1]["evidence_records"] == [
        {"source_id": "MedicationRequest/m1",
         "record": '[2024-03-01] MedicationRequest/m1\n{"resourceType":"MedicationRequest","id":"m1"}'},
    ]

    node_instructions = [entry.kwargs["instructions"] for entry in call.await_args_list[1:3]]
    assert node_instructions[0] != node_instructions[1]
    assert all("最多1题" in text and "候选题型池" not in text for text in node_instructions)
    assert all("本轮实验只允许基于既有患者记录生成查询任务" in text
               for text in [entry.kwargs["instructions"] for entry in call.await_args_list])


@pytest.mark.asyncio
async def test_failed_stage_stops_flow_and_keeps_cost(monkeypatch):
    results = [RunResult(trace=[{"raw": "provider error"}], status="failed",
                         error="RuntimeError: provider unavailable",
                         usage={"model_turns": 1, "tool_calls": 0,
                                "prompt_tokens": 10, "completion_tokens": 5}),
               completed({"questions": [first_candidate()]}),
               completed({"questions": [second_candidate()]}),
               completed(reviews())]
    call = substitute(results)
    monkeypatch.setattr("runtime.call_llm", call)

    result = await workflow.generate_tasks(patient(), environment(), directions_pool(), runtime())

    assert result.status == "failed"
    assert result.output is None
    assert result.usage["model_turns"] == 1
    assert call.await_count == 1
    assert result.trace[-1]["stage"] == "locate"


@pytest.mark.asyncio
async def test_provider_failure_at_a_generation_position_fails_the_patient(monkeypatch):
    results = [completed(located()),
               RunResult(trace=[{"raw": "provider error"}], status="failed",
                         error="RuntimeError: provider unavailable",
                         usage={"model_turns": 1, "tool_calls": 0,
                                "prompt_tokens": 10, "completion_tokens": 5}),
               completed({"questions": [second_candidate()]}),
               completed(reviews(kept=(0,), count=1))]
    call = substitute(results)
    monkeypatch.setattr("runtime.call_llm", call)

    result = await workflow.generate_tasks(patient(), environment(), directions_pool(), runtime())

    assert result.status == "failed"
    assert result.output is None
    assert result.error == "generate_0: RuntimeError: provider unavailable"
    assert result.usage["model_turns"] == 2
    assert call.await_count == 2
    assert [entry["stage"] for entry in result.trace[:2]] == ["locate", "generate_0"]


@pytest.mark.asyncio
@pytest.mark.parametrize("stage_index,payload,expected_type", [
    (1, {"questions": [first_candidate(), first_candidate()]}, "药物处方"),
    (1, {"questions": [{**first_candidate(), "instruction": "   "}]}, "药物处方"),
    (1, {"questions": [{**first_candidate(),
                        "evidence": [{"source_id": "Observation/missing", "statement": "缺失"}]}]},
     "药物处方"),
    (2, {"questions": [second_candidate(), second_candidate()]}, "诊断与解释"),
    (2, {"questions": [candidate("治疗规划", "调整治疗", "Observation/o1", "158/96")]},
     "诊断与解释"),
])
async def test_invalid_node_output_drops_only_that_position(monkeypatch, stage_index, payload,
                                                            expected_type):
    results = [completed(located()), completed({"questions": [first_candidate()]}),
               completed({"questions": [second_candidate()]}),
               completed(reviews(kept=(0,), count=1))]
    results[stage_index] = completed(payload)
    call = substitute(results)
    monkeypatch.setattr("runtime.call_llm", call)

    result = await workflow.generate_tasks(patient(), environment(), directions_pool(), runtime())

    assert result.status == "completed"
    assert call.await_count == 4
    questions = TaskBatch.model_validate(result.output).questions
    assert [question.task_type for question in questions] == [expected_type]
    drops = [entry for entry in result.trace if entry.get("type") == "drops"]
    assert [entry["drops"][0]["stage"] for entry in drops] == [f"generate_{stage_index - 1}"]
    assert result.trace[-1]["type"] == "internal_check"


@pytest.mark.asyncio
async def test_all_positions_failed_still_fails_the_patient(monkeypatch):
    broken = {**first_candidate(), "instruction": "   "}
    call = substitute([completed(located()), completed({"questions": [broken]}),
                       completed({"questions": [broken]})])
    monkeypatch.setattr("runtime.call_llm", call)

    result = await workflow.generate_tasks(patient(), environment(), directions_pool(), runtime())

    assert result.status == "failed"
    assert result.output is None
    assert result.error == "every generation position failed"
    assert call.await_count == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("verdicts", [
    "duplicate", "missing", "out_of_range",
])
async def test_duplicate_or_missing_verdicts_fail_instead_of_filtering(monkeypatch, verdicts):
    review = reviews()
    replacement = {
        "duplicate": {"reviews": [review["reviews"][0], review["reviews"][0]]},
        "missing": {"reviews": [review["reviews"][0]]},
        "out_of_range": {"reviews": [review["reviews"][0], {**review["reviews"][1], "index": 5}]},
    }[verdicts]
    call = substitute([completed(located()), completed({"questions": [first_candidate()]}),
                       completed({"questions": [second_candidate()]}), completed(replacement)])
    monkeypatch.setattr("runtime.call_llm", call)

    result = await workflow.generate_tasks(patient(), environment(), directions_pool(), runtime())

    assert result.status == "failed"
    assert result.output is None
    assert "every candidate" in result.error
    assert result.usage["model_turns"] == 4
    assert call.await_count == 4


@pytest.mark.asyncio
async def test_check_filters_candidates_without_rewriting(monkeypatch):
    call = substitute([completed(located()), completed({"questions": [first_candidate()]}),
                       completed({"questions": [second_candidate()]}),
                       completed(reviews(kept=(1,)))])
    monkeypatch.setattr("runtime.call_llm", call)

    result = await workflow.generate_tasks(patient(), environment(), directions_pool(), runtime())

    questions = TaskBatch.model_validate(result.output).questions
    assert [question.model_dump(exclude={"decision_point"}) for question in questions] == [second_candidate()]
    assert result.trace[-1]["kept"] == [1] and result.trace[-1]["dropped"] == [0]


@pytest.mark.asyncio
async def test_budget_blocks_check_at_three_turns_and_allows_full_flow_at_four(monkeypatch):
    flow = [completed(located()), completed({"questions": [first_candidate()]}),
            completed({"questions": [second_candidate()]}), completed(reviews())]
    call = substitute(flow)
    monkeypatch.setattr("runtime.call_llm", call)

    result = await workflow.generate_tasks(patient(), environment(), directions_pool(),
                                           runtime(max_model_turns=3))

    assert call.await_count == 3
    assert result.status == "budget_exhausted"
    assert result.output is None
    assert result.usage["model_turns"] == 3
    assert [entry["stage"] for entry in result.trace[:3]] == ["locate", "generate_0", "generate_1"]
    assert workflow.BatchReview not in [entry.kwargs["output_type"] for entry in call.await_args_list]

    full = substitute(flow)
    monkeypatch.setattr("runtime.call_llm", full)
    result = await workflow.generate_tasks(patient(), environment(), directions_pool(),
                                           runtime(max_model_turns=4))

    assert full.await_count == 4
    assert result.status == "completed"
    assert result.usage["model_turns"] == 4


@pytest.mark.asyncio
async def test_empty_nodes_skip_check_and_single_candidate_is_still_checked(monkeypatch):
    empty = substitute([completed(located()), completed({"questions": []}), completed({"questions": []})])
    monkeypatch.setattr("runtime.call_llm", empty)

    result = await workflow.generate_tasks(patient(), environment(), directions_pool(), runtime())

    assert empty.await_count == 3
    assert result.status == "completed"
    assert TaskBatch.model_validate(result.output).questions == []
    assert [entry["stage"] for entry in result.trace[:3]] == ["locate", "generate_0", "generate_1"]
    assert result.trace[-1] == selection_output()

    single = substitute([completed(located()), completed({"questions": []}),
                         completed({"questions": [second_candidate()]}),
                         completed(reviews(kept=(0,), count=1))])
    monkeypatch.setattr("runtime.call_llm", single)
    result = await workflow.generate_tasks(patient(), environment(), directions_pool(), runtime())

    assert single.await_count == 4
    questions = TaskBatch.model_validate(result.output).questions
    assert [question.task_type for question in questions] == ["药物处方"]
    assert result.usage["model_turns"] == 4


@pytest.mark.asyncio
async def test_generation_requires_the_five_type_candidate_pool():
    with pytest.raises(ValueError, match="five task types"):
        await workflow.generate_tasks(patient(), environment(), directions(), runtime())


@pytest.mark.asyncio
async def test_locate_with_one_unknown_source_still_generates(monkeypatch):
    selection = pool_selection()
    located = selection.model_copy(update={"focuses": [
        selection.focuses[0],
        selection.focuses[1].model_copy(update={"source_ids": ["MedicationRequest/missing"]})]})
    call = substitute([completed(located.model_dump(mode="json")),
                       completed({"questions": [first_candidate()]}),
                       completed(reviews(kept=(0,), count=1))])
    monkeypatch.setattr("runtime.call_llm", call)

    result = await workflow.generate_tasks(patient(), environment(), directions_pool(), runtime())

    assert result.status == "completed"
    assert call.await_count == 3
    assert [entry for entry in result.trace if entry.get("type") == "drops"] == [
        {"type": "drops",
         "drops": [{"focus": 3, "reason": "unknown source",
                    "source_ids": ["MedicationRequest/missing"]}]}]
    assert result.trace[-2] == {
        "type": "selection",
        "selected_directions": [0, 3],
        "focuses": [focus.model_dump(mode="json") for focus in located.focuses],
    }
