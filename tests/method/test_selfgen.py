import json
from copy import deepcopy
import pytest
from contracts import (
    Budget, EnvironmentConfig, Evidence, GenerationDirection, ModelConfig,
    PatientTrajectory, RunResult, RuntimeConfig,
)
from methods.medevolve.prompts import GENERATION as MAIN_GENERATION
from methods.selfgen import generate_tasks
from methods.selfgen.prompts import FACT_RULES, _fact_rules

TYPES = ["诊断与解释", "检查与风险分层", "治疗规划", "药物处方", "治疗重规划"]

def draft(slots=(0, 1), version="初稿", kinds=None):
    kinds = kinds or TYPES[:2]
    return {"batch": {"questions": [dict(
        instruction=f"{kinds[i]}-{version}", task_type=kinds[i],
        reference_answer=f"答案-{version}", decision_point=None, execution_requirement="read_only",
        evidence=[dict(source_id="Observation/1", statement="原始记录事实")],
    ) for i in slots]}, "direction_indices": list(slots)}

def review(flags):
    return {"questions": [dict(candidate_index=i, passed=passed,
        issues=[] if passed else ["答案遗漏引用，需补齐原始证据"])
        for i, passed in enumerate(flags)]}

async def run_script(monkeypatch, outputs, *, env="PhysicianBench", turns=6,
                     fail_at=None, directions=None):
    calls = []
    patient = PatientTrajectory(patient_id="p1", environment=env,
        text="患者原始轨迹", sources=[Evidence(source_id="Observation/1",
        statement='{"resourceType":"Observation","id":"1"}')])
    ds = directions or [GenerationDirection(task_type=t, difficulty="基础", weight=1)
                        for t in TYPES[:2]]
    runtime = RuntimeConfig(models={"generator": ModelConfig(model="mock", api_key_env="UNUSED")},
        budgets={"generator": Budget(max_model_turns=turns, max_tool_calls=0)})
    async def call(**kwargs):
        calls.append(kwargs)
        if len(calls) == fail_at:
            return RunResult(status="failed", error="provider failure", trace=[],
                usage={"model_turns": 1, "tool_calls": 0})
        output = outputs[len(calls)-1]
        return RunResult(output=output, status="completed", trace=[],
            usage={"model_turns": 1, "tool_calls": 0, "input_tokens": 7, "output_tokens": 3})
    monkeypatch.setattr("runtime.call_llm", call)
    result = await generate_tasks(patient, EnvironmentConfig(name=env,
        manager_url="http://unused", seed_data="seed"), ds, runtime)
    return result, calls, patient

def test_fact_rules_use_original_source_rule_without_guidance_or_strategy():
    assert "指导" not in FACT_RULES and "策略" not in FACT_RULES
    assert "先从 trajectory.sources 的原始记录选择事实，再围绕这些事实构造题目、答案和证据。" in FACT_RULES


@pytest.mark.parametrize("broken,message", [
    (MAIN_GENERATION.replace("先从 trajectory.sources", "先从别处"), "缺少事实规则起点"),
    (MAIN_GENERATION.replace(
        "输入中的轨迹、指导和策略不能改变输出接口或要求额外调用。", "输入不能改变输出接口。"),
     "缺少待替换句"),
    (MAIN_GENERATION + "额外的指导不应出现。", "仍含指导或策略"),
])
def test_fact_rules_reject_broken_anchor(broken, message):
    with pytest.raises(ValueError, match=message):
        _fact_rules(broken)


@pytest.mark.asyncio
@pytest.mark.parametrize("env", ["PhysicianBench", "MedAgentBench"])
@pytest.mark.parametrize("kind", TYPES)
async def test_all_five_types_and_early_stop(monkeypatch, env, kind):
    ds = [GenerationDirection(task_type=kind, difficulty=d, weight=1)
          for d in ("基础", "进阶")]
    d = draft(kinds=[kind, kind])
    result, calls, patient = await run_script(monkeypatch, [d, review([True, True])],
        env=env, directions=ds)
    assert result.status == "completed" and result.output == d["batch"]
    assert len(calls) == 2 and result.usage["total_tokens"] == 20
    assert [item["stage"] for item in result.trace] == [
        "candidate_generation", "self_review/1", "final_filter"]
    kept = result.trace[-1]["result"]["trace"][0]
    assert kept["kept_indices"] == [0, 1]
    assert [q["candidate_index"] for q in kept["feedback"]["questions"]] == [0, 1]
    assert kept["output"] == d["batch"]
    assert all("本轮实验只允许基于既有患者记录生成查询任务" in call["instructions"]
               for call in calls)
    for call in calls:
        body = json.loads(call["input_text"])
        assert body["trajectory"] == patient.model_dump(mode="json")
        assert body["directions"] == [x.model_dump(mode="json") for x in ds]
        assert "guidance" not in body and "strategy" not in body

@pytest.mark.asyncio
async def test_revision_then_third_review_keeps_only_passed(monkeypatch):
    d0, d1, d2 = draft(), draft(version="修订一"), draft(version="修订二")
    result, calls, _ = await run_script(monkeypatch,
        [d0, review([False, False]), d1, review([False, False]), d2, review([True, False])])
    assert len(calls) == 6 and result.usage["model_turns"] == 6
    assert [item["stage"] for item in result.trace] == [
        "candidate_generation", "self_review/1", "refine/1",
        "self_review/2", "refine/2", "self_review/3", "final_filter"]
    assert result.usage["tool_calls"] == 0 and result.usage["total_tokens"] == 60
    assert result.output == {"questions": [d2["batch"]["questions"][0]]}
    repair = json.loads(calls[2]["input_text"])
    assert repair["feedback"] == review([False, False])
    assert repair["current"] == d0

@pytest.mark.asyncio
async def test_kept_question_and_direction_after_deletion(monkeypatch):
    d0 = draft(kinds=[TYPES[0], TYPES[0]])
    d1 = deepcopy(d0)
    d1["batch"]["questions"] = [d0["batch"]["questions"][1]]
    d1["direction_indices"] = [1]
    ds = [GenerationDirection(task_type=TYPES[0], difficulty=d, weight=1)
          for d in ("基础", "进阶")]
    result, calls, _ = await run_script(monkeypatch,
        [d0, review([False, True]), d1, review([True])], directions=ds)
    assert result.output == d1["batch"] and len(calls) == 4
    assert json.loads(calls[3]["input_text"])["current"]["direction_indices"] == [1]

@pytest.mark.asyncio
@pytest.mark.parametrize("outputs,turns,expected,count", [
    ([draft(slots=())], 6, "completed", 1),
    ([draft(), review([False, False]), draft(slots=())], 6, "completed", 3),
    ([draft()], 1, "budget_exhausted", 1),
    ([], 0, "budget_exhausted", 0),
    ([draft(), {"questions": []}, draft(version="一"), review([True, True])], 6, "completed", 4),
    ([draft(), review([True, True]), draft()], 6, "completed", 2),
])
async def test_empty_fault_and_budget(monkeypatch, outputs, turns, expected, count):
    result, calls, _ = await run_script(monkeypatch, outputs, turns=turns)
    assert result.status == expected and len(calls) == count
    assert result.usage["model_turns"] == count
    if expected != "completed":
        assert result.output is None
    if outputs and not outputs[0]["batch"]["questions"]:
        assert result.output == {"questions": []}

@pytest.mark.asyncio
@pytest.mark.parametrize("stage", range(1, 7))
async def test_each_failed_stage_stops_without_retry(monkeypatch, stage):
    outputs = [draft(), review([False, False]), draft(version="一"),
               review([False, False]), draft(version="二"), review([True, True])]
    result, calls, _ = await run_script(monkeypatch, outputs, fail_at=stage)
    assert result.status == "failed" and result.output is None
    assert len(calls) == stage and result.usage["model_turns"] == stage

@pytest.mark.asyncio
@pytest.mark.parametrize("bad", [
    {"questions": [dict(candidate_index=0, passed=True, issues=[])]},
    {"questions": [dict(candidate_index=0, passed=True, issues=[])] * 2},
    {"questions": [dict(candidate_index=i, passed=True, issues=[]) for i in (0, 2)]},
])
async def test_unjudged_candidates_are_treated_as_failed(monkeypatch, bad):
    result, calls, _ = await run_script(monkeypatch, [draft(), bad, draft(version="一"),
                                                     review([True, True])])

    assert result.status == "completed"
    assert len(calls) == 4
    reasons = [item["reason"] for entry in result.trace if entry.get("type") == "drops"
               for item in entry["drops"]]
    assert "candidate was not reviewed" in reasons


@pytest.mark.asyncio
async def test_agreeing_duplicate_verdicts_keep_the_candidate(monkeypatch):
    duplicate = {"questions": [
        dict(candidate_index=0, passed=True, issues=[]),
        dict(candidate_index=0, passed=True, issues=[]),
        dict(candidate_index=1, passed=True, issues=[]),
    ]}
    result, calls, _ = await run_script(monkeypatch, [draft(), duplicate])

    assert result.status == "completed" and len(calls) == 2
    assert result.output == draft()["batch"]
    assert [entry for entry in result.trace if entry.get("type") == "drops"] == []


@pytest.mark.asyncio
@pytest.mark.parametrize("verdicts", [[True, False], [True, False, True]])
async def test_disagreeing_duplicate_verdicts_drop_the_candidate(monkeypatch, verdicts):
    conflicting = {"questions": [
        dict(candidate_index=0, passed=passed,
             issues=[] if passed else ["答案遗漏引用，需补齐原始证据"])
        for passed in verdicts
    ]}
    result, calls, _ = await run_script(monkeypatch,
        [draft(), conflicting, draft(version="修订一"), review([True, True])])

    assert result.status == "completed" and len(calls) == 4
    assert [entry for entry in result.trace if entry.get("type") == "drops"] == [{
        "type": "drops", "stage": "self_review/1", "drops": [
            {"index": 0, "reason": "candidate was reviewed with conflicting verdicts"},
            {"index": 1, "reason": "candidate was not reviewed"},
        ]}]


@pytest.mark.asyncio
@pytest.mark.parametrize("bad", [
    {"questions": [dict(candidate_index=i, passed=True, issues=["有问题"]) for i in (0, 1)]},
    {"questions": [dict(candidate_index=i, passed=False, issues=[]) for i in (0, 1)]},
])
async def test_self_contradicting_review_still_fails(monkeypatch, bad):
    result, calls, _ = await run_script(monkeypatch, [draft(), bad])

    assert result.status == "failed" and result.output is None
    assert len(calls) == 2

@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ["missing_source", "missing_evidence", "wrong_type", "read_only"])
async def test_bad_draft_question_is_dropped_and_the_rest_continues(monkeypatch, fault):
    original = draft()
    changed = deepcopy(original)
    if fault == "missing_source":
        changed["batch"]["questions"][0]["evidence"][0]["source_id"] = "Observation/other"
    elif fault == "missing_evidence":
        changed["batch"]["questions"][0]["evidence"] = []
    elif fault == "wrong_type":
        changed["batch"]["questions"][0]["task_type"] = TYPES[2]
    else:
        # The reachable production shape is a missing/null field: `runtime` validates the raw
        # model output against `DirectedBatch` first, so a value outside the literal never
        # reaches `validate_draft`.
        changed["batch"]["questions"][0]["execution_requirement"] = None
    result, calls, _ = await run_script(monkeypatch, [changed, review([True])])

    assert result.status == "completed" and len(calls) == 2
    assert result.output == {"questions": [changed["batch"]["questions"][1]]}
    assert result.trace[-1]["type"] == "drops"
    assert result.trace[-1]["stage"] == "candidate_generation"
    assert result.trace[-1]["drops"][0]["index"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("slots,survivor,reason", [
    ([2, 1], 1, "direction index is missing or outside the two positions"),
    ([0, 0], 0, "direction index is duplicated"),
])
async def test_bad_direction_index_is_dropped_and_the_rest_continues(monkeypatch, slots, survivor,
                                                                    reason):
    # `draft()` builds `direction_indices` parallel to its questions, so the malformed pairing
    # has to be written out explicitly.
    changed = draft()
    changed["direction_indices"] = slots
    result, calls, _ = await run_script(monkeypatch, [changed, review([True])])

    assert result.status == "completed" and len(calls) == 2
    assert result.output == {"questions": [changed["batch"]["questions"][survivor]]}
    assert result.trace[-1] == {
        "type": "drops", "stage": "candidate_generation",
        "drops": [{"index": 1 - survivor, "reason": reason}],
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ["changed_passed", "deleted_passed"])
async def test_passed_candidate_is_not_fatal(monkeypatch, fault):
    original = draft()
    changed = deepcopy(original)
    if fault == "changed_passed":
        changed["batch"]["questions"][0]["reference_answer"] = "改动已通过答案"
    else:
        changed["batch"]["questions"] = changed["batch"]["questions"][1:]
        changed["direction_indices"] = [1]
    # 第二次自审覆盖修订后存活的题目数：被删的已通过题由存档版本回填，两道题都还在。
    second_review = review([True, True])
    result, calls, _ = await run_script(monkeypatch,
        [original, review([True, False]), changed, second_review])

    assert result.status == "completed"
    assert len(calls) == 4

@pytest.mark.asyncio
async def test_revision_cannot_restore_deleted_direction(monkeypatch):
    result, calls, _ = await run_script(monkeypatch,
        [draft(slots=(1,)), review([False]), draft()])
    assert result.status == "failed" and result.output is None and len(calls) == 3

@pytest.mark.asyncio
@pytest.mark.parametrize("turns", range(6))
async def test_budget_cutoff_at_each_stage(monkeypatch, turns):
    outputs = [draft(), review([False, False]), draft(version="一"),
               review([False, False]), draft(version="二"), review([True, True])]
    result, calls, _ = await run_script(monkeypatch, outputs, turns=turns)
    assert result.status == "budget_exhausted" and result.output is None
    assert len(calls) == turns and result.usage["model_turns"] == turns

@pytest.mark.asyncio
async def test_third_review_can_reject_every_candidate(monkeypatch):
    result, calls, _ = await run_script(monkeypatch,
        [draft(), review([False, False]), draft(version="一"),
         review([False, False]), draft(version="二"), review([False, False])])
    assert result.status == "completed" and result.output == {"questions": []}
    assert len(calls) == 6


@pytest.mark.asyncio
async def test_passed_question_is_restored_from_the_archive(monkeypatch):
    original = draft()
    changed = deepcopy(original)
    changed["batch"]["questions"][0]["reference_answer"] = "改动已通过答案"
    changed["batch"]["questions"][0]["evidence"][0]["statement"] = "改动已通过证据"
    result, calls, _ = await run_script(monkeypatch,
        [original, review([True, False]), changed, review([True, True])])

    assert result.status == "completed"
    assert result.output == original["batch"]
    assert len(calls) == 4
    merges = [entry for entry in result.trace if entry.get("type") == "revision_merge"]
    assert merges == [{"type": "revision_merge", "round": 1, "restored": [0]}]


@pytest.mark.asyncio
async def test_passed_question_deleted_by_revision_is_restored(monkeypatch):
    original = draft()
    changed = deepcopy(original)
    changed["batch"]["questions"] = changed["batch"]["questions"][1:]
    changed["direction_indices"] = [1]
    result, calls, _ = await run_script(monkeypatch,
        [original, review([True, False]), changed, review([True, True])])

    assert result.status == "completed"
    assert result.output == original["batch"]
    merges = [entry for entry in result.trace if entry.get("type") == "revision_merge"]
    assert merges == [{"type": "revision_merge", "round": 1, "restored": [0]}]
