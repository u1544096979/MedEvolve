import json
from copy import deepcopy
import pytest
from contracts import (
    Budget, EnvironmentConfig, Evidence, GenerationDirection, ModelConfig,
    PatientTrajectory, RunResult, RuntimeConfig,
)
from methods.selfgen import generate_tasks

TYPES = ["诊断与解释", "检查与风险分层", "治疗规划", "药物处方", "治疗重规划"]

def draft(slots=(0, 1), version="初稿", kinds=None):
    kinds = kinds or TYPES[:2]
    return {"batch": {"questions": [dict(
        instruction=f"{kinds[i]}-{version}", task_type=kinds[i],
        reference_answer=f"答案-{version}", decision_point=None,
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
    ([draft(), {"questions": []}], 6, "failed", 2),
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
    {"questions": [dict(candidate_index=i, passed=True, issues=["有问题"]) for i in (0, 1)]},
    {"questions": [dict(candidate_index=i, passed=False, issues=[]) for i in (0, 1)]},
])
async def test_bad_feedback_does_not_trigger_repair(monkeypatch, bad):
    result, calls, _ = await run_script(monkeypatch, [draft(), bad])
    assert result.status == "failed" and result.output is None
    assert len(calls) == 2

@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ["missing_source", "missing_evidence", "changed_passed", "deleted_passed"])
async def test_source_and_passed_candidate_faults(monkeypatch, fault):
    original = draft()
    changed = deepcopy(original)
    if fault == "missing_source":
        changed["batch"]["questions"][0]["evidence"][0]["source_id"] = "Observation/other"
    elif fault == "missing_evidence":
        changed["batch"]["questions"][0]["evidence"] = []
    elif fault == "changed_passed":
        changed["batch"]["questions"][0]["reference_answer"] = "改动已通过答案"
    else:
        changed["batch"]["questions"] = changed["batch"]["questions"][1:]
        changed["direction_indices"] = [1]
    result, calls, _ = await run_script(monkeypatch,
        [original, review([True, False]), changed])
    assert result.status == "failed" and result.output is None
    assert len(calls) == 3

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
