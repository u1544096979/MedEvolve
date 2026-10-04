"""SelfGen boundaries with a substitute runtime: independence, filtering and cost."""

import json

import pytest

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
from methods.selfgen import AnswerConsistency, SelfAnswer, generate_tasks

CANDIDATES = [
    GeneratedQuestion(
        instruction="解释血糖控制情况", task_type="诊断与解释",
        reference_answer="血糖控制不佳，参考解释",
        evidence=[Evidence(source_id="Observation/1", statement="糖化血红蛋白升高")],
    ),
    GeneratedQuestion(
        instruction="给出治疗规划", task_type="治疗规划",
        reference_answer="加强血糖监测",
        evidence=[Evidence(source_id="Observation/1", statement="糖化血红蛋白升高")],
    ),
]
SELF_ANSWERS = {
    "解释血糖控制情况": ["血糖控制不佳，建议调整方案", "血糖控制不佳，考虑调整治疗"],
    "给出治疗规划": ["加强血糖监测并复诊", "加强血糖监测，两周后复诊"],
}


@pytest.fixture
def patient():
    return PatientTrajectory(
        patient_id="MRN4460967357", environment="PhysicianBench",
        text='[2025-05-02] Observation/1\n{"valueQuantity":{"value":7.8}}',
        sources=[Evidence(source_id="Observation/1", statement="糖化血红蛋白 7.8%")],
    )


@pytest.fixture
def directions():
    return [GenerationDirection(task_type=question.task_type, difficulty="基础", weight=1)
            for question in CANDIDATES]


def environment():
    return EnvironmentConfig(name="PhysicianBench", manager_url="http://unused",
                             seed_data="fhir-full:v1")


def runtime(max_model_turns=8):
    return RuntimeConfig(
        models={"generator": ModelConfig(model="gpt-4o-mini", api_key_env="TEST_UNUSED_KEY")},
        budgets={"generator": Budget(max_model_turns=max_model_turns, max_tool_calls=0)},
    )


def completed(output):
    return RunResult(output=output.model_dump(mode="json"), trace=[{"stage_output": "raw"}],
                     usage={"model_turns": 1, "tool_calls": 0,
                            "prompt_tokens": 10, "completion_tokens": 4},
                     status="completed")


def responder(calls, *, fail_at=(), conflict_at={("给出治疗规划", 1)}):
    """Stand-in model: candidate one agrees, candidate two conflicts unless overridden."""
    positions = {}

    async def call_llm(*, instructions, input_text, output_type, model):
        payload = json.loads(input_text)
        calls.append({"instructions": instructions, "input_text": input_text,
                      "payload": payload, "output_type": output_type})
        if output_type is TaskBatch:
            return completed(TaskBatch(questions=list(CANDIDATES)))
        if output_type is SelfAnswer:
            instruction = payload["question"]["instruction"]
            position = positions.get(instruction, 0)
            positions[instruction] = position + 1
            if (instruction, position) in fail_at:
                return RunResult(trace=[{"run_error": "provider unavailable"}],
                                 usage={"model_turns": 1, "tool_calls": 0}, status="failed",
                                 error="RuntimeError: provider unavailable")
            if (instruction, position) in conflict_at:
                return completed(SelfAnswer(answer="立即停止全部降糖药"))
            return completed(SelfAnswer(answer=SELF_ANSWERS[instruction][position]))
        answers = [item["answer"] for item in payload["answers"]]
        consistent = len({answer[:4] for answer in answers}) == 1
        return completed(AnswerConsistency(
            consistent=consistent,
            reason="三份答案结论一致" if consistent else "第三份答案给出停药，与其他答案结论冲突"))

    return call_llm


def stages(result):
    return {entry["stage"]: entry["result"] for entry in result.trace}


@pytest.mark.asyncio
async def test_self_answers_are_independent_and_only_consistent_candidates_survive(
    monkeypatch, patient, directions
):
    calls = []
    monkeypatch.setattr("runtime.call_llm", responder(calls))
    result = await generate_tasks(patient, environment(), directions, runtime())

    assert result.status == "completed"
    batch = TaskBatch.model_validate(result.output)
    assert [question.instruction for question in batch.questions] == ["解释血糖控制情况"]
    # The kept candidate keeps the first generated answer and its own evidence.
    assert batch.questions[0].reference_answer == CANDIDATES[0].reference_answer
    assert batch.questions[0].evidence == CANDIDATES[0].evidence

    assert calls[0]["output_type"] is TaskBatch
    self_answers = [call for call in calls if call["output_type"] is SelfAnswer]
    assert len(self_answers) == 4
    material = {"patient_id": patient.patient_id, "environment": patient.environment,
                "text": patient.text, "source_ids": ["Observation/1"]}
    for call in self_answers:
        assert call["payload"]["patient"] == material
        assert set(call["payload"]) == {"question", "patient"}
        assert set(call["payload"]["question"]) == {"instruction", "task_type"}
        for candidate in CANDIDATES:
            assert candidate.reference_answer not in call["input_text"]
        for answers in SELF_ANSWERS.values():
            for answer in answers:
                assert answer not in call["input_text"]
    # The two positions for one question carry the identical context, so no answer leaks.
    paired = [call["input_text"] for call in self_answers[:2]]
    assert paired[0] == paired[1]

    consistency = [call for call in calls if call["output_type"] is AnswerConsistency]
    assert len(consistency) == 2
    assert [item["answer"] for item in consistency[0]["payload"]["answers"]] == [
        CANDIDATES[0].reference_answer, *SELF_ANSWERS["解释血糖控制情况"]]
    assert [item["answer"] for item in consistency[1]["payload"]["answers"]] == [
        CANDIDATES[1].reference_answer, SELF_ANSWERS["给出治疗规划"][0], "立即停止全部降糖药"]

    # Consistency judgments are recorded by the method; evidence stays for public verification.
    assert stages(result)["filter/0"]["trace"][0]["kept"] is True
    dropped = stages(result)["filter/1"]["trace"][0]
    assert dropped["kept"] is False and "冲突" in dropped["reason"]
    assert dropped["evidence_verification"] == "pending public verification"
    assert "verification" not in result.output["questions"][0]
    assert result.output["questions"][0]["decision_point"] is None

    # Every internal call is part of the batch cost.
    assert result.usage["model_turns"] == 7 and result.usage["tool_calls"] == 0
    assert result.usage["input_tokens"] == 70 and result.usage["output_tokens"] == 28
    assert result.usage["total_tokens"] == 98
    assert result.usage["runs_without_known_tokens"] == 0


@pytest.mark.asyncio
async def test_budget_cutoff_keeps_cost_and_claims_no_batch(monkeypatch, patient, directions):
    calls = []
    monkeypatch.setattr("runtime.call_llm", responder(calls))
    result = await generate_tasks(patient, environment(), directions, runtime(max_model_turns=3))

    assert result.status == "budget_exhausted" and result.output is None
    assert [call["output_type"] for call in calls] == [TaskBatch, SelfAnswer, SelfAnswer]
    assert result.usage["model_turns"] == 3 and result.usage["total_tokens"] == 42
    assert stages(result)["filter/0"]["trace"][0]["kept"] is False


@pytest.mark.asyncio
async def test_failed_self_answer_drops_one_candidate_without_resampling(
    monkeypatch, patient, directions
):
    calls = []
    monkeypatch.setattr("runtime.call_llm",
                        responder(calls, fail_at={("解释血糖控制情况", 1)}, conflict_at=()))
    result = await generate_tasks(patient, environment(), directions, runtime())

    assert result.status == "completed"
    assert [call["output_type"] for call in calls] == [
        TaskBatch, SelfAnswer, SelfAnswer, SelfAnswer, SelfAnswer, AnswerConsistency]
    assert [question.instruction for question in TaskBatch.model_validate(result.output).questions] == [
        "给出治疗规划"]
    stages_seen = stages(result)
    assert set(stages_seen) == {
        "candidate_generation", "self_answer/0/0", "self_answer/0/1",
        "self_answer/1/0", "self_answer/1/1", "consistency/1", "filter/0", "filter/1"}
    assert stages_seen["self_answer/0/1"]["status"] == "failed"
    assert "without resampling" in stages_seen["filter/0"]["trace"][0]["reason"]
    # The failed run still counts as a model turn; its token usage stays unknown.
    assert result.usage["model_turns"] == 6 and result.usage["total_tokens"] == 70
    assert result.usage["runs_without_known_tokens"] == 1
