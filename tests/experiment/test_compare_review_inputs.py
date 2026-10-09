import json
from unittest.mock import AsyncMock

import pytest

from contracts import (ComplexityMetrics, DifficultyScore, Evidence,
                       EvaluationConfig, ModelConfig, PatientTrajectory,
                       QualityDimension, QualityReview, RunResult, Task,
                       Verification)
from scripts import compare_review_inputs as compare


def _case(index):
    environment = "PhysicianBench" if index < 3 else "MedAgentBench"
    return {
        "environment": environment,
        "task": Task(id=f"{environment}:{index}", patient_id=f"p{index}",
                     environment=environment, round_index=0, source_method="test",
                     task_type="治疗规划", instruction="题目", reference_answer="答案",
                     evidence=[Evidence(source_id="Condition/1", statement="依据")]),
        "trajectory": PatientTrajectory(patient_id=f"p{index}", environment=environment,
                                        text="轨迹", sources=[]),
        "verification": Verification(valid=True, reason="有效"),
        "config": EvaluationConfig(runtime={"models": {
            "quality": ModelConfig(model="original", api_key_env="KEY"),
            "verifier": ModelConfig(model="verifier", api_key_env="KEY"),
        }, "budgets": {}}),
        "run_dir": f"source-{index}", "task_path": f"source-{index}/tasks.json",
        "evaluation_path": f"source-{index}/evaluations.json",
        "trajectory_path": f"source-{index}/trajectory.json",
    }


def _scores(count=3):
    quality = QualityReview(**{
        name: QualityDimension(score=4, scores=[4, 4, 4], reason="reason")
        for name in ("factual_accuracy", "clinical_soundness",
                     "evidence_sufficiency", "task_clarity")
    })
    difficulty = DifficultyScore(reasoning_depth=3, reasoning_depths=[3, 3, 3],
                                 reason="reason", complexity_metrics=ComplexityMetrics(
                                     evidence_count=1, resource_type_count=1,
                                     time_span_days=None))
    runs = [RunResult(trace=[], usage={"input_tokens": 10, "output_tokens": 2},
                      status="completed") for _ in range(count)]
    return quality, difficulty, runs


@pytest.mark.asyncio
async def test_execute_uses_three_modes_one_shared_semaphore_and_counts_runs_once(monkeypatch):
    async def fake(*args, **kwargs):
        return _scores(6 if args[3].scoring_mode == "separate" else 3)
    fake = AsyncMock(side_effect=fake)
    monkeypatch.setattr(compare, "review_scores", fake)
    args = type("Args", (), {"concurrency": 24, "model": "gpt-4o-mini",
                              "base_url": "https://api.openlux.ai/v1",
                              "api_key_env": "OPENLUX_API_KEY"})()
    cases = [_case(index) for index in range(6)]
    records, requests = await compare._execute(cases, args)
    assert fake.await_count == 18
    assert {(call.args[3].scoring_mode, call.args[3].review_input_mode)
            for call in fake.await_args_list} == set(compare.MODES)
    assert len({id(call.kwargs["request_semaphore"])
                for call in fake.await_args_list}) == 1
    assert all(call.args[3].judges == [] for call in fake.await_args_list)
    assert all(call.args[3].runtime.models["difficulty"].model == "gpt-4o-mini"
               for call in fake.await_args_list)
    assert len(requests) == 72
    assert sum(item["request_count"] for item in records) == len(requests)
    assert compare._comparison(records)["request_count"] == len(requests)


def test_cli_persists_cases_runs_and_comparison(monkeypatch, tmp_path):
    cases = [_case(index) for index in range(6)]
    monkeypatch.setattr(compare, "_source_runs", lambda _: cases)
    async def fake(*args, **kwargs):
        return _scores(6 if args[3].scoring_mode == "separate" else 3)
    fake = AsyncMock(side_effect=fake)
    monkeypatch.setattr(compare, "review_scores", fake)
    compare.main(["--run-dir", "one", "--run-dir", "two", "--output-dir",
                  str(tmp_path)])
    written_cases = json.loads((tmp_path / "cases.json").read_text())
    written_runs = json.loads((tmp_path / "runs.json").read_text())
    comparison = json.loads((tmp_path / "comparison.json").read_text())
    assert len(written_cases) == 6
    assert all(item["trajectory_path"].endswith("trajectory.json") for item in written_cases)
    assert len(written_runs) == comparison["request_count"] == 72
    assert all(item["status"] == "completed" for item in written_runs)
    assert fake.await_count == 18
