"""SingleShot public entry: free selection over all task types, public evaluation only."""

import json
from pathlib import Path
from typing import get_args
from unittest.mock import AsyncMock

import pytest

import runtime
from contracts import (EnvironmentInstance, EvaluationResult, ExperimentConfig,
                       PatientTrajectory, RunResult, TaskType)
from experiments import runner, summarize_run


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["PhysicianBench", "MedAgentBench"])
async def test_singleshot_entry_uses_all_types_and_public_evaluation(name, tmp_path, monkeypatch):
    values = json.loads((Path(__file__).resolve().parents[2] / "configs/experiment.json").read_text())
    values.update(method="singleshot", run_id="single-test", evolution_patients=[],
                  evaluation_patients=["p"], round_patients=[], output_dir=str(tmp_path))
    values["environment"]["name"] = name
    if "compare_initial_strategy" in values:
        values["compare_initial_strategy"] = False
    # 仅保留一个配置题型，确保结果不受 initial_strategy 的配额限制。
    values["initial_strategy"]["allocation"] = [
        {"task_type": "治疗规划", "difficulty": "配置字段", "weight": 1}]
    config = ExperimentConfig.model_validate(values)
    monkeypatch.setattr(runner, "create_environment", AsyncMock(return_value=EnvironmentInstance(
        instance_id="cache", environment=name, endpoint="http://unused")))
    release = AsyncMock()
    monkeypatch.setattr(runner, "release_environment", release)
    monkeypatch.setattr(runner, "prepare_trajectory", AsyncMock(return_value=PatientTrajectory(
        patient_id="p", environment=name, text="records", sources=[
            {"source_id": "Patient/p", "statement": '{"id":"p"}'}])))
    for fn in ("explore_data", "explore_environment", "analyze_feedback", "update_strategy"):
        monkeypatch.setattr(runner, fn, AsyncMock(side_effect=AssertionError("unexpected evolution")))
    question = {"instruction": "说明记录", "task_type": "诊断与解释",
        "reference_answer": "候选答案", "evidence": [{"source_id": "Patient/p", "statement": "记录事实"}],
        "decision_point": None}
    llm = AsyncMock(return_value=RunResult(output={"questions": [question, question]},
        status="completed", trace=[], usage={"model_turns": 1, "tool_calls": 0,
                                             "input_tokens": 12, "output_tokens": 8}))
    monkeypatch.setattr(runtime, "call_llm", llm)
    evaluate = AsyncMock(side_effect=lambda task, *args: EvaluationResult(
        task_id=task.id, status="invalid", observed_state={}, runs=[]))
    monkeypatch.setattr(runner, "evaluate_task", evaluate)

    await runner.run_experiment(config)

    llm.assert_awaited_once()
    payload = json.loads(llm.await_args.kwargs["input_text"])
    assert set(payload) == {"trajectory", "task_types"}
    assert payload["task_types"] == list(get_args(TaskType))
    assert "directions" not in payload and "strategy" not in payload
    assert evaluate.await_count == 2 and release.await_count == 1
    assert evaluate.call_args.args[0].source_method == "singleshot"
    attempts = json.loads((tmp_path / "evaluation/attempts.json").read_text())
    assert attempts[0]["task_types"] == list(get_args(TaskType))
    assert "directions" not in attempts[0]
    summary = summarize_run(str(tmp_path))
    assert summary["evaluation"]["actual_questions"] == 2
    assert summary["costs"]["generation"]["model_turns"] == 1
    assert summary["costs"]["generation"]["tool_calls"] == 0
    assert not (tmp_path / "guidance.json").exists()
