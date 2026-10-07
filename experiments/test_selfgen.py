import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock
import pytest
from contracts import EnvironmentInstance, EvaluationResult, ExperimentConfig, PatientTrajectory, RunResult
from experiments import runner, summarize_run

@pytest.mark.parametrize("env", ["PhysicianBench", "MedAgentBench"])
def test_selfgen_sampling_isolation_cost_and_release(tmp_path, monkeypatch, env):
    source = "physicianbench-comparison.json" if env == "PhysicianBench" else "medagentbench-flash.json"
    values = json.loads((Path(__file__).parents[1] / "configs" / source).read_text())
    values.update(method="selfgen", evolution_patients=["train"], evaluation_patients=["heldout"],
        round_patients=[], compare_initial_strategy=False, output_dir=str(tmp_path), run_id="selfgen-test")
    config = ExperimentConfig.model_validate(values)
    calls = []
    async def generate(trajectory, environment, directions, runtime):
        calls.append((trajectory, environment, directions))
        return RunResult(output={"questions": [dict(instruction="解释记录中的临床事实",
            task_type=directions[0].task_type, reference_answer="仅供公共核验的候选答案",
            evidence=[dict(source_id="Observation/1", statement="患者记录事实")], decision_point=None,
            execution_requirement="read_only")]},
            trace=[], usage={"model_turns": 1, "tool_calls": 0, "input_tokens": 7,
            "output_tokens": 3, "total_tokens": 10}, status="completed")
    monkeypatch.setattr(runner, "resolve_generator", lambda method: generate)
    monkeypatch.setattr(runner, "create_environment", AsyncMock(return_value=EnvironmentInstance(
        instance_id="cache", endpoint="http://unused", environment=env)))
    monkeypatch.setattr(runner, "prepare_trajectory", AsyncMock(side_effect=lambda inst, p:
        PatientTrajectory(patient_id=p, environment=env, text="原始记录", sources=[])))
    release = AsyncMock()
    monkeypatch.setattr(runner, "release_environment", release)
    for name in ("explore_data", "explore_environment", "analyze_feedback", "update_strategy"):
        monkeypatch.setattr(runner, name, AsyncMock(side_effect=AssertionError("探索或进化被调用")))
    evaluate = AsyncMock(side_effect=lambda task, *args: EvaluationResult(
        task_id=task.id, status="invalid", runs=[], observed_state={}))
    monkeypatch.setattr(runner, "evaluate_task", evaluate)
    asyncio.run(runner.run_experiment(config))
    trajectory, environment, directions = calls[0]
    assert trajectory.patient_id == "heldout" and environment.name == env
    release.assert_awaited_once()
    attempts = json.loads((tmp_path / "evaluation/attempts.json").read_text())
    assert attempts[0]["directions"] == [d.model_dump(mode="json") for d in directions]
    assert evaluate.call_args.args[0].source_method == "selfgen"
    assert not (tmp_path / "guidance.json").exists()
    assert not (tmp_path / "rounds").exists()
    summary = summarize_run(str(tmp_path))
    assert summary["costs"]["generation"]["total_tokens"] == 10
