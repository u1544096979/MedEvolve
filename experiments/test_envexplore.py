import asyncio
import json
import random
from pathlib import Path
from unittest.mock import AsyncMock
import pytest
from contracts import (
    EnvironmentInstance, EvaluationResult, Evidence, ExperimentConfig,
    PatientTrajectory, RunResult,
)
from experiments import runner

@pytest.fixture
def config(tmp_path):
    values = json.loads((Path(__file__).parents[1] / "configs/experiment.json").read_text())
    values.update(method="envexplore", run_id="envexplore-test", evolution_patients=["train"],
        evaluation_patients=["p1", "p2"], round_patients=[], output_dir=str(tmp_path))
    values["runtime"]["models"] = {r: values["runtime"]["models"][r]
                                  for r in ("environment_explorer", "generator")}
    values["runtime"]["budgets"] = {r: values["runtime"]["budgets"][r]
                                   for r in ("environment_explorer", "generator")}
    return ExperimentConfig.model_validate(values)

@pytest.fixture
def wired(config, monkeypatch):
    prepared = []
    async def prepare(instance, patient):
        prepared.append(patient)
        return PatientTrajectory(patient_id=patient, environment=config.environment.name,
            text="患者原始记录", sources=[Evidence(source_id="Condition/c1", statement='{"id":"c1"}')])
    instance = EnvironmentInstance(instance_id="trajectory", endpoint="http://unused",
                                   environment=config.environment.name)
    monkeypatch.setattr(runner, "create_environment", AsyncMock(return_value=instance))
    release = AsyncMock()
    monkeypatch.setattr(runner, "release_environment", release)
    monkeypatch.setattr(runner, "prepare_trajectory", prepare)
    exploration = AsyncMock(return_value=RunResult(output={"guidance": "已验证读取能力"},
        trace=[{"type": "tool_output", "output": {"id": "trial-record"}}],
        usage={"model_turns": 2, "tool_calls": 1, "input_tokens": 8, "output_tokens": 4},
        status="completed"))
    monkeypatch.setattr(runner, "explore_environment", exploration)
    forbidden = []
    for name in ("explore_data", "analyze_feedback", "update_strategy"):
        mock = AsyncMock(side_effect=AssertionError(name + " 不应调用"))
        monkeypatch.setattr(runner, name, mock)
        forbidden.append(mock)
    async def generate(**kwargs):
        body = json.loads(kwargs["input_text"])
        return RunResult(output={"questions": [{"instruction": "解释原始诊断",
            "task_type": body["directions"][0]["task_type"], "reference_answer": "诊断解释",
            "evidence": [{"source_id": "Condition/c1", "statement": "记录事实"}],
            "decision_point": None}]}, trace=[],
            usage={"model_turns": 1, "tool_calls": 0, "input_tokens": 7, "output_tokens": 3},
            status="completed")
    call = AsyncMock(side_effect=generate)
    monkeypatch.setattr("runtime.call_llm", call)
    evaluate = AsyncMock(side_effect=lambda task, *args: EvaluationResult(
        task_id=task.id, observed_state={}, status="invalid", runs=[]))
    monkeypatch.setattr(runner, "evaluate_task", evaluate)
    return prepared, release, exploration, forbidden, call, evaluate

def test_once_exploration_fixed_strategy_and_public_sampling(config, wired):
    prepared, release, exploration, forbidden, call, evaluate = wired
    before = config.initial_strategy.model_dump(mode="json")
    asyncio.run(runner.run_experiment(config))
    exploration.assert_awaited_once_with(config.environment, ["train"], config.runtime)
    assert prepared == ["p1", "p2"] and call.await_count == evaluate.await_count == 2
    release.assert_awaited_once()
    for mock in forbidden: mock.assert_not_awaited()
    assert config.initial_strategy.model_dump(mode="json") == before
    rng = random.Random(config.random_seed)
    expected = [rng.choices(config.initial_strategy.allocation,
        weights=[d.weight for d in config.initial_strategy.allocation], k=2) for _ in range(2)]
    for index, entry in enumerate(call.await_args_list):
        body = json.loads(entry.kwargs["input_text"])
        assert body["guidance"] == {"environment_guidance": "已验证读取能力"}
        assert body["strategy"] == {"generation_guidance": before["generation_guidance"]}
        assert body["directions"] == [d.model_dump(mode="json") for d in expected[index]]
        assert "trial-record" not in entry.kwargs["input_text"]
    root = Path(config.output_dir)
    assert json.loads((root / "environment_guidance.json").read_text()) == {"guidance": "已验证读取能力"}
    assert not (root / "rounds").exists() and not (root / "data_exploration_patients.json").exists()
    runs = json.loads((root / "runs.json").read_text())
    assert sum(r["stage"] == "explore_environment" for r in runs) == 1
    assert all(e.args[0].source_method == "envexplore" for e in evaluate.await_args_list)

@pytest.mark.parametrize("status,output", [
    ("failed", None), ("budget_exhausted", None),
    ("completed", {"guidance": " "}), ("completed", {"invalid": "output"}),
])
def test_invalid_exploration_stops_generation(config, wired, status, output):
    _, _, exploration, _, call, evaluate = wired
    exploration.return_value = RunResult(output=output, trace=[],
        usage={"model_turns": 2, "input_tokens": 8, "output_tokens": 4}, status=status)
    with pytest.raises(RuntimeError):
        asyncio.run(runner.run_experiment(config))
    call.assert_not_awaited()
    evaluate.assert_not_awaited()
    root = Path(config.output_dir)
    assert json.loads((root / "manifest.json").read_text())["status"] == "failed"
    recorded = json.loads((root / "runs.json").read_text())[0]["result"]
    assert recorded["status"] == status and recorded["usage"]["total_tokens"] == 12

@pytest.mark.parametrize("fault", ["empty_exploration", "missing_role", "overlap", "comparison"])
def test_bad_config_rejected_before_environment(config, wired, fault):
    if fault == "empty_exploration": config.evolution_patients = []
    if fault == "missing_role": del config.runtime.models["environment_explorer"]
    if fault == "overlap": config.evolution_patients = ["p1"]
    if fault == "comparison": config.compare_initial_strategy = True
    with pytest.raises(ValueError):
        asyncio.run(runner.run_experiment(config))
    assert wired[0] == []
    wired[2].assert_not_awaited()
