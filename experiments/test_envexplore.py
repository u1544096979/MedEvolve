import asyncio
import json
import random
from pathlib import Path
from unittest.mock import ANY, AsyncMock
import pytest
from contracts import (
    EnvironmentExploration, EnvironmentInstance, EvaluationResult, Evidence, ExperimentConfig,
    FrozenEnvironmentGuidance,
    PatientTrajectory, RunResult,
)
from experiments import runner

@pytest.fixture
def config(tmp_path):
    values = json.loads((Path(__file__).parents[1] / "configs/experiment.json").read_text())
    values["environment"]["name"] = "PhysicianBench"
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
            "decision_point": None, "execution_requirement": "read_only"}]}, trace=[],
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
    exploration.assert_awaited_once_with(config.environment, ["train"], config.runtime,
                                         environments=ANY)
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
    frozen = FrozenEnvironmentGuidance.model_validate_json(
        (root / "environment_guidance.json").read_text()
    )
    assert frozen.environment == config.environment.name
    assert frozen.exploration_patients == ["train"]
    assert frozen.guidance.guidance == "已验证读取能力"
    assert frozen.exploration_run.status == "completed"
    assert not (root / "rounds").exists() and not (root / "data_exploration_patients.json").exists()
    runs = json.loads((root / "runs.json").read_text())
    assert sum(r["stage"] == "explore_environment" for r in runs) == 1
    assert all(e.args[0].source_method == "envexplore" for e in evaluate.await_args_list)


def test_exploration_only_writes_reusable_frozen_guidance(config, wired, tmp_path):
    prepared, _, exploration, _, call, evaluate = wired
    config.exploration_only = True
    config.evaluation_patients = []
    config.round_patients = []
    config.output_dir = str(tmp_path / "exploration")
    asyncio.run(runner.run_experiment(config))
    exploration.assert_awaited_once()
    assert prepared == [] and call.await_count == evaluate.await_count == 0
    frozen = FrozenEnvironmentGuidance.model_validate_json(
        (Path(config.output_dir) / "environment_guidance.json").read_text()
    )
    assert frozen.environment == config.environment.name
    assert frozen.exploration_run.status == "completed"
    assert frozen.exploration_patients == ["train"]


def test_frozen_guidance_reused_across_batches_without_explorer_role(config, wired, tmp_path):
    _, _, exploration, _, call, evaluate = wired
    frozen_path = tmp_path / "frozen.json"
    frozen_path.write_text(FrozenEnvironmentGuidance(
        environment=config.environment.name,
        exploration_patients=["train"],
        guidance=EnvironmentExploration(guidance="已验证读取能力"),
        exploration_run=RunResult(output={"guidance": "已验证读取能力"}, trace=[], usage={
            "model_turns": 2, "tool_calls": 1, "input_tokens": 8, "output_tokens": 4,
        }, status="completed"),
        runtime=config.runtime,
    ).model_dump_json())
    del config.runtime.models["environment_explorer"]
    del config.runtime.budgets["environment_explorer"]
    config.guidance_path = str(frozen_path)
    for index in range(2):
        config.output_dir = str(tmp_path / f"batch-{index}")
        asyncio.run(runner.run_experiment(config))
    exploration.assert_not_awaited()
    assert call.await_count == evaluate.await_count == 4
    for index in range(2):
        runs = json.loads((tmp_path / f"batch-{index}" / "runs.json").read_text())
        assert all(row["stage"] != "explore_environment" for row in runs)


@pytest.mark.parametrize("fault", ["failed", "environment", "overlap", "missing"])
def test_invalid_frozen_guidance_prevents_generation(config, wired, tmp_path, fault):
    _, _, exploration, _, call, evaluate = wired
    path = tmp_path / "frozen.json"
    config.guidance_path = str(path)
    if fault == "missing":
        with pytest.raises(ValueError, match="guidance_path"):
            asyncio.run(runner.run_experiment(config))
    else:
        payload = FrozenEnvironmentGuidance(
            environment=("MedAgentBench" if fault == "environment" else config.environment.name),
            exploration_patients=(config.evaluation_patients if fault == "overlap" else ["train"]),
            guidance=EnvironmentExploration(guidance="有效指导"),
            exploration_run=RunResult(output={"guidance": "有效指导"}, trace=[], usage={},
                status="failed" if fault == "failed" else "completed"),
            runtime=config.runtime,
        )
        path.write_text(payload.model_dump_json())
        with pytest.raises(ValueError):
            asyncio.run(runner.run_experiment(config))
    assert wired[0] == []
    exploration.assert_not_awaited()
    call.assert_not_awaited()
    evaluate.assert_not_awaited()

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
