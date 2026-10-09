import json
from pathlib import Path

import pytest

from contracts import EnvironmentExploration, FrozenEnvironmentGuidance, RunResult, RuntimeConfig
from scripts import run_full_comparison as driver


def _write_envexplore_configs(root: Path, *, overlap: bool = False) -> None:
    configs = root / "configs"
    configs.mkdir()
    exploration = {
        "output_dir": str(root / "outputs" / "exploration"),
        "environment": {"name": "PhysicianBench"},
        "evolution_patients": ["train-1", "train-2"],
    }
    (configs / "full-physicianbench-envexplore-exploration.json").write_text(
        json.dumps(exploration), encoding="utf-8"
    )
    batches = [["test-1"], ["train-2" if overlap else "test-2"]]
    for index, patients in enumerate(batches, start=1):
        payload = {"evaluation_patients": patients}
        (configs / f"full-physicianbench-envexplore-batch-{index:02}.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )


def test_driver_validates_union_of_test_patients_before_exploration_or_testing(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    _write_envexplore_configs(tmp_path, overlap=True)
    monkeypatch.setattr("sys.argv", ["run", "--slug", "physicianbench",
                                     "--only", "envexplore", "--skip-evolution"])
    monkeypatch.setattr(driver, "emit", lambda *args: None)
    calls = []
    monkeypatch.setattr(driver, "run", lambda *args: calls.append("exploration") or 0)
    monkeypatch.setattr(driver, "run_glob", lambda *args: calls.append("batches") or 0)

    assert driver.main() == 2
    assert calls == []


def test_driver_finishes_exploration_before_running_test_batches(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_envexplore_configs(tmp_path)
    monkeypatch.setattr("sys.argv", ["run", "--slug", "physicianbench",
                                     "--only", "envexplore", "--skip-evolution"])
    monkeypatch.setattr(driver, "emit", lambda *args: None)
    calls = []

    def run_exploration(config_path, *_args):
        calls.append("exploration")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        guidance_path = Path(config["output_dir"]) / "environment_guidance.json"
        guidance_path.parent.mkdir(parents=True, exist_ok=True)
        guidance_path.write_text(FrozenEnvironmentGuidance(
            environment="PhysicianBench", exploration_patients=["train-1", "train-2"],
            guidance=EnvironmentExploration(guidance="指导"),
            exploration_run=RunResult(trace=[], usage={}, status="completed"),
            runtime=RuntimeConfig(models={}, budgets={}),
        ).model_dump_json(), encoding="utf-8")
        return 0

    monkeypatch.setattr(driver, "run", run_exploration)
    monkeypatch.setattr(driver, "run_glob", lambda *args: calls.append("batches") or 0)

    assert driver.main() == 0
    assert calls == ["exploration", "batches"]
