import json
from pathlib import Path
import pytest
from contracts import ExperimentConfig
from experiments.runner import resolve_generator
from methods.selfgen import generate_tasks

@pytest.mark.parametrize("source,target", [
    ("physicianbench-comparison.json", "physicianbench-selfgen.json"),
    ("medagentbench-flash.json", "medagentbench-selfgen.json"),
])
def test_config_keeps_sampling_environment_and_model(source, target):
    root = Path(__file__).parents[1] / "configs"
    base = json.loads((root / source).read_text())
    values = json.loads((root / target).read_text())
    config = ExperimentConfig.model_validate(values)
    assert config.method == "selfgen" and config.round_patients == []
    assert values["compare_initial_strategy"] is False
    assert config.evaluation_patients
    for key in ("environment", "evolution_patients", "evaluation_patients",
                "initial_strategy", "random_seed", "evaluation"):
        assert values[key] == base[key]
    assert set(config.runtime.models) == {"generator"}
    assert set(config.runtime.budgets) == {"generator"}
    assert values["runtime"]["models"]["generator"] == base["runtime"]["models"]["generator"]
    assert values["runtime"]["budgets"]["generator"] == base["runtime"]["budgets"]["generator"]
    assert resolve_generator("selfgen") is generate_tasks
