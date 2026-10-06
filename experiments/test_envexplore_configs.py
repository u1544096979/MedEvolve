import json
from pathlib import Path
import pytest
from contracts import ExperimentConfig

@pytest.mark.parametrize("source,target", [
    ("physicianbench-comparison.json", "physicianbench-envexplore.json"),
    ("medagentbench-flash.json", "medagentbench-envexplore.json"),
])
def test_configs_keep_main_comparison_material(source, target):
    root = Path(__file__).parents[1] / "configs"
    base = json.loads((root / source).read_text())
    values = json.loads((root / target).read_text())
    config = ExperimentConfig.model_validate(values)
    assert config.method == "envexplore" and config.round_patients == []
    assert config.compare_initial_strategy is False
    for key in ("environment", "evolution_patients", "evaluation_patients",
                "initial_strategy", "random_seed", "evaluation"):
        if key == "environment":
            assert {k: v for k, v in values[key].items() if k != "tool_result_limit"} == base[key]
            assert values[key]["tool_result_limit"] == 5000
            continue
        assert values[key] == base[key]
    assert set(config.runtime.models) == {"environment_explorer", "generator"}
    assert set(config.runtime.budgets) == {"environment_explorer", "generator"}
    for role in ("environment_explorer", "generator"):
        assert values["runtime"]["models"][role] == base["runtime"]["models"][role]
        assert values["runtime"]["budgets"][role] == base["runtime"]["budgets"][role]
    assert config.evolution_patients and config.evaluation_patients
    assert not set(config.evolution_patients) & set(config.evaluation_patients)
    assert values["run_id"] != base["run_id"] and values["output_dir"] != base["output_dir"]
