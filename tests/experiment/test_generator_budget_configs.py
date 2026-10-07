import json
from pathlib import Path

from contracts import ExperimentConfig
from scripts.emit_comparison_configs import METHOD_SPEC, _runtime

ROOT = Path(__file__).resolve().parents[2]
METHODS = {"medevolve", "taskcraft", "workflow", "singleshot", "selfgen", "envexplore"}
EXPECTED = {"max_model_turns": 800, "max_tool_calls": 800}


def test_declared_generation_budgets_are_uniform():
    checked = set()
    declared = set()
    for path in sorted((ROOT / "configs").glob("*.json")):
        if path.name.startswith("full-"):
            continue
        payload = json.loads(path.read_text())
        if payload.get("method") not in METHODS:
            continue
        if "generator" not in payload.get("runtime", {}).get("budgets", {}):
            continue
        declared.add(payload["method"])
        config = ExperimentConfig.model_validate(payload)
        assert config.runtime.budgets["generator"].model_dump() == EXPECTED, path.name
        checked.add(config.method)
    assert checked == declared
    assert set(METHOD_SPEC) == METHODS


def test_emitted_runtime_uses_uniform_generator_budget():
    template = json.loads((ROOT / "configs" / "experiment.json").read_text())
    for method, spec in METHOD_SPEC.items():
        runtime = _runtime(spec, template)
        assert runtime["budgets"]["generator"] == EXPECTED, method
        for role, limits in spec["budgets"].items():
            if role != "generator":
                assert runtime["budgets"][role] == {
                    "max_model_turns": limits[0], "max_tool_calls": limits[1]
                }
