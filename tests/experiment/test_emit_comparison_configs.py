import pytest
import json
from pathlib import Path
from scripts import emit_comparison_configs

from scripts.emit_comparison_configs import _round_patients


def test_explicit_rounds_cover_evolution_patients_once():
    rounds = _round_patients(
        {"round_patients": [["p1", "p2"], ["p3"]]}, ["p1", "p2", "p3"]
    )
    assert rounds == [["p1", "p2"], ["p3"]]


def test_legacy_cohort_repeats_full_evolution_set_each_round():
    assert _round_patients({}, ["p1", "p2"]) == [["p1", "p2"], ["p1", "p2"]]


@pytest.mark.parametrize(
    "rounds",
    [[], [["p1"]], [["p1"], ["p1"]], [["p1"], ["p3"]]],
)
def test_invalid_explicit_rounds_are_rejected(rounds):
    with pytest.raises(ValueError, match="round_patients"):
        _round_patients({"round_patients": rounds}, ["p1", "p2"])


@pytest.mark.parametrize("slug,template", [
    ("physicianbench", "physicianbench-comparison.json"),
    ("medagentbench", "medagentbench-flash.json"),
])
def test_envexplore_emits_one_frozen_exploration_and_shared_batch_path(
    tmp_path, monkeypatch, slug, template
):
    source = Path(__file__).parents[2] / "configs" / template
    values = json.loads(source.read_text())
    cohort = tmp_path / "cohort.json"
    cohort.write_text(json.dumps({"evolution_patients": ["train-1", "train-2"],
        "batches": [["test-1"], ["test-2"]]}))
    configs_dir = tmp_path / "configs"
    output_root = tmp_path / "output"
    monkeypatch.setattr("sys.argv", ["emit", "--slug", slug, "--method", "envexplore",
        "--cohort", str(cohort), "--configs-dir", str(configs_dir),
        "--template-dir", str(source.parent), "--output-root", str(output_root),
        "--batch-size", "1"])
    emit_comparison_configs.main()
    exploration = json.loads((configs_dir / f"full-{slug}-envexplore-exploration.json").read_text())
    batches = sorted(configs_dir.glob(f"full-{slug}-envexplore-batch-*.json"))
    assert exploration["exploration_only"] is True
    assert exploration["evaluation_patients"] == []
    assert len(batches) == 2
    guidance_paths = {json.loads(path.read_text())["guidance_path"] for path in batches}
    assert guidance_paths == {str(output_root / "exploration" / "environment_guidance.json")}
