import pytest

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
