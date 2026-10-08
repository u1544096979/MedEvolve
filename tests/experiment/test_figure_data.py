"""实验图数据的缺失值、分母与区间边界。"""

import pytest
from pydantic import ValidationError

from scripts.plot_experiment_figures import (
    DIMENSIONS, OUTCOMES, Dimension, FigureData, MethodResult, blank_data,
    draw_quality, draw_validity,
)


def result_payload():
    return {"method": "medevolve", "generated_tasks": 10, "generated_patients": 5,
            "outcomes": {name: (10 if name == "level_3" else 0) for name in OUTCOMES}}


def test_unknown_template_roundtrips_without_results():
    data = FigureData.model_validate_json(blank_data().model_dump_json())
    assert all(row.generated_tasks is None and row.outcomes is None and row.quality is None
               for row in data.groups[0].results)


@pytest.mark.parametrize("bad_count", [-1, 1.5, True, "10"])
def test_invalid_partition_count_is_rejected(bad_count):
    payload = result_payload()
    payload["outcomes"]["level_3"] = bad_count
    with pytest.raises(ValidationError):
        MethodResult.model_validate(payload)


def test_partition_cannot_omit_generated_tasks():
    payload = result_payload()
    payload["outcomes"]["level_3"] = 9
    with pytest.raises(ValidationError, match="partition"):
        MethodResult.model_validate(payload)


def test_zero_generation_is_distinct_from_unknown_generation():
    payload = result_payload()
    payload.update(generated_tasks=0, generated_patients=0)
    payload["outcomes"] = dict.fromkeys(OUTCOMES, 0)
    assert MethodResult.model_validate(payload).generated_tasks == 0
    assert blank_data().groups[0].results[0].generated_tasks is None


@pytest.mark.parametrize("low,high", [(None, 4.5), (4.5, 3.5), (1, 3.5), (.5, 4.5)])
def test_incomplete_or_inconsistent_interval_is_rejected(low, high):
    with pytest.raises(ValidationError):
        Dimension(mean=4, ci_low=low, ci_high=high, n_tasks=10, n_patients=5)


def test_quality_sample_cannot_include_verification_failures():
    payload = result_payload()
    payload["outcomes"].update(invalid=9, level_3=1)
    quality = dict.fromkeys(DIMENSIONS)
    quality["factual_accuracy"] = {"mean": 4, "ci_low": 3.5, "ci_high": 4.5,
                                   "n_tasks": 2, "n_patients": 1}
    payload["quality"] = quality
    with pytest.raises(ValidationError, match="verification-failed"):
        MethodResult.model_validate(payload)


def test_repeated_method_and_colliding_output_groups_are_rejected():
    data = blank_data().model_dump()
    data["groups"][0]["results"][-1]["method"] = "medevolve"
    with pytest.raises(ValidationError, match="six methods"):
        FigureData.model_validate(data)
    data = blank_data().model_dump()
    data["groups"] *= 2
    with pytest.raises(ValidationError, match="collide"):
        FigureData.model_validate(data)


def test_observed_data_requires_identified_condition():
    data = blank_data().model_dump()
    data["groups"][0]["results"][0] = result_payload()
    with pytest.raises(ValidationError, match="labels"):
        FigureData.model_validate(data)


def test_missing_data_produces_no_bars_points_or_intervals():
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    validity = draw_validity(blank_data().groups[0])
    quality = draw_quality(blank_data().groups[0])
    assert not validity.axes[0].patches
    assert all(not ax.lines and not ax.collections for ax in quality.axes)
    plt.close(validity)
    plt.close(quality)


def test_display_uses_generated_denominator_and_supplied_interval():
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = blank_data().model_dump()
    group = data["groups"][0]
    group.update(environment="PhysicianBench", generator="fixture")
    row = result_payload()
    row["outcomes"].update(invalid=2, level_3=8)
    row["quality"] = dict.fromkeys(DIMENSIONS)
    row["quality"]["factual_accuracy"] = {"mean": 4, "ci_low": 3.5, "ci_high": 4.5,
                                          "n_tasks": 8, "n_patients": 4}
    group["results"][0] = row
    parsed = FigureData.model_validate(data).groups[0]
    validity = draw_validity(parsed)
    quality = draw_quality(parsed)
    assert [patch.get_width() for patch in validity.axes[0].patches] == [20, 80]
    assert list(quality.axes[0].collections[0].get_segments()[0][:, 0]) == [3.5, 4.5]
    assert sum(len(ax.collections) for ax in quality.axes) == 1
    plt.close(validity)
    plt.close(quality)
