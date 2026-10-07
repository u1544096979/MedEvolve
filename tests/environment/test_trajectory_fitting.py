"""超长患者输入的裁剪规则。"""

from contracts import Evidence, PatientTrajectory
from environments.fitting import fit_trajectory, generation_budget_chars, payload_chars


def trajectory(records):
    blocks = [f"[2020-01-01] {source_id}\n" + "x" * size
              for source_id, size in records]
    sources = [Evidence(source_id=source_id, statement="s" * size)
               for source_id, size in records]
    return PatientTrajectory(patient_id="p1", environment="PhysicianBench",
                             text="\n\n".join(blocks), sources=sources)


def test_within_budget_is_returned_unchanged():
    original = trajectory([("Patient/p1", 10), ("Condition/c1", 10)])
    assert fit_trajectory(original, 10000) is original


def test_patient_and_condition_survive_observation_flood():
    original = trajectory([("Patient/p1", 10)] + [("Observation/o1", 4000)] * 10
                          + [("Condition/c1", 200)])
    fitted = fit_trajectory(original, 9000)
    kept = [source.source_id for source in fitted.sources]
    assert kept[0] == "Patient/p1" and "Condition/c1" in kept
    assert payload_chars(fitted) <= 9000
    assert len(fitted.text.split("\n\n")) == len(fitted.sources)


def test_recent_records_are_preferred_within_a_type():
    original = trajectory([("Patient/p1", 10)]
                          + [(f"Observation/o{index}", 900) for index in range(10)])
    fitted = fit_trajectory(original, 4000)
    kept = [source.source_id for source in fitted.sources]
    assert kept[0] == "Patient/p1"
    assert kept[-1] == "Observation/o9" and "Observation/o0" not in kept


def test_alignment_mismatch_skips_fitting():
    original = trajectory([("Patient/p1", 10), ("Condition/c1", 10)])
    mismatched = original.model_copy(update={"text": original.text.split("\n\n")[0]})
    assert fit_trajectory(mismatched, 10) is mismatched


def test_budget_follows_context_window():
    assert generation_budget_chars(128000, 16384) == int((128000 - 16384 - 8192) * 2.611)
    assert generation_budget_chars(10000, 16384) == 0
