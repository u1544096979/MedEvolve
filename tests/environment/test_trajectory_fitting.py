"""超长患者输入的裁剪规则。"""

from contracts import Evidence, PatientTrajectory
from environments.fitting import fit_trajectory, generation_budget_chars, payload_chars


def trajectory(records):
    """逐段轨迹：一段一条记录，正文段落与 sources 一一对应。"""
    blocks = [f"[2020-01-01] {source_id}\n" + "x" * size for source_id, size in records]
    sources = [Evidence(source_id=source_id, statement="s" * size) for source_id, size in records]
    return PatientTrajectory(patient_id="p1", environment="PhysicianBench",
                             text="\n\n".join(blocks), sources=sources)


def line_trajectory(records, headers=("# 患者原始信息", "# 整理统计")):
    """分区式轨迹：一行一条记录，行首是记录 id。"""
    lines = [headers[0]]
    sources = []
    for source_id, size in records:
        lines.append(f"[2020-01-01] {source_id} encounter=none " + "x" * size)
        sources.append(Evidence(source_id=source_id, statement="s" * size))
    lines.append(headers[1])
    return PatientTrajectory(patient_id="p1", environment="FHIR-AgentBench",
                             text="\n".join(lines), sources=sources)


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


def test_line_mode_keeps_headers_and_drops_unrendered_records():
    original = line_trajectory([("Patient/p1", 10)]
                               + [(f"Observation/o{index}", 2000) for index in range(20)]
                               + [("Condition/c1", 200)])
    fitted = fit_trajectory(original, 20000)
    kept = [source.source_id for source in fitted.sources]
    assert kept[0] == "Patient/p1" and "Condition/c1" in kept
    assert "Observation/o0" not in kept and "Observation/o19" in kept
    assert fitted.text.startswith("# 患者原始信息") and "# 整理统计" in fitted.text
    assert payload_chars(fitted) <= 20000
    rendered = fitted.text.split("\n")[2:-1]
    assert all(any(source.source_id in line for source in fitted.sources) for line in rendered)


def test_budget_follows_context_window():
    assert generation_budget_chars(128000, 16384) == int((128000 - 16384 - 8192) * 2.611)
    assert generation_budget_chars(10000, 16384) == 0
