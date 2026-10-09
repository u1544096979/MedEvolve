import json
import base64

from contracts import (ComplexityMetrics, Evidence, PatientTrajectory, Task,
                       Verification)
from evaluation.material import review_material


def test_compact_material_keeps_conflicting_context_and_unknown_fields():
    task = Task(
        id="t1", patient_id="p1", environment="PhysicianBench", round_index=0,
        source_method="test", task_type="治疗规划", instruction="评估方案",
        reference_answer="应考虑过敏后停药", decision_point=None,
        evidence=[Evidence(source_id="MedicationRequest/start", statement="处方")],
    )
    trajectory = PatientTrajectory(
        patient_id="p1", environment="PhysicianBench", text="duplicated source render",
        sources=[
            Evidence(source_id="MedicationRequest/start", statement=json.dumps({
                "resourceType": "MedicationRequest", "id": "start",
                "authoredOn": "2024-03-05T12:00:00Z", "status": "active",
                "dosageInstruction": [{"text": "每日一次"}],
                "unknownClinicalField": {"value": "保留未知临床字段"},
            }, ensure_ascii=False)),
            Evidence(source_id="Observation/negative", statement=json.dumps({
                "resourceType": "Observation", "effectiveDateTime": "2024-03-05",
                "interpretation": "未见感染", "valueString": "未见感染",
            }, ensure_ascii=False)),
            Evidence(source_id="Observation/later", statement=json.dumps({
                "resourceType": "Observation", "effectiveDateTime": "2024-03-06",
                "valueQuantity": {"value": 8, "unit": "mg/L"},
            }, ensure_ascii=False)),
            Evidence(source_id="Condition/allergy", statement=json.dumps({
                "resourceType": "Condition", "clinicalStatus": "已缓解，因过敏停药",
                "note": [{"text": "患者报告之前出现皮疹"}],
            }, ensure_ascii=False)),
            Evidence(source_id="DocumentReference/note", statement=json.dumps({
                "resourceType": "DocumentReference", "content": [{"attachment": {
                    "contentType": "text/plain", "data": base64.b64encode(
                        "完整自由文本：继续观察，不截断。".encode()).decode()
                }}], "extension": [{"valueString": "保留extension"}],
            }, ensure_ascii=False)),
        ],
    )
    compact = review_material(
        task, trajectory, Verification(valid=True, reason="合法"),
        ComplexityMetrics(evidence_count=5, resource_type_count=4, time_span_days=1),
        mode="compact",
    )
    assert compact["cited_records"][0]["source_id"] == "MedicationRequest/start"
    serialized = json.dumps(compact, ensure_ascii=False)
    for text in ("因过敏停药", "未见感染", "2024-03-05", "2024-03-06",
                 "保留未知临床字段", "完整自由文本：继续观察，不截断。", "保留extension"):
        assert text in serialized
    assert "duplicated source render" not in serialized


def test_full_mode_remains_default_and_rejects_unknown_mode():
    task = Task(
        id="t", patient_id="p", environment="PhysicianBench", round_index=0,
        source_method="test", task_type="诊断与解释", instruction="题目",
        reference_answer="答案", evidence=[],
    )
    trajectory = PatientTrajectory(patient_id="p", environment="PhysicianBench",
                                   text="原始轨迹", sources=[])
    common = (task, trajectory, Verification(valid=True, reason="ok"),
              ComplexityMetrics(evidence_count=0, resource_type_count=0,
                               time_span_days=None))
    assert "trajectory" in review_material(*common, mode="full")
    try:
        review_material(*common, mode="other")
    except ValueError as exc:
        assert "unknown review input mode" in str(exc)
    else:
        raise AssertionError("invalid input mode should fail")
