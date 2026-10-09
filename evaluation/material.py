"""Build deterministic inputs for quality and difficulty reviewers."""

import copy
import json
from datetime import datetime, timezone

from contracts import ComplexityMetrics, PatientTrajectory, Task, Verification
from environments.fhir import decode_document_notes


def _record(source) -> tuple[dict, datetime | None]:
    try:
        parsed = json.loads(source.statement)
    except (TypeError, ValueError):
        return {"source_id": source.source_id, "record": source.statement}, None
    if not isinstance(parsed, dict):
        return {"source_id": source.source_id, "record": source.statement}, None
    original = copy.deepcopy(parsed)
    try:
        parsed = decode_document_notes(parsed)
    except (ValueError, UnicodeDecodeError):
        parsed = original
    moment = None
    for key in ("effectiveDateTime", "onsetDateTime", "performedDateTime",
                "occurrenceDateTime", "authoredOn", "sent", "start", "date",
                "recordedDate", "issued"):
        value = parsed.get(key)
        if isinstance(value, str):
            try:
                moment = datetime.fromisoformat(value.replace("Z", "+00:00"))
                if moment.tzinfo is None:
                    moment = moment.replace(tzinfo=timezone.utc)
                break
            except ValueError:
                continue
    if moment is None:
        for key in ("effectivePeriod", "onsetPeriod", "performedPeriod", "period"):
            period = parsed.get(key)
            if isinstance(period, dict):
                value = period.get("start")
                if isinstance(value, str):
                    try:
                        moment = datetime.fromisoformat(value.replace("Z", "+00:00"))
                        if moment.tzinfo is None:
                            moment = moment.replace(tzinfo=timezone.utc)
                        break
                    except ValueError:
                        continue
    return {"source_id": source.source_id, "record": parsed}, moment


def review_material(task: Task, trajectory: PatientTrajectory,
                    verification: Verification, complexity: ComplexityMetrics,
                    *, mode: str) -> dict:
    """Return full or compact clinical context for scoring calls only.

    Compact mode keeps every trajectory source and every clinical field. Its
    reduction comes from omitting the duplicated rendered trajectory text and
    placing cited sources apart from the chronologically ordered remaining
    records.
    """
    task_data = task.model_dump(include={
        "instruction", "task_type", "reference_answer", "evidence",
        "decision_point", "patient_id", "environment",
    })
    if mode == "full":
        from evaluation.evaluator import _review_material
        payload = _review_material(task, trajectory)
        payload["verification"] = verification.model_dump(mode="json")
        return payload
    if mode != "compact":
        raise ValueError(f"unknown review input mode: {mode}")

    cited_ids = {item.source_id for item in task.evidence}
    cited_records = []
    context_records = []
    for source in trajectory.sources:
        entry, moment = _record(source)
        if source.source_id in cited_ids:
            cited_records.append(entry)
        else:
            context_records.append((moment, source.source_id, entry))
    context_records.sort(key=lambda item: (
        item[0] is None,
        item[0] or datetime.max.replace(tzinfo=timezone.utc),
        item[1],
    ))
    return {
        "task": task_data,
        "cited_records": cited_records,
        "clinical_context": [entry for _, _, entry in context_records],
        "verification": verification.model_dump(mode="json"),
        "complexity_metrics": complexity.model_dump(mode="json"),
    }
