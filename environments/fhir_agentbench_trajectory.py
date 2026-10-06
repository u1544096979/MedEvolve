"""FHIR-AgentBench 患者轨迹整理：按原始字段、就诊关系与引用呈现患者材料。

时间字段不共用 ``environments/__init__.py`` 的 ``_record_time``/``_time_sort``：那份实现按
固定优先级混合不同含义的时间。本环境里 ``Observation/issued`` 是发布时间、
``Observation/valueDateTime`` 是测量值描述的日期、``Condition/recordedDate`` 是记录时间、
``MedicationStatement/dateAsserted`` 是陈述时间，字段来源必须分别保留。
"""

import json
from datetime import datetime, timezone

from contracts import EnvironmentInstance, Evidence, PatientTrajectory
from environments.fhir import references

# 就诊来源由 identifier.system 末段判定，已知取值见 MIMIC-IV FHIR 的 identifier 命名。
ENCOUNTER_KINDS = ("encounter-hosp", "encounter-ed", "encounter-icu")

# 按原始字段名读取的记录时间字段；嵌套字段写成父字段.子字段。
_FLAT_TIME_FIELDS = (
    "effectiveDateTime",
    "effectivePeriod",
    "issued",
    "valueDateTime",
    "authoredOn",
    "performedDateTime",
    "performedPeriod",
    "onsetDateTime",
    "onsetPeriod",
    "recordedDate",
    "dateAsserted",
    "whenPrepared",
    "whenHandedOver",
)

# 事件发生时间：只取该资源真正表示事件发生的字段，其余时间另作标注。
_EVENT_FIELDS = {
    "Observation": ("effectiveDateTime", "effectivePeriod"),
    "MedicationRequest": ("authoredOn",),
    "MedicationAdministration": ("effectiveDateTime", "effectivePeriod"),
    "Procedure": ("performedDateTime", "performedPeriod"),
    "Specimen": ("collection.collectedDateTime", "collection.collectedPeriod"),
    "Condition": ("onsetDateTime", "onsetPeriod"),
    "MedicationStatement": ("effectiveDateTime", "effectivePeriod"),
    "MedicationDispense": ("whenHandedOver",),
}

# 不表示事件发生的时间字段，按来源标注，避免被当成事件时间。
_TIME_LABELS = {
    "issued": "发布时间",
    "valueDateTime": "测量值日期",
    "recordedDate": "记录时间",
    "dateAsserted": "陈述时间",
    "whenPrepared": "备药时间",
}

_UNDATED = "[无事件时间]"


def encounter_kind(resource: dict) -> str | None:
    """按 identifier.system 判断就诊来源；识别不到返回 None，不看 class。"""
    for identifier in resource.get("identifier", []):
        if not isinstance(identifier, dict):
            continue
        kind = str(identifier.get("system", "")).rsplit("/", 1)[-1]
        if kind in ENCOUNTER_KINDS:
            return kind
    return None


def time_fields(resource: dict) -> dict:
    """返回记录中实际存在的时间字段，字段名保留来源（如 collection.collectedDateTime）。"""
    fields = {
        key: resource[key] for key in _FLAT_TIME_FIELDS if key in resource
    }
    if resource.get("resourceType") == "Encounter" and "period" in resource:
        fields["period"] = resource["period"]
    collection = resource.get("collection")
    if isinstance(collection, dict):
        for key in ("collectedDateTime", "collectedPeriod"):
            if key in collection:
                fields[f"collection.{key}"] = collection[key]
    return fields


def _source_id(resource: dict) -> str:
    return f"{resource.get('resourceType')}/{resource.get('id')}"


def _dedupe(resources: list[dict]) -> list[dict]:
    unique: dict[str, dict] = {}
    for resource in resources:
        unique[_source_id(resource)] = resource
    return list(unique.values())


def _parse(value: str) -> datetime | None:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _time_values(value) -> list[str]:
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, dict):
        return [
            value[part] for part in ("start", "end") if isinstance(value.get(part), str)
        ]
    return []


def _time_text(value) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        start = value.get("start") or "缺失"
        end = value.get("end") or "缺失"
        return f"{start}→{end}"
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def event_time(resource: dict) -> str | None:
    """返回事件时间的原始值；Period 取 start，其次 end；没有事件时间返回 None。"""
    fields = time_fields(resource)
    for field in _EVENT_FIELDS.get(resource.get("resourceType", ""), ()):
        value = fields.get(field)
        for part in (None, "start", "end"):
            raw = value
            if part is not None:
                raw = value.get(part) if isinstance(value, dict) else None
            if isinstance(raw, str) and raw:
                return raw
    return None


def _clinical(resource: dict) -> str:
    """记录内容：保留临床字段，去掉身份与展示用字段。"""
    compact = {
        key: value
        for key, value in resource.items()
        if key not in {"meta", "text", "id", "resourceType", "subject"}
    }
    return json.dumps(
        _compact_value(compact), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def _compact_value(value):
    """把编码、数量、引用压成一行可读形式，不改变原始字段名。

    与 ``environments/__init__.py`` 的版本分开：这里保留完整的 ``system`` URI，
    药名与编码按原始记录呈现。
    """
    if isinstance(value, list):
        return [_compact_value(item) for item in value]
    if not isinstance(value, dict):
        return value
    if "coding" in value and isinstance(value["coding"], list):
        parts = []
        for coding in value["coding"]:
            parts.append(
                "|".join(
                    part
                    for part in (
                        coding.get("system", ""),
                        coding.get("code", ""),
                        coding.get("display", ""),
                    )
                    if part
                )
            )
        if value.get("text"):
            parts.append(value["text"])
        extras = {
            key: _compact_value(item)
            for key, item in value.items()
            if key not in {"coding", "text"}
        }
        return {"concept": parts, **extras} if extras else {"concept": parts}
    if "reference" in value and set(value) <= {
        "reference",
        "display",
        "type",
        "identifier",
        "extension",
        "id",
    }:
        return value["reference"]
    return {key: _compact_value(item) for key, item in value.items()}


def _encounter_bounds(resource: dict | None) -> str:
    if resource is None:
        return "未在材料中"
    period = resource.get("period") or {}
    start = period.get("start") or "缺失"
    end = period.get("end") or "缺失"
    return f"{start}→{end}"


def _times_line(resource: dict) -> str:
    parts = []
    for field, value in time_fields(resource).items():
        label = _TIME_LABELS.get(field)
        parts.append(f"{field}={_time_text(value)}" + (f" ({label})" if label else ""))
    return " ".join(parts)


def _refs_line(resource: dict) -> str:
    refs = sorted(
        {
            reference
            for reference in references(resource)
            if not reference.startswith(("Patient/", "#"))
        }
    )
    return f" refs={','.join(refs)}" if refs else ""


def _encounter_ref(resource: dict) -> str:
    for field in ("encounter", "context"):
        value = resource.get(field)
        reference = value.get("reference") if isinstance(value, dict) else None
        if isinstance(reference, str) and reference:
            return reference
    return ""


def _record_line(
    resource: dict,
    *,
    encounter_ref: str,
    encounters: dict[str, dict],
    event: str | None,
) -> str:
    source_id = _source_id(resource)
    head = _UNDATED if event is None else f"[{event}]"
    parts = [head, source_id, f"encounter={encounter_ref or 'none'}"]
    times = _times_line(resource)
    if times:
        parts.append(times)
    if event is None and encounter_ref:
        parts.append(
            f"background={encounter_ref} {_encounter_bounds(encounters.get(encounter_ref))}"
        )
    line = " ".join(parts)
    return f"{line}{_refs_line(resource)} {_clinical(resource)}"


def _encounter_line(resource: dict, indent: str = "") -> str:
    kind = encounter_kind(resource) or "就诊类型未知"
    class_code = (resource.get("class") or {}).get("code") or "缺失"
    parts = [f"{indent}{_source_id(resource)} [{kind}] class={class_code}"]
    period = resource.get("period")
    if isinstance(period, dict):
        parts.append(f"period={_time_text(period)}")
    parent = (resource.get("partOf") or {}).get("reference")
    if parent:
        parts.append(f"partOf={parent}")
    return " ".join(parts) + " " + _clinical(resource)


def build_trajectory(patient_id: str, resources: list[dict]) -> PatientTrajectory:
    """把一份患者材料整理成轨迹：sources 保存完整原始 JSON，text 按分区呈现。"""
    records = _dedupe(resources)
    by_source = {_source_id(resource): resource for resource in records}
    patient = by_source.get(f"Patient/{patient_id}")
    if patient is None:
        raise ValueError(f"Patient material does not contain Patient/{patient_id}")

    encounters = {
        source_id: resource
        for source_id, resource in by_source.items()
        if resource.get("resourceType") == "Encounter"
    }
    hospitals = [
        resource
        for resource in encounters.values()
        if encounter_kind(resource) == "encounter-hosp"
    ]
    hospitals.sort(key=lambda resource: _encounter_sort_key(resource))
    children: dict[str, list[dict]] = {}
    for resource in encounters.values():
        parent = (resource.get("partOf") or {}).get("reference")
        if parent:
            children.setdefault(parent, []).append(resource)
    for group in children.values():
        group.sort(key=lambda resource: _encounter_sort_key(resource))

    encounter_lines = []
    presented = set()
    for hospital in hospitals:
        encounter_lines.append(_encounter_line(hospital))
        presented.add(_source_id(hospital))
        for child in children.get(_source_id(hospital), []):
            encounter_lines.append(_encounter_line(child, indent="  "))
            presented.add(_source_id(child))
    for source_id, resource in encounters.items():
        if source_id not in presented:
            encounter_lines.append(_encounter_line(resource))

    events = []
    undated = []
    for source_id, resource in by_source.items():
        if resource.get("resourceType") in {"Patient", "Encounter"}:
            continue
        event = event_time(resource)
        if event is None:
            undated.append((source_id, resource))
        else:
            events.append((source_id, resource, event))
    events.sort(key=lambda item: _event_sort_key(item[1], item[2]))
    undated.sort(key=lambda item: (_encounter_ref(item[1]), item[0]))

    unparsable = 0
    for resource in by_source.values():
        for value in time_fields(resource).values():
            for raw in _time_values(value):
                if _parse(raw) is None:
                    unparsable += 1

    sections = [
        "# 患者原始信息",
        f"{_source_id(patient)} {_clinical(patient)}",
        "",
        "# 就诊原始信息",
        *encounter_lines,
        "",
        "# 归属事件",
        *[
            _record_line(
                resource,
                encounter_ref=_encounter_ref(resource),
                encounters=encounters,
                event=event,
            )
            for _, resource, event in events
        ],
        "",
        "# 无事件时间记录",
        *[
            _record_line(
                resource,
                encounter_ref=_encounter_ref(resource),
                encounters=encounters,
                event=None,
            )
            for _, resource in undated
        ],
        "",
        "# 整理统计",
        _stats_line(by_source, encounters, events, undated, unparsable),
    ]
    return PatientTrajectory(
        patient_id=patient_id,
        environment="FHIR-AgentBench",
        text="\n".join(sections),
        sources=[
            Evidence(
                source_id=source_id,
                statement=json.dumps(resource, ensure_ascii=False, sort_keys=True),
            )
            for source_id, resource in sorted(by_source.items())
        ],
    )


def _encounter_sort_key(resource: dict) -> tuple:
    period = resource.get("period") or {}
    start = period.get("start")
    parsed = _parse(start) if isinstance(start, str) else None
    if parsed is None:
        return (1, 0.0, _source_id(resource))
    return (0, _timestamp(parsed), _source_id(resource))


def _event_sort_key(resource: dict, event: str) -> tuple:
    parsed = _parse(event)
    source_id = _source_id(resource)
    if parsed is None:
        return (1, 0.0, source_id)
    return (0, _timestamp(parsed), source_id)


def _timestamp(parsed: datetime) -> float:
    # 仅用于排序：原始字符串保留数据中的时区，缺失时区不写回记录。
    return parsed.replace(tzinfo=parsed.tzinfo or timezone.utc).timestamp()


def _stats_line(
    by_source: dict[str, dict],
    encounters: dict[str, dict],
    events: list[tuple],
    undated: list[tuple],
    unparsable: int,
) -> str:
    counts = {kind: 0 for kind in ENCOUNTER_KINDS}
    other = 0
    for resource in encounters.values():
        kind = encounter_kind(resource)
        if kind is None:
            other += 1
        else:
            counts[kind] += 1
    return (
        f"资源 {len(by_source)} 条；就诊 {len(encounters)} 条"
        f"（医院 {counts['encounter-hosp']}、急诊 {counts['encounter-ed']}、"
        f"ICU {counts['encounter-icu']}、其他 {other}）；"
        f"事件记录 {len(events)} 条；无事件时间记录 {len(undated)} 条；"
        f"时间解析失败 {unparsable} 条"
    )


async def prepare(instance: EnvironmentInstance, patient_id: str) -> PatientTrajectory:
    from .fhir_agentbench import read_patient_material

    resources = await read_patient_material(instance, patient_id)
    return build_trajectory(patient_id, resources)
