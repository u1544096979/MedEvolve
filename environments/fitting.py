"""按记录优先级裁剪超长患者输入，使生成输入落在模型上下文预算内。

生成模型上下文有限，而原始轨迹可能包含上万条记录。裁剪保留患者记录、诊断与用药等
关键类型，其余类型按预算均分后保留轨迹中靠后的记录。正文与 sources 同步裁剪：
按记录分段的轨迹逐段保留，按分区呈现的轨迹按行保留。评测始终使用未裁剪的原始轨迹，
核验读取的也是原始来源记录。
"""

import json

from contracts import Evidence, PatientTrajectory

# 关键临床类型优先保留，未列出的类型排在其后。
TYPE_PRIORITY = (
    "Patient", "Condition", "AllergyIntolerance", "MedicationRequest",
    "MedicationStatement", "MedicationAdministration", "Procedure", "ServiceRequest",
    "DiagnosticReport", "Encounter", "CarePlan", "Immunization", "Observation",
    "DocumentReference", "Communication", "Appointment",
)
# 每条来源在 JSON 输入中的固定开销：source_id、字段名与转义余量。
SOURCE_OVERHEAD_CHARS = 64
# 行式轨迹里正文按记录平摊的估算偏差，先用留有余量的预算选择记录。
LINE_MODE_MARGIN = 0.9


def _priority(resource_type: str) -> tuple[int, str]:
    return (TYPE_PRIORITY.index(resource_type) if resource_type in TYPE_PRIORITY
            else len(TYPE_PRIORITY), resource_type)


def payload_chars(trajectory: PatientTrajectory) -> int:
    """生成模型实际收到的字符数：轨迹按 JSON 序列化后的大小。"""
    return len(json.dumps(trajectory.model_dump(mode="json"), ensure_ascii=False))


def _select(records: list[dict], budget: int, always_keep: set[str]) -> list[dict]:
    """优先保留关键类型与轨迹中靠后的记录，返回选中的记录。"""
    kept: list[dict] = []
    used = 0

    def keep(record: dict) -> bool:
        nonlocal used
        if used + record["cost"] > budget:
            return False
        kept.append(record)
        used += record["cost"]
        return True

    # 患者主记录无条件保留：缺它时任务没有对象，后续记录也失去归属。
    for record in records:
        if record["type"] in always_keep:
            kept.append(record)
            used += record["cost"]

    by_type: dict[str, list[dict]] = {}
    for record in records:
        by_type.setdefault(record["type"], []).append(record)
    remaining = [resource_type for resource_type in by_type if resource_type not in always_keep]

    # 一轮均分：总量不超过当前均分的类型整类保留，剩余预算留给更大的类型。
    while remaining:
        share = (budget - used) // len(remaining)
        accepted = [resource_type for resource_type in remaining
                    if sum(record["cost"] for record in by_type[resource_type]) <= share]
        if not accepted:
            break
        for resource_type in accepted:
            for record in by_type[resource_type]:
                keep(record)
            remaining.remove(resource_type)

    # 剩余类型按优先级保留轨迹中靠后的记录，直到预算用尽。
    for resource_type in sorted(remaining, key=_priority):
        for record in sorted(by_type[resource_type], key=lambda item: -item["order"]):
            keep(record)
    return kept


def _records_for_blocks(blocks: list[str], sources: list[Evidence]) -> list[dict]:
    return [{"index": index, "type": source.source_id.split("/", 1)[0],
             "cost": len(block) + len(source.statement) + SOURCE_OVERHEAD_CHARS, "order": index}
            for index, (block, source) in enumerate(zip(blocks, sources))]


def _fit_paragraphs(trajectory: PatientTrajectory, blocks: list[str],
                    sources: list[Evidence], budget: int) -> PatientTrajectory:
    records = _records_for_blocks(blocks, sources)
    selected = sorted(record["index"] for record in _select(records, budget, {"Patient"}))
    return PatientTrajectory(
        patient_id=trajectory.patient_id,
        environment=trajectory.environment,
        text="\n\n".join(blocks[index] for index in selected),
        sources=[sources[index] for index in selected],
    )


def _ids_in_line(line: str, known: set[str]) -> list[str]:
    found = []
    for token in line.split():
        for part in token.replace("=", " ").replace(",", " ").split():
            part = part.strip("[]();:")
            if part in known:
                found.append(part)
    return found


def _fit_lines(trajectory: PatientTrajectory, budget: int) -> PatientTrajectory:
    lines = trajectory.text.split("\n")
    sources = list(trajectory.sources)
    known = {source.source_id for source in sources}
    text_share = max(len(trajectory.text) // max(len(sources), 1), 0)
    # 行首记录是这行呈现的对象，行内的其它 id 只是引用；按行首记录决定去留，
    # 否则保留一条被引用的记录会连带保留大量引用它的行。
    primary = []
    for line in lines:
        ids = _ids_in_line(line, known)
        primary.append(ids[0] if ids else None)
    records = []
    for index, source in enumerate(sources):
        line_numbers = [number for number, owner in enumerate(primary)
                        if owner == source.source_id]
        if not line_numbers:
            continue
        records.append({"index": index, "type": source.source_id.split("/", 1)[0],
                        "cost": len(source.statement) + SOURCE_OVERHEAD_CHARS
                        + sum(len(lines[number]) for number in line_numbers),
                        "order": line_numbers[0], "lines": line_numbers})
    retained = sources
    text = trajectory.text
    candidate = trajectory
    target = int(budget * LINE_MODE_MARGIN)
    for _ in range(3):
        selected = _select(records, target, {"Patient"})
        kept_ids = {sources[record["index"]].source_id for record in selected}
        text = "\n".join(line for number, line in enumerate(lines)
                         if primary[number] is None or primary[number] in kept_ids)
        retained = [sources[record["index"]] for record in
                    sorted(selected, key=lambda item: item["index"])]
        candidate = PatientTrajectory(
            patient_id=trajectory.patient_id, environment=trajectory.environment,
            text=text, sources=retained,
        )
        if payload_chars(candidate) <= budget:
            return candidate
        target = max(int(target * budget / payload_chars(candidate) * 0.95), 1)
    return candidate


def fit_trajectory(trajectory: PatientTrajectory, budget_chars: int) -> PatientTrajectory:
    """裁剪到预算内；已满足预算时原样返回。"""
    if payload_chars(trajectory) <= budget_chars:
        return trajectory
    blocks = trajectory.text.split("\n\n")
    if len(blocks) == len(trajectory.sources):
        return _fit_paragraphs(trajectory, blocks, list(trajectory.sources), budget_chars)
    return _fit_lines(trajectory, budget_chars)


def generation_budget_chars(context_tokens: int, output_tokens: int,
                            instruction_tokens: int = 8192,
                            chars_per_token: float = 2.611) -> int:
    """生成输入预算：上下文减去输出与指令预留，再按字符换算。"""
    usable = max(context_tokens - output_tokens - instruction_tokens, 0)
    return int(usable * chars_per_token)
