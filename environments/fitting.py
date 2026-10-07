"""按记录优先级裁剪超长患者输入，使生成输入落在模型上下文预算内。

生成模型上下文有限，而原始轨迹可能包含上万条记录。裁剪保留患者记录、诊断与用药等
关键类型，再按时间保留其余记录；正文段落与 sources 一一对应，方法按段落取来源的
假设不受影响。评测始终使用未裁剪的原始轨迹，核验读取的也是原始来源记录。
"""

from contracts import PatientTrajectory

# 关键临床类型优先保留，未列出的类型排在其后。
TYPE_PRIORITY = (
    "Patient", "Condition", "AllergyIntolerance", "MedicationRequest",
    "MedicationStatement", "MedicationAdministration", "Procedure", "ServiceRequest",
    "DiagnosticReport", "Encounter", "CarePlan", "Immunization", "Observation",
    "DocumentReference", "Communication", "Appointment",
)
# 每条来源在 JSON 输入中的固定开销：source_id、字段名与转义余量。
SOURCE_OVERHEAD_CHARS = 64


def _priority(resource_type: str) -> tuple[int, str]:
    return (TYPE_PRIORITY.index(resource_type) if resource_type in TYPE_PRIORITY
            else len(TYPE_PRIORITY), resource_type)


def _record_cost(block: str, statement: str) -> int:
    return len(block) + len(statement) + SOURCE_OVERHEAD_CHARS


def payload_chars(trajectory: PatientTrajectory) -> int:
    return sum(_record_cost(block, source.statement)
               for block, source in zip(trajectory.text.split("\n\n"), trajectory.sources))


def fit_trajectory(trajectory: PatientTrajectory, budget_chars: int) -> PatientTrajectory:
    """裁剪到预算内；已满足预算时原样返回。

    正文段落与 sources 数量不一致时说明轨迹不是按记录逐段生成的，此时不做裁剪，
    由调用方决定如何处理。
    """
    blocks = trajectory.text.split("\n\n")
    sources = list(trajectory.sources)
    if len(blocks) != len(sources) or payload_chars(trajectory) <= budget_chars:
        return trajectory

    records = []
    for index, (block, source) in enumerate(zip(blocks, sources)):
        resource_type = source.source_id.split("/", 1)[0]
        records.append({"index": index, "type": resource_type, "cost": _record_cost(block, source.statement)})

    kept: set[int] = set()
    used = 0

    def keep(record: dict) -> bool:
        nonlocal used
        if used + record["cost"] > budget_chars:
            return False
        kept.add(record["index"])
        used += record["cost"]
        return True

    # 患者主记录先占位，缺它时后续任务无法确定对象。
    for record in records:
        if record["type"] == "Patient":
            keep(record)

    by_type: dict[str, list[dict]] = {}
    for record in records:
        by_type.setdefault(record["type"], []).append(record)
    remaining = [resource_type for resource_type in by_type if resource_type != "Patient"]

    # 一轮均分：总量不超过当前均分的类型整类保留，剩余预算留给更大的类型。
    while remaining:
        share = (budget_chars - used) // len(remaining)
        accepted = [resource_type for resource_type in remaining
                    if sum(record["cost"] for record in by_type[resource_type]) <= share]
        if not accepted:
            break
        for resource_type in accepted:
            for record in by_type[resource_type]:
                keep(record)
            remaining.remove(resource_type)

    # 剩余类型按优先级保留最近的记录，直到预算用尽。
    for resource_type in sorted(remaining, key=_priority):
        for record in reversed(by_type[resource_type]):
            keep(record)

    selected = sorted(kept)
    return PatientTrajectory(
        patient_id=trajectory.patient_id,
        environment=trajectory.environment,
        text="\n\n".join(blocks[index] for index in selected),
        sources=[sources[index] for index in selected],
    )


def generation_budget_chars(context_tokens: int, output_tokens: int,
                            instruction_tokens: int = 8192,
                            chars_per_token: float = 2.611) -> int:
    """生成输入预算：上下文减去输出与指令预留，再按字符换算。"""
    usable = max(context_tokens - output_tokens - instruction_tokens, 0)
    return int(usable * chars_per_token)
