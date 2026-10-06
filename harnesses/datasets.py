"""Freeze verified generated tasks into one task set shared by all harnesses.

Only tasks that were verified by the generation experiment, that belong to a
single environment and that reference an existing patient enter the frozen set.
Sampling uses the sorted task IDs and a fixed seed, so every harness reads the
same manifest instead of drawing its own tasks.
"""

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from contracts import EvaluationResult, Task


MANIFEST_NAME = "manifest.json"
TASKS_NAME = "tasks.json"
EVALUATIONS_NAME = "evaluations.json"
PATIENTS_NAME = "patients.json"
# 已观测的生成成本形状：workflow 运行写 costs.generation，taskcraft/envexplore 写
# generation_usage，PhysicianBench 批量生成运行写顶层 usage。
GENERATION_COST_KEYS = ("costs.generation", "generation_usage", "usage")
MANIFEST_VERSION = 1


class DatasetFreezeError(RuntimeError):
    """Raised when the configured task sources cannot be frozen as requested."""


@dataclass
class _Source:
    name: str
    tasks_path: Path
    error: str | None = None
    status: str = "ok"
    environment: str | None = None
    tasks: list[Task] = field(default_factory=list)
    valid: list[Task] = field(default_factory=list)
    verified_valid: int = 0
    unverified: int = 0
    patient_missing: int = 0
    tasks_sha256: str = ""
    evaluations_path: Path | None = None
    evaluations_sha256: str = ""
    patients_path: Path | None = None
    patients_sha256: str | None = None
    patient_roster: str = "absent"
    patient_roster_reason: str = ""
    generation_cost: dict | None = None
    generation_cost_source: str | None = None
    generation_cost_reason: str = ""


def freeze_datasets(
    inputs: dict[str, Path], *, count: int, seed: int, destination: Path
) -> dict[str, list[Task]]:
    """Sample ``count`` verified tasks per source and write the shared manifest.

    ``inputs`` maps a task source name to its ``tasks.json``; the sibling
    ``evaluations.json`` and optional ``patients.json`` are read from the same
    directory. All sources must belong to one environment. A source that is
    missing files, unverified, mixed across environments or short of verified
    tasks raises ``DatasetFreezeError`` listing every source, and the configured
    count is never changed silently. Sampling orders the verified task IDs by a
    digest of the fixed seed and the full task ID, so the order is reproducible
    independently of the interpreter's random number generator.
    """
    if count <= 0:
        raise DatasetFreezeError(f"任务量必须为正整数：count={count}")
    if not inputs:
        raise DatasetFreezeError("任务量无法确定：没有配置任何任务来源")

    reports = [_read_source(name, path) for name, path in sorted(inputs.items())]
    if any(report.error for report in reports):
        raise DatasetFreezeError(
            "任务来源不可用，无法冻结任务集：\n" + _describe(reports)
            + "\n不自动出题或复制其他来源补齐。")

    environments = sorted({report.environment for report in reports})
    if len(environments) > 1:
        listing = "\n".join(f"  {report.name}: {report.environment}" for report in reports)
        raise DatasetFreezeError(
            "环境不一致：一次冻结只能包含一个环境，实际为 "
            + "、".join(environments) + "：\n" + listing)

    if any(len(report.valid) < count for report in reports):
        raise DatasetFreezeError(
            f"有效量不足：请求 {count} 条，各源产量：\n" + _describe(reports)
            + "\n按配置数量冻结，不静默改变已配置数量。")

    frozen: dict[str, list[Task]] = {}
    details: dict[str, dict] = {}
    for report in reports:
        by_id = {task.id: task for task in report.tasks}
        sampling_order = _sampling_order(report.valid, seed)
        sampled_ids = sampling_order[:count]
        frozen[report.name] = [by_id[task_id] for task_id in sampled_ids]
        details[report.name] = {
            "tasks_file": str(report.tasks_path),
            "tasks_sha256": report.tasks_sha256,
            "evaluations_file": str(report.evaluations_path),
            "evaluations_sha256": report.evaluations_sha256,
            "patients_file": str(report.patients_path) if report.patients_path else None,
            "patients_sha256": report.patients_sha256,
            "patient_roster": report.patient_roster,
            "patient_roster_reason": report.patient_roster_reason,
            "generation_cost": report.generation_cost,
            "generation_cost_source": report.generation_cost_source,
            "generation_cost_reason": report.generation_cost_reason,
            "task_ids": sorted(by_id),
            "sampling_order": sampling_order,
            "sampled_task_ids": sampled_ids,
            "counts": {
                "total": len(report.tasks),
                "verified_valid": report.verified_valid,
                "valid": len(report.valid),
                "invalid": len(report.tasks) - len(report.valid),
                "unverified": report.unverified,
                "patient_missing": report.patient_missing,
            },
        }

    manifest = {
        "version": MANIFEST_VERSION,
        "seed": seed,
        "count": count,
        "environment": environments[0],
        "source_methods": [report.name for report in reports],
        "sources": details,
    }
    destination.mkdir(parents=True, exist_ok=True)
    _write_json(destination / TASKS_NAME,
                {name: [task.model_dump(mode="json") for task in tasks]
                 for name, tasks in frozen.items()})
    _write_json(destination / MANIFEST_NAME, manifest)
    return frozen


def load_frozen(destination: Path) -> dict[str, list[Task]]:
    """Read the frozen task set back, checking it against the shared manifest."""
    destination = Path(destination)
    manifest = json.loads((destination / MANIFEST_NAME).read_text())
    payload = json.loads((destination / TASKS_NAME).read_text())
    frozen: dict[str, list[Task]] = {}
    for name in manifest["source_methods"]:
        if name not in manifest["sources"] or name not in payload:
            raise DatasetFreezeError(f"冻结任务集与 manifest 不一致：缺少来源 {name}")
        tasks = [Task.model_validate(item) for item in payload[name]]
        if [task.id for task in tasks] != manifest["sources"][name]["sampled_task_ids"]:
            raise DatasetFreezeError(f"冻结任务顺序与 manifest 不一致：{name}")
        frozen[name] = tasks
    return frozen


def preflight_datasets(
    environments: dict[str, dict], *, count: int, seed: int
) -> dict[str, dict]:
    """Report every configured source without freezing or writing any file.

    ``environments`` is the config ``environments`` mapping. Sources configured
    with no task file are reported as ``missing`` with their recorded reason;
    sources that exist are read through the same checks as ``freeze_datasets``.
    ``freezable`` is false while any source is missing, invalid or short of the
    configured task count.
    """
    report: dict[str, dict] = {}
    for environment, settings in environments.items():
        entries: dict[str, dict] = {}
        blockers: list[str] = []
        if not settings.get("sources"):
            blockers.append("未配置任务来源")
        for method, source in sorted((settings.get("sources") or {}).items()):
            path = source.get("tasks")
            if path is None:
                entries[method] = {"status": "missing", "reason": source.get("missing_reason")
                                   or "该环境尚未配置任务文件"}
            else:
                state = _read_source(method, Path(path))
                entry = {"status": state.status, "reason": state.error or "",
                         "tasks_file": str(state.tasks_path)}
                if state.status == "ok":
                    entry.update(total=len(state.tasks), valid=len(state.valid),
                                 verified_valid=state.verified_valid)
                entries[method] = entry
            if entries[method]["status"] != "ok":
                blockers.append(f"{method}: {entries[method]['reason']}")
            elif entries[method]["valid"] < count:
                entries[method]["reason"] = f"有效 {entries[method]['valid']}，少于配置的 {count}"
                blockers.append(f"{method}: {entries[method]['reason']}")
        report[environment] = {"environment": environment, "count": count, "seed": seed,
                               "sources": entries, "freezable": not blockers,
                               "blockers": blockers}
    return report


def _read_source(name: str, path: Path) -> _Source:
    tasks_path = Path(path)
    report = _Source(name=name, tasks_path=tasks_path)
    if not tasks_path.is_file():
        report.status = "missing"
        report.error = f"来源文件缺失：{tasks_path} 不存在"
        return report
    report.tasks_sha256 = _sha256(tasks_path)

    evaluations_path = tasks_path.with_name(EVALUATIONS_NAME)
    report.evaluations_path = evaluations_path
    if not evaluations_path.is_file():
        report.status = "missing"
        report.error = f"缺少 {EVALUATIONS_NAME}，没有核验结果"
        return report
    report.evaluations_sha256 = _sha256(evaluations_path)

    try:
        tasks = [Task.model_validate(item) for item in _load(tasks_path)]
        evaluations = [EvaluationResult.model_validate(item) for item in _load(evaluations_path)]
    except (ValueError, TypeError) as error:
        report.status = "invalid"
        report.error = f"文件解析失败：{error}"
        return report
    if not tasks:
        report.status = "invalid"
        report.error = "任务文件为空"
        return report
    if len({task.id for task in tasks}) != len(tasks):
        report.status = "invalid"
        report.error = "任务 ID 重复"
        return report
    if len({evaluation.task_id for evaluation in evaluations}) != len(evaluations):
        report.status = "invalid"
        report.error = f"{EVALUATIONS_NAME} 的 task_id 重复"
        return report

    environments = sorted({task.environment for task in tasks})
    if len(environments) > 1:
        report.status = "invalid"
        report.error = "环境不一致：同一来源包含多个环境 " + "、".join(environments)
        return report
    report.environment = environments[0]

    patients: list[str] | None = None
    roster_path = tasks_path.with_name(PATIENTS_NAME)
    if roster_path.is_file():
        try:
            roster = json.loads(roster_path.read_text())
            roster_environment = roster["environment"]
            patients = list(roster["patients"])
        except (ValueError, TypeError, KeyError) as error:
            report.status = "invalid"
            report.error = f"患者名单文件不可用：{roster_path}（{error}）"
            return report
        if roster_environment != report.environment:
            report.status = "invalid"
            report.error = (f"环境不一致：任务文件为 {report.environment}，"
                            f"患者名单为 {roster_environment}")
            return report
        report.patients_path = roster_path
        report.patients_sha256 = _sha256(roster_path)
        report.patient_roster = "checked"
        report.patient_roster_reason = f"按 {PATIENTS_NAME} 名单校验 patient_id 存在于名单"
    else:
        report.patient_roster_reason = (f"来源目录无 {PATIENTS_NAME}，"
                                        "患者存在只校验 patient_id 非空")

    report.tasks = tasks
    evaluated = {evaluation.task_id: evaluation for evaluation in evaluations}
    for task in tasks:
        evaluation = evaluated.get(task.id)
        if evaluation is None or evaluation.verification is None:
            report.unverified += 1
            continue
        if not evaluation.verification.valid:
            continue
        report.verified_valid += 1
        if not task.patient_id or (patients is not None and task.patient_id not in patients):
            report.patient_missing += 1
            continue
        report.valid.append(task)
    if report.unverified == len(tasks):
        report.status = "invalid"
        report.error = f"{EVALUATIONS_NAME} 中所有任务都没有核验结果"
        return report

    (report.generation_cost_source, report.generation_cost,
     report.generation_cost_reason) = _generation_cost(tasks_path)
    return report


def _generation_cost(tasks_path: Path) -> tuple[str | None, dict | None, str]:
    """Read the generation cost recorded next to the tasks, or explain its absence.

    Cost lives in a sibling ``summary.json`` (PhysicianBench bulk generation) or in
    the parent directory's one (workflow runs, whose tasks sit in ``evaluation/``).
    All three observed key shapes are tried; a source without a readable summary or
    without any of those fields keeps ``null`` cost and the reason why.
    """
    candidates = (tasks_path.parent / "summary.json",
                  tasks_path.parent.parent / "summary.json")
    observed = [candidate for candidate in candidates if candidate.is_file()]
    if not observed:
        return None, None, ("未找到 summary.json（已查找 "
                            + "、".join(str(candidate) for candidate in candidates) + "）")
    notes: list[str] = []
    for candidate in observed:
        try:
            summary = json.loads(candidate.read_text())
        except (ValueError, TypeError) as error:
            notes.append(f"{candidate} 读取失败：{error}")
            continue
        for key in GENERATION_COST_KEYS:
            cost = _dig(summary, key)
            if isinstance(cost, dict) and cost:
                return str(candidate), cost, f"取自 {candidate} 的 {key}"
        notes.append(f"{candidate} 没有生成成本字段（已查 "
                     + "、".join(GENERATION_COST_KEYS) + "）")
    return None, None, "；".join(notes)


def _dig(payload: dict, dotted: str) -> object:
    value: object = payload
    for part in dotted.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def _sampling_order(tasks: list[Task], seed: int) -> list[str]:
    return sorted((task.id for task in tasks),
                  key=lambda task_id: hashlib.sha256(f"{seed}:{task_id}".encode()).hexdigest())


def _describe(reports: list[_Source]) -> str:
    lines = []
    for report in reports:
        if report.error:
            lines.append(f"  {report.name}: 不可用（{report.error}）")
        else:
            lines.append(f"  {report.name}: 有效 {len(report.valid)}/{len(report.tasks)}"
                         f"（核验通过 {report.verified_valid}）")
    return "\n".join(lines)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> list:
    payload = json.loads(path.read_text())
    if not isinstance(payload, list):
        raise ValueError(f"{path} 不是 JSON 数组")
    return payload


def _write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
