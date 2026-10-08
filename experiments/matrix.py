"""Model-isolated experiment matrix orchestration and cost aggregation."""

from __future__ import annotations

import asyncio
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Awaitable, Callable

from pydantic import BaseModel, Field, model_validator

from contracts import ExperimentConfig, ModelConfig
from methods.common import normalize_usage
from .summary import summarize_difficulty, summarize_quality


SLUGS = ("physicianbench", "medagentbench", "fhir-agentbench")
METHODS = ("medevolve", "taskcraft", "workflow", "singleshot", "selfgen", "envexplore")
TOKEN_FIELDS = ("input_tokens", "output_tokens", "total_tokens")
EVALUATION_ROLES = ("verifier", "quality", "difficulty", "solver", "completion")


class MatrixConfig(BaseModel):
    """矩阵输入；所有模型路由均由调用方提供，不在此加载凭证。"""

    models: list[ModelConfig] = Field(min_length=1)
    judges: list[ModelConfig] = Field(min_length=2, max_length=3)
    slugs: list[str] = Field(min_length=1)
    methods: list[str] = Field(min_length=1)
    cohort_root: Path = Path("outputs/pool-screening")
    output_root: Path = Path("outputs/nuc-full-experiment")
    manager_url: str = "http://127.0.0.1:18084"
    template_dir: Path = Path("configs")
    seed_data_overrides: dict[str, str] = Field(default_factory=dict)
    model_concurrency: int = Field(default=2, ge=1, le=2)
    generation_concurrency: int = Field(default=12, gt=0)
    evaluation_concurrency: int = Field(default=12, gt=0)
    batch_size: int = Field(default=16, gt=0)

    @model_validator(mode="after")
    def validate_matrix(self) -> "MatrixConfig":
        if len(self.judges) not in {2, 3}:
            raise ValueError("judges must contain exactly two or three models")
        judge_signatures = [json.dumps(item.model_dump(mode="json"), sort_keys=True) for item in self.judges]
        if len(set(judge_signatures)) != len(judge_signatures):
            raise ValueError("judges must be unique model configurations")
        if len({item.model for item in self.judges}) != len(self.judges):
            raise ValueError("judge model names must be distinct")
        if len(set(self.slugs)) != len(self.slugs) or any(slug not in SLUGS for slug in self.slugs):
            raise ValueError(f"slugs must be unique members of {SLUGS}")
        if len(set(self.methods)) != len(self.methods) or any(method not in METHODS for method in self.methods):
            raise ValueError(f"methods must be unique members of {METHODS}")
        model_signatures = [json.dumps(item.model_dump(mode="json"), sort_keys=True) for item in self.models]
        if len(set(model_signatures)) != len(model_signatures):
            raise ValueError("models must be unique configurations")
        return self


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-._") or "model"


def model_key(model: ModelConfig) -> str:
    """组合模型名与非凭证路由字段，保证相同 model 名也能隔离。"""
    identity = model.model_dump(mode="json")
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:8]
    return f"{_safe_name(model.model)}-{digest}"


def apply_matrix_overrides(payload: dict[str, Any], model: ModelConfig,
                           judges: list[ModelConfig], output_dir: Path | str,
                           run_id: str, manager_url: str, slug: str,
                           seed_data: str | None, generation_concurrency: int,
                           evaluation_concurrency: int, *,
                           guidance_path: str | None = None) -> dict[str, Any]:
    """覆盖生成器、解题器、裁判和本机环境路由，并验证完整实验配置。"""
    updated = json.loads(json.dumps(payload))
    generator = model.model_dump(mode="json")
    runtime_models = updated["runtime"]["models"]
    updated["runtime"]["models"] = {role: dict(generator) for role in runtime_models}
    evaluation = updated["evaluation"]
    evaluation_models = evaluation["runtime"]["models"]
    judge = judges[0].model_dump(mode="json")
    evaluation["runtime"]["models"] = {
        role: (dict(generator) if role == "solver" else dict(judge))
        for role in evaluation_models
    }
    evaluation["judges"] = [item.model_dump(mode="json") for item in judges]
    updated["run_id"] = run_id
    updated["output_dir"] = str(output_dir)
    updated["environment"]["manager_url"] = manager_url
    if seed_data is not None:
        updated["environment"]["seed_data"] = seed_data
    updated["generation_concurrency"] = generation_concurrency
    updated["evaluation_concurrency"] = evaluation_concurrency
    if guidance_path is not None:
        updated["guidance_path"] = guidance_path
    elif updated.get("method") == "medevolve":
        updated.pop("guidance_path", None)
    ExperimentConfig.model_validate(updated)
    return updated


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _emit_configs(*, slug: str, method: str, output_root: Path,
                  configs_dir: Path, cohort_path: Path, template_dir: Path,
                  manager_url: str, batch_size: int,
                  strategy_path: Path | None = None,
                  evolution: bool = False) -> list[Path]:
    """通过现有配置生成器生成一组模板配置。"""
    configs_dir.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "scripts/emit_comparison_configs.py", "--slug", slug,
               "--method", method, "--cohort", str(cohort_path),
               "--configs-dir", str(configs_dir), "--template-dir", str(template_dir),
               "--output-root", str(output_root), "--batch-size", str(batch_size),
               "--manager-url", manager_url]
    if strategy_path is not None:
        command.extend(("--strategy", str(strategy_path)))
    pattern = f"full-{slug}-{method}-evolution.json" if evolution else f"full-{slug}-{method}-batch-*.json"
    for stale in configs_dir.glob(pattern):
        stale.unlink()
    subprocess.run(command, cwd=Path.cwd(), check=True, capture_output=True, text=True)
    return sorted(configs_dir.glob(pattern))


def _attempt_dirs(batch_root: Path) -> list[Path]:
    attempts = sorted(
        (path for path in batch_root.glob("attempt-*")
         if path.name.removeprefix("attempt-").isdigit()),
        key=lambda path: int(path.name.removeprefix("attempt-")),
    )
    if (batch_root / "manifest.json").exists():
        return [batch_root, *attempts]
    return attempts


def _manifest_status(path: Path) -> str | None:
    manifest_path = path / "manifest.json"
    if not manifest_path.exists():
        return None
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8")).get("status")
    except (OSError, json.JSONDecodeError):
        return None


def _read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def classify_run(output_dir: Path | str) -> dict[str, Any]:
    """按 manifest 与公共评测覆盖率判断运行状态。"""
    root = Path(output_dir)
    manifest = _read_json(root / "manifest.json", {"status": "unknown", "errors": []})
    tasks = _read_json(root / "evaluation" / "tasks.json", [])
    evaluations = _read_json(root / "evaluation" / "evaluations.json", [])
    by_task = {item.get("task_id"): item for item in evaluations if isinstance(item, dict)}
    missing_reviews: list[str] = []
    failed_reviews: list[str] = []
    for task in tasks:
        task_id = task.get("id")
        result = by_task.get(task_id)
        if result is None:
            missing_reviews.append(task_id)
            continue
        if result.get("status") == "failed":
            failed_reviews.append(task_id)
        verification = result.get("verification")
        if verification is None:
            missing_reviews.append(task_id)
        elif verification.get("valid") is True:
            if any(result.get(field) is None for field in ("quality", "difficulty", "completion")):
                missing_reviews.append(task_id)
    attempts = _read_json(root / "evaluation" / "attempts.json", [])
    failed_generation = [item for item in attempts
                        if item.get("produced", 0) < item.get("requested", 0)]
    review_errors = sorted(set(missing_reviews + failed_reviews))
    errors = list(manifest.get("errors", []))
    if manifest.get("status") == "failed":
        status = "failed"
    elif manifest.get("status") == "completed":
        status = "completed_with_errors" if errors or review_errors or failed_generation else "completed"
    else:
        status = "incomplete"
    return {
        "status": status,
        "manifest_status": manifest.get("status", "unknown"),
        "manifest_errors": errors,
        "missing_reviews": sorted(set(missing_reviews)),
        "failed_reviews": sorted(set(failed_reviews)),
        "failed_generation_attempts": failed_generation,
        "task_count": len(tasks),
        "valid_task_count": sum(
            1 for item in evaluations
            if isinstance(item.get("verification"), dict) and item["verification"].get("valid") is True
        ),
    }


def _find_model_identity(trace: list[dict[str, Any]]) -> dict[str, Any] | None:
    found = None
    for event in trace:
        identity = event.get("model_identity") if isinstance(event, dict) else None
        if isinstance(identity, dict):
            found = identity
    return found


def _find_provider_identity(trace: list[dict[str, Any]]) -> dict[str, Any] | None:
    found = None
    for event in trace:
        if not isinstance(event, dict):
            continue
        identity = event.get("provider_identity")
        if identity is None and event.get("type") == "provider_identity":
            identity = event
        if isinstance(identity, dict):
            found = identity
    return found


def _provider_attempts(trace: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [event for event in trace if isinstance(event, dict)
            and event.get("type") == "provider_attempt"
            and isinstance(event.get("result"), dict)]


def _leaf_cost_calls(result: dict[str, Any], *, inherited_provider: dict[str, Any] | None = None,
                     inherited_identity: dict[str, Any] | None = None,
                     inherited_stage: str | None = None
                     ) -> list[tuple[dict[str, Any], dict[str, Any], str, dict[str, Any], str]]:
    """展开聚合 trace，只返回实际模型调用，避免父级汇总用量重复计费。"""
    trace = result.get("trace", [])
    identity = _find_model_identity(trace) or inherited_identity or {}
    provider = _find_provider_identity(trace) or inherited_provider or {}
    stage = inherited_stage or "unknown"

    attempts = _provider_attempts(trace)
    if attempts:
        calls = []
        for attempt in attempts:
            attempt_result = attempt["result"]
            calls.extend(_leaf_cost_calls(
                attempt_result,
                inherited_provider=attempt.get("provider_identity", provider),
                inherited_identity=identity,
                inherited_stage=stage,
            ))
        return calls

    nested = [event for event in trace if isinstance(event, dict)
              and isinstance(event.get("stage"), str)
              and isinstance(event.get("result"), dict)
              and "usage" in event["result"]]
    if nested:
        calls = []
        for event in nested:
            calls.extend(_leaf_cost_calls(
                event["result"], inherited_provider=provider,
                inherited_identity=identity, inherited_stage=event["stage"],
            ))
        return calls

    return [(provider, result.get("usage", {}), result.get("status", "unknown"), identity, stage)]


def _role_for_stage(stage: str) -> str:
    if stage == "explore_data":
        return "data_explorer"
    if stage == "explore_environment" or stage.endswith("/explore_environment"):
        return "environment_explorer"
    if stage.endswith("/feedback") or stage == "feedback":
        return "feedback"
    if stage.endswith("/update") or stage == "update_strategy":
        return "updater"
    if "/generate/" in stage or stage.endswith("/generate"):
        return "generator"
    return "unknown"


def _phase_for(role: str, stage: str) -> str:
    if role in EVALUATION_ROLES:
        return role
    if role in {"data_explorer", "environment_explorer"}:
        return "exploration"
    if role == "generator":
        return "generation"
    if role == "feedback":
        return "feedback"
    if role == "updater":
        return "update"
    if "/evaluate/" in stage:
        return "evaluation_unknown"
    return "unknown"


def _model_from_config(config: dict[str, Any], role: str) -> dict[str, Any] | None:
    if role in EVALUATION_ROLES:
        return config.get("evaluation", {}).get("runtime", {}).get("models", {}).get(role)
    return config.get("runtime", {}).get("models", {}).get(role)


def _run_cost_rows(run_dir: Path) -> list[dict[str, Any]]:
    config = _read_json(run_dir / "config.json", {})
    records = _read_json(run_dir / "runs.json", [])
    rows = []
    for record_index, record in enumerate(records):
        result = record.get("result", {})
        stage = str(record.get("stage", "unknown"))
        leaves = _leaf_cost_calls(result, inherited_stage=stage)
        normalized_calls = [(provider, normalize_usage(usage), call_status, identity, call_stage)
                            for provider, usage, call_status, identity, call_stage in leaves]
        known_call = any(any(usage.get(field) is not None for field in TOKEN_FIELDS)
                         for _, usage, _, _, _ in normalized_calls)
        missing_call = any(
            int(usage.get("runs_without_known_tokens", 0)) > 0
            or not any(usage.get(field) is not None for field in TOKEN_FIELDS)
            for _, usage, _, _, _ in normalized_calls
        )
        merged_usage = normalize_usage(result.get("usage", {}))
        partial_record = known_call and (missing_call or
            int(merged_usage.get("runs_without_known_tokens", 0)) > 0)
        for call_index, (provider_identity, usage, call_status, call_identity, call_stage) in enumerate(normalized_calls):
            role = (call_identity.get("role") or _role_for_stage(call_stage)
                    or _role_for_stage(stage))
            if role == "unknown":
                role = _role_for_stage(stage)
            model_config = _model_from_config(config, role)
            actual_model = ((provider_identity or {}).get("model")
                            or call_identity.get("model")
                            or (model_config or {}).get("model"))
            tokens_known = any(usage.get(field) is not None for field in TOKEN_FIELDS)
            base_url = ((provider_identity or {}).get("base_url")
                        or (provider_identity or {}).get("endpoint")
                        or (provider_identity or {}).get("provider_endpoint")
                        or (provider_identity or {}).get("api_base")
                        or call_identity.get("base_url")
                        or (model_config or {}).get("base_url"))
            rows.append({
                "run_dir": str(run_dir),
                "environment": config.get("environment", {}).get("name", "unknown"),
                "method": config.get("method", "unknown"),
                "generator_model": config.get("runtime", {}).get("models", {}).get("generator", {}).get("model", "unknown"),
                "record_stage": stage,
                "call_stage": call_stage,
                "source_record_id": f"{run_dir}|{record_index}",
                "call_index": call_index,
                "phase": _phase_for(role, call_stage if call_stage != "unknown" else stage),
                "role": role,
                "model": actual_model or "unknown",
                "base_url": base_url,
                "provider": (provider_identity or {}).get("provider"),
                "status": call_status,
                "input_tokens": usage.get("input_tokens"),
                "output_tokens": usage.get("output_tokens"),
                "total_tokens": usage.get("total_tokens"),
                "tokens_known": tokens_known,
                "missing_usage_calls": max(0, int(usage.get("runs_without_known_tokens", 0)))
                or int(not tokens_known),
                "partial_source_record": partial_record,
            })
    return rows


def summarize_costs(run_dirs: list[Path | str]) -> dict[str, Any]:
    rows = [row for run_dir in run_dirs for row in _run_cost_rows(Path(run_dir))]
    totals: dict[str, int | None] = {}
    for field in TOKEN_FIELDS:
        known_values = [row[field] for row in rows if row[field] is not None]
        totals[field] = sum(known_values) if known_values else None
    grouped: dict[tuple[str, str, str, str, str, str | None, str | None], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["environment"], row["method"], row["generator_model"],
                 row["model"], row["phase"], row["base_url"], row["provider"])].append(row)
    by_group = []
    for key, entries in sorted(grouped.items()):
        group_tokens = {}
        for field in TOKEN_FIELDS:
            values = [entry[field] for entry in entries if entry[field] is not None]
            group_tokens[field] = sum(values) if values else None
        by_group.append({"environment": key[0], "method": key[1], "generator_model": key[2],
                         "model": key[3], "phase": key[4], "base_url": key[5],
                         "provider": key[6], "runs": len(entries),
                         "missing_usage_runs": sum(entry["missing_usage_calls"] for entry in entries),
                         **group_tokens})
    return {
        "records": rows,
        "totals": totals,
        "by_group": by_group,
        "known_usage_runs": sum(row["tokens_known"] for row in rows),
        "missing_usage_runs": sum(row["missing_usage_calls"] for row in rows),
        "partial_usage_runs": len({row["source_record_id"] for row in rows if row["partial_source_record"]}),
    }


def _write_cost_csv(path: Path, cost_summary: dict[str, Any]) -> None:
    fields = ("run_dir", "environment", "method", "generator_model", "record_stage", "call_stage", "phase",
              "source_record_id", "call_index", "role", "model", "base_url", "provider",
              "status", "input_tokens", "output_tokens", "total_tokens", "tokens_known",
              "missing_usage_calls", "partial_source_record")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(cost_summary["records"])


def _selected_counts(output_dir: Path) -> dict[str, int]:
    tasks = _read_json(output_dir / "evaluation" / "tasks.json", [])
    evaluations = _read_json(output_dir / "evaluation" / "evaluations.json", [])
    valid = sum(1 for item in evaluations
                if isinstance(item.get("verification"), dict)
                and item["verification"].get("valid") is True)
    return {"generated_questions": len(tasks), "valid_questions": valid}


async def run_matrix(
    config: MatrixConfig,
    *,
    emit_configs: Callable[..., list[Path]] | None = None,
    experiment_runner: Callable[[ExperimentConfig], Awaitable[None]] | None = None,
) -> dict[str, Any]:
    """运行模型隔离矩阵，优先波次完成后再启动后续模型。"""
    from experiments.runner import run_experiment

    emitter = emit_configs or _emit_configs
    runner = experiment_runner or run_experiment
    # 批内并发由已验证的 ExperimentConfig 显式写入；容器池限制由环境变量控制。
    os.environ["MEDEVOLVE_MAX_INSTANCES"] = "2"
    root = config.output_root
    root.mkdir(parents=True, exist_ok=True)
    status_path = root / "matrix-status.json"
    state: dict[str, Any] = {
        "status": "running", "jobs": [], "errors": [],
        "priority_models": [model.model for model in config.models[:2]],
    }
    _write_json(status_path, state)
    all_run_dirs: list[Path] = []

    def checkpoint() -> None:
        _write_json(status_path, state)

    async def run_one(model: ModelConfig, slug: str, method: str,
                      *, evolution: bool = False,
                      strategy_path: Path | None = None) -> tuple[dict[str, Any], Path | None]:
        key = model_key(model)
        model_output = root / key
        method_root = model_output / slug / method
        job_id = f"{key}/{slug}/{method}/{'evolution' if evolution else 'batches'}"
        job: dict[str, Any] = {
            "job_id": job_id, "model": model.model, "model_key": key,
            "slug": slug, "method": method,
            "kind": "evolution" if evolution else "batch_group",
            "status": "running", "batches": [], "attempts": [],
        }
        state["jobs"].append(job)
        checkpoint()
        config_dir = root / ".matrix-configs" / key / slug / method / ("evolution" if evolution else "batches")
        cohort_path = config.cohort_root / slug / "cohort-full.json"
        try:
            emitted = emitter(
                slug=slug, method=method, output_root=method_root,
                configs_dir=config_dir, cohort_path=cohort_path,
                template_dir=config.template_dir, manager_url=config.manager_url,
                batch_size=config.batch_size, strategy_path=strategy_path,
                evolution=evolution,
            )
            if not emitted:
                raise RuntimeError(f"configuration emitter produced no configs for {job_id}")
            selected_strategy: Path | None = None
            for emitted_path in emitted:
                payload = _read_json(Path(emitted_path), None)
                if payload is None:
                    raise ValueError(f"invalid emitted config: {emitted_path}")
                base_path = Path(payload["output_dir"])
                batch_name = base_path.name
                attempts = _attempt_dirs(base_path)
                all_run_dirs.extend(path for path in attempts if path not in all_run_dirs)
                completed = next((path for path in reversed(attempts)
                                  if _manifest_status(path) == "completed"), None)
                if completed is not None:
                    assessment = classify_run(completed)
                    item = {"batch": batch_name, "status": assessment["status"],
                            "selected_output_dir": str(completed), "skipped": True,
                            "missing_reviews": assessment["missing_reviews"],
                            "failed_reviews": assessment["failed_reviews"],
                            "review_errors": assessment["missing_reviews"] + assessment["failed_reviews"]}
                    job["batches"].append(item)
                    if evolution:
                        selected_strategy = completed / "final_strategy.json"
                    checkpoint()
                    continue
                next_attempt = max((int(path.name.rsplit("-", 1)[-1]) for path in attempts
                                    if path.name.startswith("attempt-") and path.name.rsplit("-", 1)[-1].isdigit()),
                                   default=0) + 1
                output_path = base_path / f"attempt-{next_attempt:02d}"
                run_id = f"{_safe_name(payload.get('run_id', batch_name))}-{key}-{next_attempt:02d}"
                guidance = payload.get("guidance_path")
                actual_payload = apply_matrix_overrides(
                    payload, model, config.judges, output_path, run_id,
                    config.manager_url, slug, config.seed_data_overrides.get(slug),
                    config.generation_concurrency, config.evaluation_concurrency,
                    guidance_path=guidance,
                )
                experiment = ExperimentConfig.model_validate(actual_payload)
                attempt_entry = {"batch": batch_name, "attempt": next_attempt,
                                 "output_dir": str(output_path), "status": "running"}
                job["attempts"].append(attempt_entry)
                checkpoint()
                try:
                    await runner(experiment)
                    assessment = classify_run(output_path)
                    attempt_entry.update(assessment)
                    attempt_entry["status"] = assessment["status"]
                except Exception as exc:
                    attempt_entry.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
                    state["errors"].append({"job_id": job_id, "batch": batch_name,
                                            "output_dir": str(output_path), "error": attempt_entry["error"]})
                all_run_dirs.append(output_path)
                batch_entry = {"batch": batch_name, "status": attempt_entry["status"],
                               "selected_output_dir": str(output_path)
                               if attempt_entry["status"].startswith("completed") else None,
                               "skipped": False,
                               "missing_reviews": attempt_entry.get("missing_reviews", []),
                               "failed_reviews": attempt_entry.get("failed_reviews", []),
                               "review_errors": attempt_entry.get("missing_reviews", [])
                               + attempt_entry.get("failed_reviews", [])}
                job["batches"].append(batch_entry)
                if batch_entry["selected_output_dir"]:
                    if evolution:
                        selected_strategy = output_path / "final_strategy.json"
                checkpoint()
            batch_statuses = [item["status"] for item in job["batches"]]
            if any(item == "failed" for item in batch_statuses):
                job["status"] = "failed"
            elif any(item not in {"completed", "completed_with_errors"} for item in batch_statuses):
                job["status"] = "incomplete"
            elif any(item == "completed_with_errors" for item in batch_statuses):
                job["status"] = "completed_with_errors"
            else:
                job["status"] = "completed"
            if evolution and selected_strategy is not None and not selected_strategy.exists():
                job["status"] = "failed"
                job["error"] = f"frozen strategy missing: {selected_strategy}"
                selected_strategy = None
            checkpoint()
            return job, selected_strategy
        except Exception as exc:
            job["status"] = "failed"
            job["error"] = f"{type(exc).__name__}: {exc}"
            state["errors"].append({"job_id": job_id, "error": job["error"]})
            checkpoint()
            return job, None

    async def run_model(model: ModelConfig) -> None:
        for slug in config.slugs:
            for method in config.methods:
                strategy = None
                if method == "medevolve":
                    _, strategy = await run_one(model, slug, method, evolution=True)
                    if strategy is None:
                        continue
                await run_one(model, slug, method, strategy_path=strategy)

    model_slots = config.model_concurrency
    for start in range(0, len(config.models), model_slots):
        wave = config.models[start:start + model_slots]
        await asyncio.gather(*(run_model(model) for model in wave))

    # 成本覆盖所有尝试；最终题数只取每个批次明确选择的完成运行。
    unique_run_dirs = list(dict.fromkeys(all_run_dirs))
    cost_summary = summarize_costs(unique_run_dirs)
    selected_totals: dict[tuple[str, str, str], dict[str, int]] = defaultdict(
        lambda: {"generated_questions": 0, "valid_questions": 0}
    )
    selected_evaluations: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for job in state["jobs"]:
        for batch in job.get("batches", []):
            selected = batch.get("selected_output_dir")
            if not selected:
                continue
            path = Path(selected)
            key = (job["model"], job["slug"], job["method"])
            counts = _selected_counts(path)
            for field in ("generated_questions", "valid_questions"):
                count = counts[field]
                selected_totals[key][field] += count
            selected_evaluations[key].extend(
                _read_json(path / "evaluation" / "evaluations.json", [])
            )
    state["selected_totals"] = [
        {
            "model": key[0], "slug": key[1], "method": key[2], **counts,
            "quality_summary": summarize_quality(selected_evaluations[key]),
            "difficulty_summary": summarize_difficulty(selected_evaluations[key]),
            "verification": {
                "reviews": sum(item.get("verification") is not None for item in selected_evaluations[key]),
                "valid": sum(
                    isinstance(item.get("verification"), dict)
                    and item["verification"].get("valid") is True for item in selected_evaluations[key]
                ),
                "invalid": sum(
                    isinstance(item.get("verification"), dict)
                    and item["verification"].get("valid") is False for item in selected_evaluations[key]
                ),
            },
            "completion": {
                "reviews": sum(item.get("completion") is not None for item in selected_evaluations[key]),
                "completed": sum(
                    isinstance(item.get("completion"), dict)
                    and item["completion"].get("completed") is True for item in selected_evaluations[key]
                ),
                "missing": max(0, counts["generated_questions"] - sum(
                    item.get("completion") is not None for item in selected_evaluations[key]
                )),
            },
        }
        for key, counts in sorted(selected_totals.items())
    ]
    environment_names = {
        job["slug"]: next((
            _read_json(Path(batch["selected_output_dir"]) / "config.json", {})
            .get("environment", {}).get("name")
            for batch in job.get("batches", []) if batch.get("selected_output_dir")
        ), job["slug"])
        for job in state["jobs"]
    }
    efficiency = []
    for item in state["selected_totals"]:
        slug = item["slug"]
        environment = environment_names.get(slug, slug)
        relevant = [row for row in cost_summary["records"]
                    if row["environment"] == environment and row["method"] == item["method"]
                    and row["generator_model"] == item["model"]
                    and row["phase"] in {"generation", "exploration", "feedback", "update"}]
        missing_generation_usage = sum(row["missing_usage_calls"] for row in relevant)
        partial_generation_usage = any(row["partial_source_record"] for row in relevant)
        known = (not missing_generation_usage and not partial_generation_usage
                 and all(row["tokens_known"] for row in relevant))
        token_values = [row["total_tokens"] for row in relevant if row["total_tokens"] is not None]
        generation_tokens = sum(token_values) if token_values else None
        generated = item["generated_questions"]
        valid = item["valid_questions"]
        efficiency.append({
            "model": item["model"], "slug": slug, "environment": environment,
            "method": item["method"], "generation_tokens": generation_tokens,
            "missing_generation_usage_runs": missing_generation_usage,
            "partial_generation_usage_records": len({
                row["source_record_id"] for row in relevant if row["partial_source_record"]
            }),
            "generated_questions": generated, "valid_questions": valid,
            "tokens_per_generated_question": (
                generation_tokens / generated if generation_tokens is not None and known and generated else None
            ),
            "tokens_per_valid_question": (
                generation_tokens / valid if generation_tokens is not None and known and valid else None
            ),
        })
    state["generation_efficiency"] = efficiency
    state["status"] = (
        "completed_with_errors" if state["errors"] or any(
            job["status"] in {"failed", "completed_with_errors", "incomplete"} for job in state["jobs"]
        ) else "completed"
    )
    state["cost_summary"] = {key: value for key, value in cost_summary.items() if key != "records"}
    state["run_directories"] = [str(path) for path in unique_run_dirs]
    _write_json(root / "cost-summary.json", cost_summary)
    _write_cost_csv(root / "cost-summary.csv", cost_summary)
    _write_json(root / "matrix-summary.json", {
        "status": state["status"], "jobs": state["jobs"],
        "errors": state["errors"], "selected_totals": state["selected_totals"],
        "generation_efficiency": efficiency,
        "cost_summary": {key: value for key, value in cost_summary.items() if key != "records"},
    })
    checkpoint()
    return state
