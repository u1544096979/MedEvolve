"""按环境顺序执行全量对比：进化 → 冻结策略 → 各方法批次。

    python scripts/run_full_comparison.py --slug physicianbench

环境之间串行、环境内方法与批次串行，单批失败只影响该批，重跑会跳过已完成运行。
并发与容器预算按环境设置，避免多个环境同时占用本机内存。
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from contracts import FrozenEnvironmentGuidance

BASELINES = ("singleshot", "selfgen", "workflow", "taskcraft", "envexplore")
# 每个环境的并发与容器预算：PhysicianBench 解题要独立实例，其余环境整轮共用一个实例。
SETTINGS = {
    "physicianbench": {"MEDEVOLVE_RUN_CONCURRENCY": "6", "MEDEVOLVE_MAX_INSTANCES": "4"},
    "medagentbench": {"MEDEVOLVE_RUN_CONCURRENCY": "6", "MEDEVOLVE_MAX_INSTANCES": "2"},
    "fhir-agentbench": {"MEDEVOLVE_RUN_CONCURRENCY": "6", "MEDEVOLVE_MAX_INSTANCES": "1"},
}
COHORTS = {
    "physicianbench": Path("outputs/pool-screening/physicianbench/cohort-full.json"),
    "medagentbench": Path("outputs/pool-screening/medagentbench/cohort-full.json"),
    "fhir-agentbench": Path("outputs/pool-screening/fhir-agentbench/cohort-full.json"),
}


MEMORY_LIMIT_GB = 6.0


def managed_memory_gb() -> float:
    """受管容器当前占用之和；多个环境并行时用它替代容器数量检查。"""
    listed = subprocess.run(
        ["docker", "ps", "-q", "--filter", "label=medevolve.managed=true"],
        capture_output=True, text=True, check=True,
    ).stdout.split()
    if not listed:
        return 0.0
    result = subprocess.run(
        ["docker", "stats", "--no-stream", "--format", "{{.MemUsage}}", *listed],
        capture_output=True, text=True, check=True,
    )
    total = 0.0
    for line in result.stdout.splitlines():
        used = line.split("/")[0].strip()
        for unit, factor in (("GiB", 1.0), ("MiB", 1 / 1024), ("KiB", 1 / 1048576)):
            if used.endswith(unit):
                total += float(used[: -len(unit)]) * factor
                break
    return total


def memory_guard() -> int:
    used = managed_memory_gb()
    if used > MEMORY_LIMIT_GB:
        print(f"[abort] managed containers already use {used:.1f} GiB", flush=True)
        return 2
    return 0


def status_of(config_path: Path) -> str | None:
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    manifest = Path(payload["output_dir"]) / "manifest.json"
    if not manifest.exists():
        return None
    data = json.loads(manifest.read_text(encoding="utf-8"))
    status = data.get("status")
    if status != "completed":
        return status
    if "generation" in data and "evaluation" in data:
        return status
    # 老 manifest 缺少阶段账本时，检查现存结果，避免重复信任历史误报。
    folders = [Path(payload["output_dir"]) / "evaluation",
               Path(payload["output_dir"]) / "initial_evaluation",
               *sorted((Path(payload["output_dir"]) / "rounds").glob("*"))]
    for folder in folders:
        attempts_path = folder / "attempts.json"
        evaluations_path = folder / "evaluations.json"
        tasks_path = folder / "tasks.json"
        attempts = json.loads(attempts_path.read_text(encoding="utf-8")) if attempts_path.exists() else []
        evaluations = json.loads(evaluations_path.read_text(encoding="utf-8")) if evaluations_path.exists() else []
        tasks = json.loads(tasks_path.read_text(encoding="utf-8")) if tasks_path.exists() else []
        if any(a.get("status") == "failed" or
               (a.get("status") == "budget_exhausted" and not a.get("produced")) for a in attempts):
            return "failed"
        if any(e.get("status") == "failed" for e in evaluations):
            return "failed"
        if len(evaluations) < len(tasks):
            return "failed"
    return status


def run(config_path: Path, settings: dict, log_dir: Path) -> int:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{config_path.stem}.log"
    print(f"[run] {config_path.name} -> {log_path}", flush=True)
    environment = {**os.environ, **settings}
    with log_path.open("ab") as log:
        code = subprocess.run(
            [sys.executable, "-m", "experiments.run", "--config", str(config_path)],
            stdout=log, stderr=subprocess.STDOUT, env=environment,
        ).returncode
    print(f"[exit {code}] {config_path.name}", flush=True)
    return code


def emit(slug: str, method: str, strategy: Path | None = None) -> None:
    command = [sys.executable, "scripts/emit_comparison_configs.py", "--slug", slug,
               "--method", method, "--cohort", str(COHORTS[slug])]
    if strategy is not None:
        command += ["--strategy", str(strategy)]
    subprocess.run(command, check=True)


def run_glob(pattern: str, settings: dict, log_dir: Path) -> int:
    configs = sorted(Path().glob(pattern))
    if not configs:
        print(f"[warn] no config matches {pattern}", flush=True)
        return 0
    for config_path in configs:
        if status_of(config_path) == "completed":
            print(f"[skip] {config_path.name} completed", flush=True)
            continue
        guard = memory_guard()
        if guard:
            return guard
        code = run(config_path, settings, log_dir)
        if code != 0:
            return code
        guard = memory_guard()
        if guard:
            return guard
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--slug", required=True, choices=sorted(SETTINGS))
    parser.add_argument("--only", help="只跑指定方法（含 medevolve）")
    parser.add_argument("--skip-evolution", action="store_true")
    parser.add_argument("--log-dir", type=Path, default=Path("outputs/full-comparison-logs"))
    args = parser.parse_args()
    slug, settings = args.slug, SETTINGS[args.slug]
    methods = [args.only] if args.only else list(BASELINES) + ["medevolve"]

    evolution = Path(f"configs/full-{slug}-medevolve-evolution.json")
    strategy = Path(f"outputs/full-comparison/{slug}/medevolve/evolution/final_strategy.json")
    if "medevolve" in methods and not args.skip_evolution:
        if status_of(evolution) != "completed":
            code = run(evolution, settings, args.log_dir)
            if code != 0:
                return code
        if not strategy.exists():
            print(f"[abort] frozen strategy is missing: {strategy}", flush=True)
            return 2

    for method in methods:
        if method == "medevolve":
            emit(slug, method, strategy)
        else:
            emit(slug, method)
        if method == "envexplore" and slug in {"physicianbench", "medagentbench"}:
            exploration = Path(f"configs/full-{slug}-envexplore-exploration.json")
            exploration_data = json.loads(exploration.read_text(encoding="utf-8"))
            frozen_path = Path(exploration_data["output_dir"]) / "environment_guidance.json"
            test_configs = sorted(Path().glob(f"configs/full-{slug}-envexplore-batch-*.json"))
            all_test_patients = set()
            for config_path in test_configs:
                payload = json.loads(config_path.read_text(encoding="utf-8"))
                all_test_patients.update(payload["evaluation_patients"])
            overlap = set(exploration_data["evolution_patients"]) & all_test_patients
            if overlap:
                print(f"[abort] EnvExplore patients overlap: {sorted(overlap)}", flush=True)
                return 2
            if status_of(exploration) != "completed":
                code = run(exploration, settings, args.log_dir)
                if code != 0:
                    return code
            if not frozen_path.exists():
                print(f"[abort] frozen EnvExplore guidance is missing: {frozen_path}", flush=True)
                return 2
            try:
                frozen = FrozenEnvironmentGuidance.model_validate_json(
                    frozen_path.read_text(encoding="utf-8")
                )
            except Exception as exc:
                print(f"[abort] frozen EnvExplore guidance is invalid: {exc}", flush=True)
                return 2
            if (frozen.environment != exploration_data["environment"]["name"] or
                    set(frozen.exploration_patients) != set(exploration_data["evolution_patients"]) or
                    frozen.exploration_run.status != "completed" or
                    not frozen.guidance.guidance.strip() or
                    set(frozen.exploration_patients) & all_test_patients):
                print("[abort] frozen EnvExplore guidance failed validation", flush=True)
                return 2
        code = run_glob(f"configs/full-{slug}-{method}-batch-*.json", settings, args.log_dir)
        if code != 0:
            return code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
