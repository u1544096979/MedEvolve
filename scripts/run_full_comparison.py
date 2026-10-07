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

BASELINES = ("singleshot", "selfgen", "workflow", "taskcraft", "envexplore")
# 每个环境的并发与容器预算：PhysicianBench 解题要独立实例，其余环境整轮共用一个实例。
SETTINGS = {
    "physicianbench": {"MEDEVOLVE_RUN_CONCURRENCY": "4", "MEDEVOLVE_MAX_INSTANCES": "3"},
    "medagentbench": {"MEDEVOLVE_RUN_CONCURRENCY": "6", "MEDEVOLVE_MAX_INSTANCES": "2"},
    "fhir-agentbench": {"MEDEVOLVE_RUN_CONCURRENCY": "6", "MEDEVOLVE_MAX_INSTANCES": "1"},
}
COHORTS = {
    "physicianbench": Path("outputs/pool-screening/physicianbench/cohort-full.json"),
    "medagentbench": Path("outputs/pool-screening/medagentbench/cohort-full.json"),
    "fhir-agentbench": Path("outputs/pool-screening/fhir-agentbench/cohort-full.json"),
}


def managed_containers() -> list[str]:
    result = subprocess.run(
        ["docker", "ps", "-aq", "--filter", "label=medevolve.managed=true"],
        capture_output=True, text=True, check=True,
    )
    return [line for line in result.stdout.split() if line]


def status_of(config_path: Path) -> str | None:
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    manifest = Path(payload["output_dir"]) / "manifest.json"
    if not manifest.exists():
        return None
    return json.loads(manifest.read_text(encoding="utf-8")).get("status")


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
        stray = managed_containers()
        if stray:
            print(f"[abort] {len(stray)} managed containers still running", flush=True)
            return 2
        code = run(config_path, settings, log_dir)
        if code != 0:
            return code
        stray = managed_containers()
        if stray:
            print(f"[abort] {len(stray)} managed containers left after run", flush=True)
            return 2
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
        code = run_glob(f"configs/full-{slug}-{method}-batch-*.json", settings, args.log_dir)
        if code != 0:
            return code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
