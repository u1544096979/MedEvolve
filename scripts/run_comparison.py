"""顺序执行全量对比运行配置：跳过已完成运行，逐个记录日志并核对容器回收。

    python scripts/run_comparison.py --configs 'configs/full-physicianbench-selfgen-*.json'

每个配置对应一次 experiments.run 子进程。运行前确认没有遗留的受管容器，运行后再次
核对，避免容器累积占满本机内存。
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def managed_containers() -> list[str]:
    result = subprocess.run(
        ["docker", "ps", "-aq", "--filter", "label=medevolve.managed=true"],
        capture_output=True, text=True, check=True,
    )
    return [line for line in result.stdout.split() if line]


def run_status(config_path: Path) -> str | None:
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    manifest = Path(payload["output_dir"]) / "manifest.json"
    if not manifest.exists():
        return None
    return json.loads(manifest.read_text(encoding="utf-8")).get("status")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", required=True, help="配置路径或 glob 模式")
    parser.add_argument("--log-dir", type=Path, default=Path("outputs/full-comparison-logs"))
    parser.add_argument("--force", action="store_true", help="重跑已完成的配置")
    args = parser.parse_args()

    configs = sorted(Path().glob(args.configs) if any(c in args.configs for c in "*?[")
                     else [Path(args.configs)])
    if not configs:
        raise SystemExit(f"no config matches {args.configs}")
    args.log_dir.mkdir(parents=True, exist_ok=True)

    for config_path in configs:
        status = run_status(config_path)
        if status == "completed" and not args.force:
            print(f"skip {config_path.name}: completed", flush=True)
            continue
        stray = managed_containers()
        if stray:
            print(f"abort: {len(stray)} managed containers still running", flush=True)
            return 2
        log_path = args.log_dir / f"{config_path.stem}.log"
        print(f"run {config_path.name} -> {log_path}", flush=True)
        with log_path.open("ab") as log:
            code = subprocess.run(
                [sys.executable, "-m", "experiments.run", "--config", str(config_path)],
                stdout=log, stderr=subprocess.STDOUT,
            ).returncode
        print(f"exit {code} {config_path.name}", flush=True)
        if code != 0:
            return code
        stray = managed_containers()
        if stray:
            print(f"abort: {len(stray)} managed containers left after run", flush=True)
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
