"""由筛查名单生成主方法全池运行配置：每环境 1 份进化配置与若干批配置。

    python scripts/emit_pool_configs.py \
        --template configs/main-pool-physicianbench-template.json \
        --cohort outputs/pool-screening/physicianbench/cohort.json \
        --slug physicianbench
"""

import argparse
import json
from pathlib import Path

from contracts import ExperimentConfig


def _save(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--slug", required=True)
    parser.add_argument("--configs-dir", type=Path, default=Path("configs"))
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--batches", type=int)
    args = parser.parse_args()

    template = json.loads(args.template.read_text(encoding="utf-8"))
    cohort = json.loads(args.cohort.read_text(encoding="utf-8"))
    output_root = args.output_root or Path(f"outputs/main-pool-{args.slug}")
    evolution_patients = list(cohort["evolution_patients"])
    batches = [list(batch) for batch in cohort["batches"]]
    if args.batches is not None:
        batches = batches[: args.batches]
    if any(len(batch) > args.batch_size for batch in batches):
        raise ValueError("batch exceeds batch-size")

    evolution = {
        **template,
        "run_id": f"main-pool-{args.slug}-evolution",
        "evolution_patients": evolution_patients,
        "evaluation_patients": [],
        "round_patients": [[patient] for patient in evolution_patients],
        "compare_initial_strategy": False,
        "output_dir": str(output_root / "evolution"),
    }
    ExperimentConfig.model_validate(evolution)
    evolution_path = args.configs_dir / f"main-pool-{args.slug}-evolution.json"
    _save(evolution_path, evolution)
    print(f"wrote {evolution_path} ({len(evolution_patients)} evolution patients)")

    guidance_path = str(output_root / "evolution" / "guidance.json")
    for index, batch in enumerate(batches, start=1):
        payload = {
            **template,
            "run_id": f"main-pool-{args.slug}-batch-{index:02d}",
            "evolution_patients": [],
            "evaluation_patients": batch,
            "round_patients": [],
            "compare_initial_strategy": False,
            "guidance_path": guidance_path,
            "output_dir": str(output_root / f"batch-{index:02d}"),
        }
        ExperimentConfig.model_validate(payload)
        path = args.configs_dir / f"main-pool-{args.slug}-batch-{index:02d}.json"
        _save(path, payload)
    print(f"wrote {len(batches)} batch configs, {sum(len(b) for b in batches)} patients")


if __name__ == "__main__":
    main()
