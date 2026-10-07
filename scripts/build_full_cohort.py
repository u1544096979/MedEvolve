"""把筛查出的患者规模表整理成全量运行名单：全部患者进入评测批次。

    python scripts/build_full_cohort.py --sizes outputs/pool-screening/physicianbench/pool_sizes.json \
        --environment PhysicianBench --output outputs/pool-screening/physicianbench/cohort-full.json

超出生成上下文预算的患者不剔除，由 experiments/runner.py 在准备生成输入时按记录
优先级裁剪；这里只记录哪些患者会被裁剪，便于报告。
"""

import argparse
import json
import random
from pathlib import Path

BATCH_SIZE = 16
CONTEXT_TOKENS = 128000 - 16384 - 8192


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=Path, required=True)
    parser.add_argument("--environment", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--evolution", nargs="*", help="固定进化患者，默认按种子抽取 2 名")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    rows = json.loads(args.sizes.read_text(encoding="utf-8"))
    by_id = {row["patient_id"]: row for row in rows}
    identifiers = sorted(by_id)
    if args.evolution:
        evolution = list(args.evolution)
        missing = [patient for patient in evolution if patient not in by_id]
        if missing:
            raise ValueError(f"evolution patients are not in the pool: {missing}")
        remaining = [patient for patient in identifiers if patient not in set(evolution)]
    else:
        shuffled = list(identifiers)
        random.Random(args.seed).shuffle(shuffled)
        evolution, remaining = shuffled[:2], shuffled[2:]
    batches = [remaining[index:index + args.batch_size]
               for index in range(0, len(remaining), args.batch_size)]
    cohort = {
        "environment": args.environment,
        "seed": args.seed,
        "patient_pool": len(identifiers),
        "evolution_patients": evolution,
        "batches": batches,
        "fitted": [{"patient_id": patient, "tokens": by_id[patient]["tokens"]}
                   for patient in identifiers if by_id[patient]["tokens"] > CONTEXT_TOKENS],
        "skipped": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(cohort, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{args.environment}: pool={len(identifiers)} evolution={len(evolution)} "
          f"batches={len(batches)} fitted={len(cohort['fitted'])} -> {args.output}")


if __name__ == "__main__":
    main()
