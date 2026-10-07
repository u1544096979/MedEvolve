"""汇总主方法全池运行的 summary.json：输出 Markdown 表与合并后的 JSON。

    python scripts/summarize_pool.py --root outputs --slug physicianbench --slug medagentbench
"""

import argparse
import json
from pathlib import Path

METRICS = ("requested_questions", "actual_questions", "generation_attempts", "short_batches",
           "evaluation_records", "quality_reviews", "completion_reviews_missing")
COST_FIELDS = ("model_turns", "tool_calls", "input_tokens", "output_tokens", "total_tokens")


def _read(path: Path, default):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def _ratio(record: dict, field: str):
    value = record.get(field) or {}
    return value.get("rate")


def _fold(records: list[dict]) -> dict:
    totals = {field: sum(record["evaluation"].get(field, 0) for record in records) for field in METRICS}
    costs = {field: sum(record["costs"].get(category, {}).get(field, 0)
                        for category in ("exploration", "generation", "evaluation", "strategy_update")
                        for record in records)
             for field in COST_FIELDS}
    valid = sum((record["evaluation"].get("validity_among_verified") or {}).get("numerator", 0)
                for record in records)
    verified = sum((record["evaluation"].get("validity_among_verified") or {}).get("denominator", 0)
                   for record in records)
    completed = sum((record["evaluation"].get("completion_among_generated") or {}).get("numerator", 0)
                    for record in records)
    quality = {}
    for dimension in ("factual_accuracy", "clinical_soundness", "evidence_sufficiency", "task_clarity"):
        means = [record["evaluation"]["quality_summary"][dimension]["mean"]
                 for record in records
                 if record["evaluation"].get("quality_summary", {}).get(dimension)]
        quality[dimension] = round(sum(means) / len(means), 2) if means else None
    depths = [record["evaluation"]["difficulty_summary"]["reasoning_depth"]["mean"]
              for record in records
              if record["evaluation"].get("difficulty_summary", {}).get("reasoning_depth")]
    failures = {}
    for record in records:
        for category, count in record["evaluation"].get("failure_categories", {}).items():
            failures[category] = failures.get(category, 0) + count
    return {
        **totals,
        "statuses": {record["run_id"]: record["status"] for record in records},
        "verification_pass_rate": round(valid / verified, 3) if verified else None,
        "completion_rate": round(completed / totals["actual_questions"], 3) if totals["actual_questions"] else None,
        "quality_means": quality,
        "difficulty_depth_mean": round(sum(depths) / len(depths), 2) if depths else None,
        "failure_categories": failures,
        "costs": costs,
    }


def _costs(records: list[dict]) -> dict:
    return {field: sum(record["costs"].get(category, {}).get(field, 0)
                       for category in ("exploration", "generation", "evaluation", "strategy_update")
                       for record in records)
            for field in COST_FIELDS}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("outputs"))
    parser.add_argument("--slug", action="append", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = {}
    for slug in args.slug:
        environment_root = args.root / f"main-pool-{slug}"
        records = [{"run_id": path.parent.name, **_read(path, {})}
                   for path in sorted(environment_root.glob("*/summary.json"))]
        records = [record for record in records if record.get("run_id") != "run_id"]
        result[slug] = {"runs": records, "combined": _fold(records)} if records else {}

    output = args.output or args.root / "main-pool-summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = ["| 环境 | 运行数 | 生成题数 | 核验通过率 | 解题完成率 | 质量（事实/临床/证据/清晰） | 难度深度 | model_turns | tool_calls | 总 token |",
             "| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |"]
    for slug, entry in result.items():
        if not entry:
            lines.append(f"| {slug} | 0 | 0 | - | - | - | - | 0 | 0 | 0 |")
            continue
        combined = entry["combined"]
        quality = combined["quality_means"]
        quality_text = "/".join(str(quality[key]) for key in
                                ("factual_accuracy", "clinical_soundness",
                                 "evidence_sufficiency", "task_clarity"))
        lines.append(
            f"| {slug} | {len(entry['runs'])} | {combined['actual_questions']} | "
            f"{combined['verification_pass_rate']} | {combined['completion_rate']} | {quality_text} | "
            f"{combined['difficulty_depth_mean']} | {combined['costs']['model_turns']} | "
            f"{combined['costs']['tool_calls']} | {combined['costs']['total_tokens']} |"
        )
    table = "\n".join(lines)
    markdown = output.parent / "main-pool-summary.md"
    markdown.write_text(table + "\n", encoding="utf-8")
    print(table)
    print(f"\nwritten {output} and {markdown}")


if __name__ == "__main__":
    main()
