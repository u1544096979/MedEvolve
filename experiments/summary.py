"""Read-only aggregation of experiment artifacts."""

import json
from collections import Counter, defaultdict
from pathlib import Path

from contracts import JsonObject


def _read(path, default):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def _ratio(numerator, denominator):
    return {"numerator": numerator, "denominator": denominator,
            "rate": numerator / denominator if denominator else None}


def summarize_run(output_dir: str) -> JsonObject:
    root = Path(output_dir)
    config = _read(root / "config.json", None)
    if config is None:
        raise ValueError("Run directory has no config.json")
    manifest = _read(root / "manifest.json", {"status": "unknown", "errors": []})

    def summarize_folders(folders, requested):
        tasks, evaluations, attempts = [], [], []
        for folder in folders:
            tasks.extend(_read(folder / "tasks.json", []))
            evaluations.extend(_read(folder / "evaluations.json", []))
            attempts.extend(_read(folder / "attempts.json", []))
        verification = [e for e in evaluations if e.get("verification") is not None]
        completion = [e for e in evaluations if e.get("completion") is not None]
        completed = sum(e["completion"]["completed"] for e in completion)
        quality = defaultdict(Counter)
        for evaluation in evaluations:
            for dimension, assessment in (evaluation.get("quality") or {}).get("dimensions", {}).items():
                if isinstance(assessment, dict) and "judgment" in assessment:
                    quality[dimension][str(assessment["judgment"])] += 1
        return {
            "requested_questions": requested,
            "generation_attempts": len(attempts),
            "actual_questions": len(tasks),
            "production": _ratio(len(tasks), requested),
            "evaluation_records": len(evaluations),
            "validity_among_verified": _ratio(sum(e["verification"]["valid"] for e in verification), len(verification)),
            "completion_among_generated": _ratio(completed, len(tasks)),
            "completion_among_reviewed": _ratio(completed, len(completion)),
            "completion_reviews_missing": len(tasks) - len(completion),
            "quality_judgments": dict(quality),
            "quality_reviews": sum(e.get("quality") is not None for e in evaluations),
            "evaluation_statuses": dict(Counter(e["status"] for e in evaluations)),
            "failure_categories": dict(Counter(e["failure_category"] for e in evaluations if e.get("failure_category"))),
        }

    records = _read(root / "runs.json", [])
    tokens = Counter()
    known = 0

    def collect_tokens(value, prefix=""):
        found = False
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else key
            if isinstance(item, dict):
                found = collect_tokens(item, path) or found
            elif "token" in key.lower() and isinstance(item, (int, float)) and not isinstance(item, bool):
                tokens[path] += item
                found = True
        return found

    for record in records:
        known += collect_tokens(record["result"].get("usage", {}))
    return {
        "run_id": config["run_id"], "method": config["method"],
        "status": manifest["status"], "errors": manifest.get("errors", []),
        "configured_budgets": {"method": config["runtime"]["budgets"],
                               "evaluation": config["evaluation"]["runtime"]["budgets"]},
        "evolution": summarize_folders(sorted((root / "rounds").glob("*")),
                                        2 * sum(len(p) for p in config["round_patients"])),
        "evaluation": summarize_folders([root / "evaluation"], 2 * len(config["evaluation_patients"])),
        "model_runs": {"count": len(records),
                       "statuses": dict(Counter(r["result"]["status"] for r in records)),
                       "known_token_totals": dict(tokens),
                       "runs_with_known_tokens": known,
                       "runs_without_known_tokens": len(records) - known},
    }
