"""Read-only aggregation of experiment artifacts."""

import json
from collections import Counter, defaultdict
from pathlib import Path

from contracts import JsonObject
from methods.common import normalize_usage, USAGE_FIELDS


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
            "generation_statuses": dict(Counter(a.get("status", "unknown") for a in attempts)),
            "short_batches": sum(a["produced"] < a["requested"] for a in attempts),
            "generation_errors": [a for a in attempts if a.get("error")],
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
    costs = {}
    partially_known = 0
    for record in records:
        usage = normalize_usage(record["result"].get("usage", {}))
        has_tokens = any(usage[k] is not None for k in ("input_tokens", "output_tokens", "total_tokens"))
        known += has_tokens
        missing = usage.get("runs_without_known_tokens", int(not has_tokens))
        partially_known += bool(has_tokens and missing)
        for key in ("input_tokens", "output_tokens", "total_tokens"):
            if usage[key] is not None:
                tokens[key] += usage[key]
        stage = record["stage"]
        category = ("evaluation" if "/evaluate/" in stage else
                    "generation" if "/generate/" in stage else
                    "exploration" if stage.startswith("explore_") else "strategy_update")
        aggregate = costs.setdefault(category, {**dict.fromkeys(USAGE_FIELDS, 0),
                                                "runs": 0, "runs_without_known_tokens": 0})
        aggregate["runs"] += 1
        aggregate["runs_without_known_tokens"] += missing
        for key in USAGE_FIELDS:
            aggregate[key] += usage[key] or 0
    if config["method"] == "envexplore":
        fields = (*USAGE_FIELDS, "runs", "runs_without_known_tokens")
        costs["method"] = {
            field: sum(costs.get(category, {}).get(field, 0)
                       for category in ("exploration", "generation"))
            for field in fields
        }
    return {
        "run_id": config["run_id"], "method": config["method"],
        "status": manifest["status"], "errors": manifest.get("errors", []),
        "configured_budgets": {"method": config["runtime"]["budgets"],
                               "evaluation": config["evaluation"]["runtime"]["budgets"]},
        "costs": costs,
        "evolution": summarize_folders(sorted((root / "rounds").glob("*")),
                                        2 * sum(len(p) for p in config["round_patients"]) if config["method"] == "medevolve" else 0),
        "evaluation": summarize_folders([root / "evaluation"], 2 * len(config["evaluation_patients"])),
        **({"initial_evaluation": summarize_folders([root / "initial_evaluation"],
                                                   2 * len(config["evaluation_patients"]))}
           if config.get("compare_initial_strategy") else {}),
        "model_runs": {"count": len(records),
                       "statuses": dict(Counter(r["result"]["status"] for r in records)),
                       "known_token_totals": dict(tokens),
                       "runs_with_known_tokens": known,
                       "runs_with_partial_tokens": partially_known,
                       "runs_without_known_tokens": len(records) - known},
    }
