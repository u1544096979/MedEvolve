"""Read-only aggregation of experiment artifacts."""

import json
import statistics
from collections import Counter
from pathlib import Path

from contracts import FrozenEnvironmentGuidance, JsonObject
from methods.common import normalize_usage, USAGE_FIELDS


def _read(path, default):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def _ratio(numerator, denominator):
    return {"numerator": numerator, "denominator": denominator,
            "rate": numerator / denominator if denominator else None}


QUALITY_DIMENSIONS = (
    "factual_accuracy", "clinical_soundness", "evidence_sufficiency", "task_clarity",
)


def _quality_scores(evaluation: dict) -> list[int] | None:
    review = evaluation.get("quality")
    if not isinstance(review, dict):
        return None
    scores = []
    for dimension in QUALITY_DIMENSIONS:
        assessment = review.get(dimension)
        if not isinstance(assessment, dict) or not isinstance(assessment.get("score"), int):
            return None
        scores.append(assessment["score"])
    return scores


def _difficulty_score(evaluation: dict) -> dict | None:
    score = evaluation.get("difficulty")
    if not isinstance(score, dict) or not isinstance(score.get("reasoning_depth"), int):
        return None
    metrics = score.get("complexity_metrics")
    if not isinstance(metrics, dict):
        return None
    if not all(isinstance(metrics.get(field), int)
               for field in ("evidence_count", "resource_type_count")):
        return None
    return {"reasoning_depth": score["reasoning_depth"],
            "evidence_count": metrics["evidence_count"],
            "resource_type_count": metrics["resource_type_count"]}


def summarize_quality(evaluations: list[dict]) -> JsonObject:
    # Legacy records keep the retired `dimensions` shape and cannot be aggregated.
    score_lists = [scores for evaluation in evaluations
                   if (scores := _quality_scores(evaluation)) is not None]
    if not score_lists:
        return {}
    summary = {}
    for index, dimension in enumerate(QUALITY_DIMENSIONS):
        scores = [score_list[index] for score_list in score_lists]
        summary[dimension] = {
            "mean": float(statistics.mean(scores)),
            "median": float(statistics.median(scores)),
            "distribution": {str(score): count for score, count in sorted(Counter(scores).items())},
            "high_quality_rate": sum(score >= 4 for score in scores) / len(scores),
        }
    return summary


def summarize_difficulty(evaluations: list[dict]) -> JsonObject:
    entries = [entry for evaluation in evaluations
               if (entry := _difficulty_score(evaluation)) is not None]
    if not entries:
        return {}
    depths = [entry["reasoning_depth"] for entry in entries]
    evidence_counts = [entry["evidence_count"] for entry in entries]
    resource_counts = [entry["resource_type_count"] for entry in entries]
    return {
        "reasoning_depth": {
            "mean": float(statistics.mean(depths)),
            "distribution": {str(depth): count for depth, count in sorted(Counter(depths).items())},
        },
        "complexity_metrics": {
            "avg_evidence_count": float(statistics.mean(evidence_counts)),
            "avg_resource_types": float(statistics.mean(resource_counts)),
        },
    }


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
        produced = sum(int(a.get("produced", 0)) for a in attempts)
        generation_requested = sum(int(a.get("requested", 0)) for a in attempts)
        generation_failed = sum(a.get("status") == "failed" for a in attempts)
        evaluation_failed = sum(e.get("status") == "failed" for e in evaluations)
        quality = summarize_quality(evaluations)
        difficulty = summarize_difficulty(evaluations)
        return {
            "requested_questions": requested,
            "generation_attempts": len(attempts),
            "generation_statuses": dict(Counter(a.get("status", "unknown") for a in attempts)),
            "short_batches": sum(a["produced"] < a["requested"] for a in attempts),
            "generation_errors": [a for a in attempts if a.get("error")],
            "actual_questions": len(tasks),
            "production": _ratio(len(tasks), requested),
            "generation_yield": _ratio(produced, generation_requested),
            "generation_failure_rate": _ratio(generation_failed, len(attempts)),
            "evaluation_failure_rate": _ratio(evaluation_failed, len(tasks)),
            "evaluation_records": len(evaluations),
            "validity_among_verified": _ratio(sum(e["verification"]["valid"] for e in verification), len(verification)),
            "completion_among_generated": _ratio(completed, len(tasks)),
            "completion_among_reviewed": _ratio(completed, len(completion)),
            "completion_reviews_missing": len(tasks) - len(completion),
            "quality_summary": quality,
            "difficulty_summary": difficulty,
            "quality_reviews": sum(e.get("quality") is not None for e in evaluations),
            "evaluation_statuses": dict(Counter(e["status"] for e in evaluations)),
            "failure_categories": dict(Counter(e["failure_category"] for e in evaluations if e.get("failure_category"))),
        }

    records = _read(root / "runs.json", [])
    tokens = Counter()
    known = 0
    costs = {}
    partially_known = 0
    unknown_token_runs = 0
    for record in records:
        usage = normalize_usage(record["result"].get("usage", {}))
        has_tokens = any(usage[k] is not None for k in ("input_tokens", "output_tokens", "total_tokens"))
        known += has_tokens
        missing = usage["runs_without_known_tokens"]
        unknown_token_runs += missing
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
        frozen_path = config.get("guidance_path")
        if frozen_path and Path(frozen_path).exists():
            frozen = FrozenEnvironmentGuidance.model_validate_json(
                Path(frozen_path).read_text(encoding="utf-8")
            )
            costs["frozen_exploration_reference"] = {
                **normalize_usage(frozen.exploration_run.usage),
                "status": frozen.exploration_run.status,
                "environment": frozen.environment,
                "exploration_patients": frozen.exploration_patients,
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
                       "runs_without_known_tokens": unknown_token_runs},
    }
