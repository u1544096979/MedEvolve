"""Score a fixed cross-run sample with alternative review inputs."""

import argparse
import asyncio
import json
from pathlib import Path

from contracts import (
    EvaluationConfig, ExperimentConfig, ModelConfig, PatientTrajectory, Task, Verification,
)
from evaluation.evaluator import ScoringError, review_scores


MODES = (("separate", "full"), ("joint", "full"), ("joint", "compact"))


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _source_runs(paths: list[Path]) -> list[dict]:
    candidates = []
    environments = set()
    for root in paths:
        config = ExperimentConfig.model_validate(_read(root / "config.json"))
        tasks = [Task.model_validate(item) for item in _read(root / "evaluation/tasks.json")]
        evaluations = _read(root / "evaluation/evaluations.json")
        by_id = {item["task_id"]: item for item in evaluations}
        trajectories = {}
        for trajectory_path in (root / "trajectories/evaluation").glob("*.json"):
            data = _read(trajectory_path)
            trajectories[data["patient_id"]] = (trajectory_path, data)
        environments.add(config.environment.name)
        for task in tasks:
            result = by_id.get(task.id)
            if not result or not result.get("verification", {}).get("valid"):
                continue
            trajectory_item = trajectories.get(task.patient_id)
            if trajectory_item is None:
                continue
            trajectory_path, trajectory_data = trajectory_item
            try:
                trajectory = PatientTrajectory.model_validate(trajectory_data)
                verification = Verification.model_validate(result["verification"])
            except Exception:
                continue
            candidates.append({
                "environment": task.environment,
                "task": task,
                "trajectory": trajectory,
                "verification": verification,
                "config": config.evaluation,
                "task_path": str(root / "evaluation/tasks.json"),
                "evaluation_path": str(root / "evaluation/evaluations.json"),
                "trajectory_path": str(trajectory_path),
                "run_dir": str(root),
            })
    if len(environments) != 2:
        raise ValueError(f"expected exactly two environments, found {sorted(environments)}")
    chosen = []
    for environment in sorted(environments):
        env_cases = sorted((item for item in candidates
                            if item["environment"] == environment),
                           key=lambda item: item["task"].id)
        if len(env_cases) < 3:
            raise ValueError(f"{environment} has {len(env_cases)} eligible verified tasks; need 3")
        chosen.extend(env_cases[:3])
    return chosen


def _override(config: EvaluationConfig, scoring_mode: str,
              model_name: str, base_url: str, api_key_env: str) -> EvaluationConfig:
    raw = config.model_dump(mode="json")
    model = ModelConfig(model=model_name, base_url=base_url,
                        api_key_env=api_key_env).model_dump(mode="json")
    raw["judges"] = []
    raw["scoring_mode"] = scoring_mode
    raw["review_input_mode"] = "full"
    raw["runtime"]["models"]["quality"] = model
    raw["runtime"]["models"]["difficulty"] = model
    return EvaluationConfig.model_validate(raw)


def _usage(usage: dict) -> dict:
    usage = usage or {}
    input_tokens = next((usage[k] for k in ("input_tokens", "prompt_tokens")
                         if isinstance(usage.get(k), int)), None)
    output_tokens = next((usage[k] for k in ("output_tokens", "completion_tokens")
                          if isinstance(usage.get(k), int)), None)
    return {"input_tokens": input_tokens, "output_tokens": output_tokens,
            "total_tokens": (input_tokens + output_tokens
                             if input_tokens is not None and output_tokens is not None
                             else usage.get("total_tokens"))}


def _request_count(run: dict) -> int | None:
    """返回有轨迹或显式 model_turns 证明的请求数；缺少两者时保留未知。"""
    traced = sum(item.get("type") == "model_request" for item in run.get("trace", []))
    if traced:
        return traced
    turns = (run.get("usage") or {}).get("model_turns")
    return turns if isinstance(turns, int) and turns >= 0 else None


def _run_statistics(runs: list[dict]) -> dict:
    requests = [(_request_count(run), _usage(run.get("usage", {}))) for run in runs]

    def total(key):
        values = [usage[key] for _, usage in requests if usage[key] is not None]
        return sum(values) if values else None

    failed = [run for run in runs if run["status"] != "completed"]
    return {
        "attempt_count": len(runs),
        "request_count": sum(count for count, _ in requests if count is not None),
        "unknown_request_attempts": sum(count is None for count, _ in requests),
        "failed_attempts": len(failed),
        "failed_requests": sum(count for run in failed
                                if (count := _request_count(run)) is not None),
        "usage_totals": {
            "input_tokens": total("input_tokens"),
            "output_tokens": total("output_tokens"),
            "total_tokens": total("total_tokens"),
            "unknown_requests": sum(
                count for count, usage in requests
                if count is not None and count > 0 and usage["total_tokens"] is None
            ),
            "cost_usd": None,
        },
    }


async def _execute(cases: list[dict], args) -> tuple[list[dict], list[dict]]:
    semaphore = asyncio.Semaphore(args.concurrency)
    records = []

    async def one(case_index: int, case: dict, scoring_mode: str, input_mode: str):
        config = _override(case["config"], scoring_mode, args.model,
                           args.base_url, args.api_key_env).model_copy(
                               update={"review_input_mode": input_mode})
        record = {"case_id": case["task"].id, "scoring_mode": scoring_mode,
                  "review_input_mode": input_mode, "status": "completed",
                  "runs": [], "scores": None}
        try:
            quality, difficulty, runs = await review_scores(
                case["task"], case["trajectory"], case["verification"], config,
                request_semaphore=semaphore,
            )
            record["runs"] = [run.model_dump(mode="json") for run in runs]
            record["scores"] = {"quality": quality.model_dump(mode="json"),
                                "difficulty": difficulty.model_dump(mode="json")}
        except ScoringError as exc:
            record.update(status="failed", error=str(exc),
                          runs=[run.model_dump(mode="json") for run in exc.runs],
                          partial_scores={
                              "quality": exc.quality.model_dump(mode="json") if exc.quality else None,
                              "difficulty": exc.difficulty.model_dump(mode="json") if exc.difficulty else None,
                          })
        counts = [_request_count(run) for run in record["runs"]]
        record["request_count"] = sum(count for count in counts if count is not None)
        record["unknown_request_attempts"] = sum(count is None for count in counts)
        return case_index, record

    jobs = [one(index, case, scoring_mode, input_mode)
            for index, case in enumerate(cases)
            for scoring_mode, input_mode in MODES]
    completed = await asyncio.gather(*jobs)
    completed.sort(key=lambda item: (item[0], MODES.index(
        (item[1]["scoring_mode"], item[1]["review_input_mode"]))))
    records = [record for _, record in completed]
    requests = []
    for record in records:
        for run in record["runs"]:
            identity = next((item.get("model_identity") for item in run.get("trace", [])
                             if item.get("type") == "model_identity"), {})
            requests.append({"case_id": record["case_id"],
                             "scoring_mode": record["scoring_mode"],
                             "review_input_mode": record["review_input_mode"],
                             "status": run["status"],
                             "request_count": _request_count(run),
                             "model_identity": identity, "run": run})
    return records, requests


def _comparison(records: list[dict]) -> dict:
    by_case = {}
    for record in records:
        by_case.setdefault(record["case_id"], {})[
            f"{record['scoring_mode']}/{record['review_input_mode']}"] = record
    comparisons = []
    dimensions = ("factual_accuracy", "clinical_soundness", "evidence_sufficiency",
                  "task_clarity", "reasoning_depth")
    for case_id, variants in sorted(by_case.items()):
        compact_variants = {
            name: {**{key: value for key, value in record.items()
                      if key not in {"runs", "request_count", "unknown_request_attempts"}},
                   "statistics": _run_statistics(record["runs"])}
            for name, record in variants.items()
        }
        base = variants.get("joint/full")
        deltas = {}
        if base and base.get("scores"):
            for variant, record in variants.items():
                if variant == "joint/full" or not record.get("scores"):
                    continue
                left, right = record["scores"], base["scores"]
                deltas[variant] = {}
                for dimension in dimensions:
                    section = "difficulty" if dimension == "reasoning_depth" else "quality"
                    a = left[section].get(dimension, {}).get("score") if section == "quality" else left[section].get(dimension)
                    b = right[section].get(dimension, {}).get("score") if section == "quality" else right[section].get(dimension)
                    if a is not None and b is not None:
                        deltas[variant][dimension] = a - b
        comparisons.append({"case_id": case_id, "variants": compact_variants,
                            "deltas_vs_joint_full": deltas})
    raw_runs = [run for item in records for run in item["runs"]]
    return {"cases": comparisons, **_run_statistics(raw_runs)}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", action="append", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--base-url", default="https://api.openlux.ai/v1")
    parser.add_argument("--api-key-env", default="OPENLUX_API_KEY")
    parser.add_argument("--concurrency", type=int, default=24)
    args = parser.parse_args(argv)
    if args.concurrency < 1:
        parser.error("--concurrency must be positive")
    cases = _source_runs(args.run_dir)
    records, requests = asyncio.run(_execute(cases, args))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    case_data = [{"case_id": case["task"].id, "environment": case["environment"],
                  "run_dir": case["run_dir"], "task_path": case["task_path"],
                  "evaluation_path": case["evaluation_path"],
                  "trajectory_path": case["trajectory_path"],
                  "task": case["task"].model_dump(mode="json"),
                  "verification": case["verification"].model_dump(mode="json")}
                 for case in cases]
    (args.output_dir / "cases.json").write_text(
        json.dumps(case_data, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output_dir / "runs.json").write_text(
        json.dumps(requests, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output_dir / "comparison.json").write_text(
        json.dumps(_comparison(records), ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
