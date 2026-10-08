"""在官方 Benchmark 上对比同组初始与最终冻结经验。"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import tempfile
from pathlib import Path

from contracts import EnvironmentConfig
from harnesses.benchmarks.assets import load_cases
from harnesses.benchmarks.scoring import score_case, check_scorer
from harnesses.benchmarks.tools import BenchmarkSessions
from harnesses.contracts import GroupKey, Snapshot
from harnesses.runners import create_runner
from harnesses.snapshots import directory_digest
from experiments.harness_train import (bridge_server, group_id, load_config, rooted,
    runner_config, selections, write_json, _source_probe, _version_ok, ROOT)
from experiments.harness_summary import _paired_interval, _cost


def benchmark_settings(config, environment):
    settings = dict(config["environments"][environment]["benchmark"])
    settings["source_dir"] = str(rooted(settings["source_dir"]))
    python = Path(settings.get("python_path") or ".venv/bin/python").expanduser()
    settings["python_path"] = str(python.absolute() if python.is_absolute() else (ROOT / python).absolute())
    return settings


def snapshots_for(config, environment, harness, source, seed):
    path = rooted(group_id(config, environment, harness, source, seed))
    train = json.loads((path / "train.json").read_text(encoding="utf-8"))
    expected_group = GroupKey(run_id=f"{config['run_id']}-{environment}-{harness}-{source}",
        environment=environment, harness=harness, source_method=source, seed=seed)
    if train.get("group") != expected_group.model_dump(mode="json"):
        raise ValueError("训练记录与评测组不一致")
    snapshots = {state:path / f"{state}-experience" for state in ("initial", "final")}
    for state, directory in snapshots.items():
        expected = train[f"{state}_snapshot"]["digest"]
        if not directory.is_dir() or directory_digest(directory) != expected:
            raise ValueError(f"{state} 经验快照缺失或与训练记录不一致")
    return snapshots


def check_group(config, environment, harness, source, seed):
    result = {"environment":environment, "harness":harness, "source":source, "seed":seed, "ok":False}
    try:
        benchmark = benchmark_settings(config, environment)
        cases, manifest = load_cases(environment, benchmark)
        scorer = check_scorer(environment, benchmark)
        if not scorer["ok"]:
            raise ValueError(scorer["error"])
        settings = config["local_harnesses"][harness]
        python = Path(settings.get("python_path") or ".venv/bin/python").expanduser()
        if not python.is_absolute():
            python = (ROOT / python).absolute()
        probe = _source_probe(harness, rooted(settings["source_dir"]), rooted(settings["program_path"]),
                              python, config["upstreams_lock"])
        version_ok, version = _version_ok(config["upstreams_lock"], harness, settings)
        runtime = probe.get("runtime") or {}
        runtime_ok = runtime.get("returncode") == 0 if harness == "openclaw" else bool(runtime) and all(v is True for v in runtime.values())
        if not (version_ok and probe["source_sha"] == version["expected"] and probe["program_path"]
                and all(probe["components"].values()) and runtime_ok):
            raise ValueError("基座锁定源码或依赖未就绪")
        snapshots_for(config, environment, harness, source, seed)
        output = rooted(group_id(config, environment, harness, source, seed)) / "benchmark"
        if output.exists():
            raise FileExistsError(f"评测目录已存在：{output}")
        result.update(ok=True, tasks=len(cases), assets=manifest, output=str(output))
    except (ValueError, OSError, KeyError) as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def require_execution(config, environment):
    if config.get("execute") is not True or os.environ.get("MEDEVOLVE_HARNESS_LIVE") != "1":
        raise ValueError("真实评测需要 execute: true 与 MEDEVOLVE_HARNESS_LIVE=1")
    roles = ["execution"] + (["benchmark_judge"] if environment == "PhysicianBench" else [])
    for role in roles:
        model = config["models"][role]
        if not model.get("model") or not model.get("base_url") or model.get("fallback"):
            raise ValueError(f"缺少固定模型配置：{role}")
        if not os.environ.get(model["api_key_env"]):
            raise ValueError(f"模型密钥环境变量未设置：{model['api_key_env']}")
    if not config.get("limits", {}).get("timeout_seconds"):
        raise ValueError("单题超时未配置")


def summarize_benchmark(result):
    records = result["records"]
    count = result["planned_denominator"]
    states, by_state = {}, {}
    for state in ("initial", "final"):
        rows = [row for row in records if row["state"] == state]
        scored = [row for row in rows if row["status"] == "scored"]
        successes = sum(row.get("completed") is True for row in scored)
        checkpoints = [row["checkpoint_score"] for row in scored if "checkpoint_score" in row]
        states[state] = {"successes":successes, "planned_denominator":count,
            "success_rate":successes / count if count else None,
            "valid_denominator":len(scored),
            "failures":{status:sum(row["status"] == status for row in rows)
                        for status in sorted({row["status"] for row in rows} - {"scored"})},
            "mean_checkpoint_score":sum(checkpoints) / len(checkpoints) if checkpoints else None}
        by_state[state] = {row["task_id"]:row for row in scored}
    pairs = []
    patients = {}
    for task_id in sorted(by_state["initial"].keys() & by_state["final"].keys()):
        initial, final = by_state["initial"][task_id], by_state["final"][task_id]
        delta = int(final["completed"]) - int(initial["completed"])
        pairs.append(delta)
        # 患者未知的检索题以题目为独立簇，不使用隐藏答案分组。
        patient = initial["patient_id"] or task_id
        patients.setdefault(patient, []).append(delta)
    lift = (states["final"]["success_rate"] - states["initial"]["success_rate"]) * 100 if count else None
    costs = _cost(result["runs"], result["evaluation_config"])
    return {"environment":result["environment"], "harness":result["harness"],
        "source":result["source"], "seed":result["seed"], "assets":result["assets"],
        "states":states, "lift_percentage_points":lift,
        "official_task_count":len(result["assets"].get("task_ids", [])),
        "selected_task_count":count,
        "full_benchmark":count == len(result["assets"].get("task_ids", [])),
        "paired_tasks":len(pairs), "paired_lift_percentage_points":sum(pairs) / len(pairs) * 100 if pairs else None,
        "paired_confidence_interval":_paired_interval(patients, seed=result["seed"]),
        "comparison_valid":result["snapshots_unchanged"] and len(pairs) == count and count > 0,
        "costs":{"execution":costs["testing"], "official_scoring":costs["review"]}}


async def run_benchmark(config, environment, harness, source, seed, task_ids=None):
    check = check_group(config, environment, harness, source, seed)
    if not check["ok"]:
        raise ValueError(check["error"])
    require_execution(config, environment)
    settings = benchmark_settings(config, environment)
    cases, assets = load_cases(environment, settings)
    if task_ids is not None:
        wanted = set(task_ids)
        if not wanted or wanted - {case.id for case in cases}:
            raise ValueError("指定的官方任务 ID 为空或不存在")
        cases = [case for case in cases if case.id in wanted]
    assets["selected_task_ids"] = [case.id for case in cases]
    snapshots = snapshots_for(config, environment, harness, source, seed)
    before = {state:directory_digest(path) for state, path in snapshots.items()}
    output = rooted(group_id(config, environment, harness, source, seed)) / "benchmark"
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "config.json", config, config)
    write_json(output / "assets.json", assets, config)
    records, runs = [], []
    envcfg = EnvironmentConfig(**config["environments"][environment])
    try:
        for case in cases:
            for state, snapshot in snapshots.items():
                job = output / "jobs" / state / case.id
                job.mkdir(parents=True, exist_ok=False)
                sessions = BenchmarkSessions(settings, job)
                client = f"{config['run_id']}-{environment}-{harness}-{source}-{seed}-{state}-{case.id}"
                session_id, runner = None, None
                row = {"state":state, "task_id":case.id, "patient_id":case.patient_id,
                       "status":"execution_error", "completed":None}
                try:
                    solver_input = case.solver_input()
                    if environment == "MedAgentBench":
                        solver_input = solver_input.model_copy(update={"instruction":solver_input.instruction +
                            "\n\nSubmit final answers through medical_action using FINISH([...]) before returning your final response."})
                    session_id = await sessions.open(solver_input, envcfg, client_id=client, read_only=False,
                        tool_budget=config["limits"]["execution"].get("max_tool_calls"),
                        timeout_seconds=config["limits"]["timeout_seconds"])
                    context = sessions.inspect(session_id, client_id=client)
                    tools = await sessions.list(session_id, client_id=client)
                    key = hashlib.sha256("\0".join((solver_input.patient_id, solver_input.instruction, "")).encode()).hexdigest()
                    metadata = {key:{"task_type":"", "tool_names":[tool["name"] for tool in tools]}}
                    rconfig = runner_config(config, harness, metadata, client_id=client, frozen=True)
                    rconfig["read_only"] = False
                    with tempfile.TemporaryDirectory(prefix=f"medevolve-benchmark-{harness}-") as tmp:
                        async with bridge_server(sessions) as url:
                            runner = create_runner(harness, config=rconfig, group_dir=Path(tmp), bridge_url=url)
                            group = GroupKey(run_id=client, environment=environment, harness=harness,
                                             source_method=source, seed=seed)
                            try:
                                await runner.start(group, Snapshot(path=snapshot, digest=before[state]))
                                run = await asyncio.wait_for(runner.execute(solver_input, session_id),
                                                             timeout=config["limits"]["timeout_seconds"])
                                row["run"] = run.model_dump(mode="json")
                                runs.append({"state":state, "task_id":case.id, "stage":"testing", "role":"execution", "run":row["run"]})
                                if run.status != "completed":
                                    row.update(status="execution_failed", error=run.error or run.status)
                                else:
                                    row.update(await score_case(case, settings, job, context.instance.endpoint,
                                        context.calls, run, config["models"].get("benchmark_judge")))
                                    if "judge_usage" in row:
                                        runs.append({"state":state,"task_id":case.id,"stage":"testing_review",
                                            "role":"benchmark_judge","run":{"status":row["status"],"usage":row["judge_usage"]}})
                            finally:
                                await runner.close()
                                runner = None
                except Exception as exc:
                    row.update(status="scoring_error" if row.get("run", {}).get("status") == "completed" else "execution_error",
                               completed=None, error=f"{type(exc).__name__}: {exc}")
                finally:
                    try:
                        if runner:
                            await runner.close()
                    finally:
                        if session_id:
                            row["tool_session"] = await sessions.close(session_id, client_id=client)
                        await sessions.aclose()
                records.append(row)
                write_json(job / "result.json", row, config)
                write_json(output / "records.json", records, config)
    finally:
        after = {state:directory_digest(path) for state, path in snapshots.items()}
        result = {"environment":environment, "harness":harness, "source":source, "seed":seed,
            "planned_denominator":len(cases), "assets":assets, "records":records, "runs":runs,
            "snapshot_digests_before":before, "snapshot_digests_after":after,
            "snapshots_unchanged":before == after,
            "evaluation_config":{"models":config["models"]}}
        write_json(output / "benchmark.json", result, config)
        write_json(output / "summary.json", summarize_benchmark(result), config)
    if before != after:
        raise ValueError("评测期间冻结经验发生变化，比较结果无效")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/harness-evolution.json")
    parser.add_argument("--environment", choices=["PhysicianBench", "MedAgentBench"])
    parser.add_argument("--harness"); parser.add_argument("--source")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--check-assets", action="store_true")
    parser.add_argument("--task-id", action="append")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    combos = selections(config, args.environment, args.harness, args.source)
    if not combos:
        parser.error("所选环境、基座或来源未配置")
    if args.check_assets:
        checks = []
        for environment in dict.fromkeys(e for e, _, _ in combos):
            try:
                cases, manifest = load_cases(environment, benchmark_settings(config, environment))
                scorer = check_scorer(environment, benchmark_settings(config, environment))
                checks.append({"environment":environment, "ok":scorer["ok"], "tasks":len(cases), "assets":manifest, "scorer":scorer})
            except ValueError as exc:
                checks.append({"environment":environment, "ok":False, "error":str(exc)})
        print(json.dumps(checks, ensure_ascii=False, indent=2))
        return 0 if all(row["ok"] for row in checks) else 2
    checks = [check_group(config, e, h, s, seed) for e, h, s in combos for seed in config["seeds"]]
    if args.check or not all(row["ok"] for row in checks):
        print(json.dumps(checks, ensure_ascii=False, indent=2))
        return 0 if all(row["ok"] for row in checks) else 2
    for e, h, s in combos:
        for seed in config["seeds"]:
            asyncio.run(run_benchmark(config, e, h, s, seed, args.task_id))
            print(rooted(group_id(config, e, h, s, seed)) / "benchmark/summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
