"""训练合成任务并冻结各 Harness 的最终经验。"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import shutil
import socket
from contextlib import asynccontextmanager
import subprocess
import sys
from pathlib import Path

from contracts import EnvironmentConfig, EvaluationConfig, ModelConfig, RuntimeConfig, SolverInput
from harnesses.contracts import GroupKey, Snapshot
from harnesses.datasets import DatasetFreezeError, freeze_datasets, load_frozen, preflight_datasets
from harnesses.feedback import make_training_feedback
from harnesses.local import prepare_group_directory
from harnesses.runners import create_runner
from harnesses.snapshots import directory_digest

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path) -> dict:
    path=Path(path)
    return json.loads((path if path.is_absolute() else ROOT/path).resolve().read_text(encoding="utf-8"))


def rooted(path: str | Path) -> Path:
    path = Path(path)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def selections(config, environment=None, harness=None, source=None):
    envs = [environment] if environment else config["environments"].keys()
    harnesses = [harness] if harness else config["harnesses"]
    sources = [source] if source else config["source_methods"]
    return [(e, h, s) for e in envs for h in harnesses for s in sources
            if e in config["environments"] and h in config["harnesses"]
            and s in config["source_methods"]]


def evaluation_config(config):
    model = ModelConfig(**config["models"]["completion_review"])
    return EvaluationConfig(runtime=RuntimeConfig(models={"completion": model}, budgets={}))


def group_id(config, environment, harness, source, seed):
    template = config["run_dir_template"]
    return template.replace("<run_id>", config["run_id"]).replace(
        "<environment>", environment).replace("<harness>", harness).replace(
        "<source_method>", source).replace("<seed>", str(seed))


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def freeze_root(config, environment, source):
    layout=config.get("frozen_layout", "<environment>/<source_method>")
    layout=layout.replace("<environment>",environment).replace("<source_method>",source)
    return rooted(config["frozen_root"]) / layout


def ensure_frozen(config, environment, source):
    root = freeze_root(config, environment, source)
    manifest_path, tasks_path = root / "manifest.json", root / "tasks.json"
    source_cfg = config["environments"][environment]["sources"].get(source, {})
    if not source_cfg.get("tasks"):
        raise DatasetFreezeError(source_cfg.get("missing_reason", f"缺少任务来源：{source}"))
    expected = {source: rooted(source_cfg["tasks"])}
    if manifest_path.exists() or tasks_path.exists():
        if not manifest_path.is_file() or not tasks_path.is_file():
            raise DatasetFreezeError(f"冻结清单文件不完整：{root}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("environment") != environment or manifest.get("seed") != config["seeds"][0]:
            raise DatasetFreezeError("既有冻结清单的环境或种子与配置不匹配")
        for name, source in manifest.get("sources", {}).items():
            path = expected.get(name)
            if path is None or str(path) != str(Path(source["tasks_file"]).resolve()) or _sha(path) != source["tasks_sha256"]:
                raise DatasetFreezeError(f"既有冻结清单的源文件摘要与配置不匹配：{name}")
        return load_frozen(root)
    return freeze_datasets(expected, count=int(config["tasks_per_group"]),
                           seed=int(config["seeds"][0]), destination=root)


def _version_ok(lock, harness, settings):
    locks = json.loads(rooted(lock).read_text(encoding="utf-8"))
    entry = locks.get("upstreams", {}).get(harness, {})
    wanted = settings.get("upstream_commit")
    actual = entry.get("commit")
    return wanted is not None and actual == wanted, {"expected": wanted, "actual": actual,
        "version": entry.get("version"), "dependency_lock": entry.get("dependency_lock")}


def _official_import_probe(harness, source_dir, python_path):
    # Use the same import loaders as the worker, with outbound sockets disabled and
    # placeholder credentials. Import-time failures are reported before a paid run.
    runner = {"ace": "from harnesses.runners.ace_worker import _load_official; _load_official(sys.argv[1])",
              "reflexion": "from harnesses.runners.reflexion_worker import _official; _official(sys.argv[1], 'probe-only-invalid')"}[harness]
    code = ("import json,socket,sys,urllib.request; "
            "deny=lambda *a,**k: (_ for _ in ()).throw(RuntimeError('network disabled during import probe')); "
            "socket.socket.connect=deny; socket.socket.connect_ex=deny; socket.create_connection=deny; "
            "urllib.request.urlopen=deny; " + runner + "; print(json.dumps({'official_import_ok':True}))")
    env = {"PATH": os.environ.get("PATH", ""), "PYTHON_DOTENV_DISABLED": "1",
           "OPENAI_API_KEY": "probe-only-invalid"}
    try:
        probe = subprocess.run([str(python_path), "-c", code, str(source_dir)],
            capture_output=True, text=True, timeout=30, check=True, cwd=str(ROOT), env=env)
        lines = [line for line in probe.stdout.splitlines() if line.strip()]
        return json.loads(lines[-1]) if lines else {"official_import_ok": False, "error": "probe produced no result"}
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        detail = (getattr(exc, "stderr", None) or str(exc)).strip()
        return {"official_import_ok": False, "error": detail[-1000:]}


def _source_probe(harness, source_dir, program_path, python_path, lock_path):
    result = {"source_dir": source_dir.is_dir(), "program_path": False,
              "source_sha": None, "components": {}, "runtime": None}
    if harness == "openclaw":
        result["program_path"] = program_path.is_file()
    elif harness in {"hermes"}:
        result["program_path"] = program_path.is_file()
    else:
        result["program_path"] = program_path.is_dir()
    if not source_dir.is_dir():
        return result
    lock = json.loads(rooted(lock_path).read_text(encoding="utf-8"))
    components = lock.get("upstreams", {}).get(harness, {}).get("components", [])
    result["components"] = {name:(source_dir/name).exists() for name in components}
    try:
        revision = subprocess.run(["git","-C",str(source_dir),"rev-parse","HEAD"],
            capture_output=True,text=True,timeout=5,check=True).stdout.strip()
        result["source_sha"] = revision
    except (OSError,subprocess.SubprocessError):
        pass
    if harness == "openclaw":
        node = shutil.which("node")
        if node and program_path.is_file():
            try:
                probe = subprocess.run([node,str(program_path),"--version"],capture_output=True,text=True,timeout=10)
                result["runtime"] = {"command":"--version","returncode":probe.returncode,
                                     "version":(probe.stdout or probe.stderr).strip()[:160]}
            except (OSError,subprocess.SubprocessError) as exc:
                result["runtime"] = {"error":str(exc)}
    elif harness in {"ace", "reflexion"}:
        result["runtime"] = _official_import_probe(harness, source_dir, python_path)
    else:
        py = python_path if python_path.is_file() else Path(sys.executable)
        modules = {"hermes":["hermes_cli","openai","httpx"],
                   "ace":["openai","yaml","tiktoken"],
                   "reflexion":["langchain","openai","requests"]}[harness]
        probe_code = "import importlib.util,json; print(json.dumps({m: importlib.util.find_spec(m) is not None for m in " + repr(modules) + "}))"
        try:
            probe = subprocess.run([str(py),"-c",probe_code],capture_output=True,text=True,
                timeout=10,check=True,cwd=str(source_dir))
            result["runtime"] = json.loads(probe.stdout)
        except (OSError,subprocess.SubprocessError,ValueError) as exc:
            result["runtime"] = {"error":str(exc)}
    return result


def preflight(config, combos, *, mode="train"):
    output = []
    for environment, harness, source in combos:
        settings = config["local_harnesses"][harness]
        local_paths = {key: rooted(settings[key]) for key in ("source_dir", "program_path", "initial_state_template")}
        python_path = Path(settings.get("python_path") or sys.executable).expanduser()
        if not python_path.is_absolute():
            python_path = ROOT / python_path
        python_path = Path(os.path.abspath(python_path))
        probe = _source_probe(harness, local_paths["source_dir"], local_paths["program_path"], python_path, config["upstreams_lock"])
        source_cfg = config["environments"][environment]["sources"].get(source, {})
        data_status = "missing"
        data_detail = source_cfg.get("missing_reason", "未配置任务文件")
        if source_cfg.get("tasks"):
            path = rooted(source_cfg["tasks"])
            try:
                reports = preflight_datasets({environment: {"sources": {source: {"tasks": str(path)}}}},
                    count=int(config["tasks_per_group"]), seed=int(config["seeds"][0]))
                entry = reports[environment]["sources"][source]
                data_status, data_detail = entry["status"], entry.get("reason", "")
            except Exception as exc:
                data_detail = f"{type(exc).__name__}: {exc}"
        version_ok, version = _version_ok(config["upstreams_lock"], harness, settings)
        files = {"source_dir": probe["source_dir"], "program_path":probe["program_path"],
                 "initial_state_template":local_paths["initial_state_template"].is_dir(),
                 "python_path":python_path.is_file()}
        if harness == "openclaw":
            files["program_path"] = files["program_path"] and os.access(local_paths["program_path"], os.R_OK)
        expected_sha = version.get("expected")
        version_ok = version_ok and probe["source_sha"] == expected_sha
        components_ok = all(probe["components"].values())
        runtime = probe["runtime"] or {}
        runtime_ok = (runtime.get("returncode") == 0 if harness == "openclaw" else all(value is True for value in runtime.values()))
        try:
            evaluation_config(config)
            reviewer_ok, reviewer_error = True, None
        except Exception as exc:
            reviewer_ok, reviewer_error = False, f"{type(exc).__name__}: {exc}"
        task_file=rooted(source_cfg["tasks"]) if source_cfg.get("tasks") else None
        evaluation_file=task_file.with_name("evaluations.json") if task_file else None
        data_report={"status":data_status,"detail":data_detail,
            "tasks_file":str(task_file) if task_file else None,
            "tasks_sha256":_sha(task_file) if task_file and task_file.is_file() else None,
            "evaluations_file":str(evaluation_file) if evaluation_file else None,
            "evaluations_sha256":_sha(evaluation_file) if evaluation_file and evaluation_file.is_file() else None}
        run_paths=[rooted(group_id(config,environment,harness,source,seed)) for seed in config["seeds"]]
        if mode == "train":
            directories_ok=all(not path.exists() for path in run_paths)
            directory_report={str(path):{"available":not path.exists()} for path in run_paths}
        else:
            directory_report={}
            directories_ok=True
            for path in run_paths:
                state={"train_record":(path/"train.json").is_file(),
                       "initial_snapshot":(path/"initial-experience").is_dir(),
                       "final_snapshot":(path/"final-experience").is_dir(),
                       "test_output_available":not (path/"test.json").exists()}
                directory_report[str(path)]=state
                directories_ok=directories_ok and all(state.values())
        output.append({"environment": environment, "harness": harness, "source": source,
            "directories":directory_report,"directories_ok":directories_ok,
            "data": data_report, "version": version,
            "version_ok": version_ok, "paths": {k: str(v) for k,v in local_paths.items()},
            "python_path":str(python_path),"paths_ok":files,"components":probe["components"],
            "runtime":runtime,"runtime_ok":runtime_ok,"reviewer_configured":reviewer_ok,
            "reviewer_error":reviewer_error,"ok": data_status == "ok" and version_ok and all(files.values()) and components_ok and runtime_ok and reviewer_ok and directories_ok})
    return output


def _require_execution(config):
    if config.get("execute") is not True:
        raise ValueError("实际运行需要配置 execute: true")
    if os.environ.get("MEDEVOLVE_HARNESS_LIVE") != "1":
        raise ValueError("实际 Harness 运行需要 MEDEVOLVE_HARNESS_LIVE=1")
    for key in ("execution", "learning", "completion_review"):
        item = config.get("models", {}).get(key)
        if not item or not item.get("model") or not item.get("api_key_env") or item.get("fallback"):
            raise ValueError(f"缺少固定模型配置或配置了 fallback：models.{key}")
        if not os.environ.get(item["api_key_env"]):
            raise ValueError(f"模型密钥环境变量未设置：{item['api_key_env']}")
    limits = config.get("limits", {})
    for key in ("execution", "learning"):
        budget=limits.get(key) or {}
        if not budget.get("max_model_turns") or (key=="execution" and not budget.get("max_tool_calls")):
            raise ValueError(f"缺少运行预算：limits.{key}")
    for key in ("timeout_seconds", "learning_timeout_seconds"):
        if not limits.get(key):
            raise ValueError(f"缺少运行预算：limits.{key}")


def runner_config(config, harness, task_metadata, *, client_id, frozen=False):
    cfg = dict(config)
    cfg["task_metadata"] = task_metadata
    cfg["client_id"] = client_id
    cfg["frozen"] = frozen
    return cfg


def metadata_for(task):
    key = hashlib.sha256("\0".join((task.patient_id, task.instruction, task.decision_point or "")).encode()).hexdigest()
    return key, {"task_type": task.task_type, "tool_names": []}


def _task_metadata(tasks):
    result = {}
    for task in tasks:
        key, value = metadata_for(task)
        value["tool_names"] = []
        result[key] = value
    return result


def _record_run(bucket, run, task_id, stage, role):
    bucket.append({"task_id": task_id, "stage": stage, "role": role,
                  "run": run.model_dump(mode="json")})


def _redact(value, secrets):
    if isinstance(value, dict):
        return {key: _redact(item, secrets) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact(item, secrets) for item in value]
    if isinstance(value, str):
        for secret in secrets:
            if secret:
                value = value.replace(secret, "[REDACTED]")
        return value
    return value


def write_json(path, value, config):
    secrets = [os.environ.get(model.get("api_key_env", ""), "")
               for model in config.get("models", {}).values() if model.get("api_key_env")]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(_redact(value, secrets), ensure_ascii=False, indent=2), encoding="utf-8")


@asynccontextmanager
async def bridge_server(sessions):
    import uvicorn
    from harnesses.medical_bridge import create_bridge_app
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", 0)); sock.listen(128); sock.setblocking(False)
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(create_bridge_app(sessions), host="127.0.0.1", port=port,
        log_level="critical", access_log=False, lifespan="off"))
    task = asyncio.create_task(server.serve(sockets=[sock]))
    try:
        for _ in range(100):
            if server.started: break
            if task.done():
                task.result()
            await asyncio.sleep(.01)
        if not server.started:
            raise RuntimeError("医疗工具桥接服务未启动")
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        await task


def update_task_tool_metadata(config, task, tools):
    key, metadata = metadata_for(task)
    config.setdefault("task_metadata", {})[key] = {**metadata,
        "tool_names": sorted({item.get("name") for item in tools if isinstance(item, dict) and item.get("name")})}


async def run_training(config, environment, harness, source):
    checks=preflight(config,[(environment,harness,source)],mode="train")
    if not checks[0]["ok"]:
        raise ValueError("本地预检未通过："+json.dumps(checks[0],ensure_ascii=False))
    _require_execution(config)
    all_sources = ensure_frozen(config, environment, source)
    tasks = all_sources[source]
    from harnesses.datasets import split_tasks
    split = split_tasks(tasks, seed=int(config["seeds"][0]))
    settings = config["local_harnesses"][harness]
    template = rooted(settings["initial_state_template"])
    group_path = rooted(group_id(config, environment, harness, source, config["seeds"][0]))
    if group_path.exists():
        raise FileExistsError(f"实验目录已存在：{group_path}")
    group_path.mkdir(parents=True, exist_ok=False)
    initial_copy = group_path / "initial-experience"
    prepare_group_directory(template, initial_copy)
    initial = Snapshot(path=initial_copy, digest=directory_digest(initial_copy))
    envcfg = EnvironmentConfig(**config["environments"][environment])
    from environments import prepare_trajectory
    from harnesses.medical_bridge import ToolSessions
    sessions = ToolSessions()
    review_cfg = evaluation_config(config)
    group = GroupKey(run_id=f"{config['run_id']}-{environment}-{harness}-{source}",
                     environment=environment, harness=harness, source_method=source, seed=config["seeds"][0])
    rconfig = runner_config(config, harness, _task_metadata(tasks), client_id=group.run_id)
    runner = create_runner(harness, config=rconfig, group_dir=group_path, bridge_url="")
    stages, attempts, feedback_records, learn_records = [], [], [], []
    try:
      async with bridge_server(sessions) as bridge_url:
        runner.bridge_url = bridge_url
        await runner.start(group, initial)
        for round_index in range(int(config.get("evolution_rounds", 1))):
            for task in split.train:
                solver_input = SolverInput(instruction=task.instruction, patient_id=task.patient_id, decision_point=task.decision_point)
                retry = False
                attempt_index = 0
                while True:
                    attempt_index += 1
                    session_id = None
                    summary = None
                    task_feedback = None
                    run = None
                    try:
                        session_id = await sessions.open(solver_input, envcfg, client_id=group.run_id,
                            read_only=True, tool_budget=config["limits"]["execution"].get("max_tool_calls"),
                            timeout_seconds=config["limits"]["timeout_seconds"])
                        context = sessions.inspect(session_id, client_id=group.run_id)
                        tool_catalog = await sessions.list(session_id, client_id=group.run_id)
                        update_task_tool_metadata(rconfig, task, tool_catalog)
                        trajectory = await prepare_trajectory(context.instance, task.patient_id)
                        run = await (runner.execute_retry(solver_input, session_id) if retry else runner.execute(solver_input, session_id))
                        run = run.model_copy(update={"trace": [*run.trace,
                            *context.events, *context.calls]})
                        observed_state = await sessions.environments.read_state(context.instance, sessions._state_task(context))
                        try:
                            task_feedback = await make_training_feedback(task, trajectory, run, observed_state or {}, review_cfg,
                                record=lambda rr: _record_run(stages, rr, task.id, "review", "completion"))
                            score_run = stages[-1]["run"] if stages else None
                            scoring_error = run.status == "completed" and run.output is not None and task_feedback.completed is None
                            status = "scoring_error" if scoring_error else "execution_failed" if run.status != "completed" or run.output is None else "reviewed"
                            feedback_records.append({"task_id": task.id, "completed": task_feedback.completed,
                                "status":status,"failure_category": task_feedback.failure_category, "reason": task_feedback.reason,
                                "scoring_error": scoring_error})
                        except Exception as exc:
                            task_feedback = None
                            feedback_records.append({"task_id": task.id,"completed":None,"status":"scoring_error",
                                "scoring_error":True,"error":f"{type(exc).__name__}: {exc}"})
                        summary = await sessions.close(session_id, client_id=group.run_id)
                        _record_run(stages, run, task.id, "execution", "execution")
                        attempts.append({"task_id": task.id,"attempt":attempt_index,"round":round_index+1,
                            "retry":retry,"status":"execution_failed" if run.status != "completed" or run.output is None else "scoring_error" if task_feedback and task_feedback.completed is None else "reviewed",
                            "run":run.model_dump(mode="json"),"tool_session":summary,
                            "trajectory":trajectory.model_dump(mode="json"),"observed_state":observed_state,
                            "feedback":task_feedback.model_dump(mode="json") if task_feedback else None})
                    except Exception as attempt_exc:
                        attempts.append({"task_id":task.id,"attempt":attempt_index,"round":round_index+1,
                            "retry":retry,"status":"execution_failed",
                            "error":f"{type(attempt_exc).__name__}: {attempt_exc}"})
                        feedback_records.append({"task_id":task.id,"completed":None,"status":"execution_failed",
                            "scoring_error":False,"error":f"{type(attempt_exc).__name__}: {attempt_exc}"})
                    finally:
                        if session_id:
                            try:
                                if summary is None:
                                    summary = await sessions.close(session_id, client_id=group.run_id)
                            except Exception as cleanup_exc:
                                attempts.append({"task_id":task.id,"attempt":attempt_index,"cleanup_error":f"{type(cleanup_exc).__name__}: {cleanup_exc}"})
                    if task_feedback is None or task_feedback.completed is None:
                        break
                    step = await runner.learn(task_feedback)
                    if harness in {"ace", "reflexion"} or harness == "hermes":
                        for detail in step.learning_runs:
                            _record_run(learn_records, detail, task.id, "learning", "learning")
                    elif harness == "openclaw":
                        if any(item.get("type") == "learning_feedback_turn" for item in step.run.trace):
                            _record_run(learn_records, step.run, task.id, "learning", "learning")
                        for detail in step.learning_runs:
                            _record_run(learn_records, detail, task.id, "learning", "learning")
                    attempts[-1]["learning"] = step.model_dump(mode="json")
                    if harness != "reflexion" or task_feedback.completed is not False or not runner.retry_state.get("ready"):
                        break
                    retry = True
        final_dir = group_path / "final-experience"
        snapshot = await runner.freeze(final_dir)
        return {"group": group.model_dump(mode="json"),"planned_train_denominator":len(split.train),
            "reviewed_feedback_denominator":sum(row.get("status")=="reviewed" for row in feedback_records),
            "train_task_ids":[t.id for t in split.train],"test_task_ids":[t.id for t in split.test],"attempts":attempts,"feedback":feedback_records,
            "initial_snapshot":{"path":str(initial.path),"digest":initial.digest},
            "final_snapshot":{"path":str(snapshot.path),"digest":snapshot.digest},
            "runs":stages,"learning_runs":learn_records}
    finally:
        try:
            await runner.close()
        finally:
            await sessions.aclose()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/harness-evolution.json")
    parser.add_argument("--environment"); parser.add_argument("--harness"); parser.add_argument("--source")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    combos = selections(config, args.environment, args.harness, args.source)
    if not combos:
        parser.error("所选环境、Harness 或来源未配置")
    checks=preflight(config,combos,mode="train")
    if args.check:
        print(json.dumps(checks,ensure_ascii=False,indent=2))
        return 0 if all(item["ok"] for item in checks) else 2
    if not all(item["ok"] for item in checks):
        print(json.dumps(checks,ensure_ascii=False,indent=2),file=sys.stderr)
        return 2
    _require_execution(config)
    for environment, harness, source in combos:
        result = asyncio.run(run_training(config, environment, harness, source))
        path = rooted(group_id(config, environment, harness, source, config["seeds"][0])) / "train.json"
        write_json(path, result, config)
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
