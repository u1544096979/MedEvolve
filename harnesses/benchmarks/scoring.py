"""运行原生评分进程并解析逐题结果。"""
from __future__ import annotations

import json
import os
import sys
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

from harnesses.local import run_command

WORKER = Path(__file__).with_name("worker.py")


def check_scorer(environment, settings):
    """在解题前核对独立评分解释器及原生评分模块。"""
    python = settings.get("python_path") or sys.executable
    modules = ["requests"] + (["pytest", "openai"] if environment == "PhysicianBench" else [])
    try:
        probe = subprocess.run([python, "-c", "import importlib; " +
            "; ".join(f"importlib.import_module({name!r})" for name in modules)],
            capture_output=True, text=True, timeout=30)
        if probe.returncode:
            return {"ok":False, "error":"官方评分依赖未就绪；在评分环境安装 .[benchmark]"}
        if environment == "MedAgentBench":
            with tempfile.TemporaryDirectory(prefix="medevolve-scorer-probe-") as tmp:
                request, response = Path(tmp) / "request.json", Path(tmp) / "response.json"
                request.write_text(json.dumps({"probe":True, "grader_directory":str(
                    Path(settings["source_dir"]) / "src/server/tasks/medagentbench")}), encoding="utf-8")
                probe = subprocess.run([python, str(WORKER), str(request), str(response)],
                    capture_output=True, text=True, timeout=30)
                if probe.returncode or not response.exists():
                    return {"ok":False, "error":"官方评分模块无法加载"}
                result = json.loads(response.read_text())
                if result.get("status") != "ready":
                    return {"ok":False, "error":result.get("error", "官方评分模块无法加载")}
        return {"ok":True, "python_path":python}
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        return {"ok":False, "error":f"{type(exc).__name__}: {exc}"}


def write_trajectory(job, calls, run):
    path = Path(job) / "logs/agent/trajectory.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    events = [{"type":"tool_call", "content":call["name"], "metadata":{
        "tool_name":call["name"], "input":call["arguments"],
        "output":json.dumps(call.get("output"), ensure_ascii=False)}} for call in calls]
    events.append({"type":"final_result", "content":str(run.output), "metadata":{}})
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in events), encoding="utf-8")


def parse_checkpoints(report, returncode, judge_errors=None):
    if returncode not in {0, 1} or not Path(report).is_file():
        return {"status":"scoring_error", "completed":None, "error":f"pytest exit {returncode}"}
    try:
        nodes = list(ET.parse(report).iter("testcase"))
    except ET.ParseError as exc:
        return {"status":"scoring_error", "completed":None, "error":str(exc)}
    if not nodes or any(node.find("error") is not None or node.find("skipped") is not None for node in nodes):
        return {"status":"scoring_error", "completed":None, "error":"官方检查点未完整执行"}
    errors = json.loads(Path(judge_errors).read_text()) if judge_errors and Path(judge_errors).exists() else []
    checkpoints = []
    for node in nodes:
        failure = node.find("failure")
        if failure is not None and "AssertionError" not in failure.get("message", "") and not failure.get("message", "").startswith("assert "):
            errors.append(failure.get("message", "评分异常"))
        checkpoints.append({"name":node.get("name"), "passed":failure is None,
                            "message":failure.get("message") if failure is not None else None})
    if errors:
        return {"status":"scoring_error", "completed":None, "errors":errors, "checkpoints":checkpoints}
    passed = sum(row["passed"] for row in checkpoints)
    return {"status":"scored", "completed":passed == len(checkpoints),
            "checkpoint_score":passed / len(checkpoints), "checkpoints":checkpoints}


async def score_case(case, settings, job, endpoint, calls, run, judge=None):
    job = Path(job).resolve()
    verifier = job / "logs/verifier"
    verifier.mkdir(parents=True, exist_ok=True)
    python = settings.get("python_path") or sys.executable
    timeout = settings.get("scoring_timeout_seconds", 900)
    env = {"PYTHON_DOTENV_DISABLED":"1"}
    if case.environment == "MedAgentBench":
        request, response = verifier / "request.json", verifier / "score.json"
        request.write_text(json.dumps({"grader_directory":str(case.asset_path.parent),
            "case_data":case.reference, "endpoint":endpoint, "calls":calls}), encoding="utf-8")
        code, stdout, stderr = await run_command([python, str(WORKER), str(request), str(response)],
            group_dir=job, group_env=env, timeout=timeout)
        result = json.loads(response.read_text()) if code == 0 and response.exists() else {
            "status":"scoring_error", "completed":None, "error":f"评分进程退出 {code}"}
    else:
        write_trajectory(job, calls, run)
        root = Path(settings["source_dir"]).resolve()
        judge = judge or {}
        key = os.environ.get(judge.get("api_key_env", ""))
        if not key:
            return {"status":"scoring_error", "completed":None, "error":"官方裁判密钥未配置"}
        if judge.get("backend", "openai") != "openai":
            raise ValueError("PhysicianBench 裁判须配置 OpenAI 兼容接口")
        env.update({"PYTHONPATH":os.pathsep.join([str(verifier), str(root)]),
            "FHIR_BASE_URL":endpoint.rstrip("/"), "JOB_DIR":str(job),
            "LLM_JUDGE_BACKEND":"openai", "LLM_JUDGE_MODEL":judge["model"],
            "OPENAI_API_KEY":key, "OPENAI_BASE_URL":judge["base_url"],
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD":"1"})
        # 官方助手会把裁判调用异常转成 FAIL；旁路记录异常，区分评分故障。
        errors = verifier / "judge_errors.json"
        plugin = verifier / "benchmark_judge_probe.py"
        plugin.write_text('''import json
from pathlib import Path
from utils import eval_helpers as eh
_original = eh.call_llm
_errors = []
def _call(*args, **kwargs):
    try:
        return _original(*args, **kwargs)
    except Exception as exc:
        _errors.append(type(exc).__name__)
        Path(__file__).with_name("judge_errors.json").write_text(json.dumps(_errors))
        raise
eh.call_llm = _call
''', encoding="utf-8")
        report = verifier / "checkpoints.xml"
        code, stdout, stderr = await run_command([python, "-m", "pytest", str(case.asset_path),
            "-p", "benchmark_judge_probe", "-v", "--tb=short", f"--junitxml={report}"],
            group_dir=job, group_env=env, timeout=timeout)
        result = parse_checkpoints(report, code, errors)
        result["judge_usage"] = {"known":False}
    if judge and (secret := os.environ.get(judge.get("api_key_env", ""))):
        stdout, stderr = stdout.replace(secret, "[REDACTED]"), stderr.replace(secret, "[REDACTED]")
        result = json.loads(json.dumps(result).replace(secret, "[REDACTED]"))
    (verifier / "stdout.txt").write_text(stdout, encoding="utf-8")
    (verifier / "stderr.txt").write_text(stderr, encoding="utf-8")
    return result
