"""独立评分进程；直接调用锁定仓库内的官方评分代码。"""
from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace


def load_medagent_grader(directory):
    # 官方 eval.py 使用绝对导入；仅构造包路径，避免导入 AgentBench 服务。
    name = "src.server.tasks.medagentbench"
    for part in ("src", "src.server", "src.server.tasks", name):
        package = types.ModuleType(part)
        package.__path__ = [str(directory)] if part == name else []
        sys.modules[part] = package
    for leaf in ("utils", "refsol", "eval"):
        spec = importlib.util.spec_from_file_location(f"{name}.{leaf}", directory / f"{leaf}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    return sys.modules[f"{name}.eval"].eval


def medagent_score(payload):
    grader = load_medagent_grader(Path(payload["grader_directory"]))
    if payload.get("probe"):
        return {"status":"ready"}
    history = []
    answers = None
    endpoint = payload["endpoint"].rstrip("/") + "/"
    for call in payload["calls"]:
        output = call.get("output") or {}
        request = output.get("request", {})
        action = call.get("arguments", {}).get("action", "")
        kind = output.get("kind")
        if kind in {"GET", "POST"} and request.get("url"):
            action = kind + " " + request["url"]
            if kind == "POST" and request.get("payload") is not None:
                action += "\n" + json.dumps(request["payload"])
        history.extend([SimpleNamespace(role="agent", content=action),
                        SimpleNamespace(role="user", content=str(output.get("reply", call.get("error") or "")))])
        if kind == "FINISH":
            answers = output["answers"]
            break
    if answers is None:
        return {"status":"scored", "completed":False, "reason":"missing_finish"}
    result = SimpleNamespace(result=json.dumps(answers), history=history)
    completed = grader(payload["case_data"], result, endpoint) is True
    return {"status":"scored", "completed":completed}


def main():
    request, response = map(Path, sys.argv[1:3])
    try:
        result = medagent_score(json.loads(request.read_text(encoding="utf-8")))
    except Exception as exc:
        result = {"status":"scoring_error", "error":f"{type(exc).__name__}: {exc}"}
    response.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
