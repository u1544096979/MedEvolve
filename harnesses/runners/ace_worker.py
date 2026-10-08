"""在独立 Python 环境中调用锁定的 ACE 官方组件。"""
from __future__ import annotations

import importlib
import json
import os
import re
import sys
import types
import urllib.request
from pathlib import Path
from types import SimpleNamespace


def _post(url, payload, client_id):
    request = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={
        "Content-Type": "application/json", "x-medevolve-client": client_id,
    })
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read())


def _get(url, client_id):
    request = urllib.request.Request(url, headers={"x-medevolve-client": client_id})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read())


class ACEClientShim:
    """将 ACE completion 请求接到有界的原生医疗工具循环。"""
    def __init__(self, model, key, base_url, bridge_url, session, client_id,
                 max_model_turns, max_tool_calls, max_output_tokens):
        from openai import OpenAI
        self.client = OpenAI(api_key=key, base_url=base_url, max_retries=0)
        self.model, self.bridge_url, self.session = model, bridge_url, session
        self.client_id, self.max_tool_calls = client_id, max_tool_calls
        self.max_model_turns, self.max_output_tokens = max_model_turns, max_output_tokens
        self.calls = 0
        self.native_tool_calls = 0
        self.tool_trace = []
        self.prompt_tokens = self.completion_tokens = 0
        self.usage_known = True
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        tools = (_get(f"{self.bridge_url}/sessions/{self.session}/tools", self.client_id)
                 if self.session else [])
        definitions = [{"type":"function", "function":{"name":t["name"], "description":t["description"], "parameters":t["parameters"]}} for t in tools]
        messages = list(kwargs["messages"])
        prompt = messages[0]["content"] if messages else ""
        # 保留 ACE 官方 Generator 提示词，并在系统消息中提供医疗动作约束。
        messages.insert(0, {"role":"system", "content":"使用可用医疗工具查询任务所需证据。每轮最多执行有限工具动作，最终严格按要求输出 JSON。不得尝试写入患者记录。"})
        while self.calls < self.max_model_turns:
            request_kwargs=dict(kwargs)
            request_kwargs["model"]=self.model
            request_kwargs["messages"]=messages
            request_kwargs["tools"]=definitions or None
            request_kwargs["tool_choice"]="auto" if definitions else None
            request_kwargs["temperature"]=0
            token_key="max_completion_tokens" if "max_completion_tokens" in request_kwargs else "max_tokens"
            configured=request_kwargs.get(token_key)
            if configured is None: configured=self.max_output_tokens or 2048
            request_kwargs[token_key]=int(configured if self.max_output_tokens is None else min(int(configured),self.max_output_tokens))
            response = self.client.chat.completions.create(**request_kwargs)
            self.calls += 1
            if getattr(response,"usage",None) is None:
                self.usage_known=False
            else:
                self.prompt_tokens += getattr(response.usage, "prompt_tokens", 0) or 0
                self.completion_tokens += getattr(response.usage, "completion_tokens", 0) or 0
            choice = response.choices[0]
            message = choice.message
            calls = getattr(message, "tool_calls", None) or []
            if not calls:
                return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=SimpleNamespace(
                    prompt_tokens=self.prompt_tokens, completion_tokens=self.completion_tokens))
            if self.calls >= self.max_model_turns:
                raise RuntimeError(f"ACE_MODEL_TURN_BUDGET_EXHAUSTED: calls={self.calls}")
            messages.append(message.model_dump(exclude_none=True))
            for call in calls:
                if self.native_tool_calls >= self.max_tool_calls:
                    raise RuntimeError("ACE 医疗工具调用超过配置上限")
                self.native_tool_calls += 1
                function = call.function
                result = _post(f"{self.bridge_url}/sessions/{self.session}/calls", {
                    "name": function.name, "arguments": json.loads(function.arguments or "{}")}, self.client_id)
                self.tool_trace.append({"name":function.name,"arguments":json.loads(function.arguments or "{}"),"result":result})
                messages.append({"role":"tool", "tool_call_id":call.id, "content":json.dumps(result, ensure_ascii=False)})
        raise RuntimeError(f"ACE_MODEL_TURN_BUDGET_EXHAUSTED: calls={self.calls}")


def _load_official(source_dir):
    source = Path(source_dir).resolve()
    if not (source / "ace" / "core" / "generator.py").is_file():
        raise RuntimeError(f"ACE 官方源码缺失：{source}/ace/core/generator.py；请配置锁定 checkout 的 source_dir")
    sys.path.insert(0, str(source))
    try:
        # 按锁定的上游包路径加载三个组件，避免导入无关的 Benchmark 编排模块。
        package = sys.modules.get("ace") or types.ModuleType("ace")
        package.__path__ = [str(source / "ace")]
        sys.modules["ace"] = package
        core_package = sys.modules.get("ace.core") or types.ModuleType("ace.core")
        core_package.__path__ = [str(source / "ace" / "core")]
        sys.modules["ace.core"] = core_package
        generator = importlib.import_module("ace.core.generator").Generator
        reflector = importlib.import_module("ace.core.reflector").Reflector
        curator = importlib.import_module("ace.core.curator").Curator
        core = types.SimpleNamespace(Generator=generator, Reflector=reflector, Curator=curator)
        playbook = importlib.import_module("playbook_utils")
    except Exception as exc:
        raise RuntimeError(f"无法加载锁定 ACE 官方组件；请在 python_path 环境安装 upstreams/ace/uv.lock 依赖：{type(exc).__name__}: {exc}") from exc
    required = ("Generator", "Reflector", "Curator")
    if any(not hasattr(core, name) for name in required):
        raise RuntimeError("锁定 ACE checkout 缺少 Generator/Reflector/Curator")
    return core, playbook


def main(req):
    core, playbook = _load_official(req["source_dir"])
    state = Path(req["state_dir"]); state.mkdir(parents=True, exist_ok=True)
    playbook_path = state / "playbook.md"
    current = playbook_path.read_text(encoding="utf-8") if playbook_path.exists() else "## OTHERS\n"
    model = req["model"]; key = os.environ.get(model.get("api_key_env", ""), "")
    if not key:
        raise RuntimeError(f"缺少模型 API key 环境变量：{model.get('api_key_env')}")
    shim = ACEClientShim(model["model"], key, model.get("base_url"), req["bridge_url"],
                         req.get("tool_session", ""), req["client_id"],
                         req.get("max_model_turns", 30),req.get("max_tool_calls",20),req.get("max_output_tokens",2048))
    generator = core.Generator(shim, "openai", model["model"], max_tokens=req.get("max_output_tokens", 2048))
    if req["action"] == "execute":
        metadata = req.get("metadata", {})
        experiences = json.loads((state / "experiences.json").read_text()) if (state / "experiences.json").exists() else []
        context = "\n".join(f"- {entry['procedure']}（注意：{entry['pitfall']}）" for entry in experiences[:3])
        question=req["instruction"]+f"\n患者标识：{req.get('patient_id','')}\n决策点：{req.get('decision_point') or '无'}"
        raw, bullet_ids, info = generator.generate(question, current, context=context,
            reflection=req.get("reflection", "(empty)"), use_json_mode=True, call_id="medical-generator")
        return {"raw":raw, "bullet_ids":bullet_ids, "call_info":info, "tool_calls":shim.native_tool_calls,
                "tool_trace":shim.tool_trace,
                "usage":{"known":shim.usage_known,"prompt_tokens":info.get("prompt_num_tokens",0),"completion_tokens":info.get("response_num_tokens",0),"calls":shim.calls}}
    if req["action"] == "learn":
        feedback = req["feedback"]
        if feedback.get("completed") is None:
            return {"learning_status":"not_triggered", "learning_runs":[]}
        reflector = core.Reflector(shim, "openai", model["model"], max_tokens=req.get("max_output_tokens", 2048))
        reflection, tags, reflect_info = reflector.reflect(
            req["instruction"]+f"\n患者标识：{req.get('patient_id','')}\n决策点：{req.get('decision_point') or '无'}", req.get("trajectory", ""), req.get("answer", ""),
            feedback.get("reference_answer"), feedback.get("reason", ""),
            "\n".join(req.get("bullets_used", [])), use_ground_truth=True,
            use_json_mode=True, call_id="medical-reflector")
        stats = playbook.get_playbook_stats(current)
        curator = core.Curator(shim, "openai", model["model"], max_tokens=req.get("max_output_tokens", 2048))
        safe_reflection=reflection
        sensitive=[req.get("patient_id", ""),req.get("answer", ""),feedback.get("reference_answer", ""),req.get("instruction", "")]
        sensitive.extend(str(item.get("statement", "")) for item in (feedback.get("evidence") or []) if isinstance(item,dict))
        for value in sensitive:
            if len(value)>=4: safe_reflection=safe_reflection.replace(value,"[task detail]")
        safe_reflection=re.sub(r"\b\d+(?:[.,]\d+)?\b","[clinical value]",safe_reflection)
        updated, next_id, operations, curate_info = curator.curate(
            current, safe_reflection, req.get("question_context", ""), req.get("current_step", 1),
            req.get("total_samples", 1), req.get("playbook_token_budget", 2000), stats,
            use_ground_truth=True, use_json_mode=True, call_id="medical-curator",
            next_global_id=playbook.get_next_global_id(current))
        playbook_path.write_text(updated, encoding="utf-8")
        return {"learning_status":"updated" if updated != current else "unchanged", "reflection":reflection,
            "learning_runs":[{"reflect":reflect_info,"curate":curate_info,"tool_calls":shim.calls}],
            "usage":{"known":shim.usage_known,"prompt_tokens":reflect_info.get("prompt_num_tokens",0)+curate_info.get("prompt_num_tokens",0),
                     "completion_tokens":reflect_info.get("response_num_tokens",0)+curate_info.get("response_num_tokens",0),"calls":shim.calls},
            "learning_runs":[{"kind":"reflector","usage":{"known":shim.usage_known,"prompt_tokens":reflect_info.get("prompt_num_tokens",0),"completion_tokens":reflect_info.get("response_num_tokens",0),"calls":1}},
                              {"kind":"curator","usage":{"known":shim.usage_known,"prompt_tokens":curate_info.get("prompt_num_tokens",0),"completion_tokens":curate_info.get("response_num_tokens",0),"calls":1}}],
            "next_global_id":next_id, "operations":operations}
    raise RuntimeError(f"不支持的 ACE worker action: {req['action']}")

if __name__ == "__main__":
    request_path, response_path = sys.argv[1:3]
    try:
        result = main(json.loads(Path(request_path).read_text(encoding="utf-8")))
        Path(response_path).write_text(json.dumps({"ok":True,"result":result}, ensure_ascii=False), encoding="utf-8")
    except Exception as exc:
        Path(response_path).write_text(json.dumps({"ok":False,"error":f"{type(exc).__name__}: {exc}"}, ensure_ascii=False), encoding="utf-8")
