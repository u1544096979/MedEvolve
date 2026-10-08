"""Hermes 隔离解释器入口；在导入 upstream 前设置 HERMES_HOME。"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HERMES_HOME", os.path.abspath(os.environ.get("HERMES_HOME", ".hermes")))


def main():
    from run_agent import AIAgent
    from tools.registry import registry
    import urllib.request

    request_path, response_path = sys.argv[1:3]
    req = json.loads(open(request_path, encoding="utf-8").read())
    hermes_home = Path(os.environ["HERMES_HOME"])
    hermes_home.mkdir(parents=True, exist_ok=True)
    # Keep imported Hermes automation local to this experiment and turn off
    # detached review threads whose usage cannot be joined to this run.
    (hermes_home / "config.yaml").write_text(
        "memory:\n  nudge_interval: 0\nskills:\n  creation_nudge_interval: 0\n",
        encoding="utf-8",
    )
    client = req["client_id"]
    session = req["tool_session"]
    bridge = req["bridge_url"].rstrip("/")

    def bridge_call(path, payload=None):
        headers = {"x-medevolve-client": client, "content-type": "application/json"}
        data = json.dumps(payload).encode() if payload is not None else None
        call = urllib.request.Request(bridge + path, data=data, headers=headers,
                                      method="POST" if data is not None else "GET")
        with urllib.request.urlopen(call, timeout=60) as response:
            return json.loads(response.read())

    def tools_list(args, **kwargs):
        return json.dumps(bridge_call(f"/sessions/{session}/tools"), ensure_ascii=False)

    def tool_call(args, **kwargs):
        result = bridge_call(f"/sessions/{session}/calls", {
            "name": args.get("name"), "arguments": args.get("arguments", {})})
        return json.dumps(result, ensure_ascii=False)

    registry.register(name="medical_tools_list", toolset="medevolve",
        schema={"name":"medical_tools_list", "description":"列出当前医疗会话允许使用的工具",
                "parameters":{"type":"object", "properties":{}, "required":[]}}, handler=tools_list,
        override=True, description="列出当前医疗会话允许使用的工具")
    registry.register(name="medical_tool_call", toolset="medevolve",
        schema={"name":"medical_tool_call", "description":"调用当前医疗会话中的一个工具",
                "parameters":{"type":"object", "properties":{"name":{"type":"string"},"arguments":{"type":"object"}},"required":["name","arguments"]}},
        handler=tool_call, override=True, description="调用当前医疗会话中的一个工具")

    frozen = bool(req.get("frozen"))
    if frozen:
        # 保留 memory/skills 读取接口，只拒绝有状态写操作。
        def make_guard(original):
            def guarded(args, **kwargs):
                action = str(args.get("action", "")).lower()
                if action in {"read", "list", "view", "search", "get"}:
                    return original(args, **kwargs)
                return json.dumps({"success": False, "error": "Frozen Hermes experience is read-only."})
            return guarded
        for name in ("memory", "skill_manage"):
            entry = registry.get_entry(name)
            if entry is not None:
                entry.handler = make_guard(entry.handler)

    model = req["model"]
    agent = AIAgent(model=model["model"], base_url=model.get("base_url"),
        api_key=os.environ.get(model.get("api_key_env", "")), provider="openai",
        enabled_toolsets=["medevolve", "memory", "skills"], session_id=req["session_id"],
        skip_context_files=True, skip_memory=False, quiet_mode=True, save_trajectories=False,
        max_iterations=int(req.get("max_iterations", 30)))
    result = agent.run_conversation(req["message"], conversation_history=req.get("history") or [],
                                    task_id=req["session_id"])
    result = result if isinstance(result, dict) else {}
    api_calls = result.get("api_calls")
    token_fields = ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens")
    usage_known = isinstance(api_calls, int) and api_calls > 0 and all(
        isinstance(result.get(field), int) for field in token_fields
    )
    usage = {
        "known": usage_known,
        "model": result.get("model", model.get("model")),
        "api_calls": api_calls,
        **{field: result.get(field) for field in token_fields},
        "estimated_cost_usd": result.get("estimated_cost_usd"),
        "cost_status": result.get("cost_status"),
    }
    trace = []
    for message in result.get("messages", []):
        if not isinstance(message, dict):
            continue
        for tool_call in message.get("tool_calls", []) or []:
            name = tool_call.get("function", {}).get("name") if isinstance(tool_call, dict) else None
            if name:
                trace.append({"type": "tool_call", "name": name})
        if message.get("role") == "tool":
            trace.append({"type": "tool_output", "name": message.get("name") or message.get("tool_name")})
    response = {"final_response": result.get("final_response"), "messages": result.get("messages", []),
                "completed": result.get("completed"), "failed": result.get("failed"),
                "error": result.get("error") or result.get("turn_exit_reason"),
                "usage": usage, "trace": trace, "api_calls": api_calls}
    with open(response_path, "w", encoding="utf-8") as stream:
        json.dump(response, stream, ensure_ascii=False)


if __name__ == "__main__":
    main()
