import importlib.util
import json
import sys
import types

import pytest


def _load_worker(monkeypatch, tmp_path, *, frozen):
    calls = {}
    hermes_home = tmp_path / "hermes-home"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    class Entry:
        def __init__(self, handler): self.handler = handler

    class Registry:
        def __init__(self):
            self.entries = {"memory": Entry(lambda args, **kwargs: "memory-read"),
                            "skill_manage": Entry(lambda args, **kwargs: "skill-write")}
        def register(self, name, **kwargs):
            schema = kwargs["schema"]
            assert schema["name"] == name
            assert isinstance(schema["description"], str)
            assert schema["parameters"]["type"] == "object"
            self.entries[name] = Entry(kwargs["handler"])
        def get_entry(self, name): return self.entries.get(name)
        def get_tool_names_for_toolset(self, toolset): return list(self.entries)

    registry = Registry()

    class FakeAgent:
        def __init__(self, **kwargs):
            calls["agent"] = kwargs
            calls["env_home"] = __import__("os").environ.get("HERMES_HOME")

        def run_conversation(self, message, conversation_history, task_id):
            calls["message"] = message
            calls["history"] = conversation_history
            calls["task_id"] = task_id
            calls["memory_read"] = registry.get_entry("memory").handler({"action": "read"})
            calls["memory_write"] = registry.get_entry("memory").handler({"action": "add"})
            calls["skill_write"] = registry.get_entry("skill_manage").handler({"action": "create"})
            return {"final_response": "答复", "messages": [{"role": "assistant", "content": "答复"}],
                    "completed": True, "failed": False, "usage": None}

    run_agent = types.ModuleType("run_agent")
    run_agent.AIAgent = FakeAgent
    tools = types.ModuleType("tools")
    tools.__path__ = []
    registry_module = types.ModuleType("tools.registry")
    registry_module.registry = registry
    monkeypatch.setitem(sys.modules, "run_agent", run_agent)
    monkeypatch.setitem(sys.modules, "tools", tools)
    monkeypatch.setitem(sys.modules, "tools.registry", registry_module)

    request = tmp_path / "request.json"
    response = tmp_path / "response.json"
    request.write_text(json.dumps({
        "action": "execute", "model": {"model": "fake", "api_key_env": "TEST_KEY"},
        "session_id": "session-1", "message": "任务", "history": [],
        "bridge_url": "http://127.0.0.1:1", "tool_session": "tool-1",
        "client_id": "client-1", "frozen": frozen,
    }), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["hermes_worker.py", str(request), str(response)])
    spec = importlib.util.spec_from_file_location("hermes_worker_test", "harnesses/runners/hermes_worker.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main()
    return calls, json.loads(response.read_text())


def test_hermes_worker_registers_medical_tools_and_preserves_read_access(monkeypatch, tmp_path):
    calls, result = _load_worker(monkeypatch, tmp_path, frozen=True)
    assert calls["env_home"] == str(tmp_path / "hermes-home")
    assert calls["agent"]["enabled_toolsets"] == ["medevolve", "memory", "skills"]
    assert calls["agent"]["skip_memory"] is False
    assert calls["agent"]["skip_context_files"] is True
    assert calls["memory_read"] == "memory-read"
    assert "read-only" in calls["memory_write"]
    assert "read-only" in calls["skill_write"]
    assert result["final_response"] == "答复"


def test_hermes_worker_keeps_native_memory_writes_during_training(monkeypatch, tmp_path):
    calls, _ = _load_worker(monkeypatch, tmp_path, frozen=False)
    assert calls["memory_write"] == "memory-read"
    assert calls["skill_write"] == "skill-write"
