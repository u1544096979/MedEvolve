import asyncio
import hashlib
import json
import sqlite3
import time
from pathlib import Path
import pytest

from contracts import SolverInput
from harnesses.contracts import GroupKey, Snapshot, TrainingFeedback
from harnesses.runners import create_runner


def test_factory_accepts_experiment_config_shape(tmp_path):
    runner = create_runner("openclaw", config={
        "models": {"execution": {"model": "exec", "base_url": "https://api.example/v1", "api_key_env": "KEY"},
                   "learning": {"model": "learn", "base_url": "https://api.example/v1", "api_key_env": "KEY"}},
        "limits": {"execution": {"max_model_turns": 30, "max_tool_calls": 20},
                   "learning": {"max_model_turns": 10, "max_tool_calls": 0},
                   "timeout_seconds": 900, "learning_timeout_seconds": 300},
        "local_harnesses": {"openclaw": {"program_path": "local_sources/openclaw/openclaw.mjs"}},
    }, group_dir=tmp_path / "group", bridge_url="http://127.0.0.1:9000")
    assert runner.config["execution_model"]["model"] == "exec"
    assert runner.config["learning_model"]["model"] == "learn"
    assert runner.config["openclaw_bin"].endswith("local_sources/openclaw/openclaw.mjs")


@pytest.mark.asyncio
async def test_openclaw_starts_with_isolated_configuration_and_rotates_task_sessions(tmp_path, monkeypatch):
    calls = []
    messages = []
    review_calls = 0

    class Process:
        returncode = None
        async def wait(self): self.returncode = 0
        def terminate(self): self.returncode = 0
        def kill(self): self.returncode = 0

    async def fake_process(*args, **kwargs): return Process()

    async def fake_command(command, *, group_dir, group_env, timeout):
        calls.append((list(command), dict(group_env), timeout))
        if "agent" in command:
            message_path=command[command.index("--message-file")+1]
            messages.append(Path(message_path).read_text(encoding="utf-8"))
            return 0, '{"status":"ok","result":{"payloads":[{"text":"诊断答案"}],"meta":{"durationMs":12,"agentMeta":{"usage":{"input":12,"output":4},"assistantTurns":10,"toolCalls":2}}}}', ""
        return 0, "", ""

    def review_reader():
        nonlocal review_calls
        review_calls += 1
        if review_calls == 1:
            return None
        return {"attemptedAtMs": int(time.time() * 1000) + 1000, "outcome": "proposed",
                "proposalId": "proposal-1", "usage": {"inputTokens": 20, "outputTokens": 4}}

    runner = create_runner("openclaw", config={
        "execution_model": {"model": "model-x", "base_url": "https://api.example/v1", "api_key_env": "TEST_KEY"},
        "learning_model": {"model": "model-y", "base_url": "https://api.example/v1", "api_key_env": "TEST_KEY"},
        "openclaw_bin": "/opt/openclaw/bin/openclaw", "gateway_port": 19991,
        "command_runner": fake_command, "process_factory": fake_process,
        "review_reader": review_reader, "review_poll_seconds": 0,
    }, group_dir=tmp_path / "group", bridge_url="http://127.0.0.1:9000")
    runner._wait_gateway = lambda: asyncio.sleep(0)
    await runner.start(GroupKey(run_id="run", environment="PhysicianBench", harness="openclaw", source_method="medevolve", seed=42), Snapshot(path=tmp_path, digest="x"))
    first = await runner.execute(SolverInput(instruction="任务一", patient_id="p1", decision_point="药物调整"), "tool-1")
    session = runner.current_session_id
    learned = await runner.learn(TrainingFeedback(task_id="task-1", completed=True, failure_category=None, reason="reason", reference_answer="secret", evidence=[]))
    assert runner.current_session_id == session
    assert learned.learning_status == "updated"
    assert learned.run.usage["model"] == "model-x"
    assert learned.learning_runs[0].usage["known"] is True
    await runner.execute(SolverInput(instruction="任务二", patient_id="p2"), "tool-2")
    assert runner.current_session_id != session
    assert first.output == {"answer": "诊断答案"}
    assert first.usage["known"] is True
    assert "决策点：药物调整" in messages[0]
    assert all("modelgateway" not in str(call) for call in calls)
    await runner.close()


@pytest.mark.asyncio
async def test_openclaw_completed_none_does_not_learn(tmp_path):
    runner = create_runner("openclaw", config={}, group_dir=tmp_path, bridge_url="http://127.0.0.1:9000")
    result = await runner.learn(TrainingFeedback(task_id="t", completed=None, failure_category="tool_error", reason="error", reference_answer="secret", evidence=[]))
    assert result.learning_status == "failed"
    assert result.learning_runs == []


def test_openclaw_reads_actual_workspace_review_state_key(tmp_path):
    runner = create_runner("openclaw", config={}, group_dir=tmp_path, bridge_url="http://127.0.0.1:9000")
    runner.workspace.mkdir(parents=True)
    db_path = runner.state_dir / "state/openclaw.sqlite"
    db_path.parent.mkdir(parents=True)
    workspace_key = hashlib.sha256(str(runner.workspace.resolve()).encode()).hexdigest()
    state = {"lastResult": {"experienceReviews": {workspace_key: {
        "attemptedAtMs": 1234567890, "outcome": "proposed", "proposalId": "p-1",
    }}}}
    with sqlite3.connect(db_path) as db:
        db.execute("CREATE TABLE config_machine_state (state_key TEXT PRIMARY KEY, value_json TEXT NOT NULL)")
        db.execute("INSERT INTO config_machine_state VALUES (?, ?)", ("skills.curatorState", json.dumps(state)))
    assert runner._read_review_status() == state["lastResult"]["experienceReviews"][workspace_key]


def test_openclaw_accepts_flattened_agent_exec_final_response():
    assert create_runner("openclaw", config={}, group_dir=".", bridge_url="http://127.0.0.1")._answer(
        {"status": "ok", "result": {"final": "flattened answer", "meta": {"agentMeta": {"assistantTurns": 10}}}}
    ) == "flattened answer"
