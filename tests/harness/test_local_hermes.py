import pytest
import json

from contracts import SolverInput
from harnesses.contracts import GroupKey, Snapshot, TrainingFeedback
from harnesses.runners import create_runner


@pytest.mark.asyncio
async def test_hermes_rotates_sessions_and_freeze_keeps_readable_experience(tmp_path):
    requests = []

    async def fake_command(command, *, group_dir, group_env, timeout):
        request = json.loads((group_dir / "hermes-request.json").read_text())
        requests.append(request)
        if request["action"] == "learn":
            (tmp_path / "group" / "hermes-home" / "memory" / "MEMORY.md").write_text("learned")
        response = {"final_response": "答复", "messages": [{"role": "assistant", "content": "答复"}],
                    "completed": True, "failed": False, "usage": None, "trace": []}
        (group_dir / "hermes-response.json").write_text(json.dumps(response))
        return 0, "", ""

    runner = create_runner("hermes", config={
        "execution_model": {"model": "m", "base_url": "https://api.example/v1", "api_key_env": "TEST_KEY"},
        "learning_model": {"model": "learn", "base_url": "https://api.example/v1", "api_key_env": "TEST_KEY"},
        "python_path": "/tmp/python", "source_dir": "/tmp/hermes",
        "command_runner": fake_command, "client_id": "client-1",
    }, group_dir=tmp_path / "group", bridge_url="http://127.0.0.1:9000")
    await runner.start(GroupKey(run_id="run", environment="PhysicianBench", harness="hermes", source_method="medevolve", seed=42), Snapshot(path=tmp_path, digest="x"))
    await runner.execute(SolverInput(instruction="任务一", patient_id="p1"), "tool-1")
    first = runner.current_session_id
    await runner.learn(TrainingFeedback(task_id="task-1", completed=True, failure_category=None, reason="reason", reference_answer="secret", evidence=[]))
    assert runner.current_session_id == first
    await runner.execute(SolverInput(instruction="任务二", patient_id="p2"), "tool-2")
    assert runner.current_session_id != first
    assert requests[0]["model"]["model"] == "m"
    assert requests[1]["model"]["model"] == "learn"
    assert "secret" not in json.dumps(requests[0])
    frozen = await runner.freeze(tmp_path / "frozen")
    assert (frozen.path / "skills").is_dir()
    await runner.close()


@pytest.mark.asyncio
async def test_hermes_frozen_run_keeps_read_tools_and_rejects_learning(tmp_path):
    commands = []

    async def fake_command(command, *, group_dir, group_env, timeout):
        commands.append((command, group_env))
        request = json.loads((group_dir / "hermes-request.json").read_text())
        response = {"final_response": "只读答案", "messages": [], "completed": True,
                    "failed": False, "usage": None, "trace": []}
        (group_dir / "hermes-response.json").write_text(json.dumps(response))
        return 0, "", ""

    runner = create_runner("hermes", config={"frozen": True,
        "execution_model": {"model": "m", "base_url": "https://api.example/v1", "api_key_env": "TEST_KEY"},
        "command_runner": fake_command}, group_dir=tmp_path / "group", bridge_url="http://127.0.0.1:9000")
    await runner.start(GroupKey(run_id="run", environment="PhysicianBench", harness="hermes", source_method="medevolve", seed=42), Snapshot(path=tmp_path, digest="x"))
    run = await runner.execute(SolverInput(instruction="任务", patient_id="p1"), "tool-1")
    assert run.status == "completed"
    learning = await runner.learn(TrainingFeedback(task_id="t", completed=True, failure_category=None, reason="r", reference_answer="a", evidence=[]))
    assert learning.learning_status == "failed"
    assert len(commands) == 1
    request = json.loads((tmp_path / "group" / "hermes-request.json").read_text())
    assert request["frozen"] is True
    await runner.close()


@pytest.mark.asyncio
async def test_hermes_completed_none_does_not_learn(tmp_path):
    runner = create_runner("hermes", config={}, group_dir=tmp_path, bridge_url="http://127.0.0.1:9000")
    result = await runner.learn(TrainingFeedback(task_id="t", completed=None, failure_category="tool_error", reason="error", reference_answer="secret", evidence=[]))
    assert result.learning_status == "failed"
    assert result.learning_runs == []
