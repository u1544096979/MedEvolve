"""Harness adapters: native execution, learning, freezing and answer isolation.

The harness's own runtime is substituted (a fake CLI plus the simulated Docker
backend), but everything MedEvolve owns is real here: the container worker code,
the rendered harness config, the write guard, the export filter and the phase
accounting. The one thing these tests cannot prove is that the real OpenClaw
binary behaves as its pinned documentation says; that is what the gated
integration test at the end of this file is for.
"""

import json
import re
import shutil
from pathlib import Path

import pytest
from contracts import SolverInput
from harnesses.contracts import Snapshot, TrainingFeedback
from harnesses.snapshots import directory_digest
from harnesses.worker_protocol import WorkerError
from harnesses.workers.base import WorkerPaths

from tests.harness.conftest import make_experience_snapshot

REPO_ROOT = Path(__file__).resolve().parents[2]
PLUGIN_DIR = REPO_ROOT / "docker" / "harnesses" / "openclaw-tools"
LOCK_PATH = REPO_ROOT / "docker" / "harnesses" / "upstreams.lock.json"
DOCKERFILE = REPO_ROOT / "docker" / "harnesses" / "openclaw.Dockerfile"

REFERENCE_ANSWER = "参考答案：患者对青霉素过敏，应改用阿奇霉素"
EVIDENCE_STATEMENT = "既往病历记录青霉素过敏反应"


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["openclaw"])
async def test_frozen_adapter_reads_without_learning(
    name, adapter_factory, clean_snapshot, group_factory, bridge, fake_docker
):
    adapter = adapter_factory(name, frozen=True)
    await adapter.start(group_factory(name), clean_snapshot)
    before = await adapter.freeze(bridge.temp_dir / "before")
    result = await adapter.execute(
        SolverInput(instruction="查询患者已有用药", patient_id="p1"),
        bridge.session_id,
    )
    after = await adapter.freeze(bridge.temp_dir / "after")
    assert result.status == "completed"
    assert before.digest == after.digest
    assert bridge.learning_model_calls == 0
    await adapter.close()
    assert fake_docker.container_names() == set()
    assert fake_docker.volumes == {}


def feedback(**overrides) -> TrainingFeedback:
    values = {
        "task_id": "p1-task",
        "completed": False,
        "failure_category": "clinical",
        "reason": "缺少过敏史核对",
        "reference_answer": REFERENCE_ANSWER,
        "evidence": [{"source_id": "cond-1", "statement": EVIDENCE_STATEMENT}],
    }
    values.update(overrides)
    return TrainingFeedback(**values)


async def started(adapter_factory, group_factory, snapshot, name="openclaw", frozen=False):
    adapter = adapter_factory(name, frozen=frozen)
    await adapter.start(group_factory(name), snapshot)
    return adapter


def worker_of(adapter):
    return adapter.transport


@pytest.mark.asyncio
async def test_openclaw_runs_a_persistent_gateway_and_native_agent_turns(
    adapter_factory, group_factory, clean_snapshot, fake_harness_cli, fake_docker
):
    cli = fake_harness_cli.for_harness("openclaw")
    adapter = await started(adapter_factory, group_factory, clean_snapshot)
    result = await adapter.execute(
        SolverInput(instruction="查询患者已有用药", patient_id="p1"), "tool-session-1"
    )
    assert result.status == "completed"
    spawn = cli.gateway_spawned[0]
    assert spawn[:2] == ("gateway", "run")
    assert "--allow-unconfigured" in spawn
    assert not [command for command in cli.commands if "--local" in command]
    agent = next(command for command in cli.commands if command[0] == "agent")
    assert agent[:3] == ("agent", "--agent", "medical")
    assert "--session-id" in agent and "--message-file" in agent and "--json" in agent
    install = next(command for command in cli.commands if command[:2] == ("plugins", "install"))
    assert "--link" in install
    config = cli.config(cli.envs[cli.commands.index(agent)])
    assert config["gateway"]["mode"] == "local"
    provider = config["models"]["providers"]["medevolve"]
    assert provider["baseUrl"] == adapter.gateway_base_url == "http://gateway:8081/v1"
    assert provider["request"]["headers"]["x-medevolve-phase"] == "execution"
    assert provider["models"][0]["id"] == "gpt-4o-mini"
    assert config["skills"]["workshop"]["autonomous"]["mode"] == "auto"
    await adapter.close()


@pytest.mark.asyncio
async def test_openclaw_background_learning_not_triggered_stays_not_triggered(
    adapter_factory, group_factory, clean_snapshot, fake_harness_cli
):
    cli = fake_harness_cli.for_harness("openclaw")
    cli.assistant_turns = 3
    adapter = await started(adapter_factory, group_factory, clean_snapshot)
    step = await adapter.learn(feedback())
    assert step.learning_status == "not_triggered"
    assert [run.status for run in step.learning_runs] == ["completed"]
    assert step.run.status == "completed"
    assert "未达到" in (adapter.transport.responses[-1].payload or {}).get("reason", "")
    assert cli.phases[-1] == "learning"
    assert not list((adapter.workspace.state_dir / "openclaw" / "workspace" / "skills").glob("*"))
    await adapter.close()


@pytest.mark.asyncio
async def test_openclaw_learning_waits_for_a_definite_outcome(
    adapter_factory, group_factory, clean_snapshot, fake_harness_cli
):
    cli = fake_harness_cli.for_harness("openclaw")
    cli.assistant_turns = 12
    adapter = await started(adapter_factory, group_factory, clean_snapshot)
    cli.skill_on_learning_turn = (
        "verify-allergy-history", "# 过敏史核对\n先核对过敏史再开药。"
    )
    step = await adapter.learn(feedback())
    assert step.learning_status == "updated"
    assert cli.phases[-1] == "learning"
    exported = await adapter.freeze(adapter.workspace.state_dir.parent.parent / "updated")
    assert (exported.path / "skills" / "verify-allergy-history" / "SKILL.md").is_file()
    await adapter.close()


@pytest.mark.asyncio
async def test_openclaw_learning_with_a_workshop_record_but_no_change_is_unchanged(
    adapter_factory, group_factory, clean_snapshot, fake_harness_cli
):
    cli = fake_harness_cli.for_harness("openclaw")
    cli.assistant_turns = 12
    cli.workshop_records = [{"id": "proposal-1", "status": "pending"}]
    adapter = await started(adapter_factory, group_factory, clean_snapshot)
    step = await adapter.learn(feedback())
    assert step.learning_status == "unchanged"
    await adapter.close()


@pytest.mark.asyncio
async def test_openclaw_learning_timeout_reports_not_triggered(
    adapter_factory, group_factory, clean_snapshot, fake_harness_cli, harness_limits
):
    cli = fake_harness_cli.for_harness("openclaw")
    cli.assistant_turns = 12
    limits = harness_limits.model_copy(update={"learning_timeout_seconds": 1})
    adapter = adapter_factory("openclaw", frozen=False, limits=limits)
    await adapter.start(group_factory("openclaw"), clean_snapshot)
    step = await adapter.learn(feedback())
    assert step.learning_status == "not_triggered"
    assert "未触发" in (adapter.transport.responses[-1].payload or {}).get("reason", "")
    await adapter.close()


@pytest.mark.asyncio
async def test_openclaw_learning_failure_is_reported_not_faked(
    adapter_factory, group_factory, clean_snapshot, fake_harness_cli
):
    cli = fake_harness_cli.for_harness("openclaw")
    adapter = await started(adapter_factory, group_factory, clean_snapshot)
    cli.fail_agent = True
    step = await adapter.learn(feedback())
    assert step.learning_status == "failed"
    assert step.run.status == "failed"
    await adapter.close()


@pytest.mark.asyncio
async def test_frozen_run_cannot_write_into_the_experience_snapshot(
    adapter_factory, group_factory, tmp_path, fake_harness_cli
):
    snapshot = make_experience_snapshot(
        tmp_path / "frozen-state", skill="check-allergy", body="# 过敏史核对\n"
    )
    before = directory_digest(snapshot.path)
    adapter = await started(
        adapter_factory, group_factory, snapshot, frozen=True
    )
    assert adapter.workspace.state_readonly is True
    result = await adapter.execute(
        SolverInput(instruction="查询患者已有用药", patient_id="p1"), None
    )
    assert result.status == "completed"
    assert adapter.transport.responses[-1].payload["skills_read"] == ["check-allergy"]
    assert directory_digest(snapshot.path) == before
    paths = adapter.transport.paths
    with pytest.raises(WorkerError, match="冻结运行禁止写入"):
        paths.assert_writable(paths.workspace / "skills" / "check-allergy" / "SKILL.md")
    with pytest.raises(WorkerError, match="冻结运行禁止写入"):
        paths.assert_writable(paths.durable_root / "openclaw.json")
    assert paths.runtime_state.is_relative_to(paths.writable)
    await adapter.close()


@pytest.mark.asyncio
async def test_frozen_run_refreshes_container_and_session_per_question(
    adapter_factory, group_factory, tmp_path, fake_harness_cli, fake_docker
):
    snapshot = make_experience_snapshot(
        tmp_path / "frozen-state", skill="check-allergy", body="# 过敏史核对\n"
    )
    adapter = await started(adapter_factory, group_factory, snapshot, frozen=True)
    cli = fake_harness_cli.for_harness("openclaw")
    await adapter.execute(SolverInput(instruction="第一题", patient_id="p1"), None)
    first_container = adapter.workspace.container_name
    first_mount = adapter.workspace.state_mount_source
    await adapter.execute(SolverInput(instruction="第二题", patient_id="p2"), None)
    assert adapter.workspace.container_name != first_container
    assert adapter.workspace.state_mount_source == first_mount
    assert cli.sessions[0] != cli.sessions[1]
    # One Gateway for the group, then one per frozen attempt: the container and
    # its session directory are refreshed for every test question.
    assert len(cli.gateway_spawned) == 3
    assert all(phase == "execution" for phase in cli.phases)
    assert fake_docker.mounts(adapter.workspace.container_name)["/state"]["RW"] is False
    await adapter.close()


@pytest.mark.asyncio
async def test_export_excludes_conversations_transcripts_and_answer_caches(
    adapter_factory, group_factory, tmp_path, fake_harness_cli
):
    snapshot = make_experience_snapshot(
        tmp_path / "frozen-state",
        skill="check-allergy",
        body="# 过敏史核对\n",
        decoys={
            "sessions/019.session.jsonl": "原始训练对话",
            "state/openclaw.sqlite": "会话与转写",
            "answers/answer_cache.jsonl": "逐题答案缓存",
            "MEMORY.md": "长期记忆",
        },
    )
    adapter = await started(adapter_factory, group_factory, snapshot, frozen=True)
    exported = await adapter.freeze(tmp_path / "export")
    relative = sorted(
        path.relative_to(exported.path).as_posix()
        for path in exported.path.iterdir()
    )
    assert relative == ["MEMORY.md", "memory", "skills"]
    assert (exported.path / "skills" / "check-allergy" / "SKILL.md").is_file()
    assert not list(exported.path.rglob("*.sqlite"))
    assert not list(exported.path.rglob("*.jsonl"))
    await adapter.close()


@pytest.mark.asyncio
async def test_solver_input_never_carries_reference_material(
    adapter_factory, group_factory, clean_snapshot, fake_harness_cli
):
    cli = fake_harness_cli.for_harness("openclaw")
    cli.assistant_turns = 2
    adapter = await started(adapter_factory, group_factory, clean_snapshot)
    await adapter.execute(
        SolverInput(instruction="查询患者已有用药", patient_id="p1"), None
    )
    await adapter.learn(feedback())
    assert REFERENCE_ANSWER in cli.prompts[1]
    assert EVIDENCE_STATEMENT in cli.prompts[1]
    await adapter.execute(
        SolverInput(instruction="复查患者用药清单", patient_id="p1"), None
    )
    assert REFERENCE_ANSWER not in cli.prompts[2]
    assert EVIDENCE_STATEMENT not in cli.prompts[2]
    assert "过敏" not in cli.prompts[2]
    await adapter.close()


@pytest.mark.asyncio
async def test_tool_session_reaches_the_worker_and_the_bridge(
    adapter_factory, group_factory, clean_snapshot, bridge
):
    session_id = await bridge.open()
    adapter = await started(adapter_factory, group_factory, clean_snapshot)
    await adapter.execute(
        SolverInput(instruction="查询患者已有用药", patient_id="p1"), session_id
    )
    execute = next(
        request for request in adapter.transport.requests if request.action == "execute"
    )
    assert execute.payload["tool_session"] == session_id
    assert execute.payload["message"].count("参考答案") == 0
    assert bridge.env.active == 1
    await adapter.close()


@pytest.mark.asyncio
async def test_worker_reports_phases_to_the_model_recorder(
    adapter_factory, group_factory, clean_snapshot, fake_harness_cli, bridge
):
    cli = fake_harness_cli.for_harness("openclaw")
    cli.turns = [4, 11]
    cli.skill_on_learning_turn = ("phase-accounting", "# 阶段记账\n")
    adapter = await started(adapter_factory, group_factory, clean_snapshot)
    await adapter.execute(SolverInput(instruction="查询患者已有用药", patient_id="p1"), None)
    assert bridge.learning_model_calls == 0
    await adapter.learn(feedback())
    assert bridge.learning_model_calls == 11
    await adapter.close()


@pytest.mark.asyncio
async def test_adapter_registry_reports_a_missing_harness(adapter_factory):
    from harnesses.adapters import AdapterError, create_adapter

    with pytest.raises(AdapterError, match="没有 hermes 的适配器实现"):
        create_adapter("hermes")
    assert adapter_factory("openclaw", frozen=True).name == "openclaw"


@pytest.mark.asyncio
async def test_close_is_idempotent_and_removes_only_recorded_resources(
    adapter_factory, group_factory, clean_snapshot, fake_docker
):
    adapter = await started(adapter_factory, group_factory, clean_snapshot)
    container, volume = adapter.workspace.container_name, adapter.workspace.volume_name
    await adapter.close()
    assert container not in fake_docker.containers
    assert volume not in fake_docker.volumes
    assert not [command for command in fake_docker.commands if "prune" in command]
    await adapter.close()


@pytest.mark.asyncio
async def test_medical_tool_session_is_handed_to_the_plugin(
    adapter_factory, group_factory, clean_snapshot, bridge
):
    session_id = await bridge.open()
    adapter = await started(adapter_factory, group_factory, clean_snapshot)
    await adapter.execute(
        SolverInput(instruction="查询患者已有用药", patient_id="p1"), session_id
    )
    handoff = json.loads(
        adapter.transport.paths.runtime_state.joinpath("tool-session.json").read_text(
            encoding="utf-8"
        )
    )
    assert handoff["session"] == session_id
    assert handoff["bridge"] == adapter.isolation.bridge_url
    assert handoff["client"] == adapter.client_id
    await adapter.close()


def test_plugin_declares_every_tool_it_registers():
    manifest = json.loads(
        (PLUGIN_DIR / "openclaw.plugin.json").read_text(encoding="utf-8")
    )
    source = (PLUGIN_DIR / "index.mjs").read_text(encoding="utf-8")
    registered = re.findall(r"registerTool\(\{[^}]*?name:\s*\"([^\"]+)\"", source, re.S)
    assert registered
    assert sorted(registered) == sorted(manifest["contracts"]["tools"])
    assert "definePluginEntry" in source
    assert "api.registerTool(" in source
    package = json.loads((PLUGIN_DIR / "package.json").read_text(encoding="utf-8"))
    assert package["openclaw"]["extensions"] == ["./index.mjs"]


def test_plugin_forwards_to_the_bridge_without_its_own_fhir_client():
    source = (PLUGIN_DIR / "index.mjs").read_text(encoding="utf-8")
    assert "/sessions/" in source and "/tools" in source and "/calls" in source
    assert "x-medevolve-client" in source
    for forbidden in ("fhir", "Patient/", "Observation?", "MedicationRequest"):
        assert forbidden not in source


def test_dockerfile_pins_the_locked_openclaw_commit():
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    entry = lock["upstreams"]["openclaw"]
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")
    assert entry["commit"] in dockerfile
    assert entry["ref"].split("/")[-1] in dockerfile
    assert entry["repository"] in dockerfile
    assert entry["dependency_lock"] in dockerfile
    instructions = "\n".join(
        line for line in dockerfile.splitlines() if not line.strip().startswith("#")
    )
    assert "latest" not in instructions
    assert "main" not in instructions
    assert entry["image"]["reference"].endswith(entry["commit"][:8])
    assert entry["image"]["verified"] is False


OPENCLAW_CLI = shutil.which("openclaw")


@pytest.mark.skipif(
    OPENCLAW_CLI is None, reason="本机没有 openclaw 可执行文件，无法核对官方配置模式"
)
def test_rendered_config_is_accepted_by_the_pinned_openclaw_cli(tmp_path):
    """UNVERIFIED by the experiment: only the config shape is checked here.

    The CLI validates against its own schema without starting a Gateway and
    without reading the operator's config, because state and config paths point
    at a temporary directory.
    """
    import os
    import subprocess

    from contracts import ModelConfig
    from harnesses.contracts import Budget, HarnessLimits
    from harnesses.workers.openclaw import render_openclaw_config

    limits = HarnessLimits(
        execution=Budget(max_model_turns=12, max_tool_calls=40),
        learning=Budget(max_model_turns=8, max_tool_calls=10),
        max_input_tokens=60000, max_output_tokens=8000, max_total_tokens=200000,
        timeout_seconds=120, learning_timeout_seconds=30,
    )
    model = ModelConfig(
        model="gpt-4o-mini", base_url="http://gateway:8081/v1",
        api_key_env="MEDEVOLVE_GATEWAY_TOKEN", context_window=128000,
        max_output_tokens=4096,
    )
    for frozen in (False, True):
        config = render_openclaw_config(
            workspace=tmp_path / "workspace", model=model, limits=limits,
            gateway_base_url="http://gateway:8081/v1", client_id="run-1:group",
            task_id="p1-abc", phase="execution", frozen=frozen,
        )
        config_path = tmp_path / f"openclaw-{frozen}.json"
        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        completed = subprocess.run(
            [OPENCLAW_CLI, "config", "validate", "--json"],
            capture_output=True, text=True, timeout=180,
            env={
                **os.environ,
                "HOME": str(tmp_path),
                "OPENCLAW_STATE_DIR": str(tmp_path / "state"),
                "OPENCLAW_CONFIG_PATH": str(config_path),
            },
        )
        assert completed.returncode == 0, completed.stderr
        assert json.loads(completed.stdout)["valid"] is True
