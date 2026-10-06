"""Two-layer isolation: per-group state, per-attempt scratch and container scope.

Docker is simulated, so these tests verify the arguments the isolation layer
produces and the mount visibility it guarantees, not a real container runtime.
"""

import asyncio
from pathlib import Path

import pytest

from harnesses.contracts import Snapshot
from harnesses.isolation import (
    IsolationError, HarnessIsolation, SCRATCH_MOUNT, STATE_MOUNT, state_env,
)

HARNESSES = ("openclaw", "hermes", "ace", "reflexion")


@pytest.mark.asyncio
async def test_database_is_reset_but_group_memory_survives(rig):
    rig.group_state.write_text("查询前确认患者ID", encoding="utf-8")
    first = await rig.open_session(patient="p1")
    await rig.write_message(first, "only-first-attempt")
    await rig.close_session(first)
    second = await rig.open_session(patient="p1")
    assert "only-first-attempt" not in await rig.read_messages(second)
    assert rig.group_state.read_text(encoding="utf-8") == "查询前确认患者ID"
    await rig.close_session(second)
    assert rig.active_instances == 0


@pytest.mark.asyncio
async def test_attempts_share_one_group_state_but_never_the_medical_instance(rig):
    first = await rig.open_session(patient="p1")
    first_instance = rig.env.created[0].instance_id
    await rig.write_message(first, "attempt-one")
    await rig.close_session(first)
    second = await rig.open_session(patient="p1")
    second_instance = rig.env.created[1].instance_id
    assert first_instance != second_instance
    assert rig.env.instances[second_instance]["messages"] == []
    assert rig.workspace.state_dir == rig.isolation.state_dir_for(rig.group)
    await rig.close_session(second)


@pytest.mark.asyncio
async def test_group_state_is_not_visible_to_another_group(
    isolation, base_snapshot, group_factory, fake_docker
):
    first = await isolation.start_group(group_factory("ace"), base_snapshot)
    second = await isolation.start_group(group_factory("hermes"), base_snapshot)
    assert first.state_dir != second.state_dir
    first.state_dir.joinpath("memory.txt").write_text("group-a", encoding="utf-8")
    second.state_dir.joinpath("memory.txt").write_text("group-b", encoding="utf-8")
    assert fake_docker.read(first.container_name, f"{STATE_MOUNT}/memory.txt") == "group-a"
    assert fake_docker.read(second.container_name, f"{STATE_MOUNT}/memory.txt") == "group-b"
    other_source = fake_docker.mounts(second.container_name)[STATE_MOUNT]["Source"]
    with pytest.raises(PermissionError):
        fake_docker.read(first.container_name, f"{other_source}/memory.txt")
    assert fake_docker.mounts(first.container_name)[STATE_MOUNT]["Source"] != other_source
    await isolation.close()


@pytest.mark.asyncio
async def test_container_mounts_exclude_personal_data_database_and_socket(
    tmp_path, base_snapshot, group_factory, fake_docker, monkeypatch
):
    monkeypatch.setenv("HOME", str(tmp_path))
    dataset = tmp_path / "project-dataset"
    dataset.mkdir()
    instance = HarnessIsolation(
        run_id="run-1", image="harness:test", network="medevolve-run-1",
        state_root=tmp_path / "states", bridge_url="http://bridge:8080",
        gateway_url="http://gateway:8081", runner=fake_docker,
        forbidden_paths=(dataset,),
    )
    workspace = await instance.start_group(group_factory("openclaw"), base_snapshot)
    mounts = fake_docker.mounts(workspace.container_name)
    assert set(mounts) == {STATE_MOUNT, SCRATCH_MOUNT}
    assert Path(mounts[STATE_MOUNT]["Source"]).is_relative_to(tmp_path / "states")
    assert Path(mounts[SCRATCH_MOUNT]["Source"]).is_relative_to(tmp_path / "docker" / "volumes")
    assert not [mount for mount in mounts.values() if mount["Type"] == "bind"
                and "docker.sock" in str(mount["Source"])]
    environment = fake_docker.containers[workspace.container_name]["env"]
    assert environment["MEDEVOLVE_TOOL_BRIDGE"] == "http://bridge:8080"
    assert environment["MEDEVOLVE_MODEL_GATEWAY"] == "http://gateway:8081"
    assert not [value for value in environment.values() if "fhir" in value.lower()]
    run_command = next(c for c in fake_docker.commands if c[0] == "run")
    assert not [
        arg for arg in run_command
        if "/.openclaw" in str(arg) or "docker.sock" in str(arg)
    ]
    # A personal agent directory and an experiment dataset must never be mount sources.
    personal = tmp_path / ".openclaw"
    personal.mkdir()
    with pytest.raises(IsolationError, match="不允许"):
        instance.prepare_group(
            group_factory("hermes"),
            Snapshot(path=personal, digest="sha256:personal"),
        )
    with pytest.raises(IsolationError, match="不允许"):
        instance.prepare_group(
            group_factory("hermes"), Snapshot(path=dataset, digest="sha256:dataset")
        )
    await instance.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("name", HARNESSES)
async def test_harness_state_paths_are_explicit_and_isolated(
    name, isolation, base_snapshot, group_factory, fake_docker
):
    workspace = await isolation.start_group(group_factory(name), base_snapshot)
    environment = fake_docker.containers[workspace.container_name]["env"]
    assert environment["MEDEVOLVE_WRITABLE_DIR"] == SCRATCH_MOUNT
    assert environment["MEDEVOLVE_HARNESS"] == name
    assert environment["MEDEVOLVE_FROZEN"] == "0"
    for key, value in state_env(name).items():
        assert environment[key] == value
        assert value.startswith("/state/")
    assert workspace.state_dir.name.endswith(name) or name in str(workspace.state_dir)
    await isolation.close()


@pytest.mark.asyncio
async def test_attempt_rotation_uses_a_fresh_volume_and_keeps_group_state(
    isolation, base_snapshot, group_factory, fake_docker
):
    workspace = await isolation.start_group(group_factory("ace"), base_snapshot)
    first_container, first_volume = workspace.container_name, workspace.volume_name
    await isolation.exec(workspace, "write", f"{SCRATCH_MOUNT}/session.log", "attempt one")
    await isolation.exec(workspace, "write", f"{STATE_MOUNT}/memory.txt", "learned")
    assert fake_docker.read(workspace.container_name, f"{SCRATCH_MOUNT}/session.log") == "attempt one"
    workspace = await isolation.start_attempt(workspace)
    assert workspace.container_name != first_container
    assert workspace.volume_name != first_volume
    assert first_volume not in fake_docker.volumes
    assert fake_docker.read(workspace.container_name, f"{STATE_MOUNT}/memory.txt") == "learned"
    assert not Path(
        fake_docker.mounts(workspace.container_name)[SCRATCH_MOUNT]["Source"]
    ).joinpath("session.log").exists()
    labels = fake_docker.containers[workspace.container_name]["Config"]["Labels"]
    assert labels["medevolve.run_id"] == "run-1"
    assert labels["medevolve.harness"] == "ace"
    await isolation.close()


@pytest.mark.asyncio
async def test_frozen_workspace_projects_the_snapshot_read_only(
    isolation, base_snapshot, group_factory, fake_docker
):
    workspace = await isolation.start_group(group_factory("hermes"), base_snapshot, frozen=True)
    mounts = fake_docker.mounts(workspace.container_name)
    assert mounts[STATE_MOUNT]["Source"] == str(base_snapshot.path)
    assert mounts[STATE_MOUNT]["RW"] is False
    assert mounts[SCRATCH_MOUNT]["RW"] is True
    environment = fake_docker.containers[workspace.container_name]["env"]
    assert environment["MEDEVOLVE_FROZEN"] == "1"
    assert environment["MEDEVOLVE_WRITABLE_DIR"] == SCRATCH_MOUNT
    with pytest.raises(PermissionError):
        fake_docker.write(workspace.container_name, f"{STATE_MOUNT}/memory.txt", "overwrite")
    fake_docker.write(workspace.container_name, f"{SCRATCH_MOUNT}/session.log", "log")
    assert workspace.state_readonly is True
    await isolation.close()


@pytest.mark.asyncio
async def test_teardown_removes_only_recorded_run_resources(
    isolation, base_snapshot, group_factory, fake_docker
):
    workspace = await isolation.start_group(group_factory("reflexion"), base_snapshot)
    await fake_docker(
        "run", "--detach", "--name", "medevolve-harness-other-run", "--label",
        "medevolve.managed=true", "--label", "medevolve.run_id=other", "image",
    )
    await fake_docker("volume", "create", "--label", "medevolve.run_id=other", "foreign-volume")
    container, volume = workspace.container_name, workspace.volume_name
    await isolation.close()
    assert container not in fake_docker.containers
    assert volume not in fake_docker.volumes
    assert "medevolve-harness-other-run" in fake_docker.containers
    assert "foreign-volume" in fake_docker.volumes
    assert not [command for command in fake_docker.commands if "prune" in command]
    removals = [
        command for command in fake_docker.commands
        if command[:2] in {("container", "rm"), ("volume", "rm")}
    ]
    assert removals
    for command in removals:
        assert command[-1] in {container, volume}


@pytest.mark.asyncio
async def test_start_failure_removes_partial_container_and_volume(
    isolation, base_snapshot, group_factory, fake_docker
):
    fake_docker.fail_run = "docker run failed: no such image"
    with pytest.raises(IsolationError, match="no such image"):
        await isolation.start_group(group_factory("openclaw"), base_snapshot)
    assert not fake_docker.volumes
    assert not fake_docker.container_names()


@pytest.mark.asyncio
async def test_cancellation_during_start_removes_partial_resources(
    tmp_path, base_snapshot, group_factory, fake_docker
):
    started = asyncio.Event()

    async def runner(*args, stdin=None):
        fake_docker.commands.append(args)
        if args[0] == "run":
            started.set()
            await asyncio.Event().wait()
        return await fake_docker(*args, stdin=stdin)

    instance = HarnessIsolation(
        run_id="run-1", image="harness:test", network="medevolve-run-1",
        state_root=tmp_path / "states", bridge_url="http://bridge:8080",
        gateway_url="http://gateway:8081", runner=runner,
    )
    task = asyncio.create_task(instance.start_group(group_factory("ace"), base_snapshot))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not fake_docker.volumes
    assert not fake_docker.container_names()


@pytest.mark.asyncio
async def test_group_from_another_run_is_refused(isolation, base_snapshot, group_factory, fake_docker):
    with pytest.raises(IsolationError, match="run_id"):
        isolation.prepare_group(group_factory("ace", run_id="other-run"), base_snapshot)
    assert fake_docker.commands == []


@pytest.mark.asyncio
async def test_exec_requires_a_started_container(isolation, base_snapshot, group_factory):
    workspace = isolation.prepare_group(group_factory("ace"), base_snapshot)
    with pytest.raises(IsolationError, match="No such container"):
        await isolation.exec(workspace, "cat", f"{STATE_MOUNT}/memory.txt")
    await isolation.start_attempt(workspace)
    assert await isolation.exec(workspace, "cat", f"{STATE_MOUNT}/memory.txt") == "基座初始经验"
    await isolation.close()
