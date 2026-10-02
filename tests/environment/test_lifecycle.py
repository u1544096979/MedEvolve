"""Lifecycle checks use a fake Docker backend; live image verification is opt-in."""

import copy
import json

import httpx
import pytest

from contracts import EnvironmentConfig
from environments import create_environment, release_environment
from environments.errors import EnvironmentUnavailable
from server.app import create_app
from server.docker import DockerCommandError, DockerManager


@pytest.fixture
def docker_backend(monkeypatch):
    containers = {}
    commands = []

    async def run(*args):
        commands.append(args)
        if args[:2] == ("image", "inspect"):
            return json.dumps([{"Id": "sha256:seed", "Config": {"Env": []}}])
        if args[0] == "run":
            name = args[args.index("--name") + 1]
            port = 20000 + len(containers)
            containers[name] = {
                "Config": {"Labels": {"medevolve.managed": "true"}},
                "NetworkSettings": {"Ports": {"8080/tcp": [{"HostPort": str(port)}]}},
                "Mounts": [{"Type": "volume", "Name": name + "-data"}],
                "data": copy.deepcopy({"Patient/a": {"id": "a"}}),
            }
            return name
        if args[:2] == ("container", "inspect"):
            if args[-1] not in containers:
                raise DockerCommandError("No such container")
            return json.dumps([containers[args[-1]]])
        if args[:2] == ("container", "rm"):
            containers.pop(args[-1], None)
            return args[-1]
        raise AssertionError(args)

    async def ready(self, port):
        return None

    monkeypatch.setattr("server.docker.docker_command", run)
    monkeypatch.setattr(DockerManager, "_wait_ready", ready)
    return containers, commands


@pytest.mark.asyncio
async def test_manager_client_lifecycle_isolated_storage_and_release(
    docker_backend, monkeypatch
):
    containers, commands = docker_backend
    app = create_app(DockerManager("remote.example"))
    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kwargs: real_client(transport=httpx.ASGITransport(app=app), **kwargs),
    )
    config = EnvironmentConfig(
        name="PhysicianBench", manager_url="http://manager", seed_data="fhir-full:v1"
    )
    first = await create_environment(config)
    second = await create_environment(config)
    first_store = containers[f"medevolve-{first.instance_id}"]
    second_store = containers[f"medevolve-{second.instance_id}"]
    assert first.endpoint != second.endpoint
    assert first_store["Mounts"][0]["Name"] != second_store["Mounts"][0]["Name"]
    first_store["data"]["Patient/a"]["active"] = False
    assert "active" not in second_store["data"]["Patient/a"]
    for run in (c for c in commands if c[0] == "run"):
        assert run[-1] == "sha256:seed"
        assert "--pull=never" in run
        assert "--volume" not in run and "--mount" not in run
    await release_environment(config, first)
    assert f"medevolve-{first.instance_id}" not in containers
    assert f"medevolve-{second.instance_id}" in containers
    await release_environment(
        config, first
    )  # Release remains safe after a retry/restart.
    await release_environment(config, second)
    assert not containers
    assert all("--volumes" in c for c in commands if c[:2] == ("container", "rm"))


@pytest.mark.asyncio
async def test_startup_failure_releases_partial_container(docker_backend, monkeypatch):
    async def fail(self, port):
        raise TimeoutError("FHIR startup timed out")

    monkeypatch.setattr(DockerManager, "_wait_ready", fail)
    config = EnvironmentConfig(
        name="PhysicianBench", manager_url="http://manager", seed_data="image"
    )
    with pytest.raises(TimeoutError):
        await DockerManager("host").create(config)
    assert not docker_backend[0]


@pytest.mark.asyncio
async def test_unavailable_adapter_and_foreign_container_not_modified(docker_backend):
    manager = DockerManager("host")
    config = EnvironmentConfig(
        name="MedAgentBench", manager_url="http://manager", seed_data="image"
    )
    with pytest.raises(EnvironmentUnavailable):
        await manager.create(config)
    assert not docker_backend[1]
    foreign_id = "00000000-0000-0000-0000-000000000001"
    docker_backend[0][f"medevolve-{foreign_id}"] = {"Config": {"Labels": {}}}
    with pytest.raises(ValueError, match="not managed"):
        await manager.release(foreign_id)
    assert f"medevolve-{foreign_id}" in docker_backend[0]
