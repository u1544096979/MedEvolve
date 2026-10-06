"""FHIR-AgentBench 实例生命周期机制测试。

全部使用 fake Docker CLI 后端与 httpx.MockTransport，不连接 Docker、PostgreSQL
或真实 FHIR 服务，也不创建、停止或删除本机任何容器、卷与网络。
"""

import asyncio
import copy
import hashlib
import json
import uuid
from pathlib import Path

import httpx
import pytest

from contracts import EnvironmentConfig, EnvironmentInstance
from environments import (
    FHIR_AGENTBENCH_CREATE_TIMEOUT,
    create_environment,
    release_environment,
)
from environments.errors import EnvironmentUnavailable
from server import fhir_agentbench
from server.docker import DockerCommandError, DockerManager

PROBE = {
    "resourceType": "Observation",
    "id": "o1",
    "status": "final",
    "code": {"text": "Heart rate"},
    "subject": {"reference": "Patient/p001"},
    "effectiveDateTime": "2136-04-22T18:47:00-04:00",
    "valueQuantity": {"value": 80, "unit": "beats/min"},
}

DUMP = b"PG-DUMP-CONTENT"


def options_of(command: tuple[str, ...]) -> dict[str, list[str]]:
    """把 `--key value` 形式的参数收集成字典，位置参数不参与。"""

    options: dict[str, list[str]] = {}
    key = None
    for argument in command[1:]:
        if argument.startswith("--"):
            key = argument
            options.setdefault(key, [])
        elif key is not None:
            options[key].append(argument)
    return options


def labelled(command: tuple[str, ...]) -> dict[str, str]:
    labels = {}
    for position, argument in enumerate(command):
        if argument == "--label":
            key, _, value = command[position + 1].partition("=")
            labels[key] = value
    return labels


def environment_of(command: tuple[str, ...]) -> dict[str, str]:
    values = {}
    for position, argument in enumerate(command):
        if argument == "--env":
            key, _, value = command[position + 1].partition("=")
            values[key] = value
    return values


class FakeDocker:
    """记录 HAPI 容器、数据库容器、卷与网络四类资源的 fake Docker CLI。"""

    def __init__(self):
        self.containers: dict[str, dict] = {}
        self.volumes: dict[str, dict] = {}
        self.networks: dict[str, dict] = {}
        self.commands: list[tuple[str, ...]] = []
        self.database_ready = True
        self.fail_hapi_run = False
        self.image_ids: dict[str, dict] = {}
        self.port = 30000
        # 在标签过滤之外额外返回的标识，用于覆盖列表之后的 inspect 复核。
        self.extra_listings: dict[str, list[str]] = {}
        self.fail_container_remove = False
        self.database_password = ""

    # --- 断言辅助 -----------------------------------------------------------

    @property
    def leftovers(self) -> tuple[dict, dict, dict]:
        return self.containers, self.volumes, self.networks

    def run_command(self, name: str) -> tuple[str, ...]:
        for command in self.commands:
            if command[0] == "run" and command[command.index("--name") + 1] == name:
                return command
        raise AssertionError(f"没有创建容器 {name}：{[c[0] for c in self.commands]}")

    def add_unrelated(self) -> None:
        """加入不属于本次实例、但属于其他环境的资源。"""

        self.containers["physicianbench-other"] = {
            "Name": "/physicianbench-other",
            "Config": {
                "Labels": {
                    "medevolve.managed": "true",
                    "medevolve.environment": "PhysicianBench",
                }
            },
            "NetworkSettings": {"Ports": {}},
        }
        self.volumes["physicianbench-other-data"] = {
            "medevolve.managed": "true",
            "medevolve.environment": "PhysicianBench",
        }
        self.networks["physicianbench-other-net"] = {
            "medevolve.managed": "true",
            "medevolve.environment": "PhysicianBench",
        }

    # --- Docker CLI 模拟 ----------------------------------------------------

    async def __call__(self, *args: str) -> str:
        self.commands.append(args)
        if args[:2] == ("image", "inspect"):
            return json.dumps([self._image(args[2])])
        if args[:2] == ("network", "create"):
            self.networks[args[-1]] = labelled(args)
            return args[-1]
        if args[:2] == ("volume", "create"):
            self.volumes[args[-1]] = labelled(args)
            return args[-1]
        if args[0] == "run":
            return self._run(args)
        if args[:2] == ("container", "inspect"):
            if args[-1] not in self.containers:
                raise DockerCommandError(f"No such container: {args[-1]}")
            return json.dumps([self.containers[args[-1]]])
        if args[:2] == ("container", "rm"):
            if self.fail_container_remove:
                raise DockerCommandError(
                    f"cannot remove container {args[-1]}: "
                    f"POSTGRES_PASSWORD={self.database_password}"
                )
            self.containers.pop(args[-1], None)
            return args[-1]
        if args[:2] == ("container", "ls"):
            return "\n".join(
                self._listed(self.containers, args)
                + self.extra_listings.get("container", [])
            )
        if args[:2] == ("volume", "rm"):
            self.volumes.pop(args[-1], None)
            return args[-1]
        if args[:2] == ("volume", "ls"):
            return "\n".join(self._listed(self.volumes, args))
        if args[:2] == ("network", "rm"):
            self.networks.pop(args[-1], None)
            return args[-1]
        if args[:2] == ("network", "ls"):
            return "\n".join(self._listed(self.networks, args))
        if args[0] == "exec":
            if not self.database_ready:
                raise DockerCommandError("pg_isready: no response")
            return "accepting connections"
        raise AssertionError(args)

    def _image(self, reference: str) -> dict:
        if reference in self.image_ids:
            return self.image_ids[reference]
        digest = reference.rsplit("@sha256:", 1)[-1]
        return {
            "Id": f"sha256:{digest}",
            "RepoDigests": [reference],
            "Config": {"Env": []},
        }

    def _run(self, args: tuple[str, ...]) -> str:
        options = options_of(args)
        name = options["--name"][0]
        env = environment_of(args)
        if "POSTGRES_PASSWORD" in env:
            self.database_password = env["POSTGRES_PASSWORD"]
        if self.fail_hapi_run and args[-1] == fhir_agentbench.HAPI_IMAGE:
            raise DockerCommandError(f"docker run {name}: port is already allocated")
        if "--publish" in options:
            self.port += 1
            ports = {"8080/tcp": [{"HostPort": str(self.port)}]}
        else:
            ports = {}
        self.containers[name] = {
            "Name": f"/{name}",
            "Config": {"Labels": labelled(args)},
            "NetworkSettings": {"Ports": ports},
            "Mounts": [
                {
                    "Type": "volume",
                    "Name": options["--volume"][0].split(":", 1)[0],
                }
            ]
            if "--volume" in options
            else [],
        }
        return name

    def _listed(self, resources: dict[str, dict], args: tuple[str, ...]) -> list[str]:
        filters = {}
        for position, argument in enumerate(args):
            if argument == "--filter":
                kind, _, value = args[position + 1].partition("=")
                assert kind == "label"
                label, _, label_value = value.partition("=")
                filters[label] = label_value
        names = []
        for name, record in resources.items():
            labels = record.get("Config", {}).get("Labels", record)
            if all(labels.get(label) == value for label, value in filters.items()):
                names.append(name)
        return names


@pytest.fixture(autouse=True)
def fake_docker(monkeypatch):
    backend = FakeDocker()

    async def command(*args: str) -> str:
        return await backend(*args)

    monkeypatch.setattr("server.docker.docker_command", command)
    return backend


class FakeProcess:
    def __init__(self, factory, stdin):
        self.factory = factory
        self.stdin = stdin
        self.returncode = None
        self.killed = False
        self.dump = None

    async def communicate(self):
        if self.killed:
            self.returncode = -9
            return b"", b"killed"
        if self.factory.hang:
            await asyncio.sleep(60)
        self.dump = self.stdin.read() if self.stdin is not None else b""
        self.returncode = self.factory.returncode
        return b"", (b"pg_restore: error: could not read input" if self.returncode else b"")

    def kill(self):
        self.killed = True
        self.returncode = -9


class FakeSubprocessFactory:
    """替换 asyncio.create_subprocess_exec，记录恢复命令与 stdin 内容。"""

    def __init__(self):
        self.calls: list[tuple[str, ...]] = []
        self.processes: list[FakeProcess] = []
        self.returncode = 0
        self.hang = False

    async def __call__(self, *args, stdin=None, stdout=None, stderr=None):
        self.calls.append(args)
        process = FakeProcess(self, stdin)
        self.processes.append(process)
        return process


@pytest.fixture(autouse=True)
def fake_subprocess(monkeypatch):
    factory = FakeSubprocessFactory()
    monkeypatch.setattr(asyncio, "create_subprocess_exec", factory)
    return factory


def seed_manifest(dump: bytes = DUMP) -> dict:
    return {
        "schema_version": 1,
        "data_version": "2.1.0",
        "hapi_image": fhir_agentbench.HAPI_IMAGE,
        "postgres_image": fhir_agentbench.POSTGRES_IMAGE,
        "database_sha256": hashlib.sha256(dump).hexdigest(),
        "source_sha256": "c" * 64,
        "resource_counts": {"Patient": 100, "Observation": 3},
        "probe_source_id": "Observation/o1",
        "probe_sha256": fhir_agentbench.resource_digest(PROBE),
    }


@pytest.fixture
def seed(tmp_path) -> Path:
    seed_dir = tmp_path / "seed"
    seed_dir.mkdir()
    (seed_dir / "database.dump").write_bytes(DUMP)
    (seed_dir / "manifest.json").write_text(
        json.dumps(seed_manifest()), encoding="utf-8"
    )
    return seed_dir


class FakeFhir:
    """httpx.MockTransport 后端：metadata、按类型计数与探针读取。"""

    def __init__(self, manifest: dict):
        self.manifest = manifest
        self.capability = True
        self.reachable = True
        self.fhir_version = "4.0.1"
        self.total = manifest["resource_counts"]["Patient"]
        self.probe = copy.deepcopy(PROBE)
        self.requests: list[str] = []

    def factory(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            transport=httpx.MockTransport(self._handle), trust_env=False
        )

    def _handle(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        self.requests.append(url)
        if not self.reachable:
            raise httpx.ConnectError("connection refused", request=request)
        if url.endswith("/metadata"):
            if not self.capability:
                return httpx.Response(200, json={"resourceType": "OperationOutcome"})
            return httpx.Response(
                200,
                json={
                    "resourceType": "CapabilityStatement",
                    "fhirVersion": self.fhir_version,
                },
            )
        if "Patient?" in url:
            return httpx.Response(
                200,
                json={
                    "resourceType": "Bundle",
                    "type": "searchset",
                    "total": self.total,
                },
            )
        if url.endswith(self.manifest["probe_source_id"]):
            return httpx.Response(200, json=self.probe)
        return httpx.Response(404, json={"resourceType": "OperationOutcome"})


def make_manager(backend: FakeDocker, fhir: FakeFhir, **overrides):
    settings = {"startup_timeout": 5.0, "restore_timeout": 5.0, "poll_interval": 0.01}
    settings.update(overrides)
    return fhir_agentbench.FhirAgentBenchManager(
        backend,
        "127.0.0.1",
        "127.0.0.1",
        client_factory=fhir.factory,
        **settings,
    )


def config_for(seed_dir: Path) -> EnvironmentConfig:
    return EnvironmentConfig(
        name="FHIR-AgentBench",
        manager_url="http://manager",
        seed_data=str(seed_dir),
    )


def assert_only_unrelated(backend: FakeDocker) -> None:
    """断言本次创建的四类资源全部回收，其他环境资源未受影响。"""

    assert set(backend.containers) == {"physicianbench-other"}
    assert set(backend.volumes) == {"physicianbench-other-data"}
    assert set(backend.networks) == {"physicianbench-other-net"}


# --- 创建 -------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_provisions_paired_resources_from_seed(
    fake_docker, fake_subprocess, seed
):
    fake_docker.add_unrelated()
    fhir = FakeFhir(seed_manifest())
    manager = make_manager(fake_docker, fhir)

    instance = await manager.create(config_for(seed))

    instance_id = instance.instance_id
    assert str(uuid.UUID(instance_id)) == instance_id
    assert instance.environment == "FHIR-AgentBench"
    hapi = f"medevolve-{instance_id}"
    database = f"{hapi}-db"
    volume = f"{hapi}-data"
    network = f"{hapi}-net"
    assert instance.endpoint == "http://127.0.0.1:30001/fhir/"

    expected_labels = {
        "medevolve.managed": "true",
        "medevolve.environment": "FHIR-AgentBench",
        "medevolve.instance": instance_id,
    }
    database_run = fake_docker.run_command(database)
    hapi_run = fake_docker.run_command(hapi)
    assert labelled(database_run) == expected_labels
    assert labelled(hapi_run) == expected_labels
    assert fake_docker.volumes[volume] == expected_labels
    assert fake_docker.networks[network] == expected_labels

    assert "--publish" not in database_run
    assert database_run[database_run.index("--volume") + 1] == (
        f"{volume}:/var/lib/postgresql/data"
    )
    assert database_run[database_run.index("--memory") + 1] == "1g"
    assert database_run[-1] == fhir_agentbench.POSTGRES_IMAGE
    assert database_run[database_run.index("--network") + 1] == network
    assert hapi_run[hapi_run.index("--publish") + 1] == "127.0.0.1::8080"
    assert hapi_run[hapi_run.index("--memory") + 1] == "3g"
    assert hapi_run[-1] == fhir_agentbench.HAPI_IMAGE
    assert hapi_run[hapi_run.index("--network") + 1] == network

    database_env = environment_of(database_run)
    hapi_env = environment_of(hapi_run)
    assert database_env["POSTGRES_USER"] == "medevolve"
    assert database_env["POSTGRES_DB"] == "medevolve"
    assert hapi_env["SPRING_DATASOURCE_URL"] == (
        f"jdbc:postgresql://{database}:5432/medevolve"
    )
    assert hapi_env["SPRING_DATASOURCE_USERNAME"] == "medevolve"
    assert hapi_env["HAPI_FHIR_FHIR_VERSION"] == "R4"
    assert hapi_env["HAPI_FHIR_ENFORCE_REFERENTIAL_INTEGRITY_ON_WRITE"] == "false"
    assert hapi_env["JAVA_TOOL_OPTIONS"] == "-XX:MaxRAMPercentage=60.0"
    password = database_env["POSTGRES_PASSWORD"]
    assert password and hapi_env["SPRING_DATASOURCE_PASSWORD"] == password
    assert password not in instance.model_dump_json()

    assert fake_subprocess.calls == [
        (
            "docker",
            "exec",
            "-i",
            database,
            "pg_restore",
            "--no-owner",
            "--no-acl",
            "--no-password",
            "-U",
            "medevolve",
            "-d",
            "medevolve",
        )
    ]
    assert fake_subprocess.processes[0].dump == DUMP

    assert any(request.endswith("/fhir/metadata") for request in fhir.requests)
    assert any("Patient?_summary=count" in request for request in fhir.requests)
    assert any(request.endswith("Observation/o1") for request in fhir.requests)


@pytest.mark.asyncio
async def test_database_probe_uses_tcp_to_skip_initdb_temporary_server(
    fake_docker, fake_subprocess, seed
):
    """postgres 镜像入口在 initdb 阶段会临时启动一个只监听 unix socket 的服务器。

    不带 -h 的 pg_isready 会在那个窗口误判就绪，恢复随后撞上真正的服务重启；
    探针必须显式走 TCP，因为只有完成初始化的真实服务才监听 TCP。
    """

    manager = make_manager(fake_docker, FakeFhir(seed_manifest()))

    instance = await manager.create(config_for(seed))

    probe = next(
        command
        for command in fake_docker.commands
        if command[0] == "exec" and "pg_isready" in command
    )
    assert probe == (
        "exec",
        f"medevolve-{instance.instance_id}-db",
        "pg_isready",
        "-h",
        "127.0.0.1",
        "-p",
        fhir_agentbench.DATABASE_PORT,
        "-U",
        fhir_agentbench.DATABASE_USER,
        "-d",
        fhir_agentbench.DATABASE_NAME,
    )


@pytest.mark.asyncio
async def test_create_stops_at_the_worst_case_deadline_and_reclaims(
    fake_docker, fake_subprocess, seed, monkeypatch
):
    """超过最坏路径上限就终止本次创建，已建资源仍全部回收。"""

    fake_docker.add_unrelated()
    fake_subprocess.hang = True
    monkeypatch.setattr(fhir_agentbench, "WORST_CASE_CREATE_SECONDS", 0.05)
    manager = make_manager(fake_docker, FakeFhir(seed_manifest()), restore_timeout=30.0)

    with pytest.raises(EnvironmentUnavailable, match="创建"):
        await manager.create(config_for(seed))

    assert fake_subprocess.processes[0].killed
    assert_only_unrelated(fake_docker)


def test_create_wait_limit_covers_the_managers_worst_case():
    """客户端等待上限严格大于管理器最坏路径，最坏路径覆盖各相位上限之和。"""

    phase_caps = fhir_agentbench.STARTUP_TIMEOUT * 2 + fhir_agentbench.RESTORE_TIMEOUT
    assert fhir_agentbench.WORST_CASE_CREATE_SECONDS >= phase_caps
    assert FHIR_AGENTBENCH_CREATE_TIMEOUT > fhir_agentbench.WORST_CASE_CREATE_SECONDS


@pytest.mark.asyncio
async def test_container_memory_limits_follow_the_shared_environment_variables(
    fake_docker, fake_subprocess, seed, monkeypatch
):
    """运行期容器内存预算与准备期读同一组同名环境变量。"""

    monkeypatch.setenv(fhir_agentbench.HAPI_MEMORY_ENV, "2g")
    monkeypatch.setenv(fhir_agentbench.POSTGRES_MEMORY_ENV, "512m")
    manager = make_manager(fake_docker, FakeFhir(seed_manifest()))

    instance = await manager.create(config_for(seed))

    hapi = fake_docker.run_command(f"medevolve-{instance.instance_id}")
    database = fake_docker.run_command(f"medevolve-{instance.instance_id}-db")
    assert hapi[hapi.index("--memory") + 1] == "2g"
    assert database[database.index("--memory") + 1] == "512m"


@pytest.mark.asyncio
async def test_create_rejects_image_digest_mismatch(fake_docker, fake_subprocess, seed):
    fake_docker.add_unrelated()
    fake_docker.image_ids[fhir_agentbench.POSTGRES_IMAGE] = {
        "Id": "sha256:" + "0" * 64,
        "RepoDigests": ["postgres@sha256:" + "0" * 64],
        "Config": {"Env": []},
    }
    manager = make_manager(fake_docker, FakeFhir(seed_manifest()))

    with pytest.raises(EnvironmentUnavailable, match="摘要"):
        await manager.create(config_for(seed))

    assert_only_unrelated(fake_docker)
    assert fake_subprocess.calls == []


@pytest.mark.asyncio
async def test_create_rejects_missing_seed_directory(fake_docker, fake_subprocess, tmp_path):
    manager = make_manager(fake_docker, FakeFhir(seed_manifest()))

    with pytest.raises(ValueError, match="manifest.json"):
        await manager.create(config_for(tmp_path / "absent"))

    assert fake_docker.commands == []


# --- 失败回收 ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_restore_failure_reclaims_all_created_resources(
    fake_docker, fake_subprocess, seed
):
    fake_docker.add_unrelated()
    fake_subprocess.returncode = 1
    manager = make_manager(fake_docker, FakeFhir(seed_manifest()))

    with pytest.raises(EnvironmentUnavailable, match="pg_restore") as info:
        await manager.create(config_for(seed))

    assert_only_unrelated(fake_docker)
    database_run = next(
        command
        for command in fake_docker.commands
        if command[0] == "run" and command[command.index("--name") + 1].endswith("-db")
    )
    password = environment_of(database_run)["POSTGRES_PASSWORD"]
    chain = [str(info.value), repr(info.value)]
    cause = info.value.__cause__
    while cause is not None:
        chain.extend([str(cause), repr(cause)])
        cause = cause.__cause__
    assert all(password not in text for text in chain)


@pytest.mark.asyncio
async def test_cleanup_failure_keeps_original_error_without_cleanup_exception(
    fake_docker, fake_subprocess, seed
):
    """清理失败时仍以原始错误为失败原因，清理异常只留一条脱敏说明。"""

    fake_docker.add_unrelated()
    fake_subprocess.returncode = 1
    fake_docker.fail_container_remove = True
    manager = make_manager(fake_docker, FakeFhir(seed_manifest()))

    with pytest.raises(EnvironmentUnavailable, match="pg_restore") as info:
        await manager.create(config_for(seed))

    password = fake_docker.database_password
    assert password
    assert info.value.__context__ is None
    assert not isinstance(info.value.__cause__, DockerCommandError)
    chain = [str(info.value), repr(info.value)]
    cause = info.value.__cause__
    while cause is not None:
        chain.extend([str(cause), repr(cause)])
        cause = cause.__cause__
    assert all(password not in text for text in chain)
    notes = getattr(info.value, "__notes__", [])
    assert any("清理失败" in note for note in notes)
    assert all(password not in note for note in notes)


@pytest.mark.asyncio
async def test_service_never_ready_reclaims_all_created_resources(
    fake_docker, fake_subprocess, seed
):
    fake_docker.add_unrelated()
    fhir = FakeFhir(seed_manifest())
    fhir.reachable = False
    manager = make_manager(
        fake_docker, fhir, startup_timeout=0.05, poll_interval=0.005
    )

    with pytest.raises(EnvironmentUnavailable, match="就绪"):
        await manager.create(config_for(seed))

    assert_only_unrelated(fake_docker)


@pytest.mark.asyncio
async def test_hapi_start_failure_reclaims_database_volume_and_network(
    fake_docker, fake_subprocess, seed
):
    fake_docker.add_unrelated()
    fake_docker.fail_hapi_run = True
    manager = make_manager(fake_docker, FakeFhir(seed_manifest()))

    with pytest.raises(EnvironmentUnavailable):
        await manager.create(config_for(seed))

    assert_only_unrelated(fake_docker)


@pytest.mark.asyncio
async def test_restore_timeout_kills_subprocess_and_reclaims(
    fake_docker, fake_subprocess, seed
):
    fake_docker.add_unrelated()
    fake_subprocess.hang = True
    manager = make_manager(fake_docker, FakeFhir(seed_manifest()), restore_timeout=0.05)

    with pytest.raises(EnvironmentUnavailable, match="恢复"):
        await manager.create(config_for(seed))

    assert fake_subprocess.processes[0].killed
    assert_only_unrelated(fake_docker)


@pytest.mark.asyncio
async def test_cancelled_create_kills_subprocess_and_reclaims(
    fake_docker, fake_subprocess, seed
):
    """取消也必须杀掉恢复子进程并回收已创建的资源。"""

    fake_docker.add_unrelated()
    fake_subprocess.hang = True
    manager = make_manager(fake_docker, FakeFhir(seed_manifest()), restore_timeout=30.0)

    task = asyncio.create_task(manager.create(config_for(seed)))
    for _ in range(200):
        if fake_subprocess.processes:
            break
        await asyncio.sleep(0.01)
    assert fake_subprocess.processes, "恢复子进程尚未启动"
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert fake_subprocess.processes[0].killed
    assert_only_unrelated(fake_docker)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mutation, expected",
    [
        ({"capability": False}, "就绪"),
        ({"fhir_version": "3.0.2"}, "就绪"),
        ({"total": 99}, "就绪"),
    ],
)
async def test_readiness_requires_capability_count_and_probe(
    fake_docker, fake_subprocess, seed, mutation, expected
):
    fake_docker.add_unrelated()
    fhir = FakeFhir(seed_manifest())
    for key, value in mutation.items():
        setattr(fhir, key, value)
    manager = make_manager(fake_docker, fhir, startup_timeout=0.05, poll_interval=0.005)

    with pytest.raises(EnvironmentUnavailable, match=expected):
        await manager.create(config_for(seed))

    assert_only_unrelated(fake_docker)


@pytest.mark.asyncio
async def test_readiness_rejects_tampered_probe(fake_docker, fake_subprocess, seed):
    fake_docker.add_unrelated()
    fhir = FakeFhir(seed_manifest())
    fhir.probe["valueQuantity"]["value"] = 81
    manager = make_manager(fake_docker, fhir, startup_timeout=0.05, poll_interval=0.005)

    with pytest.raises(EnvironmentUnavailable, match="就绪"):
        await manager.create(config_for(seed))

    assert_only_unrelated(fake_docker)


# --- 释放 -------------------------------------------------------------------


@pytest.mark.asyncio
async def test_release_is_idempotent(fake_docker, fake_subprocess, seed):
    fake_docker.add_unrelated()
    manager = make_manager(fake_docker, FakeFhir(seed_manifest()))
    instance = await manager.create(config_for(seed))

    await manager.release(instance.instance_id)
    removals = [command for command in fake_docker.commands if command[0] in {"container", "volume", "network"} and "rm" in command]
    await manager.release(instance.instance_id)

    assert_only_unrelated(fake_docker)
    assert [
        command
        for command in fake_docker.commands
        if command[0] in {"container", "volume", "network"} and "rm" in command
    ] == removals


@pytest.mark.asyncio
async def test_release_reclaims_when_hapi_container_is_gone(
    fake_docker, fake_subprocess, seed
):
    fake_docker.add_unrelated()
    manager = make_manager(fake_docker, FakeFhir(seed_manifest()))
    instance = await manager.create(config_for(seed))
    fake_docker.containers.pop(f"medevolve-{instance.instance_id}")

    await manager.release(instance.instance_id)

    assert_only_unrelated(fake_docker)


@pytest.mark.asyncio
async def test_release_ignores_resources_without_our_labels(
    fake_docker, fake_subprocess, seed
):
    """列表之外还有一次 inspect 复核：标签不全的同名容器不得被删除。"""

    fake_docker.add_unrelated()
    manager = make_manager(fake_docker, FakeFhir(seed_manifest()))
    instance = await manager.create(config_for(seed))
    impostor = f"medevolve-{instance.instance_id}"
    fake_docker.containers[impostor]["Config"]["Labels"] = {}
    # 让容器列表无视标签过滤返回它，从而真正走到 inspect 复核分支。
    fake_docker.extra_listings["container"] = [impostor]

    await manager.release(instance.instance_id)

    assert set(fake_docker.containers) == {"physicianbench-other", impostor}
    assert not [
        command
        for command in fake_docker.commands
        if command[:2] == ("container", "rm") and impostor in command
    ]
    assert set(fake_docker.volumes) == {"physicianbench-other-data"}
    assert set(fake_docker.networks) == {"physicianbench-other-net"}


@pytest.mark.asyncio
async def test_release_rejects_non_uuid_instance_id(fake_docker, fake_subprocess):
    manager = make_manager(fake_docker, FakeFhir(seed_manifest()))

    with pytest.raises(ValueError):
        await manager.release("not-a-uuid")

    assert fake_docker.commands == []


# --- 公共分派 ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_docker_manager_dispatches_fhir_agentbench(
    fake_docker, fake_subprocess, seed, monkeypatch
):
    fhir = FakeFhir(seed_manifest())

    def manager_factory(command, public_host, bind_host):
        return fhir_agentbench.FhirAgentBenchManager(
            command,
            public_host,
            bind_host,
            startup_timeout=5.0,
            restore_timeout=5.0,
            poll_interval=0.01,
            client_factory=fhir.factory,
        )

    monkeypatch.setattr("server.docker.FhirAgentBenchManager", manager_factory)
    manager = DockerManager("127.0.0.1", "127.0.0.1")
    config = config_for(seed)

    instance = await manager.create(config)
    assert instance.environment == "FHIR-AgentBench"
    instance_id = instance.instance_id
    assert f"medevolve-{instance_id}-db" in fake_docker.containers
    assert str(seed) not in [argument for command in fake_docker.commands for argument in command]

    await manager.release(instance_id)
    assert not fake_docker.containers
    assert not fake_docker.volumes
    assert not fake_docker.networks


# --- 公共请求等待 -----------------------------------------------------------


def recording_client():
    """替换 httpx.AsyncClient，记录构造参数并返回固定管理服务响应。"""

    calls: list[dict] = []

    class Recording:
        def __init__(self, **kwargs):
            calls.append(kwargs)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return None

        async def post(self, url, json=None):
            body = {
                "instance_id": str(uuid.uuid4()),
                "endpoint": "http://127.0.0.1:12345/fhir/",
                "environment": json["name"],
            }
            return httpx.Response(201, json=body, request=httpx.Request("POST", url))

        async def delete(self, url):
            return httpx.Response(204, request=httpx.Request("DELETE", url))

    return Recording, calls


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "environment", ["FHIR-AgentBench", "PhysicianBench", "MedAgentBench"]
)
async def test_create_environment_waits_for_seed_restore(monkeypatch, environment):
    client, calls = recording_client()
    monkeypatch.setattr(httpx, "AsyncClient", client)

    instance = await create_environment(
        EnvironmentConfig(
            name=environment, manager_url="http://manager", seed_data="seed"
        )
    )

    assert instance.environment == environment
    assert len(calls) == 1
    assert calls[0]["trust_env"] is False
    if environment == "FHIR-AgentBench":
        # 断言覆盖管理器真实最坏路径（见 WORST_CASE_CREATE_SECONDS），而不是
        # 钉死具体秒数：内部上限调整后这条断言不会再次失配。
        assert calls[0]["timeout"] > fhir_agentbench.WORST_CASE_CREATE_SECONDS
    else:
        assert calls[0]["timeout"] == 240


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "environment, expected",
    [("FHIR-AgentBench", 180), ("PhysicianBench", 90), ("MedAgentBench", 90)],
)
async def test_release_environment_waits_for_paired_cleanup(
    monkeypatch, environment, expected
):
    client, calls = recording_client()
    monkeypatch.setattr(httpx, "AsyncClient", client)
    instance = EnvironmentInstance(
        instance_id=str(uuid.uuid4()),
        endpoint="http://127.0.0.1:12345/fhir/",
        environment=environment,
    )

    await release_environment(
        EnvironmentConfig(
            name=environment, manager_url="http://manager", seed_data="seed"
        ),
        instance,
    )

    assert calls == [{"timeout": expected, "trust_env": False}]
