"""FHIR-AgentBench 种子校验、双容器实例生命周期与就绪检查。

种子目录包含 `manifest.json` 与 `database.dump`。清单记录固定的数据版本、镜像摘要、
源数据与数据库备份校验值、按类型资源数量，以及用于启动就绪检查的原始资源探针。
每个实例由一套 HAPI 服务、PostgreSQL 数据库、私有网络与独立数据卷组成，
从同一份种子恢复，互不共享可写数据库。
"""

import asyncio
import hashlib
import json
import re
import secrets
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import httpx

from contracts import EnvironmentConfig, EnvironmentInstance
from environments.errors import EnvironmentUnavailable

SEED_SCHEMA_VERSION = 1
SEED_DATA_VERSION = "2.1.0"
SEED_PATIENT_COUNT = 100

HAPI_IMAGE = (
    "hapiproject/hapi@sha256:4d59628d42b3556b2dcd274d26054a43ba53697ebdae0fd71e6a5475065ed22d"
)
POSTGRES_IMAGE = (
    "postgres@sha256:6b0faa930dc29e5af5af9b50081ad496fad106b85a88ad66aea01f11d4494733"
)

ENVIRONMENT_NAME = "FHIR-AgentBench"
MANAGED_LABEL = "medevolve.managed"
ENVIRONMENT_LABEL = "medevolve.environment"
INSTANCE_LABEL = "medevolve.instance"

DATABASE_USER = "medevolve"
DATABASE_NAME = "medevolve"
DATABASE_PORT = "5432"
HAPI_MEMORY = "3g"
POSTGRES_MEMORY = "1g"
HAPI_CONTAINER_PORT = "8080/tcp"

STARTUP_TIMEOUT = 300.0
RESTORE_TIMEOUT = 1800.0
READY_POLL_INTERVAL = 2.0
HTTP_TIMEOUT = 60.0

HAPI_ENVIRONMENT = (
    "SPRING_DATASOURCE_DRIVERCLASSNAME=org.postgresql.Driver",
    "SPRING_JPA_PROPERTIES_HIBERNATE_DIALECT="
    "ca.uhn.fhir.jpa.model.dialect.HapiFhirPostgresDialect",
    "HAPI_FHIR_FHIR_VERSION=R4",
    "HAPI_FHIR_ENFORCE_REFERENTIAL_INTEGRITY_ON_WRITE=false",
    "HAPI_FHIR_VALIDATION_REQUESTS_ENABLED=false",
    "HAPI_FHIR_VALIDATION_RESPONSES_ENABLED=false",
    "JAVA_TOOL_OPTIONS=-XX:MaxRAMPercentage=60.0",
)

MANIFEST_FIELDS = (
    "schema_version",
    "data_version",
    "hapi_image",
    "postgres_image",
    "database_sha256",
    "source_sha256",
    "resource_counts",
    "probe_source_id",
    "probe_sha256",
)

IMAGE_PATTERN = re.compile(r"^[^\s@]+@sha256:[0-9a-f]{64}$")
DIGEST_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class SeedValidationError(ValueError):
    """种子清单或种子文件不满足固定约定。"""


class FhirAgentBenchError(EnvironmentUnavailable):
    """实例的镜像、恢复、启动或就绪检查失败。"""


def file_digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def resource_digest(resource: dict) -> str:
    """服务端生成的顶层 meta 不参与比较，其余字段按规范化 JSON 计算 sha256。"""

    normalized = {key: value for key, value in resource.items() if key != "meta"}
    payload = json.dumps(
        normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _require_image(manifest: dict, field: str) -> None:
    image = manifest[field]
    if not isinstance(image, str) or not IMAGE_PATTERN.match(image):
        raise SeedValidationError(
            f"{field} 必须为 repository@sha256:digest 形式，实际为 {image!r}"
        )


def _require_digest(manifest: dict, field: str) -> None:
    digest = manifest[field]
    if not isinstance(digest, str) or not DIGEST_PATTERN.match(digest):
        raise SeedValidationError(f"{field} 必须为 sha256 十六进制摘要，实际为 {digest!r}")


def _require_resource_counts(manifest: dict) -> None:
    counts = manifest["resource_counts"]
    if not isinstance(counts, dict) or not counts:
        raise SeedValidationError("resource_counts 必须为非空对象")
    for resource_type, count in counts.items():
        if not isinstance(resource_type, str) or not resource_type:
            raise SeedValidationError(f"resource_counts 键必须为资源类型，实际为 {resource_type!r}")
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise SeedValidationError(
                f"resource_counts[{resource_type}] 必须为正整数，实际为 {count!r}"
            )
    patients = counts.get("Patient")
    if patients != SEED_PATIENT_COUNT:
        raise SeedValidationError(
            f"resource_counts.Patient 必须为 {SEED_PATIENT_COUNT}，实际为 {patients!r}"
        )


def _require_probe_source_id(manifest: dict) -> None:
    probe_source_id = manifest["probe_source_id"]
    if not isinstance(probe_source_id, str) or "/" not in probe_source_id:
        raise SeedValidationError(
            f"probe_source_id 必须为 ResourceType/id，实际为 {probe_source_id!r}"
        )
    resource_type, _, resource_id = probe_source_id.partition("/")
    if not resource_type or not resource_id:
        raise SeedValidationError(
            f"probe_source_id 必须为 ResourceType/id，实际为 {probe_source_id!r}"
        )


def load_seed(path: str) -> dict:
    """读取并校验种子目录，返回固定字段的清单；任一校验失败即抛错。"""

    seed_dir = Path(path)
    manifest_path = seed_dir / "manifest.json"
    if not manifest_path.is_file():
        raise SeedValidationError(f"种子清单不存在：{manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SeedValidationError(f"种子清单不是合法 JSON：{manifest_path}") from exc
    if not isinstance(manifest, dict):
        raise SeedValidationError(f"种子清单必须为对象：{manifest_path}")

    missing = [field for field in MANIFEST_FIELDS if field not in manifest]
    if missing:
        raise SeedValidationError(f"种子清单缺少字段：{', '.join(missing)}")

    if manifest["schema_version"] != SEED_SCHEMA_VERSION:
        raise SeedValidationError(
            f"schema_version 必须为 {SEED_SCHEMA_VERSION}，实际为 {manifest['schema_version']!r}"
        )
    if manifest["data_version"] != SEED_DATA_VERSION:
        raise SeedValidationError(
            f"data_version 必须为 {SEED_DATA_VERSION}，实际为 {manifest['data_version']!r}"
        )
    _require_image(manifest, "hapi_image")
    _require_image(manifest, "postgres_image")
    for field in ("database_sha256", "source_sha256", "probe_sha256"):
        _require_digest(manifest, field)
    _require_resource_counts(manifest)
    _require_probe_source_id(manifest)

    dump_path = seed_dir / "database.dump"
    if not dump_path.is_file():
        raise SeedValidationError(f"数据库备份不存在：{dump_path}")
    if file_digest(dump_path) != manifest["database_sha256"]:
        raise SeedValidationError(
            f"database_sha256 与 database.dump 实际内容不一致：{dump_path}"
        )

    return {field: manifest[field] for field in MANIFEST_FIELDS}


@dataclass(frozen=True)
class InstanceResources:
    """同一实例的四类资源共用一个 UUID。"""

    instance_id: str
    hapi: str
    database: str
    volume: str
    network: str


def instance_resources(instance_id: str) -> InstanceResources:
    """规范化实例标识，并派生 HAPI、数据库、卷与网络名称。"""

    if str(uuid.UUID(instance_id)) != instance_id:
        raise ValueError("instance_id must be a canonical UUID")
    prefix = f"medevolve-{instance_id}"
    return InstanceResources(
        instance_id=instance_id,
        hapi=prefix,
        database=f"{prefix}-db",
        volume=f"{prefix}-data",
        network=f"{prefix}-net",
    )


def _digest_of(reference: str) -> str:
    """取出镜像引用或镜像 ID 中的 sha256 摘要。"""

    return reference.rsplit("sha256:", 1)[-1]


class FhirAgentBenchManager:
    """FHIR-AgentBench 实例：HAPI 与 PostgreSQL 成对容器，从种子恢复。"""

    def __init__(
        self,
        command,
        public_host: str,
        bind_host: str,
        *,
        startup_timeout: float = STARTUP_TIMEOUT,
        restore_timeout: float = RESTORE_TIMEOUT,
        poll_interval: float = READY_POLL_INTERVAL,
        client_factory=None,
    ):
        self.command = command
        self.public_host = public_host
        self.bind_host = bind_host
        self.startup_timeout = startup_timeout
        self.restore_timeout = restore_timeout
        self.poll_interval = poll_interval
        self.client_factory = client_factory

    # --- 创建 ---------------------------------------------------------------

    async def create(self, config: EnvironmentConfig) -> EnvironmentInstance:
        if config.name != ENVIRONMENT_NAME:
            raise EnvironmentUnavailable(f"{config.name} is unavailable")
        if not config.seed_data or config.seed_data.startswith("-"):
            raise ValueError(
                "seed_data must be the absolute path of a prepared seed directory"
            )
        manifest = load_seed(config.seed_data)
        resources = instance_resources(str(uuid.uuid4()))
        password = secrets.token_urlsafe(32)
        try:
            await self._verify_images(manifest)
            await self._create_network(resources)
            await self._create_volume(resources)
            await self._start_database(resources, password)
            await self._wait_for_database(resources)
            await self._restore_seed(
                resources, Path(config.seed_data) / "database.dump", password
            )
            await self._start_hapi(resources, password)
            port = await self._hapi_port(resources)
            endpoint = f"http://{self.public_host}:{port}/fhir/"
            await self._wait_until_ready(endpoint, manifest, password)
        except BaseException as error:
            await self._reclaim(resources, error)
        return EnvironmentInstance(
            instance_id=resources.instance_id,
            endpoint=endpoint,
            environment=ENVIRONMENT_NAME,
        )

    async def _verify_images(self, manifest: dict) -> None:
        """启动前确认本地镜像与种子清单记录的是同一摘要。"""

        for field in ("hapi_image", "postgres_image"):
            reference = manifest[field]
            try:
                result = await self.command("image", "inspect", reference)
            except EnvironmentUnavailable as exc:
                raise FhirAgentBenchError(
                    f"镜像不可用，请先在本机加载：{reference}"
                ) from exc
            try:
                inspected = json.loads(result)[0]
            except (ValueError, IndexError) as exc:
                raise FhirAgentBenchError(f"无法读取本地镜像信息：{reference}") from exc
            pinned = {inspected.get("Id"), *(inspected.get("RepoDigests") or [])}
            if not any(
                entry and _digest_of(entry) == _digest_of(reference) for entry in pinned
            ):
                raise FhirAgentBenchError(
                    f"本地镜像与种子清单摘要不一致：{reference}"
                )

    def _labels(self, resources: InstanceResources) -> list[str]:
        return [
            "--label",
            f"{MANAGED_LABEL}=true",
            "--label",
            f"{ENVIRONMENT_LABEL}={ENVIRONMENT_NAME}",
            "--label",
            f"{INSTANCE_LABEL}={resources.instance_id}",
        ]

    async def _create_network(self, resources: InstanceResources) -> None:
        await self.command(
            "network", "create", *self._labels(resources), resources.network
        )

    async def _create_volume(self, resources: InstanceResources) -> None:
        await self.command("volume", "create", *self._labels(resources), resources.volume)

    async def _start_database(self, resources: InstanceResources, password: str) -> None:
        await self.command(
            "run",
            "--detach",
            "--pull=never",
            "--name",
            resources.database,
            *self._labels(resources),
            "--network",
            resources.network,
            "--memory",
            POSTGRES_MEMORY,
            "--memory-swap",
            POSTGRES_MEMORY,
            "--env",
            f"POSTGRES_USER={DATABASE_USER}",
            "--env",
            f"POSTGRES_DB={DATABASE_NAME}",
            "--env",
            f"POSTGRES_PASSWORD={password}",
            "--volume",
            f"{resources.volume}:/var/lib/postgresql/data",
            POSTGRES_IMAGE,
        )

    async def _start_hapi(self, resources: InstanceResources, password: str) -> None:
        await self.command(
            "run",
            "--detach",
            "--pull=never",
            "--name",
            resources.hapi,
            *self._labels(resources),
            "--network",
            resources.network,
            "--publish",
            f"{self.bind_host}::8080",
            "--memory",
            HAPI_MEMORY,
            "--memory-swap",
            HAPI_MEMORY,
            "--env",
            f"SPRING_DATASOURCE_URL=jdbc:postgresql://"
            f"{resources.database}:{DATABASE_PORT}/{DATABASE_NAME}",
            "--env",
            f"SPRING_DATASOURCE_USERNAME={DATABASE_USER}",
            "--env",
            f"SPRING_DATASOURCE_PASSWORD={password}",
            *(
                argument
                for setting in HAPI_ENVIRONMENT
                for argument in ("--env", setting)
            ),
            HAPI_IMAGE,
        )

    async def _wait_for_database(self, resources: InstanceResources) -> None:
        deadline = time.monotonic() + self.startup_timeout
        while True:
            try:
                await self.command(
                    "exec",
                    resources.database,
                    "pg_isready",
                    "-U",
                    DATABASE_USER,
                    "-d",
                    DATABASE_NAME,
                )
                return
            except EnvironmentUnavailable as exc:
                if time.monotonic() >= deadline:
                    raise FhirAgentBenchError(
                        f"数据库未在 {self.startup_timeout} 秒内就绪"
                    ) from exc
                await asyncio.sleep(self.poll_interval)

    async def _restore_seed(
        self, resources: InstanceResources, dump_path: Path, password: str
    ) -> None:
        """从 stdin 流式送入种子备份；不用普通命令结果承载二进制内容。"""

        args = (
            "docker",
            "exec",
            "-i",
            resources.database,
            "pg_restore",
            "--no-owner",
            "--no-acl",
            "--no-password",
            "-U",
            DATABASE_USER,
            "-d",
            DATABASE_NAME,
        )
        with dump_path.open("rb") as dump:
            process = await asyncio.create_subprocess_exec(
                *args,
                stdin=dump,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                _, stderr = await asyncio.wait_for(
                    process.communicate(), timeout=self.restore_timeout
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.communicate()
                raise FhirAgentBenchError(
                    f"数据库恢复超过 {self.restore_timeout} 秒"
                ) from None
            except asyncio.CancelledError:
                process.kill()
                await process.communicate()
                raise
        if process.returncode:
            detail = _redact(stderr.decode(errors="replace").strip(), password)
            raise FhirAgentBenchError(f"数据库种子恢复失败：{detail}")

    async def _hapi_port(self, resources: InstanceResources) -> int:
        try:
            inspected = json.loads(
                await self.command("container", "inspect", resources.hapi)
            )[0]
        except (ValueError, IndexError) as exc:
            raise FhirAgentBenchError("无法读取 FHIR 服务容器信息") from exc
        bindings = (inspected.get("NetworkSettings", {}).get("Ports") or {}).get(
            HAPI_CONTAINER_PORT
        )
        if not bindings:
            raise FhirAgentBenchError("FHIR 服务容器没有映射宿主端口")
        return int(bindings[0]["HostPort"])

    # --- 就绪检查 -----------------------------------------------------------

    async def _wait_until_ready(
        self, endpoint: str, manifest: dict, password: str
    ) -> None:
        deadline = time.monotonic() + self.startup_timeout
        async with self._client() as client:
            while True:
                try:
                    await self._check_ready(client, endpoint, manifest)
                    return
                except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
                    if time.monotonic() >= deadline:
                        raise FhirAgentBenchError(
                            f"FHIR 服务未在 {self.startup_timeout} 秒内就绪："
                            f"{_redact(str(exc), password)}"
                        ) from exc
                    await asyncio.sleep(self.poll_interval)

    def _client(self):
        if self.client_factory is not None:
            return self.client_factory()
        return httpx.AsyncClient(timeout=HTTP_TIMEOUT, trust_env=False)

    async def _check_ready(self, client, endpoint: str, manifest: dict) -> None:
        """三条检查：R4 能力声明、种子患者数量、固定探针的原始字段校验值。"""

        response = await client.get(
            endpoint + "metadata", headers={"Accept": "application/fhir+json"}
        )
        response.raise_for_status()
        capability = response.json()
        if capability.get("resourceType") != "CapabilityStatement":
            raise ValueError("服务未返回能力声明")
        version = str(capability.get("fhirVersion", ""))
        if not version.startswith("4."):
            raise ValueError(f"FHIR 版本不是 R4：{version!r}")

        response = await client.get(
            endpoint + "Patient?_summary=count",
            headers={"Accept": "application/fhir+json"},
        )
        response.raise_for_status()
        expected = manifest["resource_counts"]["Patient"]
        total = response.json().get("total")
        if total != expected:
            raise ValueError(f"患者数量不一致：服务端 {total!r}，种子 {expected!r}")

        response = await client.get(
            endpoint + manifest["probe_source_id"],
            headers={"Accept": "application/fhir+json"},
        )
        response.raise_for_status()
        if resource_digest(response.json()) != manifest["probe_sha256"]:
            raise ValueError(f"探针资源与种子不一致：{manifest['probe_source_id']}")

    # --- 回收 ---------------------------------------------------------------

    async def _reclaim(self, resources: InstanceResources, error: BaseException) -> None:
        """清理失败也要保留原始错误，交给调用方判断实例创建失败原因。"""

        try:
            await asyncio.shield(self.release(resources.instance_id))
        except BaseException as cleanup_error:
            raise error from cleanup_error
        raise error

    async def release(self, instance_id: str) -> None:
        resources = instance_resources(instance_id)
        for name in await self._managed_containers(resources):
            await self.command("container", "rm", "--force", "--volumes", name)
        for name in await self._managed_volumes(resources):
            await self.command("volume", "rm", "--force", name)
        for name in await self._managed_networks(resources):
            await self.command("network", "rm", name)

    def _filters(self, resources: InstanceResources) -> list[str]:
        return [
            "--filter",
            f"label={MANAGED_LABEL}=true",
            "--filter",
            f"label={ENVIRONMENT_LABEL}={ENVIRONMENT_NAME}",
            "--filter",
            f"label={INSTANCE_LABEL}={resources.instance_id}",
        ]

    @staticmethod
    def _owns(labels: dict, resources: InstanceResources) -> bool:
        return (
            labels.get(MANAGED_LABEL) == "true"
            and labels.get(ENVIRONMENT_LABEL) == ENVIRONMENT_NAME
            and labels.get(INSTANCE_LABEL) == resources.instance_id
        )

    async def _managed_containers(self, resources: InstanceResources) -> list[str]:
        identifiers = (
            await self.command(
                "container", "ls", "--all", "--quiet", *self._filters(resources)
            )
        ).split()
        names = []
        for identifier in identifiers:
            inspected = json.loads(
                await self.command("container", "inspect", identifier)
            )[0]
            labels = (inspected.get("Config", {}).get("Labels") or {})
            if self._owns(labels, resources):
                names.append(str(inspected.get("Name") or identifier).lstrip("/"))
        return names

    async def _managed_volumes(self, resources: InstanceResources) -> list[str]:
        listing = await self.command(
            "volume", "ls", "--quiet", *self._filters(resources)
        )
        return [name for name in listing.split() if name]

    async def _managed_networks(self, resources: InstanceResources) -> list[str]:
        listing = await self.command(
            "network", "ls", "--quiet", *self._filters(resources)
        )
        return [name for name in listing.split() if name]


def _redact(message: str, password: str) -> str:
    """容器凭据只存在于创建进程内，任何对外文本都不得携带它。"""

    return message.replace(password, "***") if password else message
