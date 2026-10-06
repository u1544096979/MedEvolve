"""FHIR-AgentBench 原始数据校验、按依赖导入与数据库种子准备。

用法：

    python scripts/fhir_agentbench_data.py prepare \
        --data-dir /Users/guagua/data/mimic-iv-fhir-demo/2.1.0/fhir \
        --seed-dir /Users/guagua/data/mimic-iv-fhir-demo/2.1.0/seed

导入在临时准备的 HAPI FHIR R4 + PostgreSQL 容器内完成，随后导出 PostgreSQL 备份并写清单。
准备容器、卷与网络只在本次运行内存在，结束时（含失败路径）全部删除。
"""

import argparse
import asyncio
import gzip
import hashlib
import json
import os
import secrets
import sys
import time
import uuid
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

import httpx

DATA_VERSION = "2.1.0"
EXPECTED_PATIENTS = 100
EXPECTED_TOTAL_RESOURCES = 928_935

# 源数据引用关系：被引用方先导入。
IMPORT_ORDER = (
    "Organization",
    "Location",
    "Patient",
    "Encounter",
    "Medication",
    "Specimen",
    "Condition",
    "MedicationRequest",
    "MedicationAdministration",
    "MedicationStatement",
    "MedicationDispense",
    "Procedure",
    "Observation",
)

BATCH_SIZE = 200
PROGRESS_BATCH_INTERVAL = 20
SEARCH_PAGE_SIZE = 500

HTTP_TIMEOUT = 300.0
DOCKER_TIMEOUT = 60.0
SERVICE_START_TIMEOUT = 300.0
READY_POLL_INTERVAL = 2.0
DATABASE_TIMEOUT = 1800.0

HAPI_IMAGE = (
    "hapiproject/hapi@sha256:4d59628d42b3556b2dcd274d26054a43ba53697ebdae0fd71e6a5475065ed22d"
)
POSTGRES_IMAGE = (
    "postgres@sha256:6b0faa930dc29e5af5af9b50081ad496fad106b85a88ad66aea01f11d4494733"
)
SUPPORTED_ARCHITECTURES = {"arm64", "aarch64"}

MANAGED_LABEL = "medevolve.managed"
ENVIRONMENT_LABEL = "medevolve.environment"
ROLE_LABEL = "medevolve.role"
ENVIRONMENT_NAME = "FHIR-AgentBench"

DEFAULT_HAPI_MEMORY = "3g"
DEFAULT_POSTGRES_MEMORY = "1g"
HAPI_MEMORY_ENV = "MEDEVOLVE_FHIR_HAPI_MEMORY"
POSTGRES_MEMORY_ENV = "MEDEVOLVE_FHIR_PG_MEMORY"

DATABASE_USER = "medevolve"
DATABASE_NAME = "medevolve"
DATABASE_PORT = "5432"

FHIR_HEADERS = {
    "Accept": "application/fhir+json",
    "Content-Type": "application/fhir+json",
}

MANIFEST_NAME = "manifest.json"
DATABASE_DUMP_NAME = "database.dump"
CHECKSUM_LEDGER_NAME = "SHA256SUMS.txt"


class SeedPreparationError(RuntimeError):
    """种子准备流程失败。"""


class SourceDataError(SeedPreparationError):
    """原始数据不满足导入约定。"""


class ResourceImportError(SeedPreparationError):
    """批量写入或全量核对失败。"""


class InventoryMismatchError(ResourceImportError):
    """服务端资源 ID 集合与源数据不一致。"""


SENSITIVE_ENV_HINTS = ("PASSWORD", "SECRET", "TOKEN")


def _sensitive_env_values(args: Sequence[str]) -> tuple[str, ...]:
    """取出 docker 参数里 --env 携带的敏感取值。

    这些取值会被注入容器凭据，绝不能出现在日志或异常文本中，
    因此命令失败时需要先从命令行与命令输出里擦掉。
    """

    values: list[str] = []
    for position, argument in enumerate(args):
        if argument != "--env" or position + 1 >= len(args):
            continue
        key, separator, value = args[position + 1].partition("=")
        if separator and value and any(hint in key.upper() for hint in SENSITIVE_ENV_HINTS):
            values.append(value)
    return tuple(values)


def _redact(text: str, *secrets: str) -> str:
    for secret in secrets:
        if secret:
            text = text.replace(secret, "***")
    return text


# --- 源数据读取 -------------------------------------------------------------


def iter_ndjson(path: Path) -> Iterator[dict]:
    try:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for number, line in enumerate(handle, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise SourceDataError(
                        f"{path.name} 第 {number} 行不是合法 JSON"
                    ) from exc
                if not isinstance(record, dict):
                    raise SourceDataError(f"{path.name} 第 {number} 行不是 JSON 对象")
                yield record
    except (OSError, EOFError, UnicodeDecodeError) as exc:
        raise SourceDataError(f"无法读取 {path.name}：{exc}") from exc


def source_files(data_dir: Path) -> dict[str, list[Path]]:
    """按资源类型归并源文件；每个文件的首条记录决定其类型。"""

    if not data_dir.is_dir():
        raise SourceDataError(f"数据目录不存在：{data_dir}")
    paths = sorted(data_dir.glob("*.ndjson.gz"))
    if not paths:
        raise SourceDataError(f"数据目录下没有 .ndjson.gz 文件：{data_dir}")
    grouped: dict[str, list[Path]] = {}
    for path in paths:
        first = next(iter_ndjson(path), None)
        if first is None:
            raise SourceDataError(f"{path.name} 为空文件")
        resource_type = first.get("resourceType")
        if not isinstance(resource_type, str) or not resource_type:
            raise SourceDataError(f"{path.name} 首条记录缺少 resourceType")
        grouped.setdefault(resource_type, []).append(path)
    return grouped


def iter_resources(
    resource_type: str, paths: Sequence[Path]
) -> Iterator[dict]:
    for path in paths:
        for record in iter_ndjson(path):
            if record.get("resourceType") != resource_type:
                raise SourceDataError(
                    f"{path.name} 混装了 {record.get('resourceType')!r}，"
                    f"按类型导入要求文件只含 {resource_type}"
                )
            yield record


def source_inventory(data_dir: Path) -> dict[str, set[str]]:
    """返回 {resourceType: {resource_id, ...}}；同一 ID 出现两次即报错。"""

    grouped = source_files(data_dir)
    inventory: dict[str, set[str]] = {}
    for resource_type, paths in grouped.items():
        identifiers = inventory.setdefault(resource_type, set())
        for record in iter_resources(resource_type, paths):
            resource_id = record.get("id")
            if not isinstance(resource_id, str) or not resource_id:
                raise SourceDataError(f"{resource_type} 记录缺少 id")
            if resource_id in identifiers:
                raise SourceDataError(f"重复资源 ID：{resource_type}/{resource_id}")
            identifiers.add(resource_id)
    return inventory


def require_known_types(inventory: dict[str, set[str]]) -> None:
    unknown = sorted(set(inventory) - set(IMPORT_ORDER))
    if unknown:
        raise SourceDataError(f"源数据含未定义导入顺序的资源类型：{', '.join(unknown)}")


def require_published_dataset(inventory: dict[str, set[str]]) -> None:
    """确认输入就是已发布的 MIMIC-IV-FHIR Demo 2.1.0 数据集。"""

    require_known_types(inventory)
    total = sum(len(identifiers) for identifiers in inventory.values())
    if total != EXPECTED_TOTAL_RESOURCES:
        raise SourceDataError(
            f"资源总数必须为 {EXPECTED_TOTAL_RESOURCES}，实际为 {total}"
        )
    patients = len(inventory.get("Patient", ()))
    if patients != EXPECTED_PATIENTS:
        raise SourceDataError(f"Patient 数量必须为 {EXPECTED_PATIENTS}，实际为 {patients}")


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def source_digest(entries: Sequence[tuple[str, str]]) -> str:
    payload = "".join(f"{rel} {digest}\n" for rel, digest in sorted(entries))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def hash_dataset(data_dir: Path) -> str:
    """没有发布校验清单时，用全部源文件内容计算校验值。"""

    paths = sorted(data_dir.glob("*.ndjson.gz"))
    if not paths:
        raise SourceDataError(f"数据目录下没有 .ndjson.gz 文件：{data_dir}")
    hasher = hashlib.sha256()
    for path in paths:
        hasher.update(path.name.encode("utf-8"))
        hasher.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                hasher.update(chunk)
    return hasher.hexdigest()


def verify_checksums(data_dir: Path) -> str | None:
    """校验发布清单（存在时），返回 source_sha256；清单缺失时返回 None。"""

    ledger = data_dir.parent / CHECKSUM_LEDGER_NAME
    if not ledger.is_file():
        return None
    entries: list[tuple[str, str]] = []
    for number, raw in enumerate(ledger.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        digest, _, relative = line.partition(" ")
        digest = digest.strip()
        relative = relative.strip()
        if len(digest) != 64 or relative == "":
            raise SourceDataError(f"{ledger.name} 第 {number} 行格式不合法")
        for path in (data_dir.parent / relative, data_dir / relative):
            if path.is_file():
                target = path
                break
        else:
            raise SourceDataError(f"校验清单引用的文件不存在：{relative}")
        if file_sha256(target) != digest:
            raise SourceDataError(f"校验失败：{relative}")
        entries.append((relative, digest))
    if not entries:
        raise SourceDataError(f"{ledger.name} 为空")
    return source_digest(entries)


def resource_source_id(resource: dict) -> str:
    return f"{resource['resourceType']}/{resource['id']}"


def resource_digest(resource: dict) -> str:
    """服务端生成的 meta 不参与比较，其余字段按规范化 JSON 计算 sha256。"""

    normalized = {key: value for key, value in resource.items() if key != "meta"}
    payload = json.dumps(
        normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# --- 批量导入 ---------------------------------------------------------------


def transaction_entry(resource: dict) -> dict:
    return {
        "resource": resource,
        "request": {"method": "PUT", "url": resource_source_id(resource)},
    }


def require_transaction_success(bundle: dict, batch: Sequence[dict] | None = None) -> None:
    """逐条检查 transaction-response；任何一条失败即抛错，并给出源端 ID。"""

    if not isinstance(bundle, dict) or bundle.get("resourceType") != "Bundle":
        raise ResourceImportError("写入响应不是 FHIR Bundle")
    if bundle.get("type") != "transaction-response":
        raise ResourceImportError(f"写入响应类型不是 transaction-response：{bundle.get('type')!r}")
    entries = bundle.get("entry", [])
    if not isinstance(entries, list):
        raise ResourceImportError("写入响应 entry 不是数组")
    if batch is not None and len(entries) != len(batch):
        raise ResourceImportError(
            f"写入响应数量与提交数量不一致：响应 {len(entries)} 条，提交 {len(batch)} 条"
        )
    for position, entry in enumerate(entries):
        status = str(entry.get("response", {}).get("status", ""))
        code = status.split(" ", 1)[0]
        if code not in {"200", "201"}:
            if batch is not None and position < len(batch):
                source_id = resource_source_id(batch[position])
            else:
                source_id = str(entry.get("response", {}).get("location", "未知资源"))
            raise ResourceImportError(
                f"导入失败：{source_id} 返回 {status or '空状态'}；{entry}"
            )


def batches(records, size: int | None = None) -> Iterator[list]:
    size = size or BATCH_SIZE
    current: list = []
    for record in records:
        current.append(record)
        if len(current) == size:
            yield current
            current = []
    if current:
        yield current


async def submit_batch(client: httpx.AsyncClient, endpoint: str, batch: Sequence[dict]) -> None:
    payload = {
        "resourceType": "Bundle",
        "type": "transaction",
        "entry": [transaction_entry(resource) for resource in batch],
    }
    response = await client.post(endpoint, json=payload)
    response.raise_for_status()
    require_transaction_success(response.json(), batch)


def _default_client_factory() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        timeout=HTTP_TIMEOUT, trust_env=False, headers=FHIR_HEADERS
    )


def _fhir_base(endpoint: str) -> str:
    return endpoint.rstrip("/") + "/"


async def import_resources(
    endpoint: str,
    data_dir: Path,
    *,
    client_factory: Callable[[], httpx.AsyncClient] | None = None,
    log: Callable[[str], None] | None = None,
) -> None:
    """按资源依赖顺序分批写入；每个批次验证成功后才计入导入量。"""

    data_dir = Path(data_dir)
    inventory = source_inventory(data_dir)
    if not inventory.get("Patient"):
        raise SourceDataError("源数据缺少 Patient 资源")
    require_known_types(inventory)
    grouped = source_files(data_dir)
    url = _fhir_base(endpoint)
    factory = client_factory or _default_client_factory
    async with factory() as client:
        for resource_type in IMPORT_ORDER:
            identifiers = inventory.get(resource_type)
            if not identifiers:
                continue
            imported = 0
            for batch_index, batch in enumerate(
                batches(iter_resources(resource_type, grouped[resource_type])), start=1
            ):
                await submit_batch(client, url, batch)
                imported += len(batch)
                if log and batch_index % PROGRESS_BATCH_INTERVAL == 0:
                    log(f"{resource_type} 已导入 {imported}/{len(identifiers)}")
            if imported != len(identifiers):
                raise ResourceImportError(
                    f"{resource_type} 导入数量与源数据不一致：{imported}/{len(identifiers)}"
                )
            if log:
                log(f"{resource_type} 导入完成：{imported}")


# --- 全量 ID 核对 -----------------------------------------------------------


def _next_link(bundle: dict) -> str | None:
    for link in bundle.get("link", []) or []:
        if link.get("relation") == "next":
            url = link.get("url")
            if not isinstance(url, str) or not url:
                raise InventoryMismatchError("分页链接缺少 url")
            return url
    return None


async def _search_ids(client: httpx.AsyncClient, endpoint: str, resource_type: str) -> set[str]:
    url: str | None = f"{_fhir_base(endpoint)}{resource_type}"
    params: dict[str, Any] | None = {"_count": SEARCH_PAGE_SIZE}
    identifiers: set[str] = set()
    visited: set[str] = set()
    while url:
        if url in visited:
            raise InventoryMismatchError(f"{resource_type} 分页链接循环：{url}")
        visited.add(url)
        response = await client.get(url, params=params)
        response.raise_for_status()
        page = response.json()
        if not isinstance(page, dict) or page.get("resourceType") != "Bundle":
            raise InventoryMismatchError(f"{resource_type} 检索响应不是 Bundle")
        if page.get("type") != "searchset":
            raise InventoryMismatchError(
                f"{resource_type} 检索响应类型不是 searchset：{page.get('type')!r}"
            )
        for entry in page.get("entry", []) or []:
            resource = entry.get("resource")
            if not isinstance(resource, dict):
                raise InventoryMismatchError(f"{resource_type} 检索结果缺少 resource")
            if resource.get("resourceType") != resource_type:
                raise InventoryMismatchError(
                    f"{resource_type} 检索结果含 {resource.get('resourceType')!r}"
                )
            resource_id = resource.get("id")
            if not isinstance(resource_id, str) or not resource_id:
                raise InventoryMismatchError(f"{resource_type} 检索结果缺少 id")
            identifiers.add(resource_id)
        url = _next_link(page)
        params = None
    return identifiers


def _mismatch_message(resource_type: str, expected: set[str], found: set[str]) -> str:
    extra = sorted(found - expected)[:5]
    missing = sorted(expected - found)[:5]
    return (
        f"{resource_type} 服务端 ID 集合与源数据不一致："
        f"服务端共 {len(found)}，源数据共 {len(expected)}；"
        f"多出 {extra}，缺少 {missing}"
    )


async def verify_inventory(
    endpoint: str,
    expected: dict[str, set[str]],
    *,
    client_factory: Callable[[], httpx.AsyncClient] | None = None,
    log: Callable[[str], None] | None = None,
) -> None:
    """完整分页读取服务端每种资源的 ID 集合，与源数据严格比较。"""

    factory = client_factory or _default_client_factory
    async with factory() as client:
        for resource_type, expected_ids in expected.items():
            found = await _search_ids(client, endpoint, resource_type)
            if found != set(expected_ids):
                raise InventoryMismatchError(
                    _mismatch_message(resource_type, set(expected_ids), found)
                )
            if log:
                log(f"{resource_type} 核对通过：{len(found)}")


# --- 探针 -------------------------------------------------------------------


def select_probe(data_dir: Path, inventory: dict[str, set[str]]) -> dict[str, str]:
    """选定固定探针：编号最小的患者，以及该患者编号最小的 Observation。"""

    patient_ids = inventory.get("Patient")
    if not patient_ids:
        raise SourceDataError("源数据缺少 Patient 资源，无法选定探针")
    observation_ids = inventory.get("Observation")
    if not observation_ids:
        raise SourceDataError("源数据缺少 Observation 资源，无法选定探针")
    patient_id = min(patient_ids)
    grouped = source_files(data_dir)
    matching = sorted(
        {
            record["id"]
            for record in iter_resources("Observation", grouped["Observation"])
            if record.get("subject", {}).get("reference") == f"Patient/{patient_id}"
        }
    )
    if not matching:
        raise SourceDataError(f"患者 {patient_id} 没有 Observation，无法选定探针")
    return {"patient": patient_id, "observation": matching[0]}


def probe_records(data_dir: Path, inventory: dict[str, set[str]]) -> dict[str, dict]:
    probe = select_probe(data_dir, inventory)
    grouped = source_files(data_dir)
    wanted = {
        "patient": ("Patient", probe["patient"]),
        "observation": ("Observation", probe["observation"]),
    }
    records: dict[str, dict] = {}
    for name, (resource_type, resource_id) in wanted.items():
        for record in iter_resources(resource_type, grouped[resource_type]):
            if record["id"] == resource_id:
                records[name] = record
                break
        else:
            raise SourceDataError(f"源数据缺少探针资源 {resource_type}/{resource_id}")
    return records


async def read_resource(
    client: httpx.AsyncClient, endpoint: str, source_id: str
) -> dict:
    response = await client.get(f"{_fhir_base(endpoint)}{source_id}")
    response.raise_for_status()
    resource = response.json()
    if not isinstance(resource, dict) or resource.get("resourceType") is None:
        raise SeedPreparationError(f"读取 {source_id} 返回的不是 FHIR 资源")
    return resource


# --- Docker 后端 ------------------------------------------------------------


class DockerBackend:
    """限制命令等待时间，并支持把长输出流式写入文件。"""

    def __init__(self, binary: str = "docker"):
        self.binary = binary

    async def run(self, *args: str, timeout: float | None = DOCKER_TIMEOUT) -> str:
        secrets_in_use = _sensitive_env_values(args)
        process = await asyncio.create_subprocess_exec(
            self.binary,
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            process.kill()
            await process.communicate()
            raise SeedPreparationError(
                _redact(
                    f"docker {' '.join(args)} 超过 {timeout} 秒未结束",
                    *secrets_in_use,
                )
            )
        if process.returncode:
            raise SeedPreparationError(
                _redact(
                    f"docker {' '.join(args)} 退出码 {process.returncode}："
                    f"{stderr.decode().strip()}",
                    *secrets_in_use,
                )
            )
        return stdout.decode().strip()

    async def stream_to_file(
        self, args: Sequence[str], sink: BinaryIO, timeout: float | None = DATABASE_TIMEOUT
    ) -> None:
        secrets_in_use = _sensitive_env_values(args)
        process = await asyncio.create_subprocess_exec(
            self.binary,
            *args,
            stdout=sink,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            _, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            process.kill()
            await process.communicate()
            raise SeedPreparationError(
                _redact(
                    f"docker {' '.join(args)} 超过 {timeout} 秒未结束",
                    *secrets_in_use,
                )
            )
        if process.returncode:
            raise SeedPreparationError(
                _redact(
                    f"docker {' '.join(args)} 退出码 {process.returncode}："
                    f"{stderr.decode().strip()}",
                    *secrets_in_use,
                )
            )


@dataclass(frozen=True)
class PrepareResources:
    prefix: str
    hapi: str
    database: str
    volume: str
    network: str


def prepare_names() -> PrepareResources:
    prefix = f"medevolve-prepare-{uuid.uuid4()}"
    return PrepareResources(
        prefix=prefix,
        hapi=prefix,
        database=f"{prefix}-db",
        volume=f"{prefix}-data",
        network=f"{prefix}-net",
    )


def _labels(role: str) -> list[str]:
    return [
        "--label",
        f"{MANAGED_LABEL}=true",
        "--label",
        f"{ENVIRONMENT_LABEL}={ENVIRONMENT_NAME}",
        "--label",
        f"{ROLE_LABEL}={role}",
    ]


async def _verify_image(backend, image: str, log: Callable[[str], None]) -> None:
    try:
        inspected = json.loads(await backend.run("image", "inspect", image))
    except (SeedPreparationError, json.JSONDecodeError) as exc:
        raise SeedPreparationError(
            f"镜像不可用，请先执行 docker pull {image}：{exc}"
        ) from exc
    image_info = inspected[0] if inspected else {}
    architecture = image_info.get("Architecture")
    platform = image_info.get("Os")
    if platform != "linux" or architecture not in SUPPORTED_ARCHITECTURES:
        raise SeedPreparationError(
            f"镜像平台不受支持：{image} 为 {platform}/{architecture}，"
            f"需要 linux/{'|'.join(sorted(SUPPORTED_ARCHITECTURES))}"
        )
    log(f"镜像就绪：{image}（{platform}/{architecture}）")


async def _start_database(
    backend, names: PrepareResources, password: str, memory: str, log: Callable[[str], None]
) -> None:
    await backend.run("volume", "create", *_labels("database-volume"), names.volume)
    await backend.run(
        "run",
        "--detach",
        "--pull=never",
        "--name",
        names.database,
        *_labels("database"),
        "--network",
        names.network,
        "--memory",
        memory,
        "--memory-swap",
        memory,
        "--env",
        f"POSTGRES_USER={DATABASE_USER}",
        "--env",
        f"POSTGRES_DB={DATABASE_NAME}",
        "--env",
        f"POSTGRES_PASSWORD={password}",
        "--volume",
        f"{names.volume}:/var/lib/postgresql/data",
        POSTGRES_IMAGE,
    )
    log(f"数据库容器已启动：{names.database}")


async def _start_hapi(
    backend,
    names: PrepareResources,
    password: str,
    memory: str,
    log: Callable[[str], None],
) -> None:
    await backend.run(
        "run",
        "--detach",
        "--pull=never",
        "--name",
        names.hapi,
        *_labels("fhir"),
        "--network",
        names.network,
        "--publish",
        "127.0.0.1::8080",
        "--memory",
        memory,
        "--memory-swap",
        memory,
        "--env",
        f"SPRING_DATASOURCE_URL=jdbc:postgresql://{names.database}:{DATABASE_PORT}/{DATABASE_NAME}",
        "--env",
        f"SPRING_DATASOURCE_USERNAME={DATABASE_USER}",
        "--env",
        f"SPRING_DATASOURCE_PASSWORD={password}",
        "--env",
        "SPRING_DATASOURCE_DRIVERCLASSNAME=org.postgresql.Driver",
        "--env",
        "SPRING_JPA_PROPERTIES_HIBERNATE_DIALECT="
        "ca.uhn.fhir.jpa.model.dialect.HapiFhirPostgresDialect",
        "--env",
        "HAPI_FHIR_FHIR_VERSION=R4",
        "--env",
        "HAPI_FHIR_ENFORCE_REFERENTIAL_INTEGRITY_ON_WRITE=false",
        "--env",
        "HAPI_FHIR_VALIDATION_REQUESTS_ENABLED=false",
        "--env",
        "HAPI_FHIR_VALIDATION_RESPONSES_ENABLED=false",
        "--env",
        "JAVA_TOOL_OPTIONS=-XX:MaxRAMPercentage=60.0",
        HAPI_IMAGE,
    )
    log(f"FHIR 服务容器已启动：{names.hapi}")


async def _inspect(backend, name: str) -> dict:
    try:
        inspected = json.loads(await backend.run("container", "inspect", name))
    except (SeedPreparationError, json.JSONDecodeError) as exc:
        raise SeedPreparationError(f"无法读取容器信息：{name}：{exc}") from exc
    return inspected[0]


async def _host_port(backend, name: str, container_port: str = "8080/tcp") -> int:
    container = await _inspect(backend, name)
    bindings = (container.get("NetworkSettings", {}).get("Ports", {}) or {}).get(
        container_port
    )
    if not bindings:
        raise SeedPreparationError(f"容器 {name} 没有映射 {container_port}")
    return int(bindings[0]["HostPort"])


async def _wait_for_database(
    backend, names: PrepareResources, timeout: float, log: Callable[[str], None]
) -> None:
    deadline = time.monotonic() + timeout
    while True:
        try:
            await backend.run(
                "exec",
                names.database,
                "pg_isready",
                "-U",
                DATABASE_USER,
                "-d",
                DATABASE_NAME,
                timeout=DOCKER_TIMEOUT,
            )
            log("数据库已就绪")
            return
        except SeedPreparationError:
            if time.monotonic() >= deadline:
                raise SeedPreparationError(f"等待数据库就绪超过 {timeout} 秒")
            await asyncio.sleep(READY_POLL_INTERVAL)


async def _wait_for_fhir(
    client: httpx.AsyncClient, endpoint: str, timeout: float, log: Callable[[str], None]
) -> None:
    url = f"{_fhir_base(endpoint)}metadata"
    deadline = time.monotonic() + timeout
    while True:
        try:
            response = await client.get(url)
            response.raise_for_status()
            if response.json().get("resourceType") == "CapabilityStatement":
                log("FHIR 服务已就绪")
                return
        except (httpx.HTTPError, ValueError):
            pass
        if time.monotonic() >= deadline:
            raise SeedPreparationError(f"等待 FHIR 服务就绪超过 {timeout} 秒")
        await asyncio.sleep(READY_POLL_INTERVAL)


async def _cleanup(backend, names: PrepareResources, log: Callable[[str], None]) -> None:
    commands = (
        ("container", "rm", "--force", "--volumes", names.hapi),
        ("container", "rm", "--force", "--volumes", names.database),
        ("volume", "rm", "--force", names.volume),
        ("network", "rm", names.network),
    )
    for command in commands:
        try:
            await backend.run(*command, timeout=DOCKER_TIMEOUT)
        except Exception as exc:  # noqa: BLE001 - 清理失败不掩盖导入或备份错误
            message = str(exc)
            if "No such container" in message or "No such volume" in message:
                continue
            if "No such network" in message:
                continue
            log(f"清理准备资源失败（需手工删除）：{' '.join(command)}：{message}")


# --- 种子准备 ---------------------------------------------------------------


def build_manifest(
    *,
    inventory: dict[str, set[str]],
    source_sha256: str,
    database_sha256: str,
    probe_source_id: str,
    probe_sha256: str,
) -> dict:
    return {
        "schema_version": 1,
        "data_version": DATA_VERSION,
        "hapi_image": HAPI_IMAGE,
        "postgres_image": POSTGRES_IMAGE,
        "database_sha256": database_sha256,
        "source_sha256": source_sha256,
        "resource_counts": {
            resource_type: len(identifiers)
            for resource_type, identifiers in sorted(inventory.items())
        },
        "probe_source_id": probe_source_id,
        "probe_sha256": probe_sha256,
    }


def write_manifest(seed_dir: Path, manifest: dict) -> Path:
    path = seed_dir / MANIFEST_NAME
    temporary = seed_dir / f"{MANIFEST_NAME}.tmp"
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)
    return path


def _discard_incomplete_seed(seed_dir: Path) -> None:
    """删掉未写完的种子产物，让失败的种子目录回到可重跑的空状态。"""

    artifacts = (
        f"{MANIFEST_NAME}.tmp",
        f"{DATABASE_DUMP_NAME}.tmp",
        DATABASE_DUMP_NAME,
    )
    for name in artifacts:
        (seed_dir / name).unlink(missing_ok=True)


async def prepare_dataset(
    data_dir: Path,
    seed_dir: Path,
    *,
    backend: DockerBackend | None = None,
    client_factory: Callable[[], httpx.AsyncClient] | None = None,
    require_published: bool = True,
    log: Callable[[str], None] = print,
) -> dict:
    """校验源数据、导入并导出种子；本次创建的资源无论成败都删除。"""

    data_dir = Path(data_dir)
    seed_dir = Path(seed_dir)
    if not data_dir.is_dir():
        raise SourceDataError(f"数据目录不存在：{data_dir}")

    seed_dir.mkdir(parents=True, exist_ok=True)
    if (seed_dir / MANIFEST_NAME).exists():
        raise SeedPreparationError(
            f"种子目录已存在 {MANIFEST_NAME}，不覆盖已有成功种子：{seed_dir}"
        )
    existing = sorted(path.name for path in seed_dir.iterdir())
    if existing:
        raise SeedPreparationError(f"种子目录非空：{seed_dir}（{', '.join(existing)}）")

    started = time.monotonic()
    checksum_digest = verify_checksums(data_dir)
    source_sha256 = checksum_digest or hash_dataset(data_dir)
    inventory = source_inventory(data_dir)
    if require_published:
        require_published_dataset(inventory)
    probe = probe_records(data_dir, inventory)
    log(
        f"源数据校验通过：{sum(len(ids) for ids in inventory.values())} 条资源，"
        f"{len(inventory)} 种类型"
    )

    backend = backend or DockerBackend()
    factory = client_factory or _default_client_factory
    password = secrets.token_urlsafe(32)
    names = prepare_names()
    hapi_memory = os.environ.get(HAPI_MEMORY_ENV, DEFAULT_HAPI_MEMORY)
    postgres_memory = os.environ.get(POSTGRES_MEMORY_ENV, DEFAULT_POSTGRES_MEMORY)
    manifest: dict = {}

    try:
        await _verify_image(backend, POSTGRES_IMAGE, log)
        await _verify_image(backend, HAPI_IMAGE, log)
        await backend.run("network", "create", *_labels("prepare-network"), names.network)
        await _start_database(backend, names, password, postgres_memory, log)
        await _wait_for_database(backend, names, SERVICE_START_TIMEOUT, log)
        await _start_hapi(backend, names, password, hapi_memory, log)
        port = await _host_port(backend, names.hapi)
        endpoint = f"http://127.0.0.1:{port}/fhir/"

        async with factory() as client:
            await _wait_for_fhir(client, endpoint, SERVICE_START_TIMEOUT, log)

        import_started = time.monotonic()
        await import_resources(
            endpoint, data_dir, client_factory=factory, log=log
        )
        log(f"导入完成，用时 {time.monotonic() - import_started:.1f} 秒")

        await verify_inventory(endpoint, inventory, client_factory=factory, log=log)

        async with factory() as client:
            probe_source_id = f"Observation/{probe['observation']['id']}"
            stored_patient = await read_resource(
                client, endpoint, f"Patient/{probe['patient']['id']}"
            )
            if resource_digest(stored_patient) != resource_digest(probe["patient"]):
                raise SeedPreparationError(
                    f"探针 Patient/{probe['patient']['id']} 与服务端字段不一致"
                )
            stored_observation = await read_resource(client, endpoint, probe_source_id)
            if resource_digest(stored_observation) != resource_digest(probe["observation"]):
                raise SeedPreparationError(
                    f"探针 {probe_source_id} 与服务端字段不一致"
                )
            log(f"探针核对通过：{probe_source_id}")

        dump_started = time.monotonic()
        dump_path = seed_dir / DATABASE_DUMP_NAME
        temporary_dump = seed_dir / f"{DATABASE_DUMP_NAME}.tmp"
        with temporary_dump.open("wb") as sink:
            await backend.stream_to_file(
                (
                    "exec",
                    names.database,
                    "pg_dump",
                    "-Fc",
                    "--no-owner",
                    "--no-acl",
                    "--no-password",
                    "-U",
                    DATABASE_USER,
                    "-d",
                    DATABASE_NAME,
                ),
                sink,
                DATABASE_TIMEOUT,
            )
        if temporary_dump.stat().st_size == 0:
            raise SeedPreparationError("pg_dump 输出为空，未生成种子备份")
        os.replace(temporary_dump, dump_path)
        log(f"数据库备份完成，用时 {time.monotonic() - dump_started:.1f} 秒")

        manifest = build_manifest(
            inventory=inventory,
            source_sha256=source_sha256,
            database_sha256=file_sha256(dump_path),
            probe_source_id=probe_source_id,
            probe_sha256=resource_digest(probe["observation"]),
        )
        write_manifest(seed_dir, manifest)
        log(
            f"种子准备完成：{seed_dir}，总用时 {time.monotonic() - started:.1f} 秒"
        )
        return manifest
    except BaseException as exc:
        if not (seed_dir / MANIFEST_NAME).exists():
            _discard_incomplete_seed(seed_dir)
        message = _redact(str(exc), password)
        if message == str(exc):
            raise
        raise SeedPreparationError(f"{type(exc).__name__}: {message}") from None
    finally:
        await _cleanup(backend, names, log)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fhir_agentbench_data", description="FHIR-AgentBench 本地数据准备"
    )
    subcommands = parser.add_subparsers(dest="command", required=True)
    prepare = subcommands.add_parser(
        "prepare", help="导入 MIMIC-IV-FHIR Demo 并生成数据库种子"
    )
    prepare.add_argument("--data-dir", required=True, type=Path, help="fhir 数据目录")
    prepare.add_argument("--seed-dir", required=True, type=Path, help="种子输出目录")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if arguments.command != "prepare":
        raise SystemExit(f"未知子命令：{arguments.command}")
    try:
        manifest = asyncio.run(
            prepare_dataset(arguments.data_dir, arguments.seed_dir)
        )
    except SeedPreparationError as exc:
        print(f"准备失败：{exc}", file=sys.stderr)
        return 1
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
