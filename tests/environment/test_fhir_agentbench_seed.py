"""FHIR-AgentBench 种子准备机制测试。

全部使用 fake Docker 后端与 httpx.MockTransport，不连接 Docker、PostgreSQL 或真实 FHIR 服务。
"""

import copy
import gzip
import hashlib
import json
import sys
from pathlib import Path

import httpx
import pytest

from scripts import fhir_agentbench_data as data
from server.fhir_agentbench import SeedValidationError, load_seed


def patient(rid: str) -> dict:
    return {
        "resourceType": "Patient",
        "id": rid,
        "gender": "female",
        "birthDate": "2130-01-01",
    }


def observation(rid: str, patient_id: str) -> dict:
    return {
        "resourceType": "Observation",
        "id": rid,
        "status": "final",
        "code": {"text": "Heart rate"},
        "subject": {"reference": f"Patient/{patient_id}"},
        "effectiveDateTime": "2136-04-22T18:47:00-04:00",
    }


def condition(rid: str, patient_id: str) -> dict:
    return {
        "resourceType": "Condition",
        "id": rid,
        "subject": {"reference": f"Patient/{patient_id}"},
        "code": {"text": "Sepsis"},
    }


def source_id(resource: dict) -> str:
    return f"{resource['resourceType']}/{resource['id']}"


def write_dataset(root: Path, files: dict[str, list[dict]]) -> Path:
    """把 {文件名: 记录列表} 写成 <root>/2.1.0/fhir/<文件名>，返回数据目录。"""

    data_dir = root / "2.1.0" / "fhir"
    data_dir.mkdir(parents=True, exist_ok=True)
    for name, records in files.items():
        with gzip.open(data_dir / name, "wt", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return data_dir


def write_checksums(data_dir: Path, rel_paths: list[str]) -> None:
    lines = []
    for rel in rel_paths:
        digest = hashlib.sha256((data_dir.parent / rel).read_bytes()).hexdigest()
        lines.append(f"{digest} {rel}\n")
    (data_dir.parent / "SHA256SUMS.txt").write_text("".join(lines), encoding="utf-8")


class FakeFhir:
    """最小 FHIR 服务端：transaction 写入、按 ID 读取、分页检索。"""

    def __init__(self, records: list[dict], page_size: int = 2):
        self.records = {source_id(record): copy.deepcopy(record) for record in records}
        self.page_size = page_size
        self.transactions: list[list[dict]] = []
        self.fail_entry: tuple[int, int] | None = None
        self.shorten_transaction = False
        self.mutate_on_read: dict[str, dict] = {}
        self.metadata_ready = True

    def client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            transport=httpx.MockTransport(self.handle),
            base_url="http://fhir.local",
            trust_env=False,
            headers={"Accept": "application/fhir+json"},
        )

    def handle(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/fhir/metadata":
            if not self.metadata_ready:
                return httpx.Response(503, json={"resourceType": "OperationOutcome"})
            return httpx.Response(
                200,
                json={"resourceType": "CapabilityStatement", "fhirVersion": "4.0.1"},
            )
        if request.method == "POST":
            return self._transaction(request)
        parts = [part for part in path.split("/") if part]
        if parts[:1] != ["fhir"]:
            return httpx.Response(404, json={"resourceType": "OperationOutcome"})
        if len(parts) == 2:
            return self._search(parts[1], request.url.params)
        if len(parts) == 3:
            return self._read(parts[1], parts[2])
        return httpx.Response(404, json={"resourceType": "OperationOutcome"})

    def _transaction(self, request: httpx.Request) -> httpx.Response:
        bundle = json.loads(request.content)
        batch = [entry["resource"] for entry in bundle["entry"]]
        index = len(self.transactions)
        self.transactions.append(batch)
        entries = []
        for position, resource in enumerate(batch):
            self.records.setdefault(source_id(resource), copy.deepcopy(resource))
            status = "201 Created"
            if self.fail_entry == (index, position):
                status = "400 Bad Request"
            entries.append(
                {
                    "response": {
                        "status": status,
                        "location": f"{source_id(resource)}/_history/1",
                    }
                }
            )
        if self.shorten_transaction:
            entries = entries[:-1]
        return httpx.Response(
            200, json={"resourceType": "Bundle", "type": "transaction-response", "entry": entries}
        )

    def _read(self, resource_type: str, resource_id: str) -> httpx.Response:
        record = self.records.get(f"{resource_type}/{resource_id}")
        if record is None:
            return httpx.Response(404, json={"resourceType": "OperationOutcome"})
        mutate = self.mutate_on_read.get(f"{resource_type}/{resource_id}")
        if mutate:
            record = mutate(copy.deepcopy(record))
        return httpx.Response(200, json=record)

    def _search(self, resource_type: str, params: httpx.QueryParams) -> httpx.Response:
        identifiers = sorted(
            key.split("/", 1)[1]
            for key in self.records
            if key.startswith(f"{resource_type}/")
        )
        count = int(params.get("_count", self.page_size))
        offset = int(params.get("_getpagesoffset", 0))
        page = identifiers[offset : offset + count]
        entries = [
            {"resource": self.records[f"{resource_type}/{rid}"]} for rid in page
        ]
        bundle = {
            "resourceType": "Bundle",
            "type": "searchset",
            "total": len(identifiers),
            "entry": entries,
        }
        next_offset = offset + count
        if next_offset < len(identifiers):
            bundle["link"] = [
                {
                    "relation": "next",
                    "url": str(
                        httpx.URL(
                            f"http://fhir.local/fhir/{resource_type}",
                            params={"_count": count, "_getpagesoffset": next_offset},
                        )
                    ),
                }
            ]
        return httpx.Response(200, json=bundle)


class FakeBackend:
    """记录被创建资源的 fake Docker 后端。"""

    def __init__(self, dump: bytes = b"FAKE-DUMP"):
        self.containers: dict[str, dict] = {}
        self.volumes: set[str] = set()
        self.networks: set[str] = set()
        self.commands: list[tuple[str, ...]] = []
        self.dump = dump
        self.fail_dump = False
        self.dump_error: Exception | None = None
        self.fail_run = False

    @property
    def env_values(self) -> list[str]:
        values = []
        for command in self.commands:
            for position, argument in enumerate(command):
                if argument == "--env":
                    values.append(command[position + 1])
        return values

    async def run(self, *args: str, timeout: float | None = None) -> str:
        self.commands.append(args)
        if args[:2] == ("image", "inspect"):
            return json.dumps(
                [{"Id": "sha256:image", "Architecture": "arm64", "Os": "linux"}]
            )
        if args[:2] == ("network", "create"):
            self.networks.add(args[-1])
            return args[-1]
        if args[:2] == ("volume", "create"):
            self.volumes.add(args[-1])
            return args[-1]
        if args[0] == "run":
            if self.fail_run:
                raise data.SeedPreparationError(
                    f"docker {' '.join(args)} 退出码 125：port is already allocated"
                )
            name = args[args.index("--name") + 1]
            self.containers[name] = {
                "Config": {"Labels": self._labels(args)},
                "NetworkSettings": {
                    "Ports": {"8080/tcp": [{"HostPort": "23456"}]},
                },
            }
            return name
        if args[:2] == ("container", "inspect"):
            if args[-1] not in self.containers:
                raise data.SeedPreparationError("No such container")
            return json.dumps([self.containers[args[-1]]])
        if args[:2] == ("container", "rm"):
            self.containers.pop(args[-1], None)
            return args[-1]
        if args[:2] == ("volume", "rm"):
            self.volumes.discard(args[-1])
            return args[-1]
        if args[:2] == ("network", "rm"):
            self.networks.discard(args[-1])
            return args[-1]
        if args[0] == "exec":
            return "accepting connections"
        raise AssertionError(args)

    async def stream_to_file(self, args, sink, timeout: float | None = None) -> None:
        self.commands.append(tuple(args))
        if self.dump_error is not None:
            raise self.dump_error
        if self.fail_dump:
            raise data.SeedPreparationError("pg_dump 退出码 1")
        sink.write(self.dump)

    @staticmethod
    def _labels(args: tuple[str, ...]) -> dict[str, str]:
        labels = {}
        for position, argument in enumerate(args):
            if argument == "--label":
                key, _, value = args[position + 1].partition("=")
                labels[key] = value
        return labels

    @property
    def leftovers(self) -> tuple[dict, set, set]:
        return self.containers, self.volumes, self.networks


MINIMAL_FILES = {
    "MimicPatient.ndjson.gz": [patient("p1"), patient("p2")],
    "MimicObservation.ndjson.gz": [observation("o1", "p1"), observation("o2", "p2")],
}


PUBLISHED_FILES = {
    "MimicPatient.ndjson.gz": [patient(f"p{index:03d}") for index in range(1, 101)],
    "MimicObservation.ndjson.gz": [observation("o1", "p001"), observation("o2", "p001")],
}


def minimal_data_dir(tmp_path: Path) -> Path:
    return write_dataset(tmp_path, MINIMAL_FILES)


def fake_client_factory(server: FakeFhir):
    return server.client


# --- 源数据校验 -------------------------------------------------------------


def test_source_inventory_groups_ids_by_type(tmp_path):
    data_dir = write_dataset(
        tmp_path,
        {
            "MimicPatient.ndjson.gz": [patient("p1")],
            "MimicObservation.ndjson.gz": [observation("o1", "p1"), observation("o2", "p1")],
        },
    )
    inventory = data.source_inventory(data_dir)
    assert inventory == {"Patient": {"p1"}, "Observation": {"o1", "o2"}}


def test_source_inventory_merges_multiple_files_of_same_type(tmp_path):
    data_dir = write_dataset(
        tmp_path,
        {
            "MimicCondition.ndjson.gz": [condition("c1", "p1")],
            "MimicConditionED.ndjson.gz": [condition("c2", "p1")],
        },
    )
    assert data.source_inventory(data_dir) == {"Condition": {"c1", "c2"}}


def test_source_inventory_rejects_duplicate_resource_id(tmp_path):
    data_dir = write_dataset(
        tmp_path,
        {
            "MimicPatient.ndjson.gz": [patient("p1")],
            "MimicEncounterED.ndjson.gz": [patient("p1")],
        },
    )
    with pytest.raises(data.SourceDataError, match="Patient/p1"):
        data.source_inventory(data_dir)


def test_source_inventory_rejects_record_without_id(tmp_path):
    broken = patient("p1")
    del broken["id"]
    data_dir = write_dataset(tmp_path, {"MimicPatient.ndjson.gz": [broken]})
    with pytest.raises(data.SourceDataError, match="id"):
        data.source_inventory(data_dir)


def test_source_inventory_rejects_missing_data_directory(tmp_path):
    with pytest.raises(data.SourceDataError):
        data.source_inventory(tmp_path / "absent")


def test_published_dataset_check_rejects_unexpected_counts():
    inventory = {"Patient": {f"p{i}" for i in range(100)}, "Observation": {"o1"}}
    with pytest.raises(data.SourceDataError, match="928935"):
        data.require_published_dataset(inventory)


def test_checksum_ledger_mismatch_fails(tmp_path):
    data_dir = minimal_data_dir(tmp_path)
    write_checksums(data_dir, ["fhir/MimicPatient.ndjson.gz", "fhir/MimicObservation.ndjson.gz"])
    (data_dir / "MimicObservation.ndjson.gz").write_bytes(b"tampered")
    with pytest.raises(data.SourceDataError, match="MimicObservation.ndjson.gz"):
        data.verify_checksums(data_dir)


# --- 批量导入 ---------------------------------------------------------------


def test_transaction_entry_uses_original_id_and_put():
    entry = data.transaction_entry(patient("p1"))
    assert entry["request"] == {"method": "PUT", "url": "Patient/p1"}
    assert entry["resource"]["id"] == "p1"


def test_require_transaction_success_reports_failed_source_id():
    batch = [patient("p1"), patient("p2")]
    bundle = {
        "resourceType": "Bundle",
        "type": "transaction-response",
        "entry": [
            {"response": {"status": "201 Created"}},
            {"response": {"status": "400 Bad Request"}},
        ],
    }
    with pytest.raises(data.ResourceImportError, match="Patient/p2") as failure:
        data.require_transaction_success(bundle, batch)
    assert "400 Bad Request" in str(failure.value)


def test_require_transaction_success_rejects_entry_count_mismatch():
    batch = [patient("p1"), patient("p2")]
    bundle = {
        "resourceType": "Bundle",
        "type": "transaction-response",
        "entry": [{"response": {"status": "201 Created"}}],
    }
    with pytest.raises(data.ResourceImportError, match="数量"):
        data.require_transaction_success(bundle, batch)


@pytest.mark.asyncio
async def test_import_raises_on_failed_entry_with_http_200(tmp_path):
    data_dir = minimal_data_dir(tmp_path)
    server = FakeFhir(list(MINIMAL_FILES["MimicPatient.ndjson.gz"]) + [observation("o1", "p1")])
    server.fail_entry = (0, 1)  # Patient 批次第二条返回 400
    with pytest.raises(data.ResourceImportError, match="Patient/p2"):
        await data.import_resources(
            "http://fhir.local/fhir/", data_dir, client_factory=server.client
        )


@pytest.mark.asyncio
async def test_import_raises_on_short_response(tmp_path):
    data_dir = minimal_data_dir(tmp_path)
    server = FakeFhir([])
    server.shorten_transaction = True
    with pytest.raises(data.ResourceImportError, match="数量"):
        await data.import_resources(
            "http://fhir.local/fhir/", data_dir, client_factory=server.client
        )


@pytest.mark.asyncio
async def test_import_rejects_data_without_patients(tmp_path):
    data_dir = write_dataset(
        tmp_path, {"MimicLocation.ndjson.gz": [{"resourceType": "Location", "id": "l1"}]}
    )
    server = FakeFhir([])
    with pytest.raises(data.SourceDataError, match="Patient"):
        await data.import_resources(
            "http://fhir.local/fhir/", data_dir, client_factory=server.client
        )


@pytest.mark.asyncio
async def test_import_writes_dependency_order_in_batches(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "BATCH_SIZE", 2)
    data_dir = write_dataset(
        tmp_path,
        {
            "MimicObservation.ndjson.gz": [observation(f"o{i}", "p1") for i in range(3)],
            "MimicPatient.ndjson.gz": [patient("p1")],
            "MimicOrganization.ndjson.gz": [{"resourceType": "Organization", "id": "org1"}],
            "MimicEncounter.ndjson.gz": [{"resourceType": "Encounter", "id": "e1"}],
        },
    )
    server = FakeFhir([])
    await data.import_resources(
        "http://fhir.local/fhir/", data_dir, client_factory=server.client
    )
    written = [source_id(resource) for batch in server.transactions for resource in batch]
    assert written[0] == "Organization/org1"
    assert written[1] == "Patient/p1"
    assert written[2] == "Encounter/e1"
    assert written[3:] == ["Observation/o0", "Observation/o1", "Observation/o2"]
    assert [len(batch) for batch in server.transactions] == [1, 1, 1, 2, 1]
    assert set(server.records) == set(written)


# --- 全量核对 ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_verify_inventory_matches_paged_server_ids():
    server = FakeFhir([observation(f"o{i}", "p1") for i in range(5)], page_size=2)
    expected = {"Observation": {f"o{i}" for i in range(5)}}
    await data.verify_inventory(
        "http://fhir.local/fhir/", expected, client_factory=server.client
    )


@pytest.mark.asyncio
async def test_verify_inventory_rejects_extra_server_id():
    server = FakeFhir([observation("o1", "p1"), observation("o2", "p1")])
    with pytest.raises(data.InventoryMismatchError, match="o2"):
        await data.verify_inventory(
            "http://fhir.local/fhir/",
            {"Observation": {"o1"}},
            client_factory=server.client,
        )


@pytest.mark.asyncio
async def test_verify_inventory_rejects_missing_server_id():
    server = FakeFhir([observation("o1", "p1")])
    with pytest.raises(data.InventoryMismatchError, match="o2"):
        await data.verify_inventory(
            "http://fhir.local/fhir/",
            {"Observation": {"o1", "o2"}},
            client_factory=server.client,
        )


# --- 种子准备 CLI -----------------------------------------------------------


def manifest_template(tmp_path: Path) -> tuple[Path, dict]:
    seed_dir = tmp_path / "seed"
    seed_dir.mkdir()
    (seed_dir / "database.dump").write_bytes(b"DUMP-BYTES")
    manifest = {
        "schema_version": 1,
        "data_version": "2.1.0",
        "hapi_image": "hapiproject/hapi@sha256:" + "a" * 64,
        "postgres_image": "postgres@sha256:" + "b" * 64,
        "database_sha256": hashlib.sha256(b"DUMP-BYTES").hexdigest(),
        "source_sha256": "c" * 64,
        "resource_counts": {"Patient": 100, "Observation": 3},
        "probe_source_id": "Observation/o1",
        "probe_sha256": "d" * 64,
    }
    return seed_dir, manifest


def test_load_seed_returns_validated_manifest(tmp_path):
    seed_dir, manifest = manifest_template(tmp_path)
    (seed_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert load_seed(str(seed_dir)) == manifest


@pytest.mark.parametrize(
    "mutation, expected",
    [
        ({"schema_version": 2}, "schema_version"),
        ({"data_version": "2.0.0"}, "data_version"),
        ({"hapi_image": "hapiproject/hapi:v8.12.0-2"}, "hapi_image"),
        ({"postgres_image": "postgres@sha256:short"}, "postgres_image"),
        ({"resource_counts": {"Patient": 99}}, "Patient"),
        ({"database_sha256": "e" * 64}, "database_sha256"),
        ({"probe_source_id": ""}, "probe_source_id"),
        ({"probe_sha256": "not-a-digest"}, "probe_sha256"),
    ],
)
def test_load_seed_rejects_invalid_manifest(tmp_path, mutation, expected):
    seed_dir, manifest = manifest_template(tmp_path)
    manifest.update(mutation)
    (seed_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(SeedValidationError, match=expected):
        load_seed(str(seed_dir))


def test_load_seed_requires_fixed_fields(tmp_path):
    seed_dir, manifest = manifest_template(tmp_path)
    manifest.pop("probe_sha256")
    (seed_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(SeedValidationError, match="probe_sha256"):
        load_seed(str(seed_dir))


def test_load_seed_requires_database_dump(tmp_path):
    seed_dir, manifest = manifest_template(tmp_path)
    (seed_dir / "database.dump").unlink()
    (seed_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(SeedValidationError, match="database.dump"):
        load_seed(str(seed_dir))


def test_load_seed_rejects_missing_manifest(tmp_path):
    seed_dir = tmp_path / "seed"
    seed_dir.mkdir()
    with pytest.raises(SeedValidationError, match="manifest.json"):
        load_seed(str(seed_dir))


@pytest.mark.asyncio
async def test_prepare_writes_seed_manifest_and_cleans_up(tmp_path):
    data_dir = write_dataset(tmp_path, PUBLISHED_FILES)
    write_checksums(
        data_dir,
        ["fhir/MimicPatient.ndjson.gz", "fhir/MimicObservation.ndjson.gz"],
    )
    seed_dir = tmp_path / "seed"
    backend = FakeBackend()
    server = FakeFhir(
        list(PUBLISHED_FILES["MimicPatient.ndjson.gz"])
        + list(PUBLISHED_FILES["MimicObservation.ndjson.gz"])
    )

    manifest = await data.prepare_dataset(
        data_dir,
        seed_dir,
        backend=backend,
        client_factory=server.client,
        require_published=False,
        log=lambda message: None,
    )

    assert manifest == load_seed(str(seed_dir))
    assert manifest["schema_version"] == 1
    assert manifest["data_version"] == "2.1.0"
    assert manifest["resource_counts"] == {"Patient": 100, "Observation": 2}
    assert manifest["probe_source_id"] == "Observation/o1"
    assert manifest["probe_sha256"] == data.resource_digest(
        PUBLISHED_FILES["MimicObservation.ndjson.gz"][0]
    )
    assert manifest["source_sha256"] == data.verify_checksums(data_dir)
    assert manifest["database_sha256"] == hashlib.sha256(backend.dump).hexdigest()
    assert (seed_dir / "database.dump").read_bytes() == backend.dump
    assert backend.leftovers == ({}, set(), set())
    assert "Observation/o1" in server.records and "Patient/p100" in server.records

    secrets_in_use = [
        value.split("=", 1)[1]
        for value in backend.env_values
        if value.startswith("SPRING_DATASOURCE_PASSWORD=")
    ]
    assert len(secrets_in_use) == 1
    password = secrets_in_use[0]
    assert password
    assert password not in (seed_dir / "manifest.json").read_text(encoding="utf-8")
    assert all(password not in command for command in backend.commands)
    backup = next(command for command in backend.commands if "pg_dump" in command)
    assert backup[0] == "exec"
    assert {"-Fc", "--no-owner", "--no-acl", "--no-password"} <= set(backup)
    assert backup[-4:] == ("-U", "medevolve", "-d", "medevolve")


@pytest.mark.asyncio
async def test_prepare_pins_images_and_memory_limits(tmp_path):
    data_dir = minimal_data_dir(tmp_path)
    backend = FakeBackend()
    server = FakeFhir(list(MINIMAL_FILES["MimicPatient.ndjson.gz"]) + MINIMAL_FILES["MimicObservation.ndjson.gz"])
    await data.prepare_dataset(
        data_dir,
        tmp_path / "seed",
        backend=backend,
        client_factory=server.client,
        require_published=False,
        log=lambda message: None,
    )
    runs = [command for command in backend.commands if command and command[0] == "run"]
    assert len(runs) == 2
    database, hapi = runs
    assert data.POSTGRES_IMAGE in database and data.HAPI_IMAGE in hapi
    assert "--publish" not in database and "127.0.0.1::8080" in hapi
    assert database[database.index("--memory") + 1] == "1g"
    assert hapi[hapi.index("--memory") + 1] == "3g"
    assert database[database.index("--memory-swap") + 1] == "1g"
    assert hapi[hapi.index("--memory-swap") + 1] == "3g"
    assert f"{data.MANAGED_LABEL}=true" in " ".join(hapi)
    database_name = database[database.index("--name") + 1]
    url = next(value for value in backend.env_values if value.startswith("SPRING_DATASOURCE_URL="))
    assert url == f"SPRING_DATASOURCE_URL=jdbc:postgresql://{database_name}:5432/medevolve"


@pytest.mark.asyncio
async def test_prepare_rejects_existing_successful_seed(tmp_path):
    data_dir = minimal_data_dir(tmp_path)
    seed_dir = tmp_path / "seed"
    seed_dir.mkdir()
    (seed_dir / "manifest.json").write_text("{}", encoding="utf-8")
    backend = FakeBackend()
    with pytest.raises(data.SeedPreparationError, match="manifest.json"):
        await data.prepare_dataset(
            data_dir, seed_dir, backend=backend, require_published=False
        )
    assert backend.commands == []


@pytest.mark.asyncio
async def test_prepare_rejects_broken_checksum_before_docker(tmp_path):
    data_dir = minimal_data_dir(tmp_path)
    write_checksums(
        data_dir,
        ["fhir/MimicPatient.ndjson.gz", "fhir/MimicObservation.ndjson.gz"],
    )
    (data_dir / "MimicObservation.ndjson.gz").write_bytes(b"tampered")
    backend = FakeBackend()
    with pytest.raises(data.SourceDataError, match="MimicObservation"):
        await data.prepare_dataset(
            data_dir, tmp_path / "seed", backend=backend, require_published=False
        )
    assert backend.commands == []


@pytest.mark.asyncio
async def test_prepare_cleans_up_when_import_fails(tmp_path, monkeypatch):
    data_dir = minimal_data_dir(tmp_path)
    backend = FakeBackend()
    server = FakeFhir([])

    async def failing_import(*args, **kwargs):
        raise data.ResourceImportError("导入中途失败")

    monkeypatch.setattr(data, "import_resources", failing_import)
    with pytest.raises(data.ResourceImportError):
        await data.prepare_dataset(
            data_dir,
            tmp_path / "seed",
            backend=backend,
            client_factory=server.client,
            require_published=False,
            log=lambda message: None,
        )
    assert backend.leftovers == ({}, set(), set())
    assert not (tmp_path / "seed" / "manifest.json").exists()


@pytest.mark.asyncio
async def test_prepare_cleans_up_when_probe_differs(tmp_path):
    data_dir = minimal_data_dir(tmp_path)
    backend = FakeBackend()
    records = list(MINIMAL_FILES["MimicPatient.ndjson.gz"]) + list(
        MINIMAL_FILES["MimicObservation.ndjson.gz"]
    )
    server = FakeFhir(records)
    server.mutate_on_read["Observation/o1"] = lambda record: {
        **record,
        "status": "amended",
    }
    with pytest.raises(data.SeedPreparationError, match="Observation/o1"):
        await data.prepare_dataset(
            data_dir,
            tmp_path / "seed",
            backend=backend,
            client_factory=server.client,
            require_published=False,
            log=lambda message: None,
        )
    assert backend.leftovers == ({}, set(), set())
    assert not (tmp_path / "seed" / "manifest.json").exists()


@pytest.mark.asyncio
async def test_prepare_cleans_up_when_service_never_becomes_ready(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "SERVICE_START_TIMEOUT", 0.05)
    monkeypatch.setattr(data, "READY_POLL_INTERVAL", 0.01)
    data_dir = minimal_data_dir(tmp_path)
    backend = FakeBackend()
    server = FakeFhir(list(MINIMAL_FILES["MimicPatient.ndjson.gz"]))
    server.metadata_ready = False
    with pytest.raises(data.SeedPreparationError, match="FHIR 服务就绪"):
        await data.prepare_dataset(
            data_dir,
            tmp_path / "seed",
            backend=backend,
            client_factory=server.client,
            require_published=False,
            log=lambda message: None,
        )
    assert backend.leftovers == ({}, set(), set())
    assert not (tmp_path / "seed" / "manifest.json").exists()


@pytest.mark.asyncio
async def test_prepare_cleans_up_when_dump_fails(tmp_path):
    data_dir = minimal_data_dir(tmp_path)
    seed_dir = tmp_path / "seed"
    backend = FakeBackend()
    backend.fail_dump = True
    server = FakeFhir(list(MINIMAL_FILES["MimicPatient.ndjson.gz"]) + MINIMAL_FILES["MimicObservation.ndjson.gz"])
    with pytest.raises(data.SeedPreparationError, match="pg_dump"):
        await data.prepare_dataset(
            data_dir,
            seed_dir,
            backend=backend,
            client_factory=server.client,
            require_published=False,
            log=lambda message: None,
        )
    assert backend.leftovers == ({}, set(), set())
    assert not (seed_dir / "manifest.json").exists()
    assert not (seed_dir / "database.dump").exists()
    assert not (seed_dir / "database.dump.tmp").exists()
    assert list(seed_dir.iterdir()) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["nonzero-exit", "timeout", "empty-output"])
async def test_prepare_removes_dump_artifacts_and_can_rerun(tmp_path, failure):
    data_dir = minimal_data_dir(tmp_path)
    seed_dir = tmp_path / "seed"
    records = list(MINIMAL_FILES["MimicPatient.ndjson.gz"]) + list(
        MINIMAL_FILES["MimicObservation.ndjson.gz"]
    )
    backend = FakeBackend()
    if failure == "nonzero-exit":
        backend.dump_error = data.SeedPreparationError("docker exec db pg_dump 退出码 1")
    elif failure == "timeout":
        backend.dump_error = data.SeedPreparationError(
            "docker exec db pg_dump 超过 1800 秒未结束"
        )
    else:
        backend.dump = b""

    with pytest.raises(data.SeedPreparationError, match="pg_dump"):
        await data.prepare_dataset(
            data_dir,
            seed_dir,
            backend=backend,
            client_factory=FakeFhir(records).client,
            require_published=False,
            log=lambda message: None,
        )

    assert backend.leftovers == ({}, set(), set())
    assert list(seed_dir.iterdir()) == []

    retry = FakeBackend(dump=b"RETRY-DUMP")
    manifest = await data.prepare_dataset(
        data_dir,
        seed_dir,
        backend=retry,
        client_factory=FakeFhir(records).client,
        require_published=False,
        log=lambda message: None,
    )
    assert manifest["resource_counts"] == {"Patient": 2, "Observation": 2}
    assert (
        json.loads((seed_dir / "manifest.json").read_text(encoding="utf-8")) == manifest
    )
    assert (seed_dir / "database.dump").read_bytes() == b"RETRY-DUMP"
    assert not (seed_dir / "database.dump.tmp").exists()
    assert retry.leftovers == ({}, set(), set())


@pytest.mark.asyncio
async def test_prepare_removes_incomplete_seed_when_manifest_write_fails(
    tmp_path, monkeypatch
):
    data_dir = minimal_data_dir(tmp_path)
    seed_dir = tmp_path / "seed"
    records = list(MINIMAL_FILES["MimicPatient.ndjson.gz"]) + list(
        MINIMAL_FILES["MimicObservation.ndjson.gz"]
    )

    def failing_manifest_write(directory, manifest):
        (Path(directory) / "manifest.json.tmp").write_text("{", encoding="utf-8")
        raise data.SeedPreparationError("清单写入失败")

    monkeypatch.setattr(data, "write_manifest", failing_manifest_write)
    backend = FakeBackend()
    with pytest.raises(data.SeedPreparationError, match="清单写入失败"):
        await data.prepare_dataset(
            data_dir,
            seed_dir,
            backend=backend,
            client_factory=FakeFhir(records).client,
            require_published=False,
            log=lambda message: None,
        )

    assert backend.leftovers == ({}, set(), set())
    assert list(seed_dir.iterdir()) == []

    monkeypatch.undo()
    retry = FakeBackend(dump=b"RETRY-DUMP")
    manifest = await data.prepare_dataset(
        data_dir,
        seed_dir,
        backend=retry,
        client_factory=FakeFhir(records).client,
        require_published=False,
        log=lambda message: None,
    )
    assert manifest["database_sha256"] == hashlib.sha256(b"RETRY-DUMP").hexdigest()
    assert retry.leftovers == ({}, set(), set())


@pytest.mark.asyncio
async def test_prepare_docker_run_failure_never_exposes_password(tmp_path):
    data_dir = minimal_data_dir(tmp_path)
    backend = FakeBackend()
    backend.fail_run = True

    with pytest.raises(data.SeedPreparationError) as info:
        await data.prepare_dataset(
            data_dir,
            tmp_path / "seed",
            backend=backend,
            require_published=False,
            log=lambda message: None,
        )

    secrets_in_use = [
        value.split("=", 1)[1]
        for value in backend.env_values
        if value.startswith("POSTGRES_PASSWORD=")
    ]
    assert len(secrets_in_use) == 1
    password = secrets_in_use[0]
    assert password

    chain = [str(info.value), repr(info.value)]
    cause = info.value.__cause__
    while cause is not None:
        chain.append(str(cause))
        chain.append(repr(cause))
        cause = cause.__cause__
    assert info.value.__cause__ is None
    assert all(password not in text for text in chain)
    assert "docker run" in str(info.value)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["nonzero-exit", "timeout"])
async def test_docker_backend_redacts_password_env_values(mode):
    password = "s3cret-password-value"
    backend = data.DockerBackend(binary=sys.executable)
    if mode == "nonzero-exit":
        args = (
            "-c",
            "import sys; sys.stderr.write(' '.join(sys.argv[1:])); sys.exit(3)",
            "--env",
            f"POSTGRES_PASSWORD={password}",
            "--env",
            "POSTGRES_USER=medevolve",
        )
        options: dict = {}
    else:
        args = (
            "-c",
            "import time; time.sleep(30)",
            "--env",
            f"POSTGRES_PASSWORD={password}",
        )
        options = {"timeout": 0.2}

    with pytest.raises(data.SeedPreparationError) as info:
        await backend.run(*args, **options)

    message = str(info.value)
    assert password not in message
    assert "POSTGRES_PASSWORD=***" in message
    if mode == "nonzero-exit":
        assert "POSTGRES_USER=medevolve" in message


@pytest.mark.asyncio
async def test_docker_backend_stream_error_never_exposes_password(tmp_path):
    password = "s3cret-password-value"
    backend = data.DockerBackend(binary=sys.executable)
    sink = (tmp_path / "dump").open("wb")
    try:
        with pytest.raises(data.SeedPreparationError) as info:
            await backend.stream_to_file(
                (
                    "-c",
                    "import time; time.sleep(30)",
                    "--env",
                    f"PGPASSWORD={password}",
                ),
                sink,
                0.2,
            )
    finally:
        sink.close()
    assert password not in str(info.value)
    assert "PGPASSWORD=***" in str(info.value)


def test_probe_digest_ignores_server_generated_meta():
    source = observation("o1", "p1")
    stored = copy.deepcopy(source)
    stored["meta"] = {"versionId": "1", "lastUpdated": "2024-01-01T00:00:00Z"}
    assert data.resource_digest(stored) == data.resource_digest(source)
