"""FHIR-AgentBench 种子校验与清单读取。

种子目录包含 `manifest.json` 与 `database.dump`。清单记录固定的数据版本、镜像摘要、
源数据与数据库备份校验值、按类型资源数量，以及用于启动就绪检查的原始资源探针。
"""

import hashlib
import json
import re
from pathlib import Path

SEED_SCHEMA_VERSION = 1
SEED_DATA_VERSION = "2.1.0"
SEED_PATIENT_COUNT = 100

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


def file_digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


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
