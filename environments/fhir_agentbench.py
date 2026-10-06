"""FHIR-AgentBench 原生检索工具：官方多轮资源检索路径的本地适配。

工具名、参数名、参数 schema 与 description 沿用官方提交
``bbb42909a5a7eb907d1cd91f72a560729e7037ea`` 的 ``tools/resource_tools.py``；
返回结构同为按 ``resourceType`` 分组、移除 ``text`` 与 ``meta`` 的记录字典。

与官方的差异：

- 官方 ``fhir_client.py`` 删除记录里的 ``text``、``meta`` 后仍用 ``id=`` 检索并按固定
  Google Cloud 端点连接；本地服务用 FHIR R4 支持的 ``_id``，客户端连接当前实例地址。
- 官方 ``except Exception: return {"error": str(e)}`` 会把失败变成看起来成功的返回；
  这里失败抛异常，交给既有运行错误通道，空结果仍是 ``{}``。
- 官方磁盘缓存会在实例之间共享结果；本适配不写磁盘，读取缓存只在实例内生效。
- 官方不核对分页链接归属与患者范围；这里分页必须留在当前实例内，共享资源只能读到
  开放患者引用可达的记录。
"""

import json

from agents import Tool, function_tool

from contracts import EnvironmentInstance, JsonObject
from environments.errors import PatientAccessError
from environments.fhir import FHIRClient, patient_id, references

SUPPORTED_TYPES = (
    "Patient",
    "Encounter",
    "Condition",
    "MedicationRequest",
    "Procedure",
    "Observation",
    "MedicationAdministration",
    "Location",
    "Specimen",
    "Medication",
)

# 官方 supported_types 里本地不支持 patient= 检索的共享资源类型。
SHARED_RESOURCE_TYPES = ("Medication", "Location")

# 共享资源没有患者归属：只有从开放患者可读记录中引用可达的记录才可读。
SHARED_REFERENCING_TYPES = {
    "Medication": (
        "MedicationRequest",
        "MedicationAdministration",
        "MedicationStatement",
        "MedicationDispense",
    ),
    "Location": ("Encounter",),
}

PAGE_SIZE = 200

TOOL_SCHEMAS = [
    {
        "name": "get_resources_by_resource_id",
        "description": "Get FHIR resources by resource ID",
        "parameters": {
            "type": "object",
            "properties": {
                "resource_type": {"type": "string", "description": "FHIR resource type"},
                "resource_id": {"type": "string", "description": "FHIR resource ID"},
            },
            "required": ["resource_type", "resource_id"],
        },
    },
    {
        "name": "get_resources_by_patient_fhir_id",
        "description": "Get FHIR resources by patient FHIR ID",
        "parameters": {
            "type": "object",
            "properties": {
                "resource_type": {"type": "string", "description": "FHIR resource type"},
                "patient_fhir_id": {"type": "string", "description": "Patient FHIR ID"},
            },
            "required": ["resource_type", "patient_fhir_id"],
        },
    },
]


def _check_resource_type(resource_type: str) -> None:
    if resource_type not in SUPPORTED_TYPES:
        raise ValueError(
            f"Unsupported FHIR resource type: {resource_type}. "
            f"Expected one of: {', '.join(SUPPORTED_TYPES)}"
        )


class ScopedReader:
    """把一次运行的读取限制在当前实例与开放患者集合内。"""

    def __init__(self, instance: EnvironmentInstance, patients: frozenset[str]):
        self.client = FHIRClient(instance.endpoint, sorted(patients))
        self._referenced: dict[tuple[str, str], set[str]] = {}

    @property
    def patient_ids(self) -> frozenset[str]:
        return self.client.patient_ids

    def _scope(self, identifier: str) -> set[str]:
        if not identifier:
            return set(self.patient_ids)
        selected = {patient_id(identifier)}
        if not selected <= self.patient_ids:
            raise PatientAccessError(
                f"Patient {identifier} is outside the allowed patient scope"
            )
        return selected

    async def by_patient(self, resource_type: str, patient_fhir_id: str) -> list[dict]:
        _check_resource_type(resource_type)
        selected = self._scope(patient_fhir_id)
        if resource_type in SHARED_RESOURCE_TYPES:
            return await self._shared_records(resource_type, selected)
        params = {"_count": PAGE_SIZE}
        if resource_type == "Patient":
            # 本地 HAPI R4 的 Patient 检索参数是 `_id`，不是官方的 `id=`。
            params["_id"] = ",".join(sorted(selected))
        else:
            params["patient"] = ",".join(sorted(selected))
        return await self.client.search_all(resource_type, params, selected)

    async def by_id(self, resource_type: str, resource_id: str) -> list[dict]:
        _check_resource_type(resource_type)
        identifiers = [
            part.strip() for part in (resource_id or "").split(",") if part.strip()
        ]
        if not identifiers:
            return await self.by_patient(resource_type, "")
        selected = set(self.patient_ids)
        if resource_type in SHARED_RESOURCE_TYPES:
            allowed = await self._reachable_ids(resource_type, selected)
            for identifier in identifiers:
                if identifier not in allowed:
                    raise PatientAccessError(
                        f"{resource_type}/{identifier} is not referenced by an allowed patient"
                    )
        if resource_type == "Patient":
            for identifier in identifiers:
                self._scope(identifier)
        return await self.client.search_all(
            resource_type,
            {"_id": ",".join(identifiers), "_count": PAGE_SIZE},
            selected,
        )

    async def _reachable_ids(self, resource_type: str, selected: set[str]) -> set[str]:
        allowed: set[str] = set()
        for patient in sorted(selected):
            allowed |= await self._referenced_by(patient, resource_type)
        return allowed

    async def _referenced_by(self, patient: str, resource_type: str) -> set[str]:
        key = (patient, resource_type)
        if key not in self._referenced:
            referenced: set[str] = set()
            for referencing in SHARED_REFERENCING_TYPES[resource_type]:
                records = await self.client.search_all(
                    referencing, {"_count": PAGE_SIZE, "patient": patient}, {patient}
                )
                for record in records:
                    for reference in references(record):
                        referenced_type, _, identifier = reference.partition("/")
                        if referenced_type == resource_type and identifier:
                            referenced.add(identifier)
            self._referenced[key] = referenced
        return self._referenced[key]

    async def _shared_records(self, resource_type: str, selected: set[str]) -> list[dict]:
        allowed = await self._reachable_ids(resource_type, selected)
        if not allowed:
            return []
        records = await self.client.search_all(
            resource_type,
            {"_id": ",".join(sorted(allowed)), "_count": PAGE_SIZE},
            selected,
        )
        for record in records:
            if record.get("id") not in allowed:
                raise PatientAccessError(
                    f"FHIR server returned an unreachable {resource_type} record"
                )
        return records

    async def read_source(self, patient: str, source_id: str) -> dict:
        resource_type, _, identifier = source_id.partition("/")
        if resource_type in SHARED_RESOURCE_TYPES:
            allowed = await self._reachable_ids(resource_type, self._scope(patient))
            if identifier not in allowed:
                raise PatientAccessError(
                    f"{source_id} is not referenced by an allowed patient"
                )
            return await self.client.read_shared(source_id)
        return await self.client.read(source_id)

    async def patient_resources(self, patient: str) -> list[dict]:
        selected = self._scope(patient)
        if len(selected) != 1:
            raise PatientAccessError("Patient resources require exactly one patient")
        return await self.client.snapshot(next(iter(selected)))


_READERS: dict[tuple[str, str, frozenset[str]], ScopedReader] = {}


def _reader_for(
    instance: EnvironmentInstance, patient_ids: list[str]
) -> ScopedReader:
    patients = frozenset(patient_id(item) for item in patient_ids)
    key = (instance.instance_id, instance.endpoint, patients)
    reader = _READERS.get(key)
    if reader is None:
        reader = ScopedReader(instance, patients)
        _READERS[key] = reader
    return reader


def forget_instance(instance_id: str) -> None:
    """实例释放后丢弃按实例隔离的读取缓存。"""
    for key in [key for key in _READERS if key[0] == instance_id]:
        del _READERS[key]


def _grouped_tool_output(records: list[dict]) -> dict:
    result: dict[str, list[dict]] = {}
    for record in records:
        cleaned = {
            key: value for key, value in record.items() if key not in {"text", "meta"}
        }
        result.setdefault(record["resourceType"], []).append(cleaned)
    return result


def get_native_tools(
    instance: EnvironmentInstance, patient_ids: list[str]
) -> list[Tool]:
    reader = _reader_for(instance, patient_ids)
    schemas = {schema["name"]: schema for schema in TOOL_SCHEMAS}

    async def get_resources_by_resource_id(resource_type: str, resource_id: str) -> dict:
        records = await reader.by_id(resource_type, resource_id)
        return json.dumps(_grouped_tool_output(records), ensure_ascii=False)

    async def get_resources_by_patient_fhir_id(
        resource_type: str, patient_fhir_id: str
    ) -> dict:
        records = await reader.by_patient(resource_type, patient_fhir_id)
        return json.dumps(_grouped_tool_output(records), ensure_ascii=False)

    tools = []
    for native in (get_resources_by_resource_id, get_resources_by_patient_fhir_id):
        schema = schemas[native.__name__]
        # 失败不交给失败格式化函数：异常继续向上走运行错误通道，不返回看似成功的返回。
        tool = function_tool(native, strict_mode=False, failure_error_function=None)
        tool.description = schema["description"]
        tool.params_json_schema = schema["parameters"]
        tools.append(tool)
    return tools


async def read_patient_resources(
    instance: EnvironmentInstance, patient_id: str
) -> list[dict]:
    return await _reader_for(instance, [patient_id]).patient_resources(patient_id)


async def read_source(
    instance: EnvironmentInstance, patient_id: str, source_id: str
) -> JsonObject:
    return await _reader_for(instance, [patient_id]).read_source(patient_id, source_id)
