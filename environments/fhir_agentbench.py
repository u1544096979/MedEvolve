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
- 患者材料读取优先用 ``Patient/$everything``，仅在服务端明确不支持时逐类型分页回退；
  回退与引用补齐都先核对能力声明的检索参数，再按 ``ResourceType/id`` 去重。
- 引用补齐与共享资源可达性都迭代到不动点：``$everything`` 返回的传递闭包、逐类型回退的
  一跳读取与共享记录之间的内部引用必须得到同一份材料。
"""

import json

import httpx
from agents import Tool, function_tool

from contracts import EnvironmentInstance, JsonObject
from environments.errors import EnvironmentUnavailable, PatientAccessError
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

# 官方 supported_types 里本地不支持 patient= 检索的共享资源类型；Organization 由
# 患者记录引用后才可读，用于补齐机构名称等原始信息。
SHARED_RESOURCE_TYPES = ("Medication", "Location", "Organization")

# 共享资源没有患者归属：只有从开放患者可读记录中引用可达的记录才可读。
SHARED_REFERENCING_TYPES = {
    "Medication": (
        "MedicationRequest",
        "MedicationAdministration",
        "MedicationStatement",
        "MedicationDispense",
    ),
    "Location": ("Encounter",),
    "Organization": ("Encounter", "Procedure"),
}

# 共享资源可达性与引用补齐的迭代上限：引用链远长于此说明数据或实现有问题，
# 应当报错而不是静默截断成一个看起来完整的子集。
REFERENCE_ROUNDS = 16

# 服务端不支持 Patient/$everything 时逐类型读取的患者临床类型。
FALLBACK_RESOURCE_TYPES = (
    "Patient",
    "Encounter",
    "Condition",
    "Observation",
    "MedicationRequest",
    "MedicationAdministration",
    "MedicationStatement",
    "MedicationDispense",
    "Procedure",
    "Specimen",
)

# 患者记录引用后需要补齐的记录类型：共享资源加上回退读取可能漏掉的临床关联记录。
COMPLETION_RESOURCE_TYPES = tuple(
    resource_type for resource_type in FALLBACK_RESOURCE_TYPES if resource_type != "Patient"
) + ("Medication", "Location", "Organization")

# 只有这些状态明确表示服务端不支持完整患者读取，其余状态照常上报。
UNSUPPORTED_STATUSES = frozenset({400, 404, 405, 501})

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


def _dedupe(resources: list[dict]) -> list[dict]:
    """按 ResourceType/id 去重，保留完整原始 JSON。"""
    unique: dict[tuple[str, str], dict] = {}
    for resource in resources:
        unique[(resource.get("resourceType"), resource.get("id"))] = resource
    return list(unique.values())


class ScopedReader:
    """把一次运行的读取限制在当前实例与开放患者集合内。"""

    def __init__(self, instance: EnvironmentInstance, patients: frozenset[str]):
        self.client = FHIRClient(instance.endpoint, sorted(patients))
        self._referenced: dict[tuple[str, str], set[str]] = {}
        self._shared: dict[frozenset[str], dict[str, set[str]]] = {}
        self._search_params: dict[str, set[str]] | None = None

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
        """开放患者引用可达的共享记录 ID：沿共享记录内部引用取传递闭包。"""
        reachable = await self._shared_reachability(selected)
        return set(reachable.get(resource_type, ()))

    async def _shared_reachability(
        self, selected: set[str]
    ) -> dict[str, set[str]]:
        """按开放患者为唯一入口，展开共享资源之间的引用闭包。

        起点是开放患者所属记录对共享类型的直接引用（如 MedicationRequest 的
        medicationReference），再沿共享记录内部的共享引用继续展开（如
        ``Medication.ingredient.itemReference``、``Location.partOf``），因此与
        ``Patient/$everything`` 的传递闭包口径一致。
        """
        key = frozenset(selected)
        cached = self._shared.get(key)
        if cached is not None:
            return cached
        allowed = {resource_type: set() for resource_type in SHARED_RESOURCE_TYPES}
        for patient in sorted(key):
            for resource_type in SHARED_RESOURCE_TYPES:
                allowed[resource_type] |= await self._referenced_by(
                    patient, resource_type
                )
        scanned = {resource_type: set() for resource_type in SHARED_RESOURCE_TYPES}
        for _ in range(REFERENCE_ROUNDS):
            pending: dict[str, list[str]] = {}
            for resource_type, identifiers in allowed.items():
                fresh = identifiers - scanned[resource_type]
                if fresh:
                    pending[resource_type] = sorted(fresh)
            if not pending:
                break
            for resource_type, identifiers in sorted(pending.items()):
                scanned[resource_type].update(identifiers)
                records = await self.client.search_all(
                    resource_type,
                    {"_id": ",".join(identifiers), "_count": PAGE_SIZE},
                    key,
                )
                missing = set(identifiers) - {record.get("id") for record in records}
                if missing:
                    raise EnvironmentUnavailable(
                        f"FHIR server did not return reachable {resource_type} records: "
                        + ", ".join(sorted(missing))
                    )
                for record in records:
                    for reference in references(record):
                        referenced_type, _, referenced_id = reference.partition("/")
                        if referenced_type in allowed and referenced_id:
                            allowed[referenced_type].add(referenced_id)
        else:
            raise EnvironmentUnavailable(
                "FHIR shared references did not converge within "
                f"{REFERENCE_ROUNDS} rounds"
            )
        self._shared[key] = allowed
        return allowed

    async def _referenced_by(self, patient: str, resource_type: str) -> set[str]:
        key = (patient, resource_type)
        if key not in self._referenced:
            referenced: set[str] = set()
            for referencing in SHARED_REFERENCING_TYPES.get(resource_type, ()):
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

    async def supported_search_params(self, resource_type: str) -> set[str]:
        """读取服务端能力声明，返回该类型实际可用的检索参数名。"""
        if self._search_params is None:
            statement = await self.client._request("GET", "metadata")
            declared: dict[str, set[str]] = {}
            for rest in statement.get("rest", []):
                for resource in rest.get("resource", []):
                    name = resource.get("type")
                    if not name:
                        continue
                    names = declared.setdefault(name, set())
                    for param in resource.get("searchParam", []):
                        if param.get("name"):
                            names.add(param["name"])
            self._search_params = declared
        return self._search_params.get(resource_type, set())

    async def _patient_param(self, resource_type: str) -> str:
        supported = await self.supported_search_params(resource_type)
        for name in ("patient", "subject"):
            if name in supported:
                return name
        raise EnvironmentUnavailable(
            f"FHIR server does not support patient search for {resource_type}"
        )

    async def _read_by_type(
        self, identifier: str, selected: set[str]
    ) -> list[dict]:
        """完整患者读取不可用时的逐类型回退读取。"""
        resources: list[dict] = []
        for resource_type in FALLBACK_RESOURCE_TYPES:
            if resource_type == "Patient":
                params = {"_id": identifier, "_count": PAGE_SIZE}
            else:
                params = {
                    await self._patient_param(resource_type): identifier,
                    "_count": PAGE_SIZE,
                }
            resources.extend(
                await self.client.search_all(resource_type, params, selected)
            )
        if not any(
            resource.get("resourceType") == "Patient"
            and resource.get("id") == identifier
            for resource in resources
        ):
            raise ValueError(f"Patient search did not return Patient/{identifier}")
        return resources

    async def _read_referenced(
        self, resource_type: str, identifiers: set[str], selected: set[str]
    ) -> list[dict]:
        """按 ID 读取引用记录；服务端没有 _id 时退回按患者检索后过滤。"""
        supported = await self.supported_search_params(resource_type)
        if "_id" in supported:
            return await self.client.search_all(
                resource_type,
                {"_id": ",".join(sorted(identifiers)), "_count": PAGE_SIZE},
                selected,
            )
        if resource_type in SHARED_RESOURCE_TYPES:
            raise EnvironmentUnavailable(
                f"FHIR server cannot read {resource_type} by ID"
            )
        patient = next(iter(selected))
        records = await self.client.search_all(
            resource_type,
            {await self._patient_param(resource_type): patient, "_count": PAGE_SIZE},
            selected,
        )
        return [record for record in records if record.get("id") in identifiers]

    async def _complete_references(
        self, resources: list[dict], selected: set[str]
    ) -> list[dict]:
        """扫描引用并按 ID 补齐材料里缺少的关联记录，迭代到闭包不再变化。

        只扫描一轮会让第二跳之后的记录（例如 ``Medication.ingredient`` 引用的复方成分）
        在逐类型回退路径上丢掉，而 ``$everything`` 路径已经把它们带进材料。
        """
        material = list(resources)
        existing = {(r.get("resourceType"), r.get("id")) for r in material}
        for _ in range(REFERENCE_ROUNDS):
            wanted: dict[str, set[str]] = {}
            for resource in material:
                for reference in references(resource):
                    resource_type, _, identifier = reference.partition("/")
                    if not identifier or resource_type not in COMPLETION_RESOURCE_TYPES:
                        continue
                    if (resource_type, identifier) in existing:
                        continue
                    wanted.setdefault(resource_type, set()).add(identifier)
            if not wanted:
                return material
            for resource_type, identifiers in sorted(wanted.items()):
                records = await self._read_referenced(resource_type, identifiers, selected)
                missing = identifiers - {record.get("id") for record in records}
                if missing:
                    raise ValueError(
                        "FHIR server did not return referenced records: "
                        + ", ".join(
                            f"{resource_type}/{identifier}"
                            for identifier in sorted(missing)
                        )
                    )
                material.extend(records)
                existing.update(
                    (resource_type, record.get("id")) for record in records
                )
        raise ValueError(
            "FHIR reference completion did not converge within "
            f"{REFERENCE_ROUNDS} rounds"
        )

    async def patient_resources(self, patient: str) -> list[dict]:
        selected = self._scope(patient)
        if len(selected) != 1:
            raise PatientAccessError("Patient resources require exactly one patient")
        identifier = next(iter(selected))
        try:
            resources = await self.client.snapshot(identifier)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code not in UNSUPPORTED_STATUSES:
                raise
        else:
            return _dedupe(resources)
        return _dedupe(await self._read_by_type(identifier, selected))

    async def patient_material(self, patient: str) -> list[dict]:
        """完整患者材料：患者关联记录加上引用可达的关联记录。"""
        selected = self._scope(patient)
        resources = await self.patient_resources(patient)
        return _dedupe(await self._complete_references(resources, selected))


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


async def read_patient_material(
    instance: EnvironmentInstance, patient_id: str
) -> list[dict]:
    return await _reader_for(instance, [patient_id]).patient_material(patient_id)


async def read_source(
    instance: EnvironmentInstance, patient_id: str, source_id: str
) -> JsonObject:
    return await _reader_for(instance, [patient_id]).read_source(patient_id, source_id)
