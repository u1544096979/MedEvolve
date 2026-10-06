"""Public environment interfaces for the experiment framework."""

import inspect
import json
from datetime import datetime, timezone
from functools import wraps
from typing import get_type_hints

import httpx
from agents import Tool, function_tool

from contracts import (
    EnvironmentConfig,
    EnvironmentInstance,
    Evidence,
    JsonObject,
    PatientTrajectory,
    Task,
)
from environments.errors import EnvironmentUnavailable, UnsupportedDecisionPoint
from environments.fhir import FHIRClient, decode_document_notes
from environments.medagentbench import NativeTools, load_functions
from environments.physicianbench import FHIR_TOOL_SCHEMAS, NATIVE_FUNCTIONS
from environments.tool_results import serialize_tool_result, validate_tool_result_limit

_WRITABLE_RESOURCE_TYPES = {
    "Communication",
    "ServiceRequest",
    "MedicationRequest",
    "Appointment",
}

TOOL_RESULT_LIMIT = 10000

# 管理器一次创建的真实最坏路径由 server/fhir_agentbench.py 的 WORST_CASE_CREATE_SECONDS
# 推导（各相位等待上限之和 + 镜像校验、卷与网络创建、容器启动等固定开销上限），客户端
# 等待上限必须严格大于它并留出余量：客户端先放弃时管理器仍可能创建成功，返回的实例 id
# 没有任何一方收到，容器、卷与网络会成为只能人工按 id 回收的孤儿。
FHIR_AGENTBENCH_CREATE_TIMEOUT = 3000


def _short_system(system: str) -> str:
    known = {
        "http://loinc.org": "LOINC",
        "http://snomed.info/sct": "SNOMED",
        "http://www.nlm.nih.gov/research/umls/rxnorm": "RxNorm",
        "http://hl7.org/fhir/sid/icd-10-cm": "ICD-10-CM",
        "http://hl7.org/fhir/sid/icd-10": "ICD-10",
        "http://hl7.org/fhir/sid/cvx": "CVX",
    }
    return known.get(system, system)


def _compact_value(value):
    """Keep clinical values while making common FHIR datatypes concise."""
    if isinstance(value, list):
        return [_compact_value(item) for item in value]
    if not isinstance(value, dict):
        return value

    if "coding" in value and isinstance(value["coding"], list):
        parts = []
        for coding in value["coding"]:
            system = _short_system(coding.get("system", ""))
            code = coding.get("code", "")
            display = coding.get("display", "")
            parts.append("|".join(part for part in (system, code, display) if part))
        if value.get("text"):
            parts.append(value["text"])
        extras = {
            key: _compact_value(item)
            for key, item in value.items()
            if key not in {"coding", "text"}
        }
        return {"concept": parts, **extras} if extras else {"concept": parts}

    if "system" in value and "code" in value and set(value) <= {
        "system", "code", "display", "userSelected"
    }:
        return "|".join(
            part
            for part in (
                _short_system(value.get("system", "")),
                value.get("code", ""),
                value.get("display", ""),
            )
            if part
        )

    if "value" in value and ("unit" in value or "system" in value):
        extras = {
            key: _compact_value(item)
            for key, item in value.items()
            if key not in {"value", "unit", "system", "code"}
        }
        quantity = " ".join(
            str(part)
            for part in (value.get("value"), value.get("unit") or value.get("code"))
            if part is not None
        )
        return {"quantity": quantity, **extras} if extras else quantity

    if "reference" in value and set(value) <= {
        "reference", "display", "type", "identifier", "extension", "id"
    }:
        reference = value.get("reference", "")
        if value.get("display"):
            reference += f" ({value['display']})"
        extras = {
            key: _compact_value(item)
            for key, item in value.items()
            if key not in {"reference", "display"}
        }
        return {"reference": reference, **extras} if extras else reference

    return {
        key: _compact_value(item)
        for key, item in value.items()
        if key != "meta"
    }


def _trajectory_resource(resource: dict) -> dict:
    compact = {
        key: value
        for key, value in decode_document_notes(resource).items()
        if key not in {"meta", "id", "resourceType", "subject"}
    }
    return _compact_value(compact)


def _check_environment(name: str) -> None:
    if name not in {"PhysicianBench", "MedAgentBench", "FHIR-AgentBench"}:
        raise EnvironmentUnavailable(
            f"{name} is unavailable: original implementation is not integrated"
        )


def _check_decision_point(decision_point: str | None) -> None:
    if decision_point:
        raise UnsupportedDecisionPoint(
            "PhysicianBench does not yet support decision-point time filtering"
        )


async def create_environment(config: EnvironmentConfig) -> EnvironmentInstance:
    if config.name not in {"PhysicianBench", "MedAgentBench", "FHIR-AgentBench"}:
        raise EnvironmentUnavailable(f"{config.name} is unavailable")
    # 本地 FHIR 实例需要恢复完整数据库种子，等待时间远长于镜像启动。
    timeout = (
        FHIR_AGENTBENCH_CREATE_TIMEOUT if config.name == "FHIR-AgentBench" else 240
    )
    async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
        response = await client.post(
            config.manager_url.rstrip("/") + "/instances", json=config.model_dump()
        )
        response.raise_for_status()
        instance = EnvironmentInstance.model_validate(response.json())
        if instance.environment != config.name:
            raise EnvironmentUnavailable("Manager returned a different environment")
        return instance


async def get_tools(
    instance: EnvironmentInstance,
    patient_ids: list[str],
    decision_point: str | None = None,
    *,
    tool_result_limit: int = TOOL_RESULT_LIMIT,
) -> list[Tool]:
    _check_environment(instance.environment)
    _check_decision_point(decision_point)
    if instance.environment == "FHIR-AgentBench":
        from .fhir_agentbench import get_native_tools

        return get_native_tools(instance, patient_ids)
    if instance.environment == "MedAgentBench":
        return NativeTools(instance.endpoint, patient_ids, load_functions())
    validate_tool_result_limit(tool_result_limit)
    client = FHIRClient(instance.endpoint, patient_ids)
    schemas = {schema["name"]: schema for schema in FHIR_TOOL_SCHEMAS}

    def tool_error(context, error):
        if isinstance(error, (EnvironmentUnavailable, UnsupportedDecisionPoint)):
            raise error
        return serialize_tool_result(
            {"error": f"{type(error).__name__}: {error}. Please retry with valid arguments."},
            tool_result_limit,
            error=True,
        )

    def bind(native):
        @wraps(native)
        async def invoke(*args, **kwargs):
            value = await native(client, *args, **kwargs)
            return serialize_tool_result(
                value, tool_result_limit, write=native.__name__ in {
                    "fhir_communication_create_message", "fhir_service_request_create",
                    "fhir_medication_request_create", "fhir_appointment_create",
                },
            )

        signature = inspect.signature(native)
        invoke.__signature__ = signature.replace(
            parameters=list(signature.parameters.values())[1:]
        )
        invoke.__annotations__ = {
            key: value
            for key, value in get_type_hints(native).items()
            if key != "client"
        }
        schema = schemas[native.__name__]
        tool = function_tool(invoke, strict_mode=False, failure_error_function=tool_error)
        tool.description = schema["description"]
        tool.params_json_schema = schema["parameters"]
        return tool

    return [bind(native) for native in NATIVE_FUNCTIONS]


async def read_state(instance: EnvironmentInstance, task: Task) -> JsonObject:
    _check_environment(instance.environment)
    if task.environment != instance.environment:
        raise ValueError("Task environment does not match the instance")
    _check_decision_point(task.decision_point)
    if instance.environment == "FHIR-AgentBench":
        # 只读环境没有写后状态：完成判断依据 solver 检索 trace、答案与原始证据。
        # 因此既不读全量病历，也不套用 PhysicianBench 的可写资源筛选。
        return {
            "patient_id": task.patient_id,
            "environment": instance.environment,
            "operation_semantics": {"read_only": True},
        }
    client = FHIRClient(instance.endpoint, [task.patient_id])
    if instance.environment == "MedAgentBench":
        resources = await _snapshot(client, task.patient_id)
    else:
        resources = await client.snapshot(task.patient_id)
    if instance.environment == "MedAgentBench":
        return {
            "patient_id": task.patient_id,
            "environment": instance.environment,
            "resources": resources,
            "operation_semantics": {
                "post": "accept_request_without_persistence",
                "acceptance_proves_database_write": False,
            },
        }
    return {
        "patient_id": task.patient_id,
        "environment": instance.environment,
        "resources": [
            resource
            for resource in resources
            if resource.get("resourceType") in _WRITABLE_RESOURCE_TYPES
        ],
    }


async def read_evidence(
    instance: EnvironmentInstance, patient_id: str, source_id: str
) -> JsonObject:
    _check_environment(instance.environment)
    if instance.environment == "FHIR-AgentBench":
        from .fhir_agentbench import read_source

        return await read_source(instance, patient_id, source_id)
    client = FHIRClient(instance.endpoint, [patient_id])
    if instance.environment == "MedAgentBench":
        parts = source_id.split("/")
        if len(parts) != 2:
            raise ValueError("Evidence must be a local ResourceType/id reference")
        resource = await client._request("GET", source_id)
        if resource.get("resourceType") != parts[0] or resource.get("id") != parts[1]:
            raise ValueError("FHIR server returned a different evidence resource")
        client._check_resource(resource, set(client.patient_ids))
        return resource
    return await client.read(source_id)


async def release_environment(
    config: EnvironmentConfig, instance: EnvironmentInstance
) -> None:
    if config.name != instance.environment:
        raise ValueError("Configuration environment does not match the instance")
    if config.name == "FHIR-AgentBench":
        from .fhir_agentbench import forget_instance

        forget_instance(instance.instance_id)
    # 双容器实例要依次回收两个容器、数据库卷与私有网络。
    timeout = 180 if config.name == "FHIR-AgentBench" else 90
    async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
        response = await client.delete(
            config.manager_url.rstrip("/") + "/instances/" + instance.instance_id
        )
        response.raise_for_status()


def _record_time(resource: dict) -> str | None:
    for field in (
        "effectiveDateTime",
        "onsetDateTime",
        "performedDateTime",
        "occurrenceDateTime",
        "authoredOn",
        "sent",
        "start",
        "date",
        "recordedDate",
        "issued",
    ):
        if isinstance(resource.get(field), str):
            return resource[field]
    for field in ("effectivePeriod", "onsetPeriod", "performedPeriod", "period"):
        if resource.get(field, {}).get("start"):
            return resource[field]["start"]
    return resource.get("context", {}).get("period", {}).get("start")


def _time_sort(resource: dict) -> tuple:
    value = _record_time(resource)
    if value:
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            timestamp = parsed.replace(tzinfo=parsed.tzinfo or timezone.utc).timestamp()
            return (
                0,
                timestamp,
                resource.get("resourceType", ""),
                resource.get("id", ""),
            )
        except ValueError:
            pass
    return (1, 0, resource.get("resourceType", ""), resource.get("id", ""))


async def prepare_trajectory(
    instance: EnvironmentInstance, patient_id: str
) -> PatientTrajectory:
    _check_environment(instance.environment)
    if instance.environment == "FHIR-AgentBench":
        from .fhir_agentbench_trajectory import prepare

        return await prepare(instance, patient_id)
    client = FHIRClient(instance.endpoint, [patient_id])
    if instance.environment == "MedAgentBench":
        resources = await _snapshot(client, patient_id)
    else:
        resources = await client.snapshot(patient_id)
    sources = []
    text = []
    seen = set()
    for resource in sorted(resources, key=_time_sort):
        source_id = f"{resource['resourceType']}/{resource['id']}"
        if source_id in seen:
            continue
        seen.add(source_id)
        # Original JSON is retained in Evidence; text attachments are decoded only for the body.
        sources.append(
            Evidence(
                source_id=source_id,
                statement=json.dumps(resource, ensure_ascii=False, sort_keys=True),
            )
        )
        text.append(
            f"[{_record_time(resource) or 'undated'}] {source_id}\n"
            + json.dumps(
                _trajectory_resource(resource), ensure_ascii=False, sort_keys=True,
                separators=(",", ":"),
            )
        )
    return PatientTrajectory(
        patient_id=patient_id,
        environment=instance.environment,
        text="\n\n".join(text),
        sources=sources,
    )


async def _snapshot(client: FHIRClient, identifier: str) -> list[dict]:
    """Read complete patient material, falling back when $everything is unavailable."""
    try:
        return await client.snapshot(identifier)
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code not in {400, 404, 405, 501}:
            raise
    resources = []
    for resource_type in (
        "Patient", "Encounter", "Condition", "Observation",
        "MedicationRequest", "Procedure", "ServiceRequest",
    ):
        params = {"_count": 200}
        params["_id" if resource_type == "Patient" else "patient"] = identifier
        result = await client._collect(resource_type, params, None, {identifier})
        resources.extend(result["entries"])
    return resources
