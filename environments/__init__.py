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
from environments.physicianbench import NATIVE_FUNCTIONS


def _check_environment(name: str) -> None:
    if name != "PhysicianBench":
        raise EnvironmentUnavailable(
            f"{name} is unavailable: original implementation is not integrated"
        )


def _check_decision_point(decision_point: str | None) -> None:
    if decision_point:
        raise UnsupportedDecisionPoint(
            "PhysicianBench does not yet support decision-point time filtering"
        )


async def create_environment(config: EnvironmentConfig) -> EnvironmentInstance:
    _check_environment(config.name)
    async with httpx.AsyncClient(timeout=240, trust_env=False) as client:
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
) -> list[Tool]:
    _check_environment(instance.environment)
    _check_decision_point(decision_point)
    client = FHIRClient(instance.endpoint, patient_ids)

    def bind(native):
        @wraps(native)
        async def invoke(*args, **kwargs):
            return await native(client, *args, **kwargs)

        signature = inspect.signature(native)
        invoke.__signature__ = signature.replace(
            parameters=list(signature.parameters.values())[1:]
        )
        invoke.__annotations__ = {
            key: value
            for key, value in get_type_hints(native).items()
            if key != "client"
        }
        return function_tool(invoke, failure_error_function=None)

    return [bind(native) for native in NATIVE_FUNCTIONS]


async def read_state(instance: EnvironmentInstance, task: Task) -> JsonObject:
    _check_environment(instance.environment)
    if task.environment != instance.environment:
        raise ValueError("Task environment does not match the instance")
    _check_decision_point(task.decision_point)
    resources = await FHIRClient(instance.endpoint, [task.patient_id]).snapshot(
        task.patient_id
    )
    return {
        "patient_id": task.patient_id,
        "environment": instance.environment,
        "resources": resources,
    }


async def release_environment(
    config: EnvironmentConfig, instance: EnvironmentInstance
) -> None:
    if config.name != instance.environment:
        raise ValueError("Configuration environment does not match the instance")
    async with httpx.AsyncClient(timeout=90, trust_env=False) as client:
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
    resources = await FHIRClient(instance.endpoint, [patient_id]).snapshot(patient_id)
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
                decode_document_notes(resource), ensure_ascii=False, sort_keys=True
            )
        )
    return PatientTrajectory(
        patient_id=patient_id,
        environment=instance.environment,
        text="\n\n".join(text),
        sources=sources,
    )
