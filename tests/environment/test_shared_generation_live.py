"""Opt-in five-patient read-only checks against an already prepared environment.

Set one of MEDEVOLVE_TEST_IMAGE, MEDEVOLVE_MEDAGENTBENCH_TEST_IMAGE, or
MEDEVOLVE_FHIR_AGENTBENCH_SEED. The checks call no models and pull no assets.
"""

import asyncio
import os

import httpx
import pytest

from contracts import EnvironmentConfig
from environments import get_tools, prepare_trajectory, read_evidence
from environments.capabilities import get_capabilities
from environments.fhir import FHIRClient
from environments.medagentbench import ReadOnlyNativeTools
from environments.errors import PatientAccessError
from evaluation.session import EvaluationEnvironments


ASSETS = (
    ("PhysicianBench", "MEDEVOLVE_TEST_IMAGE", "http://127.0.0.1:18084"),
    ("MedAgentBench", "MEDEVOLVE_MEDAGENTBENCH_TEST_IMAGE", "http://127.0.0.1:18084"),
    ("FHIR-AgentBench", "MEDEVOLVE_FHIR_AGENTBENCH_SEED", "http://127.0.0.1:18086"),
)
STAGES = ("prepare", "explore", "generate", "verify", "solve")


def configured_asset():
    for name, variable, manager_url in ASSETS:
        value = os.environ.get(variable)
        if value:
            return name, value, manager_url
    return None


def assert_read_only_tools(name, tools):
    if isinstance(tools, ReadOnlyNativeTools):
        assert all(item["name"].startswith("GET ") for item in tools.functions)
        return
    declared_queries = set(get_capabilities(name)["query_tools"])
    if declared_queries != {"all"}:
        assert {tool.name for tool in tools} <= declared_queries


async def read_only_query(instance, patient_id, tools):
    if isinstance(tools, ReadOnlyNativeTools):
        result = await tools.execute("GET Patient?_id=" + patient_id)
        assert result["kind"] == "GET"
        data = result["data"]
        if isinstance(data, str):
            import json
            data = json.loads(data)
        assert any(entry["resource"]["id"] == patient_id for entry in data.get("entry", []))


@pytest.mark.asyncio
@pytest.mark.skipif(configured_asset() is None, reason="Set one prepared local environment asset")
async def test_five_patients_share_one_read_only_instance_across_stages(monkeypatch):
    name, asset, manager_url = configured_asset()
    config = EnvironmentConfig(name=name, manager_url=manager_url, seed_data=asset)
    session = EvaluationEnvironments(instance_budget=1)
    try:
        async with session.instance(config, phase="prepare") as instance:
            async with httpx.AsyncClient(trust_env=False, timeout=30) as client:
                response = await client.get(instance.endpoint + "Patient", params={"_count": 10})
                response.raise_for_status()
            patient_ids = [entry["resource"]["id"] for entry in response.json().get("entry", [])
                           if entry.get("resource", {}).get("resourceType") == "Patient"][:5]
            if len(patient_ids) < 5:
                pytest.skip("configured seed exposes fewer than five patients")

        # Prove the FHIR client rejects writes before it creates an HTTP client.
        client_creations = 0
        class UnexpectedHTTPClient:
            def __init__(self, *args, **kwargs):
                nonlocal client_creations
                client_creations += 1
                raise AssertionError("write request reached the HTTP client")
        with monkeypatch.context() as patch:
            patch.setattr(httpx, "AsyncClient", UnexpectedHTTPClient)
            readonly_client = FHIRClient(instance.endpoint, [patient_ids[0]], allow_database_write=False)
            with pytest.raises(PatientAccessError, match="read-only FHIR queries"):
                await readonly_client._request("POST", "Observation", json={})
        assert client_creations == 0

        async def exercise(patient_id):
            stage_instances = {}
            trajectory = None
            source_id = None
            for stage in STAGES:
                async with session.instance(config, phase=stage) as instance:
                    stage_instances[stage] = instance.instance_id
                    if stage == "prepare":
                        trajectory = await prepare_trajectory(instance, patient_id)
                        assert trajectory.patient_id == patient_id
                        assert trajectory.sources
                        source_id = trajectory.sources[0].source_id
                    else:
                        tools = await get_tools(instance, [patient_id], read_only=True)
                        assert_read_only_tools(name, tools)
                        await read_only_query(instance, patient_id, tools)
                    evidence = await read_evidence(instance, patient_id, source_id or trajectory.sources[0].source_id)
                    assert evidence.get("id") == (source_id or trajectory.sources[0].source_id).split("/", 1)[1]
                    if stage == "prepare":
                        source_id = trajectory.sources[0].source_id
            assert set(stage_instances) == set(STAGES)
            assert len(set(stage_instances.values())) == 1

            async with session.instance(config, phase="explore") as instance:
                tools = await get_tools(instance, [patient_id], read_only=True)
                if isinstance(tools, ReadOnlyNativeTools):
                    source_calls = 0
                    original_execute = tools._source.execute
                    async def count_source(action):
                        nonlocal source_calls
                        source_calls += 1
                        return await original_execute(action)
                    tools._source.execute = count_source
                    denied = await tools.execute("POST " + instance.endpoint + "Observation\n{}")
                    assert denied["kind"] == "error"
                    assert "GET and FINISH only" in denied["reply"]
                    assert source_calls == 0

        await asyncio.gather(*(exercise(patient_id) for patient_id in patient_ids))
    finally:
        await session.close()
    assert session._shared == {}
