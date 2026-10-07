"""Opt-in five-patient read-only check against an already prepared environment.

Set one of MEDEVOLVE_TEST_IMAGE, MEDEVOLVE_MEDAGENTBENCH_TEST_IMAGE, or
MEDEVOLVE_FHIR_AGENTBENCH_SEED. The check calls no models and pulls no assets.
"""

import asyncio
import os

import httpx
import pytest

from contracts import EnvironmentConfig, GeneratedQuestion
from environments import get_tools, prepare_trajectory, read_evidence
from environments.medagentbench import ReadOnlyNativeTools
from evaluation.session import EvaluationEnvironments


ASSETS = (
    ("PhysicianBench", "MEDEVOLVE_TEST_IMAGE", "http://127.0.0.1:18084"),
    ("MedAgentBench", "MEDEVOLVE_MEDAGENTBENCH_TEST_IMAGE", "http://127.0.0.1:18084"),
    ("FHIR-AgentBench", "MEDEVOLVE_FHIR_AGENTBENCH_SEED", "http://127.0.0.1:18086"),
)


def configured_asset():
    for name, variable, manager_url in ASSETS:
        value = os.environ.get(variable)
        if value:
            return name, value, manager_url
    return None


@pytest.mark.asyncio
@pytest.mark.skipif(configured_asset() is None, reason="Set one prepared local environment asset")
async def test_five_patients_share_one_read_only_instance_across_stages():
    name, asset, manager_url = configured_asset()
    config = EnvironmentConfig(name=name, manager_url=manager_url, seed_data=asset)
    session = EvaluationEnvironments(instance_budget=1)
    created = []
    try:
        async with session.instance(config, phase="prepare") as instance:
            async with httpx.AsyncClient(trust_env=False, timeout=30) as client:
                response = await client.get(instance.endpoint + "Patient", params={"_count": 10})
                response.raise_for_status()
            patient_ids = [entry["resource"]["id"] for entry in response.json().get("entry", [])
                           if entry.get("resource", {}).get("resourceType") == "Patient"][:5]
            assert len(patient_ids) == 5
            created.append(instance.instance_id)

        async def exercise(patient_id):
            async with session.instance(config, phase="prepare") as instance:
                trajectory = await prepare_trajectory(instance, patient_id)
                assert trajectory.patient_id == patient_id
                evidence = await read_evidence(instance, patient_id, trajectory.sources[0].source_id)
                assert evidence.get("id") == trajectory.sources[0].source_id.split("/", 1)[1]
                created.append(instance.instance_id)

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
                else:
                    assert tools
                created.append(instance.instance_id)

            generated = GeneratedQuestion(
                instruction="解释已有记录中的主要发现", task_type="诊断与解释",
                reference_answer="基于既有记录说明发现", evidence=[],
                execution_requirement="read_only",
            )
            assert generated.execution_requirement == "read_only"
            async with session.instance(config, phase="solve") as instance:
                solve_tools = await get_tools(instance, [patient_id], read_only=True)
                assert solve_tools
                created.append(instance.instance_id)

        await asyncio.gather(*(exercise(patient_id) for patient_id in patient_ids))
        assert set(created) == {created[0]}
    finally:
        await session.close()
    assert session._shared == {}
