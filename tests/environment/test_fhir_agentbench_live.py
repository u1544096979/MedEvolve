"""真实本机实例的串行验收：实例、轨迹、两个 SDK 工具、证据回查与只读状态。

MEDEVOLVE_FHIR_AGENTBENCH_SEED=/Users/guagua/data/mimic-iv-fhir-demo/2.1.0/seed \
  python -m pytest tests/environment/test_fhir_agentbench_live.py -q

不调用任何模型；实例在 finally 中释放。
"""

import json
import os

import pytest
from agents.tool_context import ToolContext

from contracts import EnvironmentConfig, Task
from environments import (
    create_environment,
    get_tools,
    prepare_trajectory,
    read_evidence,
    read_state,
    release_environment,
)

MANAGER_URL = "http://127.0.0.1:18086"
# 三名开放患者里资源最少的一名：轨迹与工具读取的最小真实材料。
PATIENT_ID = "afa7c67f-82b9-5f51-bd04-8b7d7c4456c0"


def without_meta(resource):
    return {key: value for key, value in resource.items() if key != "meta"}


async def invoke(tools, name, **params):
    tool = next(tool for tool in tools if tool.name == name)
    arguments = json.dumps(params)
    context = ToolContext(
        context=None, tool_name=name, tool_call_id="live", tool_arguments=arguments
    )
    return json.loads(await tool.on_invoke_tool(context, arguments))


@pytest.mark.asyncio
@pytest.mark.skipif(
    not os.environ.get("MEDEVOLVE_FHIR_AGENTBENCH_SEED"),
    reason="Set MEDEVOLVE_FHIR_AGENTBENCH_SEED to the verified local seed directory",
)
async def test_live_instance_trajectory_tools_evidence_and_read_only_state():
    config = EnvironmentConfig(
        name="FHIR-AgentBench",
        manager_url=MANAGER_URL,
        seed_data=os.environ["MEDEVOLVE_FHIR_AGENTBENCH_SEED"],
    )
    instance = await create_environment(config)
    try:
        assert instance.environment == "FHIR-AgentBench"
        assert instance.endpoint

        trajectory = await prepare_trajectory(instance, PATIENT_ID)
        assert trajectory.patient_id == PATIENT_ID
        assert trajectory.environment == "FHIR-AgentBench"
        source_ids = [source.source_id for source in trajectory.sources]
        assert len(source_ids) == len(set(source_ids))
        assert f"Patient/{PATIENT_ID}" in source_ids
        assert any(source_id.startswith("Encounter/") for source_id in source_ids)
        observation_source = next(
            source_id for source_id in source_ids if source_id.startswith("Observation/")
        )
        assert "# 就诊原始信息" in trajectory.text

        tools = await get_tools(instance, [PATIENT_ID])
        by_patient = await invoke(
            tools,
            "get_resources_by_patient_fhir_id",
            resource_type="Patient",
            patient_fhir_id=PATIENT_ID,
        )
        assert [resource["id"] for resource in by_patient["Patient"]] == [PATIENT_ID]
        assert "text" not in by_patient["Patient"][0]
        assert "meta" not in by_patient["Patient"][0]

        identifier = observation_source.split("/", 1)[1]
        by_id = await invoke(
            tools,
            "get_resources_by_resource_id",
            resource_type="Observation",
            resource_id=identifier,
        )
        assert [resource["id"] for resource in by_id["Observation"]] == [identifier]

        evidence = await read_evidence(instance, PATIENT_ID, observation_source)
        statement = next(
            source.statement
            for source in trajectory.sources
            if source.source_id == observation_source
        )
        assert evidence["resourceType"] == "Observation"
        assert evidence["id"] == identifier
        assert without_meta(evidence) == without_meta(json.loads(statement))

        task = Task(
            id="live:1",
            patient_id=PATIENT_ID,
            environment="FHIR-AgentBench",
            round_index=0,
            source_method="medevolve",
            instruction="报告该患者最近一次心率并说明依据",
            task_type="检查与风险分层",
            reference_answer="",
            evidence=[],
            decision_point=None,
        )
        assert await read_state(instance, task) == {
            "patient_id": PATIENT_ID,
            "environment": "FHIR-AgentBench",
            "operation_semantics": {"read_only": True},
        }
    finally:
        await release_environment(config, instance)
