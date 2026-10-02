"""TaskCraft 端到端检查：真实 PhysicianBench 实例 + 已授权的 gpt-4o-mini。

MEDEVOLVE_TASKCRAFT_MANAGER=http://10.7.0.15:18084 \
MEDEVOLVE_TASKCRAFT_PATIENT=MRN9838448928 \
MEDEVOLVE_TASKCRAFT_BUDGET=200 \
/Users/guagua/code/MedEvolve/.venv/bin/python -m pytest tests/method/test_taskcraft_live.py -q

未配置环境变量时跳过；会真实调用模型并占用一个独立容器实例。
"""

import json
import os

import pytest

from contracts import (
    Budget, EnvironmentConfig, GenerationDirection, ModelConfig, RuntimeConfig, TaskBatch,
)
from methods.taskcraft import generate_tasks

pytestmark = pytest.mark.skipif(
    not os.environ.get("MEDEVOLVE_TASKCRAFT_MANAGER"),
    reason="set MEDEVOLVE_TASKCRAFT_MANAGER and MEDEVOLVE_TASKCRAFT_PATIENT to run live",
)


@pytest.mark.asyncio
async def test_live_batch_binds_patient_evidence_and_records_cost():
    from environments import create_environment, get_tools, prepare_trajectory, release_environment

    config = EnvironmentConfig(name="PhysicianBench",
                              manager_url=os.environ["MEDEVOLVE_TASKCRAFT_MANAGER"],
                              seed_data=os.environ.get("MEDEVOLVE_TASKCRAFT_SEED", "fhir-full:v1"))
    patient = os.environ["MEDEVOLVE_TASKCRAFT_PATIENT"]
    limits = int(os.environ.get("MEDEVOLVE_TASKCRAFT_BUDGET", "200"))
    runtime = RuntimeConfig(
        models={"generator": ModelConfig(model=os.environ.get("MEDEVOLVE_TASKCRAFT_MODEL", "gpt-4o-mini"),
                                        base_url="https://api.openlux.ai/v1",
                                        api_key_env="OPENLUX_API_KEY")},
        budgets={"generator": Budget(max_model_turns=limits, max_tool_calls=limits)},
    )
    directions = [GenerationDirection(task_type="诊断与解释", difficulty="基础", weight=1),
                  GenerationDirection(task_type="治疗规划", difficulty="基础", weight=1)]

    instance = await create_environment(config)
    try:
        trajectory = await prepare_trajectory(instance, patient)
        tools = await get_tools(instance, [patient])
        assert all(tool.name.startswith("fhir_") for tool in tools)
    finally:
        await release_environment(config, instance)

    result = await generate_tasks(trajectory, config, directions, runtime)
    print("status:", result.status, "error:", result.error)
    print("usage:", json.dumps(result.usage, ensure_ascii=False))
    entry = next(item for item in result.trace if item.get("stage") == "taskcraft/pipeline")
    print("pool:", json.dumps(entry["pool"], ensure_ascii=False))

    assert result.status in ("completed", "budget_exhausted")
    assert result.usage["model_turns"] > 0
    batch = TaskBatch.model_validate(result.output) if result.output else TaskBatch(questions=[])
    known = {source.source_id for source in trajectory.sources}
    for question in batch.questions:
        assert question.evidence
        assert {item.source_id for item in question.evidence} <= known
