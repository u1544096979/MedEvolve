from unittest.mock import AsyncMock
import pytest
from contracts import (
    Budget, EnvironmentConfig, EnvironmentInstance, ModelConfig, RunResult, RuntimeConfig,
)
from environments.medagentbench import NativeTools
from methods import medevolve

@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["PhysicianBench", "MedAgentBench"])
@pytest.mark.parametrize("failure", ["none", "budget", "release", "tools"])
async def test_shared_exploration_paths_release_and_cost(monkeypatch, name, failure):
    config = EnvironmentConfig(name=name, manager_url="http://unused", seed_data="seed")
    instance = EnvironmentInstance(instance_id="isolated", endpoint="http://unused", environment=name)
    runtime = RuntimeConfig(
        models={"environment_explorer": ModelConfig(model="mock", api_key_env="UNUSED")},
        budgets={"environment_explorer": Budget(max_model_turns=2, max_tool_calls=2)},
    )
    if name == "MedAgentBench":
        tools = NativeTools(instance.endpoint, ["train"], [])
    else:
        from agents import function_tool
        @function_tool
        async def lookup() -> str:
            """Read current patient."""
            return "record"
        tools = [lookup]
    get = AsyncMock(return_value=tools)
    if failure == "tools": get.side_effect = RuntimeError("tools unavailable")
    release = AsyncMock()
    if failure == "release": release.side_effect = RuntimeError("release unavailable")
    original = RunResult(output={"guidance": "读取能力已验证"} if failure != "budget" else None,
        trace=[{"type": "tool_output", "output": "record"}],
        usage={"model_turns": 2, "tool_calls": 1, "input_tokens": 8, "output_tokens": 4},
        status="budget_exhausted" if failure == "budget" else "completed")
    agent = AsyncMock(return_value=original)
    monkeypatch.setattr(medevolve, "create_environment", AsyncMock(return_value=instance))
    monkeypatch.setattr(medevolve, "get_tools", get)
    monkeypatch.setattr(medevolve, "release_environment", release)
    monkeypatch.setattr(medevolve, "run_agent", agent)
    result = await medevolve.explore_environment(config, ["train"], runtime)
    get.assert_awaited_once_with(instance, ["train"], tool_result_limit=config.tool_result_limit)
    release.assert_awaited_once_with(config, instance)
    if failure == "tools":
        agent.assert_not_awaited()
        assert result.status == "failed" and result.output is None
        return
    assert agent.await_args.kwargs["tools"] is tools
    instructions = agent.await_args.kwargs["instructions"]
    assert instructions.startswith(medevolve.ENVIRONMENT_EXPLORATION)
    assert "本轮实验只允许基于既有患者记录生成查询任务" in instructions
    assert agent.await_args.kwargs["model"] is runtime.models["environment_explorer"]
    assert result.usage == original.usage
    assert result.trace[0] == original.trace[0]
    if failure == "release":
        assert result.status == "failed" and result.output is None
        assert "release unavailable" in result.error
    else:
        assert result.status == original.status

@pytest.mark.asyncio
async def test_create_failure_has_no_instance_to_release(monkeypatch):
    config = EnvironmentConfig(name="PhysicianBench", manager_url="http://unused", seed_data="seed")
    runtime = RuntimeConfig(
        models={"environment_explorer": ModelConfig(model="mock", api_key_env="UNUSED")},
        budgets={"environment_explorer": Budget(max_model_turns=2, max_tool_calls=2)},
    )
    release, agent = AsyncMock(), AsyncMock()
    monkeypatch.setattr(medevolve, "create_environment",
                        AsyncMock(side_effect=RuntimeError("manager unavailable")))
    monkeypatch.setattr(medevolve, "release_environment", release)
    monkeypatch.setattr(medevolve, "run_agent", agent)
    result = await medevolve.explore_environment(config, ["train"], runtime)
    assert result.status == "failed" and result.output is None
    assert "manager unavailable" in result.error
    assert result.usage["model_turns"] == 0
    release.assert_not_awaited()
    agent.assert_not_awaited()
