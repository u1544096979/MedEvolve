"""Method boundaries and evolution flow with substitute runtime/environment calls."""

import importlib
import importlib.util
import json
import sys
from types import ModuleType
from unittest.mock import AsyncMock

import pytest
from agents.tool_context import ToolContext

from contracts import (
    Budget,
    DataExploration,
    EnvironmentConfig,
    EnvironmentExploration,
    EnvironmentInstance,
    EvaluationResult,
    Evidence,
    ExplorationGuidance,
    Feedback,
    GeneratedQuestion,
    GenerationDirection,
    ModelConfig,
    PatientTrajectory,
    RunResult,
    RuntimeConfig,
    Strategy,
    Task,
    TaskBatch,
)


@pytest.fixture
def method(monkeypatch):
    missing = []
    for name, functions in (
        ("runtime", ("call_llm", "run_agent")),
        ("environments", ("create_environment", "get_tools", "release_environment")),
    ):
        if importlib.util.find_spec(name) is None:
            module = ModuleType(name)
            for function in functions:
                setattr(module, function, AsyncMock())
            monkeypatch.setitem(sys.modules, name, module)
            missing.append(name)
    module = importlib.import_module("methods.medevolve")
    for name in ("call_llm", "run_agent", "create_environment", "get_tools", "release_environment"):
        monkeypatch.setattr(module, name, AsyncMock(side_effect=AssertionError("unexpected call")))
    yield module
    if missing:
        sys.modules.pop("methods.medevolve", None)


@pytest.fixture
def runtime():
    roles = ("data_explorer", "environment_explorer", "generator", "feedback", "updater")
    return RuntimeConfig(
        models={role: ModelConfig(model=role, api_key_env="TEST_UNUSED_KEY") for role in roles},
        budgets={role: Budget(max_model_turns=5, max_tool_calls=8) for role in roles[:2]},
    )


def trajectory(patient_id):
    return PatientTrajectory(
        patient_id=patient_id,
        environment="test",
        text=f"{patient_id} 的时间序列记录",
        sources=[Evidence(source_id=f"record-{patient_id}", statement="缓存证据")],
    )


def completed(output):
    return RunResult(output=output.model_dump(mode="json"), trace=[], usage={}, status="completed")


@pytest.mark.asyncio
async def test_data_exploration_reads_only_supplied_cache(method, runtime, monkeypatch):
    supplied = [trajectory("evolution-1"), trajectory("evolution-2")]
    held_out = trajectory("evaluation-1")

    async def agent(**kwargs):
        assert kwargs["output_type"] is DataExploration
        assert held_out.patient_id not in kwargs["input_text"]
        assert supplied[0].text not in kwargs["input_text"]
        tools = {tool.name: tool for tool in kwargs["tools"]}
        assert set(tools) == {"list_patients", "read_trajectory", "compress_trajectory"}
        async def invoke(name, arguments):
            context = ToolContext(
                context=None, tool_name=name, tool_call_id="test-call", tool_arguments=arguments
            )
            return await tools[name].on_invoke_tool(context, arguments)

        patients = await invoke("list_patients", "{}")
        assert patients == [item.patient_id for item in supplied]
        for item in supplied:
            raw = await invoke("read_trajectory", json.dumps({"patient_id": item.patient_id}))
            payload = json.loads(raw)
            assert payload["patient_id"] == item.patient_id
            assert payload["text"] == item.text
            assert payload["sources"] == [source.model_dump(mode="json") for source in item.sources]
        with pytest.raises(ValueError, match="outside"):
            await invoke("read_trajectory", json.dumps({"patient_id": held_out.patient_id}))
        return completed(DataExploration(guidance="关注时间变化与证据来源"))

    substitute = AsyncMock(side_effect=agent)
    monkeypatch.setattr(method, "run_agent", substitute)
    result = await method.explore_data(supplied, runtime)
    assert result.status == "completed"
    substitute.assert_awaited_once()
    method.create_environment.assert_not_awaited()
    method.call_llm.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [None, "tools", "agent", "failed_result"])
async def test_environment_exploration_releases_independent_instance(
    method, runtime, monkeypatch, failure
):
    config = EnvironmentConfig(name="test", manager_url="http://unused", seed_data="seed")
    instance = EnvironmentInstance(instance_id="isolated-exploration", endpoint="http://unused", environment="test")
    create = AsyncMock(return_value=instance)
    tools = AsyncMock(return_value=[])
    release = AsyncMock()
    result = (
        RunResult(trace=[{"tool_error": "unavailable"}], usage={}, status="failed", error="unavailable")
        if failure == "failed_result"
        else completed(EnvironmentExploration(guidance="原生工具约束"))
    )
    agent = AsyncMock(return_value=result)
    if failure == "tools":
        tools.side_effect = RuntimeError("tools unavailable")
    if failure == "agent":
        agent.side_effect = RuntimeError("agent unavailable")
    for name, substitute in (
        ("create_environment", create), ("get_tools", tools),
        ("release_environment", release), ("run_agent", agent),
    ):
        monkeypatch.setattr(method, name, substitute)
    if failure in ("tools", "agent"):
        with pytest.raises(RuntimeError, match="unavailable"):
            await method.explore_environment(config, ["evolution-1"], runtime)
    else:
        assert await method.explore_environment(config, ["evolution-1"], runtime) is result
        assert agent.await_args.kwargs["output_type"] is EnvironmentExploration
    create.assert_awaited_once_with(config)
    tools.assert_awaited_once_with(instance, ["evolution-1"])
    release.assert_awaited_once_with(config, instance)


@pytest.mark.asyncio
async def test_generation_feedback_update_and_next_generation(method, runtime, monkeypatch):
    patient = trajectory("evolution-1")
    guidance = ExplorationGuidance(data_guidance="整合纵向证据", environment_guidance="仅用已开放能力")
    direction = GenerationDirection(task_type="诊断与解释", difficulty="基础", weight=1)
    strategy = Strategy(generation_guidance="初始指导", allocation=[direction])
    updated = Strategy(
        generation_guidance="增加有证据支持的跨时间整合",
        allocation=[GenerationDirection(task_type="治疗重规划", difficulty="整合纵向证据", weight=1)],
    )
    question = GeneratedQuestion(
        instruction="综合病程解释诊断", task_type=direction.task_type,
        reference_answer="参考解释", evidence=patient.sources,
    )
    task = Task(
        **question.model_dump(), id="task-1", patient_id=patient.patient_id,
        environment=patient.environment, round_index=0, source_method="medevolve",
    )
    evaluation = EvaluationResult(
        task_id=task.id, status="failed", failure_category="tool_error",
        observed_state={"write_applied": False},
        solver_run=RunResult(trace=[{"tool_error": "offline"}], usage={}, status="failed", error="offline"),
        runs=[],
    )
    feedback = Feedback(
        quality_findings=[], difficulty_findings=[],
        failure_findings=["工具故障，不能据此推断难度"], suggestions=["依据可用证据整合时间变化"],
    )
    responses = [completed(TaskBatch(questions=[question])), completed(feedback), completed(updated), completed(TaskBatch(questions=[]))]
    call = AsyncMock(side_effect=responses)
    monkeypatch.setattr(method, "call_llm", call)

    first = await method.generate_tasks(patient, guidance, strategy, [direction, direction], runtime)
    assert len(TaskBatch.model_validate(first.output).questions) == 1
    analysis = await method.analyze_feedback([task], [patient], [evaluation], runtime)
    update = await method.update_strategy(strategy, Feedback.model_validate(analysis.output), runtime)
    next_strategy = Strategy.model_validate(update.output)
    second = await method.generate_tasks(patient, guidance, next_strategy, next_strategy.allocation * 2, runtime)
    assert TaskBatch.model_validate(second.output).questions == []

    assert call.await_count == 4
    calls = [entry.kwargs for entry in call.await_args_list]
    assert [entry["output_type"] for entry in calls] == [TaskBatch, Feedback, Strategy, TaskBatch]
    payloads = [json.loads(entry["input_text"]) for entry in calls]
    assert payloads[0]["trajectory"]["patient_id"] == patient.patient_id
    assert len(payloads[0]["directions"]) == 2
    assert payloads[0]["trajectory"]["sources"] == [s.model_dump(mode="json") for s in patient.sources]
    assert payloads[0]["strategy"] == {"generation_guidance": strategy.generation_guidance}
    assert "药物处方" not in calls[0]["instructions"].split("题型与难度方向：")[-1]
    assert payloads[1]["evaluations"][0]["failure_category"] == "tool_error"
    assert payloads[1]["evaluations"][0]["solver_run"]["trace"] == [{"tool_error": "offline"}]
    assert payloads[1]["evaluations"][0]["observed_state"] == {"write_applied": False}
    assert set(payloads[2]) == {"strategy", "feedback"}
    assert payloads[3]["guidance"] == payloads[0]["guidance"]
    assert payloads[3]["strategy"]["generation_guidance"] == updated.generation_guidance
    method.run_agent.assert_not_awaited()
    method.create_environment.assert_not_awaited()

    with pytest.raises(ValueError, match="exactly two"):
        await method.generate_tasks(patient, guidance, strategy, [direction], runtime)
    assert call.await_count == 4


@pytest.mark.asyncio
async def test_generation_accepts_only_requested_task_types_and_allows_short_batch(
    method, runtime, monkeypatch
):
    patient = trajectory("evolution-1")
    guidance = ExplorationGuidance(data_guidance="数据", environment_guidance="环境")
    allowed = GenerationDirection(task_type="诊断与解释", difficulty="基础", weight=1)
    skipped = GenerationDirection(task_type="治疗规划", difficulty="基础", weight=1)
    strategy = Strategy(generation_guidance="指导", allocation=[allowed])
    good = GeneratedQuestion(
        instruction="诊断", task_type=allowed.task_type, reference_answer="答案",
        evidence=patient.sources,
    )
    call = AsyncMock(return_value=completed(TaskBatch(questions=[good])))
    monkeypatch.setattr(method, "call_llm", call)

    result = await method.generate_tasks(
        patient, guidance, strategy, [allowed, skipped], runtime
    )
    assert result.status == "completed"
    assert len(TaskBatch.model_validate(result.output).questions) == 1

    unsupported = good.model_copy(update={"task_type": "药物处方"})
    call.return_value = completed(TaskBatch(questions=[unsupported]))
    result = await method.generate_tasks(
        patient, guidance, strategy, [allowed, skipped], runtime
    )
    assert result.status == "failed"
    assert "exceed requested directions" in result.error
