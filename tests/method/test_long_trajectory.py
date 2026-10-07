"""六方法长轨迹请求经真实运行层适配，HTTP 和病历工具使用本地替身。"""

import json
from typing import get_args

import httpx
import pytest
from openai import AsyncOpenAI

from contracts import (Budget, EnvironmentConfig, EnvironmentExploration, ExplorationGuidance,
                       GenerationDirection, RuntimeConfig, Strategy, TaskBatch, TaskType)
from methods import envexplore, medevolve, selfgen, singleshot, workflow
from methods.taskcraft import TaskCraftParameters, generate_tasks
from methods.taskcraft import schemas
from runtime import execution
from runtime.context import count_tokens, dumps
from tests.runtime.test_context import MODEL, patient
from tests.runtime.test_execution import install_transport
from tests.method.test_taskcraft import (FakeRuntime, directions, patched,
                                       trajectory as taskcraft_trajectory)


def request_tokens(request, model):
    # JSON 传输转义不属于模型看到的正文；计入消息框架与结构化输出声明。
    return (sum(count_tokens(message.get("content") or "", model) + 8
                for message in request["messages"])
            + count_tokens(dumps(request.get("response_format", {})), model)
            + count_tokens(dumps(request.get("tools", [])), model))


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["singleshot", "envexplore", "medevolve", "selfgen", "workflow"])
async def test_long_patient_generates_without_context_overflow(monkeypatch, method):
    original = patient()
    before = original.model_dump()
    model = MODEL.model_copy(update={"base_url": "http://model.test/v1"})
    runtime = RuntimeConfig(models={"generator": model},
                            budgets={"generator": Budget(max_model_turns=20, max_tool_calls=0)})
    allocation = [GenerationDirection(task_type=value, difficulty="基础", weight=1)
                  for value in get_args(TaskType)]
    selected = allocation[:2]
    strategy = Strategy(generation_guidance="策略", allocation=selected)
    environment = EnvironmentConfig(name="PhysicianBench", manager_url="http://unused", seed_data="seed")

    def question(index):
        return {"instruction": "根据记录查询患者编号", "task_type": allocation[index].task_type,
                "reference_answer": "p", "evidence": [{"source_id": "Patient/p", "statement": "编号为 p"}],
                "decision_point": None, "execution_requirement": "read_only"}

    batch = {"questions": [question(0)]}
    outputs = [batch]
    if method == "selfgen":
        outputs = [{"batch": batch, "direction_indices": [0]},
                   {"questions": [{"candidate_index": 0, "passed": True, "issues": []}]}]
    if method == "workflow":
        outputs = [{"focuses": [{"direction_index": index, "clinical_focus": "患者资料",
                                  "source_ids": ["Patient/p"]} for index in (0, 1)]},
                   {"questions": [question(0)]}, {"questions": [question(1)]},
                   {"reviews": [{"index": index, "clarity": True, "answer_alignment": True,
                                  "citations_supported": True, "reason": "依据记录"} for index in (0, 1)]}]
    requests = install_transport(monkeypatch, [{"role": "assistant", "content": dumps(output)}
                                               for output in outputs])
    if method == "singleshot":
        result = await singleshot.generate_tasks(original, environment, list(get_args(TaskType)), runtime)
    elif method == "envexplore":
        result = await envexplore.generate_tasks(original, environment, selected, runtime,
            environment_guidance=EnvironmentExploration(guidance="环境指导" * 100), strategy=strategy)
    elif method == "medevolve":
        result = await medevolve.generate_tasks(original,
            ExplorationGuidance(data_guidance="数据指导" * 100, environment_guidance="环境指导" * 100),
            strategy, selected, runtime)
    elif method == "selfgen":
        result = await selfgen.generate_tasks(original, environment, selected, runtime)
    else:
        result = await workflow.generate_tasks(original, environment, allocation, runtime)
    assert result.status == "completed", result.error
    assert TaskBatch.model_validate(result.output).questions
    assert len(requests) == len(outputs)
    for request in requests:
        assert request_tokens(request, model) + request["max_tokens"] < model.context_window
        assert request["max_tokens"] == model.max_output_tokens
    assert json.loads(requests[0]["messages"][-1]["content"])["trajectory"]["input_truncated"]
    assert original.model_dump() == before


@pytest.mark.asyncio
async def test_taskcraft_long_patient_keeps_atomic_chunks_and_search(monkeypatch, patched):
    fake = FakeRuntime()
    patched(fake)
    # 仅替换服务端响应，客户端、公共预算、分块和搜索步骤均真实执行。
    monkeypatch.setattr("runtime.call_llm", execution.call_llm)
    model = MODEL.model_copy(update={"base_url": "http://model.test/v1"})
    runtime = RuntimeConfig(models={"generator": model},
                            budgets={"generator": Budget(max_model_turns=160, max_tool_calls=40)})
    requests = []
    output_types = {value.__name__: value for value in vars(schemas).values()
                    if isinstance(value, type) and hasattr(value, "model_json_schema")}

    async def handle(request):
        body = json.loads(request.content)
        requests.append(body)
        system, user = body["messages"]
        schema = json.loads(system["content"].rsplit("Return only a JSON object matching this schema:\n", 1)[1])
        output = await fake.call_llm(model=model, instructions=system["content"],
                                    input_text=user["content"], output_type=output_types[schema["title"]])
        return httpx.Response(200, json={"id": "mock", "object": "chat.completion", "created": 0,
            "model": model.model, "choices": [{"index": 0, "finish_reason": "stop",
            "message": {"role": "assistant", "content": dumps(output.output)}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}})

    monkeypatch.setattr(execution, "_client", lambda _: AsyncOpenAI(api_key="unused", base_url=model.base_url,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)), max_retries=0))
    original = taskcraft_trajectory()
    long = patient()
    original = original.model_copy(update={"text": original.text + "\n\n" + long.text,
                                          "sources": [*original.sources, *long.sources]})
    before = original.model_dump()
    result = await generate_tasks(original,
        EnvironmentConfig(name="PhysicianBench", manager_url="http://unused", seed_data="seed"),
        directions(), runtime, parameters=TaskCraftParameters(max_candidate_conclusions=2, max_candidate_atomic=2))
    assert result.status == "completed", result.error
    assert TaskBatch.model_validate(result.output).questions
    assert fake.agent_calls
    assert requests
    assert all(request_tokens(request, model) + request["max_tokens"] < model.context_window
               for request in requests)
    assert original.model_dump() == before
