import json

import httpx
import pytest
from openai import AsyncOpenAI

from contracts import Budget, EnvironmentExploration, ModelConfig, SolverAnswer
from environments.medagentbench import NativeTools
from runtime import execution, run_agent


MODEL = ModelConfig(model="local-test", base_url="http://model.test/v1",
                    api_key_env="TEST_MODEL_KEY")


def install_transport(monkeypatch, responses):
    requests = []
    monkeypatch.setenv("TEST_MODEL_KEY", "test-secret")

    def handle(request):
        payload = json.loads(request.content)
        requests.append(payload)
        response = responses[len(requests) - 1]
        if isinstance(response, Exception):
            raise response
        return httpx.Response(200, json={
            "id": f"chat-{len(requests)}", "object": "chat.completion", "created": 0,
            "model": "local-test", "choices": [{"index": 0, "message": response,
            "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15,
                      "prompt_tokens_details": {"cached_tokens": 3}},
        })

    monkeypatch.setattr(execution, "_client", lambda model: AsyncOpenAI(
        api_key="test-secret", base_url=model.base_url, max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle))))
    return requests


def native_tools():
    return NativeTools("http://fhir.test/fhir/", ["p1"],
                       [{"name": "GET {api_base}/Observation"}])


@pytest.mark.asyncio
async def test_native_text_loop_and_finish_mapping(monkeypatch):
    requests = install_transport(monkeypatch, [
        {"role": "assistant", "content": "GET Observation?patient=p1"},
        {"role": "assistant", "content": "POST Observation\n{\"resourceType\":\"Observation\"}"},
        {"role": "assistant", "content": 'FINISH(["result"])'},
    ])
    tools = native_tools()
    actions = []

    async def execute(action):
        actions.append(action)
        if action.startswith("FINISH("):
            return {"kind": "FINISH", "request": {"action": action},
                    "reply": "FINISH", "answers": ["result"]}
        return {"kind": "GET" if action.startswith("GET") else "POST",
                "request": {"action": action}, "reply": f"feedback-{len(actions)}"}

    monkeypatch.setattr(tools, "execute", execute)
    result = await run_agent(instructions="SOLVER instructions name SolverAnswer schema",
        input_text="patient data", tools=tools, output_type=SolverAnswer,
        model=MODEL, budget=Budget(max_model_turns=3, max_tool_calls=2))
    assert result.status == "completed", result.error
    assert json.loads(result.output["answer"]) == ["result"]
    assert result.usage["model_turns"] == 3
    assert result.usage["requests"] == 3
    assert result.usage["tool_calls"] == 2
    assert result.usage["input_tokens"] == 30
    assert result.usage["total_tokens"] == 45
    assert result.trace[1]["response"]["usage"]["prompt_tokens_details"]["cached_tokens"] == 3
    assert [item["type"] for item in result.trace].count("tool_output") == 2
    model_requests = [item for item in result.trace if item["type"] == "model_request"]
    assert len(model_requests[0]["messages"]) == 2
    assert all("GET Observation" not in message["content"]
               and "feedback-" not in message["content"]
               for message in model_requests[0]["messages"])
    assert model_requests[1]["messages"][-2:] == [
        {"role": "assistant", "content": "GET Observation?patient=p1"},
        {"role": "user", "content": "feedback-1"},
    ]
    for request in requests:
        assert "tools" not in request and "response_format" not in request
    prompt = requests[0]["messages"][1]["content"]
    assert "Use http://fhir.test/fhir as the api_base" in prompt
    assert '"name": "GET http://fhir.test/fhir/Observation"' in prompt
    assert "{api_base}" not in prompt
    assert "one function/action each time" in prompt
    assert "FINISH([answer1, answer2, ...])" in prompt
    assert "Task material:\npatient data" in prompt
    assert requests[1]["messages"][-1] == {"role": "user", "content": "feedback-1"}
    assert requests[2]["messages"][-1] == {"role": "user", "content": "feedback-2"}
    assert all("Return only a JSON object" not in m["content"]
               for m in requests[0]["messages"] if isinstance(m["content"], str))


@pytest.mark.asyncio
async def test_native_zero_tool_budget_and_last_turn_exhaustion(monkeypatch):
    requests = install_transport(monkeypatch, [
        {"role": "assistant", "content": "GET Observation?patient=p1"},
        {"role": "assistant", "content": "GET Observation?patient=p1"},
    ])
    tools = native_tools()
    called = []

    async def execute(action):
        called.append(action)
        return {"kind": "GET", "request": {"action": action}, "reply": "ok"}

    monkeypatch.setattr(tools, "execute", execute)
    result = await run_agent(instructions="Explore", input_text="patient", tools=tools,
        output_type=EnvironmentExploration, model=MODEL,
        budget=Budget(max_model_turns=2, max_tool_calls=1))
    assert result.status == "budget_exhausted"
    assert len(called) == 1
    assert result.usage["tool_calls"] == 1

    requests = install_transport(monkeypatch, [
        {"role": "assistant", "content": "GET Observation?patient=p1"},
    ])
    called.clear()
    result = await run_agent(instructions="Explore", input_text="patient", tools=tools,
        output_type=EnvironmentExploration, model=MODEL,
        budget=Budget(max_model_turns=1, max_tool_calls=0))
    assert result.status == "budget_exhausted"
    assert not called
    assert result.usage["tool_calls"] == 0


@pytest.mark.asyncio
async def test_exploration_prompt_requires_guidance_in_finish_answers(monkeypatch):
    tools = native_tools()

    async def execute(action):
        return {"kind": "FINISH", "request": {"action": action}, "reply": "FINISH", "answers": ["指导"]}

    monkeypatch.setattr(tools, "execute", execute)
    requests = install_transport(monkeypatch, [{"role": "assistant", "content": 'FINISH(["指导"])'}])
    result = await run_agent(instructions="Explore", input_text="patient", tools=tools,
        output_type=EnvironmentExploration, model=MODEL,
        budget=Budget(max_model_turns=1, max_tool_calls=1))
    assert result.status == "completed", result.error
    prompt = requests[0]["messages"][1]["content"]
    assert "没有待回答的题目" in prompt and "不能提交空列表" in prompt
    assert prompt.index("FINISH([answer1, answer2, ...])") < prompt.index("没有待回答的题目")

    requests = install_transport(monkeypatch, [{"role": "assistant", "content": 'FINISH(["a"])'}])
    solution = await run_agent(instructions="Solve", input_text="patient", tools=tools,
        output_type=SolverAnswer, model=MODEL, budget=Budget(max_model_turns=1, max_tool_calls=1))
    assert solution.status == "completed", solution.error
    assert "没有待回答的题目" not in requests[0]["messages"][1]["content"]


@pytest.mark.asyncio
async def test_native_environment_error_is_returned_as_feedback_and_not_clinical(monkeypatch):
    requests = install_transport(monkeypatch, [
        {"role": "assistant", "content": "GET Observation?patient=p1"},
        {"role": "assistant", "content": 'FINISH(["result"])'},
    ])
    tools = native_tools()

    async def execute(action):
        raise httpx.ConnectError("test-secret unavailable")

    monkeypatch.setattr(tools, "execute", execute)
    result = await run_agent(instructions="Solve", input_text="patient", tools=tools,
        output_type=SolverAnswer, model=MODEL,
        budget=Budget(max_model_turns=2, max_tool_calls=1))
    assert result.status == "failed"
    assert any(item.get("failure_category") == "environment_error"
               for item in result.trace)
    assert "test-secret" not in result.model_dump_json()
    assert "test-secret" not in json.dumps(requests[1]["messages"])
    assert requests[1]["messages"][-1]["content"].startswith("Error: ConnectError")
    assert "[REDACTED]" in requests[1]["messages"][-1]["content"]


@pytest.mark.asyncio
async def test_get_http_feedback_continues_but_finish_is_environment_failure(monkeypatch):
    requests = install_transport(monkeypatch, [
        {"role": "assistant", "content": "GET Observation?patient=p1"},
        {"role": "assistant", "content": 'FINISH(["result"])'},
    ])
    tools = native_tools()

    async def execute(action):
        if action.startswith("FINISH("):
            return {"kind": "FINISH", "request": {"action": action},
                    "reply": "FINISH", "answers": ["result"]}
        return {"kind": "GET", "request": {"action": action},
                "reply": "Error in sending the GET request: HTTPStatusError",
                "failure_category": "environment_error"}

    monkeypatch.setattr(tools, "execute", execute)
    result = await run_agent(instructions="Solve", input_text="patient", tools=tools,
        output_type=SolverAnswer, model=MODEL,
        budget=Budget(max_model_turns=2, max_tool_calls=1))
    assert result.status == "failed"
    assert result.output == {"answer": '["result"]'}
    assert len(requests) == 2
    assert requests[1]["messages"][-1]["content"].startswith(
        "Error in sending the GET request: HTTPStatusError")
    assert result.trace[-1]["failure_category"] == "environment_error"


@pytest.mark.asyncio
async def test_native_invalid_action_fails_without_another_model_call(monkeypatch):
    requests = install_transport(monkeypatch, [
        {"role": "assistant", "content": "I will now retrieve the chart."},
        {"role": "assistant", "content": 'FINISH(["should not be requested"])'},
    ])
    result = await run_agent(instructions="Solve", input_text="patient", tools=native_tools(),
        output_type=SolverAnswer, model=MODEL,
        budget=Budget(max_model_turns=24, max_tool_calls=200))
    assert result.status == "failed"
    assert len(requests) == 1
    assert result.usage["model_turns"] == 1
    assert result.trace[-1]["type"] == "run_error"
    assert result.trace[-1]["failure_category"] == "model_error"
    assert "Invalid native action" in result.error


@pytest.mark.asyncio
async def test_native_invalid_finish_keeps_request_reply_error_and_usage(monkeypatch):
    requests = install_transport(monkeypatch, [
        {"role": "assistant", "content": "FINISH(not-json)"},
        {"role": "assistant", "content": 'FINISH(["should not be requested"])'},
    ])
    result = await run_agent(instructions="Solve", input_text="patient", tools=native_tools(),
        output_type=SolverAnswer, model=MODEL,
        budget=Budget(max_model_turns=24, max_tool_calls=200))
    assert result.status == "failed"
    assert len(requests) == 1
    assert result.usage["model_turns"] == 1
    finish_error = next(item for item in result.trace if item["type"] == "finish_error")
    assert finish_error["request"]["action"] == "FINISH(not-json)"
    assert finish_error["reply"] == "Invalid FINISH request"
    assert "Invalid FINISH action" in finish_error["error"]
    assert result.trace[-1]["failure_category"] == "model_error"


@pytest.mark.asyncio
async def test_invalid_finish_does_not_replace_prior_environment_failure(monkeypatch):
    requests = install_transport(monkeypatch, [
        {"role": "assistant", "content": "GET Observation?patient=p1"},
        {"role": "assistant", "content": "FINISH(not-json)"},
    ])
    tools = native_tools()
    native_execute = tools.execute

    async def execute(action):
        if action.startswith("GET"):
            raise httpx.ConnectError("FHIR unavailable")
        return await native_execute(action)

    monkeypatch.setattr(tools, "execute", execute)
    result = await run_agent(instructions="Solve", input_text="patient", tools=tools,
        output_type=SolverAnswer, model=MODEL,
        budget=Budget(max_model_turns=24, max_tool_calls=200))
    assert result.status == "failed"
    assert len(requests) == 2
    assert any(item["type"] == "finish_error" for item in result.trace)
    assert result.trace[-1]["type"] == "run_error"
    assert result.trace[-1]["failure_category"] == "environment_error"
    assert result.usage["model_turns"] == 2
    assert result.usage["tool_calls"] == 1
