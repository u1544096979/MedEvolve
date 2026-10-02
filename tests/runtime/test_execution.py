import json

import httpx
import pytest
from agents import function_tool
from openai import AsyncOpenAI

from contracts import Budget, ModelConfig, QualityReview, SolverAnswer
from runtime import call_llm, run_agent
from runtime import execution


MODEL = ModelConfig(model="local-test", base_url="http://model.test/v1", api_key_env="TEST_MODEL_KEY")


def install_transport(monkeypatch, responses):
    requests = []
    monkeypatch.setenv("TEST_MODEL_KEY", "test-secret")

    def handle(request):
        requests.append(json.loads(request.content))
        response = responses[len(requests) - 1]
        if isinstance(response, Exception):
            raise response
        response = json.loads(json.dumps(response))
        for tool_call in response.get("tool_calls", []):
            tool_call["id"] += f"-{len(requests)}"
        return httpx.Response(200, json={
            "id": f"chat-{len(requests)}", "object": "chat.completion", "created": 0,
            "model": "local-test", "choices": [{"index": 0, "message": response,
            "finish_reason": "tool_calls" if response.get("tool_calls") else "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        })

    monkeypatch.setattr(execution, "_client", lambda model: AsyncOpenAI(
        api_key="test-secret", base_url=model.base_url, max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle))))
    return requests


TOOL_MESSAGE = {"role": "assistant", "content": None, "tool_calls": [
    {"id": "lookup-1", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}
]}
FINAL_MESSAGE = {"role": "assistant", "content": '{"answer":"patient found"}'}


@pytest.mark.asyncio
@pytest.mark.parametrize("output_type,output", [
    (SolverAnswer, {"answer": "patient found"}),
    (QualityReview, {"dimensions": {"correctness": {"judgment": "supported", "reason": "patient found"}}}),
])
async def test_sdk_tool_loop(monkeypatch, output_type, output):
    requests = install_transport(monkeypatch, [TOOL_MESSAGE,
        {"role": "assistant", "content": json.dumps(output)}])
    calls = []

    @function_tool
    async def lookup() -> str:
        """Read patient facts."""
        calls.append(1)
        return "patient found"

    result = await run_agent(instructions="Read and answer", input_text="patient",
        tools=[lookup], output_type=output_type, model=MODEL,
        budget=Budget(max_model_turns=2, max_tool_calls=1))
    assert result.status == "completed", result.error
    assert result.output == output
    assert calls == [1]
    assert requests[1]["messages"][-1]["role"] == "tool"
    assert requests[0]["response_format"]["type"] == "json_schema"
    assert result.usage["total_tokens"] == 30
    assert result.usage["model_turns"] == 2
    assert result.usage["tool_calls"] == 1
    assert any(item["type"] == "tool_output" for item in result.trace)


@pytest.mark.asyncio
@pytest.mark.parametrize("turns,tool_limit,expected_requests,expected_calls", [
    (0, 1, 0, 0), (1, 1, 1, 1), (2, 0, 1, 0), (3, 1, 2, 1),
])
async def test_budgets_stop_execution(monkeypatch, turns, tool_limit, expected_requests, expected_calls):
    requests = install_transport(monkeypatch, [TOOL_MESSAGE, TOOL_MESSAGE, FINAL_MESSAGE])
    calls = []

    @function_tool
    async def lookup() -> str:
        """Read patient facts."""
        calls.append(1)
        return "patient found"

    result = await run_agent(instructions="Read", input_text="patient", tools=[lookup],
        output_type=SolverAnswer, model=MODEL,
        budget=Budget(max_model_turns=turns, max_tool_calls=tool_limit))
    assert result.status == "budget_exhausted", result.error
    assert len(requests) == expected_requests
    assert len(calls) == expected_calls
    assert result.output is None
    assert result.usage["total_tokens"] == expected_requests * 15
    if expected_calls:
        assert any(item["type"] == "tool_output" for item in result.trace)


@pytest.mark.asyncio
async def test_failure_preserves_tool_trace_and_usage(monkeypatch):
    requests = install_transport(monkeypatch, [TOOL_MESSAGE, httpx.ConnectError("test-secret offline")])

    @function_tool
    async def lookup() -> str:
        """Read patient facts."""
        return "patient found"

    result = await run_agent(instructions="Read", input_text="patient", tools=[lookup],
        output_type=SolverAnswer, model=MODEL,
        budget=Budget(max_model_turns=3, max_tool_calls=2))
    assert result.status == "failed"
    assert len(requests) == 2
    assert result.usage["total_tokens"] == 15
    assert any(item["type"] == "tool_output" for item in result.trace)
    assert result.trace[-1]["failure_category"] == "model_error"
    assert "test-secret" not in result.model_dump_json()


@pytest.mark.asyncio
async def test_tool_exception_has_stable_tool_error_marker(monkeypatch):
    install_transport(monkeypatch, [TOOL_MESSAGE])

    @function_tool(failure_error_function=None)
    async def lookup() -> str:
        """Read patient facts."""
        raise RuntimeError("tool offline")

    result = await run_agent(instructions="Read", input_text="patient", tools=[lookup],
        output_type=SolverAnswer, model=MODEL,
        budget=Budget(max_model_turns=2, max_tool_calls=1))
    assert result.status == "failed"
    assert result.trace[-1]["type"] == "run_error"
    assert result.trace[-1]["failure_category"] == "tool_error"


@pytest.mark.asyncio
async def test_single_request_parse_failure_is_not_repaired(monkeypatch):
    requests = install_transport(monkeypatch, [{"role": "assistant", "content": "invalid JSON"}])
    result = await call_llm(instructions="Generate", input_text="patient",
                            output_type=SolverAnswer, model=MODEL)
    assert result.status == "failed"
    assert len(requests) == 1
    assert result.trace[-1]["response"]["choices"][0]["message"]["content"] == "invalid JSON"
    assert result.usage["total_tokens"] == 15
