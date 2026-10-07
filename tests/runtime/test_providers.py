import pytest

from contracts import Budget, ModelConfig, RunResult, SolverAnswer
from runtime import call_llm, run_agent
from runtime import providers
from runtime import execution


PRIMARY = ModelConfig(
    model="deepseek-v4.1",
    base_url="https://opencode.ai/zen/go/v1",
    api_key_env="TEST_OPENCODE_KEY",
    fallback=ModelConfig(
        model="deepseek-v4.1",
        base_url="https://api.openlux.ai/v1",
        api_key_env="TEST_OPENLUX_KEY",
    ),
)


def result(status="completed", *, output=None, usage=None, error=None, trace=None):
    return RunResult(status=status, output=output, usage=usage or {}, error=error,
                     trace=trace or [])


@pytest.fixture(autouse=True)
def clear_exhausted_providers():
    providers._EXHAUSTED_PROVIDERS.clear()
    yield
    providers._EXHAUSTED_PROVIDERS.clear()


@pytest.mark.asyncio
async def test_call_llm_success_does_not_use_fallback(monkeypatch):
    calls = []

    async def fake_call(**kwargs):
        calls.append(kwargs["model"])
        return result(output={"answer": "ok"})

    monkeypatch.setattr(execution, "call_llm", fake_call)
    actual = await call_llm(instructions="", input_text="", output_type=SolverAnswer,
                            model=PRIMARY)
    assert calls == [PRIMARY]
    assert actual.output == {"answer": "ok"}
    assert actual.trace[0] == {
        "type": "provider_identity", "model": "deepseek-v4.1",
        "base_url": "https://opencode.ai/zen/go/v1", "api_key_env": "TEST_OPENCODE_KEY",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("quota_error", [
    "Error code 402: insufficient_quota",
    "RateLimitError: Error code: 429 - {'error': {'type': 'GoUsageLimitError', "
    "'message': 'Go usage limit exceeded'}, 'metadata': {'limitName': '5 hour'}}",
])
async def test_quota_exhaustion_falls_back_once_and_caches_provider(monkeypatch, quota_error):
    calls = []

    async def fake_call(**kwargs):
        calls.append(kwargs["model"])
        if kwargs["model"].api_key_env == "TEST_OPENCODE_KEY":
            return result("failed", error=quota_error)
        return result(output={"answer": "fallback"}, usage={
            "prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5,
        })

    monkeypatch.setattr(execution, "call_llm", fake_call)
    first = await call_llm(instructions="", input_text="", output_type=SolverAnswer,
                           model=PRIMARY)
    second = await call_llm(instructions="", input_text="", output_type=SolverAnswer,
                            model=PRIMARY)
    assert calls == [PRIMARY, PRIMARY.fallback, PRIMARY.fallback]
    assert first.output == second.output == {"answer": "fallback"}
    assert first.status == second.status == "completed"
    assert first.usage["input_tokens"] == 3
    assert first.usage["output_tokens"] == 2
    assert first.usage["total_tokens"] == 5
    assert first.trace[0]["type"] == "provider_attempt"
    assert first.trace[0]["provider_identity"]["api_key_env"] == "TEST_OPENCODE_KEY"
    assert any(item["type"] == "provider_switch" for item in first.trace)
    assert first.trace[0]["result"]["error"] == quota_error


@pytest.mark.asyncio
async def test_rate_limit_does_not_trigger_fallback(monkeypatch):
    calls = []

    async def fake_call(**kwargs):
        calls.append(kwargs["model"])
        return result("failed", error="Error code 429: rate_limit_exceeded")

    monkeypatch.setattr(execution, "call_llm", fake_call)
    actual = await call_llm(instructions="", input_text="", output_type=SolverAnswer,
                            model=PRIMARY)
    assert actual.status == "failed"
    assert calls == [PRIMARY]
    assert not providers._EXHAUSTED_PROVIDERS


def test_fallback_requires_same_model_and_no_nested_fallback():
    with pytest.raises(ValueError, match="model must match"):
        ModelConfig(model="main", api_key_env="A", fallback=ModelConfig(
            model="other", api_key_env="B"))
    with pytest.raises(ValueError, match="another fallback"):
        ModelConfig(model="same", api_key_env="A", fallback=ModelConfig(
            model="same", api_key_env="B", fallback=ModelConfig(
                model="same", api_key_env="C")))


@pytest.mark.asyncio
async def test_usage_aliases_merge_once_and_unknown_counts_are_retained(monkeypatch):
    async def fake_call(**kwargs):
        if kwargs["model"].api_key_env == "TEST_OPENCODE_KEY":
            return result("failed", error="insufficient balance", usage={
                "prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 9,
                "model_turns": 1, "tool_calls": 1, "runs_without_known_tokens": 1,
            }, trace=[{"type": "tool_output", "output": "one"}])
        return result(output={"answer": "ok"}, usage={
            "input_tokens": 5, "output_tokens": 4, "total_tokens": 9,
            "model_turns": 2, "tool_calls": 3, "runs_without_known_tokens": 2,
        }, trace=[{"type": "tool_output", "output": "two"}])

    monkeypatch.setattr(execution, "call_llm", fake_call)
    actual = await call_llm(instructions="", input_text="", output_type=SolverAnswer,
                            model=PRIMARY)
    assert actual.usage == {
        "model_turns": 3, "tool_calls": 4, "input_tokens": 12,
        "output_tokens": 6, "total_tokens": 18, "runs_without_known_tokens": 3,
    }
    attempts = [item for item in actual.trace if item["type"] == "provider_attempt"]
    tool_outputs = [tool["output"] for attempt in attempts
                    for tool in attempt["result"]["trace"]
                    if tool["type"] == "tool_output"]
    assert tool_outputs == ["one", "two"]
    assert actual.usage["runs_without_known_tokens"] == 3


@pytest.mark.asyncio
async def test_run_agent_uses_same_fallback_router(monkeypatch):
    calls = []

    async def fake_run(**kwargs):
        calls.append(kwargs["model"])
        if len(calls) == 1:
            return result("failed", error="credit balance exhausted")
        return result(output={"answer": "agent answer"})

    monkeypatch.setattr(execution, "run_agent", fake_run)
    actual = await run_agent(instructions="", input_text="", tools=[],
        output_type=SolverAnswer, model=PRIMARY, budget=Budget(max_model_turns=2,
                                                               max_tool_calls=1))
    assert calls == [PRIMARY, PRIMARY.fallback]
    assert actual.output == {"answer": "agent answer"}


@pytest.mark.asyncio
async def test_run_agent_keeps_both_attempts_completion_events_at_top_level(monkeypatch):
    calls = []
    primary_events = [
        {"type": "tool_start", "name": "lookup", "call_id": "primary-1"},
        {"type": "tool_output", "name": "lookup", "call_id": "primary-1",
         "output": "primary facts"},
        {"type": "run_error", "failure_category": "model_error",
         "error": "insufficient_quota"},
    ]
    fallback_events = [
        {"type": "tool_start", "name": "lookup", "call_id": "fallback-1"},
        {"type": "tool_output", "name": "lookup", "call_id": "fallback-1",
         "output": "fallback facts"},
    ]

    async def fake_run(**kwargs):
        calls.append(kwargs["model"])
        if len(calls) == 1:
            return result("failed", error="insufficient_quota", trace=primary_events)
        return result(output={"answer": "agent answer"}, trace=fallback_events)

    monkeypatch.setattr(execution, "run_agent", fake_run)
    actual = await run_agent(instructions="", input_text="", tools=[],
        output_type=SolverAnswer, model=PRIMARY, budget=Budget(max_model_turns=2,
                                                               max_tool_calls=1))
    completion_events = [event for event in actual.trace
                         if event["type"] in {"tool_start", "tool_output", "run_error",
                                              "tool_budget"}]
    assert completion_events == [*primary_events, *fallback_events]
    assert [event["type"] for event in actual.trace].count("provider_attempt") == 2
    assert actual.output == {"answer": "agent answer"}
    # Provider envelopes and top-level evidence remain serializable without cycles.
    actual.model_dump_json()


@pytest.mark.asyncio
async def test_result_redacts_both_provider_keys(monkeypatch):
    monkeypatch.setenv("TEST_OPENCODE_KEY", "opencode-secret")
    monkeypatch.setenv("TEST_OPENLUX_KEY", "openlux-secret")
    async def fake_call(**kwargs):
        return result("failed", error="opencode-secret leaked openlux-secret")

    monkeypatch.setattr(execution, "call_llm", fake_call)
    actual = await call_llm(instructions="", input_text="", output_type=SolverAnswer,
                            model=PRIMARY)
    assert "opencode-secret" not in actual.model_dump_json()
    assert "openlux-secret" not in actual.model_dump_json()
