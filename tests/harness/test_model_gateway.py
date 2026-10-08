"""Model gateway: the single billable path for every harness model call.

The provider is a fake HTTP backend, so no paid model call is ever made. These
tests cover hard call/tool limits before forwarding, token clamping, accounting
attribution, key redaction and the preflight that refuses a bypass.
"""

import json

import httpx
import pytest
from contracts import Budget, ModelConfig
from harnesses.contracts import GroupKey, HarnessLimits
from harnesses.model_gateway import (
    CLIENT_HEADER, PHASE_HEADER, TASK_HEADER, BudgetExhausted, GatewayPool,
    GatewayPreflightError, ModelGateway, ModelGatewayError, ModelNotAllowed,
    ProviderError, create_gateway_app, preflight_gateway,
)

GROUP = GroupKey(
    run_id="run-1", environment="PhysicianBench", harness="openclaw",
    source_method="medevolve", seed=42,
)
PROVIDER_TRANSPORT = object()
LIMITS = {
    "execution": {"max_model_turns": 2, "max_tool_calls": 1},
    "learning": {"max_model_turns": 3, "max_tool_calls": 0},
    "max_input_tokens": 100,
    "max_output_tokens": 50,
    "max_total_tokens": 120,
    "timeout_seconds": 60,
    "learning_timeout_seconds": 30,
}


class CountingTokenizer:
    def __init__(self, model="locked-model", count=25):
        self.model = model
        self.count_value = count
        self.calls = 0

    def count(self, messages):
        self.calls += 1
        return self.count_value


def message(text="你好"):
    return [{"role": "user", "content": text}]


@pytest.fixture
def gateway_factory(provider, api_key):
    created = []

    def make(*, client_id="container-1", limits=None, model=None,
             transport=PROVIDER_TRANSPORT, **kwargs):
        values = dict(LIMITS)
        values.update(limits or {})
        config = model or ModelConfig(
            model="locked-model", base_url="https://opencode.ai/zen/go/v1",
            api_key_env="GATEWAY_API_KEY", max_output_tokens=40,
        )
        gateway = ModelGateway(
            group=GROUP, client_id=client_id, model=config,
            limits=HarnessLimits(
                execution=Budget(**values["execution"]),
                learning=Budget(**values["learning"]),
                max_input_tokens=values["max_input_tokens"],
                max_output_tokens=values["max_output_tokens"],
                max_total_tokens=values["max_total_tokens"],
                timeout_seconds=values["timeout_seconds"],
                learning_timeout_seconds=values["learning_timeout_seconds"],
            ),
            transport=provider.transport() if transport is PROVIDER_TRANSPORT else transport,
            **kwargs,
        )
        created.append(gateway)
        return gateway

    return make


@pytest.mark.asyncio
async def test_forwards_locked_model_with_session_header(gateway_factory, provider):
    gateway = gateway_factory()
    await gateway.complete(
        {"model": "some-client-model", "messages": message()},
        phase="execution", task_id="task-1",
    )
    assert provider.requests[0]["model"] == "locked-model"
    headers = provider.headers[0]
    assert headers["authorization"] == "Bearer secret-key"
    assert headers["x-opencode-session"]
    call = gateway.calls()[0]
    assert call.model == "locked-model"
    assert call.requested_model == "some-client-model"
    assert call.task_id == "task-1"
    assert call.phase == "execution"
    assert call.status == "forwarded"
    assert call.input_tokens == 10 and call.output_tokens == 5 and call.total_tokens == 15


@pytest.mark.asyncio
async def test_client_is_told_the_locked_model(gateway_factory):
    gateway = gateway_factory()
    response = await gateway.complete(
        {"model": "some-client-model", "messages": message()},
        phase="execution", task_id="task-1",
    )
    assert response["model"] == "locked-model"


@pytest.mark.asyncio
async def test_call_limit_stops_further_forwarding(gateway_factory, provider):
    gateway = gateway_factory()
    for _ in range(2):
        await gateway.complete({"messages": message()}, phase="execution", task_id="task-1")
    with pytest.raises(BudgetExhausted, match="调用预算"):
        await gateway.complete({"messages": message()}, phase="execution", task_id="task-1")
    assert len(provider.requests) == 2
    assert gateway.summary()["calls"] == 2


@pytest.mark.asyncio
async def test_tool_call_budget_blocks_before_forwarding(provider, gateway_factory):
    provider.tool_calls = 1
    gateway = gateway_factory()
    await gateway.complete({"messages": message()}, phase="execution", task_id="task-1")
    assert gateway.summary()["tool_calls"] == 1
    with pytest.raises(BudgetExhausted, match="工具调用预算"):
        await gateway.complete({"messages": message()}, phase="execution", task_id="task-1")
    assert len(provider.requests) == 1


@pytest.mark.asyncio
async def test_learning_phase_refuses_tools_with_zero_tool_budget(gateway_factory, provider):
    gateway = gateway_factory()
    with pytest.raises(BudgetExhausted, match="不允许工具调用"):
        await gateway.complete(
            {"messages": message(), "tools": [{"type": "function"}]},
            phase="learning", task_id="task-1",
        )
    assert provider.requests == []
    await gateway.complete({"messages": message()}, phase="learning", task_id="task-1")
    assert len(provider.requests) == 1


@pytest.mark.asyncio
async def test_output_tokens_are_clamped_by_the_remaining_budget(gateway_factory, provider):
    gateway = gateway_factory()
    await gateway.complete(
        {"messages": message(), "max_tokens": 1000}, phase="execution", task_id="task-1"
    )
    assert provider.requests[0]["max_tokens"] == 50
    call = gateway.calls()[0]
    assert call.max_tokens_sent == 50
    assert call.output_clamped is True
    assert call.input_counted is False
    assert call.soft_token_limit is True


@pytest.mark.asyncio
async def test_tokenizer_must_match_the_locked_model(gateway_factory):
    with pytest.raises(ModelGatewayError, match="tokenizer"):
        gateway_factory(tokenizer=CountingTokenizer(model="other-model"))


@pytest.mark.asyncio
async def test_counted_input_tokens_reserve_the_remaining_budget(gateway_factory, provider):
    tokenizer = CountingTokenizer(count=100)
    gateway = gateway_factory(
        tokenizer=tokenizer,
        limits={"execution": {"max_model_turns": 5, "max_tool_calls": 0}},
    )
    first = await gateway.complete({"messages": message()}, phase="execution", task_id="task-1")
    assert provider.requests[0]["max_tokens"] == 20
    assert gateway.calls()[0].input_counted is True
    assert gateway.calls()[0].soft_token_limit is False
    await gateway.complete({"messages": message()}, phase="execution", task_id="task-1")
    assert provider.requests[1]["max_tokens"] == 5
    with pytest.raises(BudgetExhausted, match="token"):
        await gateway.complete({"messages": message()}, phase="execution", task_id="task-1")
    assert len(provider.requests) == 2
    assert gateway.summary()["total_tokens"] == 30
    assert first["usage"]["total_tokens"] == 15


@pytest.mark.asyncio
async def test_unknown_usage_is_recorded_as_unknown_not_zero(provider, gateway_factory):
    provider.emit_usage = False
    gateway = gateway_factory()
    for _ in range(2):
        await gateway.complete({"messages": message()}, phase="execution", task_id="task-1")
    assert all(call.usage_known is False for call in gateway.calls())
    assert all(call.total_tokens is None for call in gateway.calls())
    summary = gateway.summary()
    assert summary["input_tokens"] is None
    assert summary["output_tokens"] is None
    assert summary["total_tokens"] is None
    assert summary["unknown_usage_calls"] == 2


@pytest.mark.asyncio
async def test_soft_token_limit_discloses_a_possible_over_cap_request(provider, gateway_factory):
    gateway = gateway_factory(limits={"max_total_tokens": 10})
    await gateway.complete({"messages": message()}, phase="execution", task_id="task-1")
    assert provider.requests
    call = gateway.calls()[0]
    assert call.soft_token_limit is True
    assert call.over_cap is True
    summary = gateway.summary()
    assert summary["over_cap_calls"] == 1
    assert summary["soft_token_limit"] is True
    with pytest.raises(BudgetExhausted):
        await gateway.complete({"messages": message()}, phase="execution", task_id="task-1")


@pytest.mark.asyncio
async def test_fallback_model_is_refused(provider, gateway_factory):
    gateway = gateway_factory(forbidden_models=["gpt-4o-mini"])
    with pytest.raises(ModelNotAllowed, match="gpt-4o-mini"):
        await gateway.complete(
            {"model": "gpt-4o-mini", "messages": message()},
            phase="execution", task_id="task-1",
        )
    assert provider.requests == []


@pytest.mark.asyncio
async def test_streaming_is_refused_before_forwarding(provider, gateway_factory):
    gateway = gateway_factory()
    with pytest.raises(ModelGatewayError, match="流式"):
        await gateway.complete(
            {"messages": message(), "stream": True}, phase="execution", task_id="task-1"
        )
    assert provider.requests == []


@pytest.mark.asyncio
async def test_gateway_without_an_injected_transport_is_refused_not_sent(
    api_key, gateway_factory
):
    """A gateway built with the default transport must fail, not dial out."""
    gateway = gateway_factory(transport=None)
    with pytest.raises(ProviderError, match="禁止外发 HTTP 请求"):
        await gateway.complete({"messages": message()}, phase="execution", task_id="task-1")
    call = gateway.calls()[0]
    assert call.status == "provider_error"
    assert "禁止外发 HTTP 请求" in call.error


@pytest.mark.asyncio
async def test_fhir_service_also_routes_a_default_transport_gateway(
    api_key, fhir_service, gateway_factory
):
    fhir_service.route("/chat/completions", httpx.Response(404, text="no provider here"))
    gateway = gateway_factory(transport=None)
    with pytest.raises(ProviderError, match="404"):
        await gateway.complete({"messages": message()}, phase="execution", task_id="task-1")
    assert [str(request.url) for request in fhir_service.requests] == [
        "https://opencode.ai/zen/go/v1/chat/completions"
    ]


@pytest.mark.asyncio
async def test_provider_error_is_recorded_with_the_key_redacted(provider, gateway_factory, api_key):
    provider.status = 500
    provider.error = f"upstream rejected key {api_key}"
    gateway = gateway_factory()
    with pytest.raises(ProviderError) as failure:
        await gateway.complete({"messages": message()}, phase="execution", task_id="task-1")
    assert api_key not in str(failure.value)
    assert "[REDACTED]" in str(failure.value)
    call = gateway.calls()[0]
    assert call.status == "provider_error"
    assert api_key not in json.dumps(call.model_dump(mode="json"))
    assert call.usage_known is False


@pytest.mark.asyncio
async def test_every_call_is_attributed_to_group_task_and_phase(gateway_factory):
    gateway = gateway_factory(
        limits={"execution": {"max_model_turns": 5, "max_tool_calls": 0},
                "learning": {"max_model_turns": 5, "max_tool_calls": 0}}
    )
    await gateway.complete({"messages": message()}, phase="execution", task_id="task-1")
    await gateway.complete({"messages": message()}, phase="learning", task_id="task-1")
    await gateway.complete({"messages": message()}, phase="execution", task_id="task-2")
    summary = gateway.summary()
    assert summary["group"]["harness"] == "openclaw"
    assert summary["client_id"] == "container-1"
    assert summary["tasks"]["task-1"]["phases"]["execution"]["calls"] == 1
    assert summary["tasks"]["task-1"]["phases"]["learning"]["calls"] == 1
    assert summary["tasks"]["task-2"]["phases"]["execution"]["calls"] == 1


@pytest.mark.asyncio
async def test_http_gateway_binds_client_task_and_phase(gateway_factory, provider):
    gateway = gateway_factory()
    pool = GatewayPool([gateway])
    app = create_gateway_app(pool)
    transport = httpx.ASGITransport(app=app)
    headers = {"x-opencode-session": "claw-1"}
    async with httpx.AsyncClient(transport=transport, base_url="http://gateway") as client:
        body = {"messages": message(), "model": "locked-model"}
        assert (await client.post("/v1/chat/completions", json=body)).status_code == 403
        assert (await client.post(
            "/v1/chat/completions", json=body, headers={CLIENT_HEADER: "container-1"}
        )).status_code == 400
        assert (await client.post(
            "/v1/chat/completions", json=body,
            headers={CLIENT_HEADER: "container-1", TASK_HEADER: "task-1"},
        )).status_code == 400
        response = await client.post(
            "/v1/chat/completions", json=body,
            headers={**headers, CLIENT_HEADER: "container-1", TASK_HEADER: "task-1",
                     PHASE_HEADER: "execution"},
        )
        assert response.status_code == 200
        assert response.json()["model"] == "locked-model"
        assert provider.requests[0]["messages"] == message()
        assert gateway.calls()[0].task_id == "task-1"
        models = await client.get(
            "/v1/models", headers={CLIENT_HEADER: "container-1"}
        )
        assert models.json()["data"][0]["id"] == "locked-model"


@pytest.mark.asyncio
async def test_http_gateway_reports_budget_exhaustion_and_unknown_client(gateway_factory):
    gateway = gateway_factory(limits={"execution": {"max_model_turns": 1, "max_tool_calls": 0}})
    app = create_gateway_app(GatewayPool([gateway]))
    transport = httpx.ASGITransport(app=app)
    headers = {CLIENT_HEADER: "container-1", TASK_HEADER: "task-1", PHASE_HEADER: "execution"}
    async with httpx.AsyncClient(transport=transport, base_url="http://gateway") as client:
        body = {"messages": message()}
        assert (await client.post("/v1/chat/completions", json=body, headers=headers)).status_code == 200
        assert (await client.post("/v1/chat/completions", json=body, headers=headers)).status_code == 429
        assert (await client.post(
            "/v1/chat/completions", json=body, headers={**headers, CLIENT_HEADER: "other"}
        )).status_code == 403


def test_preflight_names_components_that_cannot_use_the_gateway():
    gateway_url = "http://gateway:8081/"
    components = {
        "openclaw.execution": "http://gateway:8081",
        "openclaw.embedding": None,
        "hermes.auxiliary": "https://api.other.example/v1",
    }
    with pytest.raises(GatewayPreflightError) as failure:
        preflight_gateway(gateway_url, components)
    message = str(failure.value)
    assert "openclaw.embedding" in message
    assert "hermes.auxiliary" in message
    assert "openclaw.execution" not in message
    report = preflight_gateway(
        gateway_url, {"ace.execution": gateway_url, "ace.learning": "http://gateway:8081/"}
    )
    assert report["ace.execution"]["status"] == "ok"
    assert report["ace.learning"]["status"] == "ok"
