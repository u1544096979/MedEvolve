"""Model gateway: one accountable path for every harness model call.

Every execution, auxiliary and background-learning client of every harness points
at this gateway. It forwards OpenAI Chat Completions requests to the authorised
provider, but only after enforcing this group's limits:

- the call count and the tool-call count are hard limits checked before forwarding;
- the output-token parameter is clamped by the remaining token budget;
- the model id is forced to the locked model, and configured fallback models are
  refused outright;
- every call is attributed to (group, task, phase) and recorded.

When a reliable tokenizer count for the locked model is available, the input and
output budget are reserved before the request. Without one, the token cap is a
soft stop: the request is forwarded, settled on the reported `usage`, and the
last request may exceed the cap - which the record and summary disclose. Missing
usage is recorded as unknown, never as zero.

If a locked component cannot be pointed at the gateway, `preflight_gateway`
fails and names that component instead of silently skipping its accounting.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Literal, Mapping, Protocol, Sequence
from urllib.parse import urlparse
from uuid import uuid4

import httpx
from contracts import ModelConfig
from fastapi import Body, FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from harnesses.contracts import GroupKey, HarnessLimits

CLIENT_HEADER = "x-medevolve-client"
TASK_HEADER = "x-medevolve-task"
PHASE_HEADER = "x-medevolve-phase"
PHASES = ("execution", "auxiliary", "learning")
_OPENCODE_SESSION = str(uuid4())
_TOKEN_FIELDS = ("input_tokens", "output_tokens", "total_tokens")


class ModelGatewayError(RuntimeError):
    """The gateway refused or could not forward a request."""


class BudgetExhausted(ModelGatewayError):
    """A hard call, tool-call or token limit is already reached."""


class ModelNotAllowed(ModelGatewayError):
    """The requested model is a configured fallback and must not be used."""


class ProviderError(ModelGatewayError):
    """The upstream provider rejected the forwarded request."""


class GatewayPreflightError(ModelGatewayError):
    """A component cannot route its model calls through the gateway."""


class Tokenizer(Protocol):
    """A tokenizer whose `model` must match the locked model."""

    model: str

    def count(self, messages: list[dict]) -> int | None: ...


class GatewayCall(BaseModel):
    """One accounted model request."""

    group: dict
    client_id: str
    task_id: str
    phase: str
    requested_model: str | None = None
    model: str
    status: Literal["forwarded", "provider_error"]
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    usage_known: bool = False
    tool_calls: int = 0
    max_tokens_sent: int | None = None
    output_clamped: bool = False
    input_counted: bool = False
    soft_token_limit: bool = False
    over_cap: bool = False
    duration_ms: float = 0.0
    error: str | None = None


@dataclass
class _Ledger:
    calls: int = 0
    tool_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    known: bool = True
    unknown_usage: int = 0
    over_cap: int = 0


class ModelGateway:
    """One group's accounted forwarder to the authorised provider."""

    def __init__(
        self,
        *,
        group: GroupKey,
        client_id: str,
        model: ModelConfig,
        limits: HarnessLimits,
        tokenizer: Tokenizer | None = None,
        transport: httpx.BaseTransport | httpx.AsyncBaseTransport | None = None,
        timeout: float = 600.0,
        forbidden_models: Sequence[str] = (),
        env: Mapping[str, str] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        if not client_id:
            raise ModelGatewayError("网关必须绑定调用容器标识")
        if tokenizer is not None and getattr(tokenizer, "model", None) != model.model:
            raise ModelGatewayError(
                f"tokenizer 与锁定模型不一致：{getattr(tokenizer, 'model', None)!r} != {model.model!r}"
            )
        if not model.base_url:
            raise ModelGatewayError("锁定模型缺少 base_url，无法配置网关转发")
        self.group = group
        self.client_id = client_id
        self.model = model
        self.limits = limits
        self.tokenizer = tokenizer
        self.forbidden_models = frozenset(forbidden_models)
        self._transport = transport
        self.timeout = timeout
        self._env = env
        self._clock = clock
        self._lock = asyncio.Lock()
        self._closed = False
        self._calls: list[GatewayCall] = []
        self._ledgers: dict[tuple[str, str], _Ledger] = {}

    # -- public API --------------------------------------------------------
    async def complete(self, payload: dict, *, phase: str, task_id: str) -> dict:
        if self._closed:
            raise ModelGatewayError("模型网关已关闭")
        if phase not in PHASES:
            raise ModelGatewayError(f"未知阶段：{phase}")
        if not task_id:
            raise ModelGatewayError("缺少任务归属，无法计入预算")
        messages = payload.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ModelGatewayError("messages 为必需项")
        if payload.get("stream"):
            raise ModelGatewayError("网关不支持流式响应，基座客户端必须使用非流式调用")
        async with self._lock:
            ledger = self._ledgers.setdefault((task_id, phase), _Ledger())
            self._check_limits(ledger, payload, phase)
            requested = payload.get("model")
            if isinstance(requested, str) and requested in self.forbidden_models:
                raise ModelNotAllowed(
                    f"拒绝备用模型 {requested}：锁定模型为 {self.model.model}"
                )
            forward = dict(payload)
            forward["model"] = self.model.model
            plan = self._plan_tokens(forward, messages, ledger)
            ledger.calls += 1
            started = self._clock()
            call = GatewayCall(
                group=self.group.model_dump(mode="json"),
                client_id=self.client_id,
                task_id=task_id,
                phase=phase,
                requested_model=requested if isinstance(requested, str) else None,
                model=self.model.model,
                status="forwarded",
                max_tokens_sent=plan["max_tokens"],
                output_clamped=plan["clamped"],
                input_counted=plan["input_tokens"] is not None,
                soft_token_limit=plan["input_tokens"] is None,
            )
            try:
                response = await self._post(forward)
            except Exception as exc:
                call.status = "provider_error"
                call.error = self._redact(str(exc))
                call.duration_ms = self._duration(started)
                self._calls.append(call)
                if isinstance(exc, ModelGatewayError):
                    raise
                raise ProviderError(call.error) from exc
            self._settle(call, ledger, response)
            call.duration_ms = self._duration(started)
            self._calls.append(call)
            response["model"] = self.model.model
            return response

    def calls(self) -> list[GatewayCall]:
        return list(self._calls)

    def summary(self) -> dict:
        tasks: dict[str, dict] = {}
        for call in self._calls:
            phase_entry = tasks.setdefault(call.task_id, {"phases": {}})
            phase_entry["phases"].setdefault(call.phase, {
                "calls": 0, "tool_calls": 0, "input_tokens": 0, "output_tokens": 0,
                "total_tokens": 0, "usage_known": True,
            })
            entry = phase_entry["phases"][call.phase]
            entry["calls"] += 1
            entry["tool_calls"] += call.tool_calls
            for field in _TOKEN_FIELDS:
                value = getattr(call, field)
                if entry[field] is None or value is None:
                    entry[field] = None
                else:
                    entry[field] += value
            entry["usage_known"] = entry["usage_known"] and call.usage_known
        return {
            "group": self.group.model_dump(mode="json"),
            "client_id": self.client_id,
            "model": self.model.model,
            "calls": len(self._calls),
            "tool_calls": sum(call.tool_calls for call in self._calls),
            **{field: _sum_or_none(self._calls, field) for field in _TOKEN_FIELDS},
            "unknown_usage_calls": sum(
                1 for call in self._calls if not call.usage_known
            ),
            "over_cap_calls": sum(1 for call in self._calls if call.over_cap),
            "soft_token_limit": any(call.soft_token_limit for call in self._calls),
            "limits": {
                "execution": self.limits.execution.model_dump(mode="json"),
                "learning": self.limits.learning.model_dump(mode="json"),
                "max_input_tokens": self.limits.max_input_tokens,
                "max_output_tokens": self.limits.max_output_tokens,
                "max_total_tokens": self.limits.max_total_tokens,
            },
            "tasks": tasks,
        }

    async def aclose(self) -> None:
        self._closed = True

    # -- limits and accounting --------------------------------------------
    def _phase_budget(self, phase: str):
        return self.limits.learning if phase == "learning" else self.limits.execution

    def _check_limits(self, ledger: _Ledger, payload: dict, phase: str) -> None:
        budget = self._phase_budget(phase)
        if ledger.calls >= budget.max_model_turns:
            raise BudgetExhausted(
                f"{phase} 阶段模型调用预算已用尽：{budget.max_model_turns}"
            )
        if budget.max_tool_calls == 0:
            if payload.get("tools"):
                raise BudgetExhausted(f"{phase} 阶段不允许工具调用")
        elif ledger.tool_calls >= budget.max_tool_calls:
            raise BudgetExhausted(
                f"{phase} 阶段工具调用预算已用尽：{budget.max_tool_calls}"
            )

    def _plan_tokens(self, forward: dict, messages: list[dict], ledger: _Ledger) -> dict:
        remaining_total = self.limits.max_total_tokens - ledger.total_tokens
        remaining_output = self.limits.max_output_tokens - ledger.output_tokens
        input_tokens = self._count_input(messages)
        if input_tokens is not None:
            remaining_total -= input_tokens
        if remaining_total <= 0:
            raise BudgetExhausted(
                f"token 额度已用尽：剩余 {remaining_total}，无法转发 {self.model.model} 请求"
            )
        if remaining_output <= 0:
            raise BudgetExhausted(f"输出 token 额度已用尽：{remaining_output}")
        requested = forward.get("max_tokens", forward.get("max_completion_tokens"))
        key = "max_completion_tokens" if "max_completion_tokens" in forward else "max_tokens"
        if requested is None:
            requested = self.model.max_output_tokens
        allowed = remaining_total if requested is None else min(requested, remaining_total)
        allowed = min(allowed, remaining_output)
        if allowed <= 0:
            raise BudgetExhausted("剩余 token 额度不足以容纳任何输出")
        forward[key] = allowed
        return {
            "max_tokens": allowed,
            "clamped": requested is not None and allowed < requested,
            "input_tokens": input_tokens,
        }

    def _count_input(self, messages: list[dict]) -> int | None:
        if self.tokenizer is None:
            return None
        counted = self.tokenizer.count(messages)
        if counted is None or not isinstance(counted, int) or counted < 0:
            return None
        return counted

    def _settle(self, call: GatewayCall, ledger: _Ledger, response: dict) -> None:
        call.tool_calls = self._count_tool_calls(response)
        ledger.tool_calls += call.tool_calls
        usage = response.get("usage")
        if not isinstance(usage, dict):
            call.usage_known = False
            ledger.known = False
            ledger.unknown_usage += 1
            return
        input_tokens = _as_int(usage.get("input_tokens", usage.get("prompt_tokens")))
        output_tokens = _as_int(usage.get("output_tokens", usage.get("completion_tokens")))
        total_tokens = _as_int(usage.get("total_tokens"))
        if total_tokens is None and input_tokens is not None and output_tokens is not None:
            total_tokens = input_tokens + output_tokens
        if input_tokens is None or output_tokens is None or total_tokens is None:
            call.usage_known = False
            ledger.known = False
            ledger.unknown_usage += 1
            return
        call.usage_known = True
        call.input_tokens, call.output_tokens, call.total_tokens = (
            input_tokens, output_tokens, total_tokens,
        )
        ledger.input_tokens += input_tokens
        ledger.output_tokens += output_tokens
        ledger.total_tokens += total_tokens
        call.over_cap = (
            ledger.total_tokens > self.limits.max_total_tokens
            or ledger.input_tokens > self.limits.max_input_tokens
            or ledger.output_tokens > self.limits.max_output_tokens
        )
        if call.over_cap:
            ledger.over_cap += 1

    @staticmethod
    def _count_tool_calls(response: dict) -> int:
        choices = response.get("choices") or []
        if not choices:
            return 0
        tool_calls = choices[0].get("message", {}).get("tool_calls") or []
        return len(tool_calls) if isinstance(tool_calls, list) else 0

    def _duration(self, started: float) -> float:
        return round((self._clock() - started) * 1000, 3)

    # -- transport ---------------------------------------------------------
    def _api_key(self) -> str:
        env = self._env if self._env is not None else os.environ
        key = env.get(self.model.api_key_env)
        if not key:
            raise ModelGatewayError(
                f"缺少 API key 环境变量：{self.model.api_key_env}"
            )
        return key

    def _redact(self, text: str) -> str:
        key = (self._env if self._env is not None else os.environ).get(
            self.model.api_key_env
        )
        return text.replace(key, "[REDACTED]") if key else text

    def _session_headers(self) -> dict:
        if urlparse(self.model.base_url or "").hostname == "opencode.ai":
            return {"x-opencode-session": _OPENCODE_SESSION, "User-Agent": "MedEvolve/0.1"}
        return {}

    async def _post(self, payload: dict) -> dict:
        url = (self.model.base_url or "").rstrip("/") + "/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._api_key()}",
            "Content-Type": "application/json",
            **self._session_headers(),
        }
        async with httpx.AsyncClient(
            timeout=self.timeout, trust_env=False, transport=self._transport
        ) as client:
            response = await client.post(url, json=payload, headers=headers)
        if response.status_code >= 400:
            raise ProviderError(
                f"provider 返回 {response.status_code}：{self._redact(response.text[:2000])}"
            )
        try:
            return response.json()
        except ValueError as exc:
            raise ProviderError(f"provider 返回非 JSON 响应：{exc}") from exc


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _sum_or_none(calls: list[GatewayCall], field: str) -> int | None:
    if not calls:
        return 0
    for call in calls:
        if getattr(call, field) is None:
            return None
    return sum(getattr(call, field) for call in calls)


class GatewayPool:
    """Gateways addressable by calling-container identity."""

    def __init__(self, gateways: Iterable[ModelGateway] = ()):
        self._gateways: dict[str, ModelGateway] = {}
        for gateway in gateways:
            self.add(gateway)

    def add(self, gateway: ModelGateway) -> None:
        self._gateways[gateway.client_id] = gateway

    def get(self, client_id: str) -> ModelGateway | None:
        return self._gateways.get(client_id)

    def values(self) -> list[ModelGateway]:
        return list(self._gateways.values())


def create_gateway_app(pool: GatewayPool) -> FastAPI:
    """OpenAI-compatible routes; the client, task and phase come from headers."""
    app = FastAPI(title="MedEvolve Harness Model Gateway")

    def gateway_for(client_id: str | None) -> ModelGateway:
        if not client_id:
            raise HTTPException(403, f"缺少 {CLIENT_HEADER} 调用容器标识")
        gateway = pool.get(client_id)
        if gateway is None:
            raise HTTPException(403, f"调用容器 {client_id} 没有可用网关")
        return gateway

    @app.get("/v1/models")
    async def list_models(
        x_medevolve_client: str | None = Header(None, alias=CLIENT_HEADER),
    ) -> dict:
        gateway = gateway_for(x_medevolve_client)
        return {"object": "list", "data": [{
            "id": gateway.model.model, "object": "model",
            "owned_by": "medevolve-gateway",
        }]}

    @app.post("/v1/chat/completions")
    async def chat_completions(
        payload: dict = Body(...),
        x_medevolve_client: str | None = Header(None, alias=CLIENT_HEADER),
        x_medevolve_task: str | None = Header(None, alias=TASK_HEADER),
        x_medevolve_phase: str | None = Header(None, alias=PHASE_HEADER),
    ):
        gateway = gateway_for(x_medevolve_client)
        if not x_medevolve_task or not x_medevolve_phase:
            raise HTTPException(400, f"需要 {TASK_HEADER} 与 {PHASE_HEADER}")
        try:
            return await gateway.complete(
                payload, phase=x_medevolve_phase, task_id=x_medevolve_task
            )
        except BudgetExhausted as exc:
            return JSONResponse({"error": {"message": str(exc)}}, status_code=429)
        except ModelNotAllowed as exc:
            return JSONResponse({"error": {"message": str(exc)}}, status_code=400)
        except (ProviderError, ModelGatewayError) as exc:
            return JSONResponse({"error": {"message": str(exc)}}, status_code=502)

    return app


def preflight_gateway(
    gateway_url: str, components: Mapping[str, str | None]
) -> dict[str, dict]:
    """Fail when a locked component cannot route its calls through the gateway."""
    target = (gateway_url or "").strip().rstrip("/")
    if not target:
        raise GatewayPreflightError("模型网关地址为空，无法预检")
    report: dict[str, dict] = {}
    blocked: list[str] = []
    for name, value in components.items():
        if value and value.strip().rstrip("/") == target:
            report[name] = {"base_url": value, "status": "ok", "reason": ""}
            continue
        reason = "未配置 base_url" if not value else f"base_url={value} 未指向模型网关 {target}"
        report[name] = {"base_url": value, "status": "blocked", "reason": reason}
        blocked.append(name)
    if blocked:
        details = "；".join(f"{name}（{report[name]['reason']}）" for name in sorted(blocked))
        raise GatewayPreflightError(
            "以下组件无法把模型请求指向网关，不能声称预算一致：" + details
        )
    return report
