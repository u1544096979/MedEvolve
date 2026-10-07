"""Quota-aware routing for configured model providers."""

import os
import re
from collections.abc import Awaitable, Callable
from typing import Any

from agents import Tool
from pydantic import BaseModel

from contracts import Budget, ModelConfig, RunResult
from . import execution


_EXHAUSTED_PROVIDERS: set[tuple[str | None, str]] = set()
_COMPLETION_EVENT_TYPES = {"tool_start", "tool_output", "run_error", "tool_budget"}
_QUOTA_PATTERNS = (
    r"(?:gousagelimiterror|go\s+usage\s+limit\s+exceeded)",
    r"insufficient[_ ]quota",
    r"quota\s+(?:is\s+)?(?:exhausted|exceeded|depleted)",
    r"(?:exhausted|exceeded|depleted)\s+(?:the\s+)?quota",
    r"insufficient\s+balance",
    r"(?:credit|credits|budget)(?:\s+balance)?\s+(?:is\s+)?(?:exhausted|exceeded|depleted)",
    r"(?:exhausted|exceeded|depleted)\s+(?:the\s+)?(?:credit|credits|budget)(?:\s+balance)?",
    r"billing\s+(?:hard\s+)?limit\s+(?:has\s+been\s+)?(?:reached|exceeded)",
    r"(?:out\s+of\s+credits|no\s+credits\s+remaining|monthly\s+spend\s+limit\s+exceeded)",
)


def _provider_key(model: ModelConfig) -> tuple[str | None, str]:
    return model.base_url, model.api_key_env


def _quota_exhausted(error: str | None) -> bool:
    text = (error or "").lower()
    return any(re.search(pattern, text) for pattern in _QUOTA_PATTERNS)


def _identity(model: ModelConfig) -> dict[str, Any]:
    return {"type": "provider_identity", "model": model.model,
            "base_url": model.base_url, "api_key_env": model.api_key_env}


def _usage_total(usage: dict) -> int | float | None:
    total = usage.get("total_tokens")
    if total is not None:
        return total
    input_tokens = usage.get("input_tokens", usage.get("prompt_tokens"))
    output_tokens = usage.get("output_tokens", usage.get("completion_tokens"))
    if input_tokens is not None and output_tokens is not None:
        return input_tokens + output_tokens
    return None


def _merge_usage(*usages: dict) -> dict:
    """Normalize aliases before summing so prompt/input aliases are never doubled."""
    normalized = []
    for usage in usages:
        total = _usage_total(usage)
        normalized.append({
            "model_turns": usage.get("model_turns", 0),
            "tool_calls": usage.get("tool_calls", 0),
            "input_tokens": usage.get("input_tokens", usage.get("prompt_tokens")),
            "output_tokens": usage.get("output_tokens", usage.get("completion_tokens")),
            "total_tokens": total,
            "runs_without_known_tokens": usage.get(
                "runs_without_known_tokens", int(total is None)),
        })
    merged = {}
    for field in ("model_turns", "tool_calls", "input_tokens", "output_tokens",
                  "total_tokens", "runs_without_known_tokens"):
        values = [entry[field] for entry in normalized if entry[field] is not None]
        if values:
            merged[field] = sum(values)
    return merged


def _redact(result: RunResult, models: list[ModelConfig]) -> RunResult:
    secrets = [os.environ.get(model.api_key_env) for model in models]
    secrets = [secret for secret in secrets if secret]

    def clean(value):
        if isinstance(value, str):
            for secret in secrets:
                value = value.replace(secret, "[REDACTED]")
            return value
        if isinstance(value, dict):
            return {clean(key): clean(item) for key, item in value.items()}
        if isinstance(value, list):
            return [clean(item) for item in value]
        return value

    return RunResult.model_validate(clean(result.model_dump(mode="json")))


def _attempt(model: ModelConfig, result: RunResult) -> dict:
    return {
        "type": "provider_attempt",
        "provider_identity": _identity(model),
        "result": {
            "usage": result.usage,
            "status": result.status,
            "error": result.error,
            "trace": result.trace,
        },
    }


def _completion_events(result: RunResult) -> list[dict]:
    """Keep evaluator evidence visible without copying model usage events."""
    return [event for event in result.trace
            if event.get("type") in _COMPLETION_EVENT_TYPES]


async def _route(call: Callable[..., Awaitable[RunResult]], *, model: ModelConfig,
                 kwargs: dict) -> RunResult:
    fallback = model.fallback
    first = None
    if fallback is not None and _provider_key(model) in _EXHAUSTED_PROVIDERS:
        switch_reason = "provider quota previously exhausted"
    else:
        first = await call(**kwargs, model=model)
        if first.status != "failed" or fallback is None or not _quota_exhausted(first.error):
            first.trace.insert(0, _identity(model))
            return _redact(first, [model, *([fallback] if fallback else [])])
        _EXHAUSTED_PROVIDERS.add(_provider_key(model))
        switch_reason = first.error

    attempts = []
    if first is not None:
        attempts.append(_attempt(model, first))
        attempts.extend(_completion_events(first))
    attempts.append({"type": "provider_switch", "from": _identity(model),
                     "to": _identity(fallback), "reason": switch_reason})
    backup = await call(**kwargs, model=fallback)
    attempts.append(_attempt(fallback, backup))
    attempts.extend(_completion_events(backup))
    usage = _merge_usage(*(result.usage for result in ([first] if first is not None else [])),
                         backup.usage)
    if backup.status == "failed" and backup.error:
        backup.error = (f"Primary provider failed: {first.error if first else switch_reason}; "
                        f"fallback failed: {backup.error}")
    backup.trace = attempts
    backup.usage = usage
    return _redact(backup, [model, fallback])


async def call_llm(*, instructions: str, input_text: str, output_type: type[BaseModel],
                   model: ModelConfig) -> RunResult:
    return await _route(execution.call_llm, model=model, kwargs={
        "instructions": instructions, "input_text": input_text, "output_type": output_type,
    })


async def run_agent(*, instructions: str, input_text: str, tools: list[Tool],
                    output_type: type[BaseModel], model: ModelConfig,
                    budget: Budget) -> RunResult:
    return await _route(execution.run_agent, model=model, kwargs={
        "instructions": instructions, "input_text": input_text, "tools": tools,
        "output_type": output_type, "budget": budget,
    })
