"""Text action loop for environments with native GET/POST/FINISH protocols."""

import json
from typing import Callable

from pydantic import BaseModel

from contracts import Budget, EnvironmentExploration, ModelConfig, RunResult, SolverAnswer
from environments.medagentbench import NativeTools


def _usage(response) -> dict:
    usage = response.usage
    return usage.model_dump(mode="json", exclude_none=True) if usage is not None else {}


def _add_usage(totals: dict, current: dict) -> None:
    for key, value in current.items():
        if key in {"prompt_tokens", "completion_tokens", "input_tokens", "output_tokens", "total_tokens"}:
            continue
        if isinstance(value, int):
            totals[key] = totals.get(key, 0) + value
        elif isinstance(value, dict):
            nested = totals.setdefault(key, {})
            for nested_key, nested_value in value.items():
                if isinstance(nested_value, int):
                    nested[nested_key] = nested.get(nested_key, 0) + nested_value


def _finish_output(answers: list, output_type: type[BaseModel]) -> dict:
    if output_type is SolverAnswer:
        return SolverAnswer(answer=json.dumps(answers, ensure_ascii=False)).model_dump(mode="json")
    if output_type is EnvironmentExploration:
        return EnvironmentExploration(
            guidance="\n".join(str(item) for item in answers)
        ).model_dump(mode="json")
    if len(answers) != 1 or not isinstance(answers[0], dict):
        raise ValueError("FINISH must contain one object matching the requested output type")
    return output_type.model_validate(answers[0]).model_dump(mode="json")


def _is_finish(action: str) -> bool:
    normalized = action.replace("```tool_code", "").replace("```", "").strip()
    return normalized.startswith("FINISH(") and normalized.endswith(")")


async def run_native_agent(*, instructions: str, input_text: str, tools: NativeTools,
                           output_type: type[BaseModel], model: ModelConfig,
                           budget: Budget, client_factory: Callable,
                           result_factory: Callable) -> RunResult:
    trace = []
    usage = {
        "requests": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
        "model_turns": 0, "tool_calls": 0,
        "max_model_turns": budget.max_model_turns,
        "max_tool_calls": budget.max_tool_calls,
    }
    messages = [
        {"role": model.instructions_role, "content": instructions},
        {"role": "user", "content": (
            "Interact using one plain-text action per response. Use GET URL, POST URL followed by JSON, "
            "or terminate with FINISH([answer, ...]). FINISH is the only terminal action. "
            "Available native interface declarations:\n"
            + json.dumps(tools.functions, ensure_ascii=False)
            + "\n\nTask material:\n" + input_text
        )},
    ]
    env_failed = False
    try:
        if budget.max_model_turns == 0:
            return result_factory(model=model, trace=trace, usage=usage,
                                  status="budget_exhausted", error="Model turn budget exhausted")
        async with client_factory(model) as client:
            for _ in range(budget.max_model_turns):
                request = {"model": model.model, "messages": messages}
                trace.append({"type": "model_request", **request})
                usage["model_turns"] += 1
                response = await client.chat.completions.create(**request)
                response_dump = response.model_dump(mode="json")
                response_usage = _usage(response)
                _add_usage(usage, response_usage)
                usage["requests"] += 1
                usage["prompt_tokens"] = usage.get("prompt_tokens", 0) + response_usage.get("prompt_tokens", 0)
                usage["completion_tokens"] = usage.get("completion_tokens", 0) + response_usage.get("completion_tokens", 0)
                usage["input_tokens"] += response_usage.get(
                    "input_tokens", response_usage.get("prompt_tokens", 0)
                )
                usage["output_tokens"] += response_usage.get(
                    "output_tokens", response_usage.get("completion_tokens", 0)
                )
                usage["total_tokens"] += response_usage.get("total_tokens", 0)
                trace.append({"type": "model_response", "response": response_dump})
                message = response.choices[0].message
                if message.refusal:
                    raise ValueError(f"Model refused: {message.refusal}")
                action = message.content or ""
                if _is_finish(action):
                    result = await tools.execute(action)
                    output = _finish_output(result["answers"], output_type)
                    if env_failed:
                        trace.append({"type": "run_error", "failure_category": "environment_error",
                                      "error": "FHIR access failed during this run"})
                        return result_factory(model=model, output=output, trace=trace,
                                              usage=usage, status="failed",
                                              error="FHIR access failed during this run")
                    return result_factory(model=model, output=output, trace=trace,
                                          usage=usage, status="completed")

                if usage["tool_calls"] >= budget.max_tool_calls:
                    trace.append({"type": "tool_budget", "action": action})
                    return result_factory(model=model, trace=trace, usage=usage,
                                          status="budget_exhausted",
                                          error="Native action budget exhausted")
                usage["tool_calls"] += 1
                trace.append({"type": "tool_start", "name": "native_action",
                              "arguments": action})
                try:
                    result = await tools.execute(action)
                    reply = result.get("reply", "")
                except Exception as exc:
                    env_failed = True
                    reply = f"Error: {type(exc).__name__}: {exc}"
                    trace.append({"type": "run_error", "failure_category": "environment_error",
                                  "error": reply})
                    result = {"kind": "error", "request": {"action": action}, "reply": reply}
                trace.append({"type": "tool_output", "name": "native_action",
                              "output": result})
                messages.append({"role": "assistant", "content": action})
                messages.append({"role": "user", "content": reply})
        trace.append({"type": "run_error", "failure_category": "model_error",
                      "error": "Model turn budget exhausted before FINISH"})
        return result_factory(model=model, trace=trace, usage=usage,
                              status="budget_exhausted",
                              error="Model turn budget exhausted before FINISH")
    except Exception as exc:
        trace.append({"type": "run_error", "failure_category": "model_error",
                      "error": f"{type(exc).__name__}: {exc}"})
        return result_factory(model=model, trace=trace, usage=usage, status="failed",
                              error=f"{type(exc).__name__}: {exc}")
