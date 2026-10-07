"""Text action loop for environments with native GET/POST/FINISH protocols."""

import copy
import json
import os
from typing import Callable

from pydantic import BaseModel

from contracts import Budget, EnvironmentExploration, ModelConfig, RunResult, SolverAnswer
from environments.medagentbench import NativeTools
from .context import budget_entry, count_tokens, dumps, fit_history, output_limit, prepare_input


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


def _redact_secret(text: str, model: ModelConfig) -> str:
    key = os.environ.get(model.api_key_env)
    return text.replace(key, "[REDACTED]") if key else text


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
    api_base = tools.endpoint.rstrip("/")
    function_declarations = json.dumps(tools.functions, ensure_ascii=False).replace(
        "{api_base}", api_base
    )
    exploration_delivery = (
        "本任务是环境探索，没有待回答的题目：用 FINISH 提交时，必须把抽象环境指导正文作为 "
        "answers 中的文本项提交，例如 FINISH([\"<此处写你实际验证过的能力、参数限制与故障>\"])，"
        "不能提交空列表，也不能把未验证的能力写成已可用。\n\n"
        if output_type is EnvironmentExploration else ""
    )
    solution_delivery = (
        "本任务是解题：用 FINISH 提交时，必须把最终答案文本作为 answers 里的文本项提交，"
        "写明你执行的操作、依据与结论，不能提交空列表。\n\n"
        if output_type is SolverAnswer else ""
    )
    native_prompt = (
        "You are an expert in using FHIR functions to assist medical professionals. "
        "Based on the task and available functions, make one or more function calls to achieve it.\n\n"
        "1. If you decide to invoke a GET function, you MUST put it in the format of\n"
        "GET url?param_name1=param_value1&param_name2=param_value2...\n\n"
        "2. If you decide to invoke a POST function, you MUST put it in the format of\n"
        "POST url\n[your payload data in JSON format]\n\n"
        "3. When all questions are answered and all requested tasks are finished, you MUST call FINISH "
        "in this format (the list must be JSON loadable):\nFINISH([answer1, answer2, ...])\n\n"
        + exploration_delivery
        + solution_delivery
        + "Your response must be exactly one of these three cases. You can call only one function/action "
        "each time and SHOULD NOT include any other text in the response.\n\n"
        f"Here is a list of functions in JSON format that you can invoke. Use {api_base} as the api_base.\n"
        f"{function_declarations}\n\n"
        f"Task material:\n{input_text}"
    )
    messages = [
        {"role": model.instructions_role, "content": instructions},
        {"role": "user", "content": native_prompt},
    ]
    env_failed = False
    # 参数错误恢复判定只适用于解题路径：探索运行以探测参数边界并产出指导为目的，
    # 服务拒绝某个参数是探索的预期结果而非故障。
    solving = output_type is SolverAnswer
    pending_parameter_error = False
    try:
        if budget.max_model_turns == 0:
            return result_factory(model=model, trace=trace, usage=usage,
                                  status="budget_exhausted", error="Model turn budget exhausted")
        prompt_prefix = native_prompt[:-len(input_text)] if input_text else native_prompt
        input_text, entry = prepare_input(input_text, model,
            instructions=instructions + prompt_prefix, schema={})
        messages[1]["content"] = prompt_prefix + input_text
        trace.append(entry)
        async with client_factory(model) as client:
            for _ in range(budget.max_model_turns):
                messages, dropped = fit_history(messages, model, fixed_tokens=0, prefix_count=2,
                    group_start=lambda item: item.get("role") == "assistant")
                trace.append(budget_entry(model, count_tokens(dumps(messages), model),
                                          history_items_dropped=dropped))
                request = {"model": model.model, "messages": messages,
                           "max_tokens": output_limit(model)}
                trace.append({"type": "model_request", **copy.deepcopy(request)})
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
                    if result.get("kind") != "FINISH":
                        error = f"Invalid FINISH action: {result.get('reply', 'invalid response')}"
                        failure_category = "environment_error" if env_failed else "model_error"
                        trace.append({"type": "finish_error", "request": result.get("request"),
                                      "reply": result.get("reply"), "error": error})
                        trace.append({"type": "run_error", "failure_category": failure_category,
                                      "error": error})
                        return result_factory(model=model, trace=trace, usage=usage,
                                              status="failed", error=error)
                    try:
                        output = _finish_output(result["answers"], output_type)
                    except Exception as exc:
                        error = f"Invalid FINISH output: {type(exc).__name__}: {exc}"
                        failure_category = "environment_error" if env_failed else "model_error"
                        trace.append({"type": "finish_error", "request": result.get("request"),
                                      "reply": result.get("reply"), "error": error})
                        trace.append({"type": "run_error", "failure_category": failure_category,
                                      "error": error})
                        return result_factory(model=model, trace=trace, usage=usage,
                                              status="failed", error=error)
                    if env_failed:
                        trace.append({"type": "run_error", "failure_category": "environment_error",
                                      "error": "FHIR access failed during this run"})
                        return result_factory(model=model, output=output, trace=trace,
                                              usage=usage, status="failed",
                                              error="FHIR access failed during this run")
                    if solving and pending_parameter_error:
                        trace.append({"type": "run_error", "failure_category": "tool_error",
                                      "error": "Query parameter error was not corrected"})
                        return result_factory(model=model, output=output, trace=trace,
                                              usage=usage, status="failed",
                                              error="Query parameter error was not corrected")
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
                tool_exception = False
                try:
                    result = await tools.execute(action)
                    reply = result.get("reply", "")
                    if result.get("failure_category") == "environment_error":
                        env_failed = True
                        trace.append({"type": "run_error", "failure_category": "environment_error",
                                      "error": _redact_secret(str(reply), model)})
                    if solving:
                        if result.get("failure_category") == "tool_error":
                            pending_parameter_error = True
                        elif result.get("kind") == "GET" and not result.get("failure_category"):
                            pending_parameter_error = False
                except Exception as exc:
                    tool_exception = True
                    env_failed = True
                    reply = _redact_secret(f"Error: {type(exc).__name__}: {exc}", model)
                    trace.append({"type": "run_error", "failure_category": "environment_error",
                                  "error": reply})
                    result = {"kind": "error", "request": {"action": action}, "reply": reply}
                trace.append({"type": "tool_output", "name": "native_action",
                              "output": result})
                if result.get("kind") == "error" and not tool_exception:
                    error = f"Invalid native action: {result.get('reply', 'invalid action')}"
                    failure_category = "environment_error" if env_failed else "model_error"
                    trace.append({"type": "run_error", "failure_category": failure_category,
                                  "error": error})
                    return result_factory(model=model, trace=trace, usage=usage,
                                          status="failed", error=error)
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
