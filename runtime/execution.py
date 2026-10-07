"""Chat Completions execution with local records and explicit budgets."""

import json
import os
from urllib.parse import urlparse
from uuid import uuid4

from agents import (
    Agent, AgentOutputSchema, FunctionTool, ModelSettings, OpenAIChatCompletionsModel,
    RunConfig, RunHooks, Runner, Tool,
)
from agents.exceptions import AgentsException, MaxTurnsExceeded
from agents.models.chatcmpl_converter import Converter
from agents.run_config import ModelInputData
from openai import AsyncOpenAI
from pydantic import BaseModel
from pydantic_core import to_jsonable_python

from contracts import Budget, ModelConfig, RunResult
from .context import (budget_entry, count_tokens, dumps, fit_history, input_limit,
                      output_limit, prepare_input)


_OPENCODE_SESSION = str(uuid4())


class _OpenCodeOutputSchema(AgentOutputSchema):
    def validate_json(self, json_str):
        # OpenCode can return literal newlines inside quoted answer strings.
        try:
            parsed = json.loads(json_str, strict=False)
        except ValueError:
            return super().validate_json(json_str)
        return super().validate_json(json.dumps(parsed, ensure_ascii=False))


class _ToolBudgetExceeded(AgentsException):
    pass


# 收尾调用可携带的工具结果上限（字符）：预算用尽后只靠已有结果作答。
FINAL_ANSWER_CHARS = 120_000
TOOL_RESULT_SNIPPET_CHARS = 4000
async def _final_answer(*, client: AsyncOpenAI, model: ModelConfig, instructions: str,
                        input_text: str, trace: list[dict],
                        output_type: type[BaseModel]) -> RunResult:
    """轮次或工具预算用尽后的收尾调用：不再提供工具，只要求结构化答案。"""
    ending = (
        "轮次或工具调用预算已用尽，不能再调用任何工具。\n"
        "只依据任务材料与下列已获得的工具结果，直接输出最终结构化答案；\n"
        "信息不足时按已有结果作答并说明限制，不要声称未验证的能力或未执行的操作。\n\n"
        "已获得的工具结果（可能被截断）：\n"
    )
    note = ("\nReturn only a JSON object matching this schema:\n"
            + json.dumps(output_type.model_json_schema(), ensure_ascii=False))
    input_text, entry = prepare_input(input_text, model,
        instructions=instructions + note + ending, schema={})
    base = count_tokens(instructions + note + input_text + ending, model)
    digest = _tool_digest(trace, model=model, token_budget=input_limit(model) - base - 64)
    payload = f"{input_text}\n\n{ending}{digest or '（没有完成的工具结果）'}"
    entry["input_tokens_estimate"] = count_tokens(instructions + note + payload, model)
    messages = [{"role": model.instructions_role, "content": instructions + note},
                {"role": "user", "content": payload}]
    usage = {"model_turns": 1, "tool_calls": 0}
    trace_items = [entry, {"type": "model_request", "stage": "final_answer", "messages": messages}]
    response = await client.chat.completions.create(
        model=model.model, messages=messages,
        response_format={"type": "json_object"},
        max_tokens=output_limit(model),
    )
    trace_items.append({"type": "model_response", "stage": "final_answer",
                        "response": response.model_dump(mode="json")})
    if response.usage is not None:
        usage.update(response.usage.model_dump(mode="json"))
    output = output_type.model_validate_json(
        response.choices[0].message.content or ""
    ).model_dump(mode="json")
    return RunResult(output=output, trace=trace_items, usage=usage, status="completed")


def _tool_digest(trace: list[dict], *, model: ModelConfig | None = None,
                 token_budget: int | None = None) -> str:
    """按时间倒序取工具结果，超出上限即停，最后按时间顺序拼接。"""
    outputs = {item.get("call_id"): item.get("output") for item in trace
               if item.get("type") == "tool_output"}
    calls = [(item.get("name"), item.get("arguments"), item.get("call_id"))
             for item in trace if item.get("type") == "tool_start"]
    chunks, used = [], 0
    for name, arguments, call_id in reversed(calls):
        text = str(outputs.get(call_id, ""))
        if len(text) > TOOL_RESULT_SNIPPET_CHARS:
            text = text[:TOOL_RESULT_SNIPPET_CHARS] + "… [截断]"
        chunk = f"{name}({arguments}) -> {text}"
        if used + len(chunk) > FINAL_ANSWER_CHARS:
            break
        if model is not None and count_tokens("\n".join([*chunks, chunk]), model) > token_budget:
            break
        chunks.append(chunk)
        used += len(chunk)
    return "\n".join(reversed(chunks))


def _merge_usage(loop: dict, final: dict) -> dict:
    merged = dict(loop)
    for field in ("model_turns", "tool_calls", "requests", "input_tokens", "output_tokens",
                  "total_tokens", "prompt_tokens", "completion_tokens"):
        values = [value for value in (loop.get(field), final.get(field)) if value is not None]
        if values:
            merged[field] = sum(values)
    return merged


def _client(model: ModelConfig) -> AsyncOpenAI:
    key = os.environ.get(model.api_key_env)
    if not key:
        raise ValueError(f"Missing API key environment variable: {model.api_key_env}")
    headers = {}
    if urlparse(model.base_url or "").hostname == "opencode.ai":
        headers = {"x-opencode-session": _OPENCODE_SESSION, "User-Agent": "MedEvolve/0.1"}
    return AsyncOpenAI(api_key=key, base_url=model.base_url, max_retries=0,
                       default_headers=headers)


def _result(*, model: ModelConfig, **fields) -> RunResult:
    # Provider errors and tool outputs can echo credentials; redact before saving.
    key = os.environ.get(model.api_key_env)

    def redact(value):
        if isinstance(value, str):
            return value.replace(key, "[REDACTED]") if key else value
        if isinstance(value, dict):
            return {redact(k): redact(v) for k, v in value.items()}
        if isinstance(value, list):
            return [redact(v) for v in value]
        return value

    return RunResult(**redact(fields))


async def call_llm(*, instructions: str, input_text: str,
                   output_type: type[BaseModel], model: ModelConfig) -> RunResult:
    trace = []
    usage = {"model_turns": 0, "tool_calls": 0}
    try:
        input_text, entry = prepare_input(input_text, model, instructions=instructions,
                                          schema=output_type.model_json_schema())
        trace.append(entry)
        async with _client(model) as client:
            messages = [
                {"role": model.instructions_role, "content": instructions +
                 "\nReturn only a JSON object matching this schema:\n" +
                 json.dumps(output_type.model_json_schema(), ensure_ascii=False)},
                {"role": "user", "content": input_text},
            ]
            trace.append({"type": "model_request", "messages": messages})
            usage["model_turns"] = 1
            response = await client.chat.completions.create(
                model=model.model, messages=messages,
                response_format={"type": "json_object"},
                max_tokens=output_limit(model),
            )
            trace.append({"type": "model_response", "response": response.model_dump(mode="json")})
            if response.usage is not None:
                usage.update(response.usage.model_dump(mode="json"))
            message = response.choices[0].message
            if message.refusal:
                raise ValueError(f"Model refused: {message.refusal}")
            output = output_type.model_validate_json(message.content or "").model_dump(mode="json")
        return _result(model=model, output=output, trace=trace, usage=usage, status="completed")
    except Exception as exc:
        return _result(model=model, trace=trace, usage=usage, status="failed",
                       error=f"{type(exc).__name__}: {exc}")


class _Records(RunHooks):
    def __init__(self, budget: Budget):
        self.budget = budget
        self.trace = []
        self.model_turns = 0
        self.tool_calls = 0
        self.tokens = {"requests": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    async def on_llm_start(self, context, agent, system_prompt, input_items):
        self.model_turns += 1
        self.trace.append({"type": "model_request", "instructions": system_prompt,
                           "input": to_jsonable_python(input_items)})

    async def on_llm_end(self, context, agent, response):
        self.tokens["requests"] += 1
        for field in ("input_tokens", "output_tokens", "total_tokens"):
            self.tokens[field] += getattr(response.usage, field)
        self.trace.append({"type": "model_response", "response_id": response.response_id,
                           "output": [item.model_dump(mode="json") for item in response.output],
                           "usage": to_jsonable_python(response.usage)})

    async def on_tool_start(self, context, agent, tool):
        if self.tool_calls >= self.budget.max_tool_calls:
            self.trace.append({"type": "tool_budget", "name": tool.name,
                               "call_id": context.tool_call_id})
            raise _ToolBudgetExceeded("Function tool call budget exhausted")
        self.tool_calls += 1
        self.trace.append({"type": "tool_start", "name": tool.name,
                           "call_id": context.tool_call_id, "arguments": context.tool_arguments})

    async def on_tool_end(self, context, agent, tool, result):
        self.trace.append({"type": "tool_output", "name": tool.name,
                           "call_id": context.tool_call_id,
                           "output": to_jsonable_python(result, fallback=str)})

    def usage(self):
        return {**self.tokens,
                "model_turns": self.model_turns, "tool_calls": self.tool_calls,
                "max_model_turns": self.budget.max_model_turns,
                "max_tool_calls": self.budget.max_tool_calls}


async def run_agent(*, instructions: str, input_text: str, tools: list[Tool],
                    output_type: type[BaseModel], model: ModelConfig,
                    budget: Budget) -> RunResult:
    from environments.medagentbench import NativeTools

    if isinstance(tools, NativeTools):
        from .native import run_native_agent

        return await run_native_agent(
            instructions=instructions, input_text=input_text, tools=tools,
            output_type=output_type, model=model, budget=budget,
            client_factory=_client, result_factory=_result,
        )

    records = _Records(budget)
    try:
        if any(not isinstance(tool, FunctionTool) for tool in tools):
            raise ValueError("Chat Completions runtime supports only SDK FunctionTool tools")
        if budget.max_model_turns == 0:
            raise MaxTurnsExceeded("Model turn budget exhausted")
        tool_schemas = [Converter.tool_to_openai(tool) for tool in tools]
        input_text, entry = prepare_input(input_text, model, instructions=instructions,
            schema=output_type.model_json_schema(), tools=tool_schemas)
        records.trace.append(entry)

        def filter_input(data):
            model_data = data.model_data
            fixed = count_tokens((model_data.instructions or "") + dumps(tool_schemas)
                                 + dumps(output_type.model_json_schema()), model)
            seen_output = False

            def group_start(item):
                nonlocal seen_output
                output = item.get("type") == "function_call_output"
                start = seen_output and not output
                seen_output = output
                return start

            items, dropped = fit_history(model_data.input, model, fixed_tokens=fixed,
                prefix_count=2 if model.instructions_role == "user" else 1,
                group_start=group_start)
            records.trace.append(budget_entry(model, fixed + count_tokens(dumps(items), model),
                                               history_items_dropped=dropped))
            return ModelInputData(input=items, instructions=model_data.instructions)

        async with _client(model) as client:
            agent_instructions = instructions + (
                "\n完成工具查询后，最终输出只返回符合以下 schema 的 JSON 对象，不加 Markdown 或正文。"
                "\nReturn only a JSON object matching this schema:\n"
                + json.dumps(output_type.model_json_schema(), ensure_ascii=False)
            )
            agent_input = input_text
            if model.instructions_role == "user":
                agent_input = [
                    {"role": "user", "content": agent_instructions},
                    {"role": "user", "content": input_text},
                ]
                agent_instructions = None
            agent = Agent(
                name="MedEvolve", instructions=agent_instructions, tools=tools,
                # Shared contracts include open dictionaries, which strict schemas reject.
                output_type=(
                    _OpenCodeOutputSchema if urlparse(model.base_url or "").hostname == "opencode.ai"
                    else AgentOutputSchema
                )(output_type, strict_json_schema=False),
                model=OpenAIChatCompletionsModel(model=model.model, openai_client=client),
                model_settings=ModelSettings(
                    max_tokens=output_limit(model),
                    parallel_tool_calls=model.parallel_tool_calls,
                    extra_body={"response_format": {"type": "json_object"}}
                    if urlparse(model.base_url or "").hostname == "opencode.ai" else None,
                ),
            )
            # 留一轮做收尾：预算用尽时不再调工具，直接产出结构化答案。
            reserved = budget.max_model_turns >= 2
            loop_turns = budget.max_model_turns - 1 if reserved else budget.max_model_turns
            try:
                result = await Runner.run(
                    agent, agent_input, hooks=records, max_turns=loop_turns,
                    run_config=RunConfig(tracing_disabled=True, call_model_input_filter=filter_input),
                )
                output = output_type.model_validate(result.final_output).model_dump(mode="json")
            except (MaxTurnsExceeded, _ToolBudgetExceeded) as exc:
                if not reserved:
                    raise
                final = await _final_answer(
                    client=client, model=model, instructions=instructions,
                    input_text=input_text, trace=records.trace, output_type=output_type,
                )
                return _result(
                    model=model, output=final.output,
                    trace=[*records.trace, *final.trace],
                    usage=_merge_usage(records.usage(), final.usage), status="completed",
                )
        return _result(model=model, output=output, trace=records.trace,
                       usage=records.usage(), status="completed")
    except Exception as exc:
        status = "budget_exhausted" if isinstance(exc, (MaxTurnsExceeded, _ToolBudgetExceeded)) else "failed"
        completed_tools = {
            item.get("call_id") for item in records.trace if item.get("type") == "tool_output"
        }
        pending_tool = next(
            (item for item in reversed(records.trace)
             if item.get("type") == "tool_start"
             and item.get("call_id") not in completed_tools),
            None,
        )
        failure_category = (
            "tool_error" if pending_tool is not None else "model_error"
        )
        records.trace.append({"type": "run_error", "failure_category": failure_category,
                              "error": f"{type(exc).__name__}: {exc}"})
        return _result(model=model, trace=records.trace, usage=records.usage(),
                       status=status, error=f"{type(exc).__name__}: {exc}")
