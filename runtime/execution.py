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
from openai import AsyncOpenAI
from pydantic import BaseModel
from pydantic_core import to_jsonable_python

from contracts import Budget, ModelConfig, RunResult


_OPENCODE_SESSION = str(uuid4())


class _ToolBudgetExceeded(AgentsException):
    pass


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
    records = _Records(budget)
    try:
        if any(not isinstance(tool, FunctionTool) for tool in tools):
            raise ValueError("Chat Completions runtime supports only SDK FunctionTool tools")
        if budget.max_model_turns == 0:
            raise MaxTurnsExceeded("Model turn budget exhausted")
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
                output_type=AgentOutputSchema(output_type, strict_json_schema=False),
                model=OpenAIChatCompletionsModel(model=model.model, openai_client=client),
                model_settings=ModelSettings(
                    parallel_tool_calls=model.parallel_tool_calls,
                    extra_body={"response_format": {"type": "json_object"}}
                    if urlparse(model.base_url or "").hostname == "opencode.ai" else None,
                ),
            )
            result = await Runner.run(
                agent, agent_input, hooks=records, max_turns=budget.max_model_turns,
                run_config=RunConfig(tracing_disabled=True),
            )
            output = output_type.model_validate(result.final_output).model_dump(mode="json")
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
