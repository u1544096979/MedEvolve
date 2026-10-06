"""HTTP medical tool bridge: one attempt-scoped session per task.

The bridge is the only path a harness container has to the medical environment.
It exposes the existing environment tools unchanged: PhysicianBench tool names,
parameter schemas and native functions come from `environments.get_tools()`, so
the existing patient-scope checks still run. MedAgentBench keeps its native text
protocol behind one `medical_action(action)` tool: GET and POST keep their native
reply text, POST stays accepted without being persisted, and FINISH keeps the
answer list.

Every call is recorded as `tool_start`/`tool_output` with order, duration and
write state, and the tool budget is deducted here. A session is bound to the
calling container and to one medical instance; closing it saves the task's
`read_state()` result before releasing the instance.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from importlib import import_module
from typing import Any, Awaitable, Callable
from uuid import uuid4

from agents import RunConfig
from agents.tool_context import ToolContext
from contracts import EnvironmentConfig, SolverInput, Task
from environments.medagentbench import NativeTools
from fastapi import Body, FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse

SESSION_HEADER = "x-medevolve-client"
MEDAGENTBENCH_TOOL = "medical_action"
MEDAGENTBENCH_NATIVE = "environments.medagentbench.NativeTools.execute"
WRITE_TOOL_NAMES = frozenset({
    "fhir_communication_create_message",
    "fhir_service_request_create",
    "fhir_medication_request_create",
    "fhir_appointment_create",
})
MEDAGENTBENCH_DESCRIPTION = (
    "执行一次 MedAgentBench 原生文本动作。只能提交以下三种之一，且每次只提交一个：\n"
    "1. 查询：GET <ResourceType>?<参数名1>=<值1>&<参数名2>=<值2>...\n"
    "2. 提交：POST <ResourceType>\\n<JSON payload>（环境接受请求但不写入数据库）\n"
    "3. 结束：FINISH([answer1, answer2, ...])，answers 必须是 JSON 列表。\n"
    "路径只写相对路径，即以资源类型开头、紧跟在服务地址之后的部分，"
    "例如 GET Condition?patient=p1；不要写服务地址、域名或完整 URL，"
    "工具会把相对路径接到本会话的 FHIR 服务上。\n"
    "保持 ResourceType、参数名与 payload 字段与上游原生实现一致。"
)


class BridgeError(RuntimeError):
    """Tool session misuse."""


class UnknownSession(BridgeError):
    """The session id is unknown to this bridge process."""


class SessionIdentityMismatch(BridgeError):
    """The calling container does not own this session."""


class ToolArgumentError(BridgeError):
    """Tool arguments do not match the declared schema."""


def _is_write_tool(name: str) -> bool:
    return name in WRITE_TOOL_NAMES or name.endswith("_create")


def _parse_tool_output(raw: Any) -> tuple[Any, str | None]:
    """Return (output, error) for an openai-agents FunctionTool result."""
    if not isinstance(raw, str):
        return raw, None
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        error = raw if raw.startswith("An error occurred") else None
        return {"raw": raw}, error
    if isinstance(value, dict) and set(value) == {"error"}:
        return value, str(value["error"])
    return value, None


@dataclass
class ToolSession:
    session_id: str
    client_id: str | None
    task: SolverInput
    environment: EnvironmentConfig
    instance: Any
    tools: list[Any]
    native: NativeTools | None
    tool_budget: int | None
    timeout_seconds: float | None
    events: list[dict] = field(default_factory=list)
    calls: list[dict] = field(default_factory=list)
    closed: bool = False
    summary: dict | None = None


class ToolSessions:
    """Registry of attempt-scoped medical tool sessions."""

    def __init__(
        self,
        *,
        environments=None,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.environments = environments if environments is not None else import_module("environments")
        self._clock = clock
        self._sessions: dict[str, ToolSession] = {}

    # -- lifecycle ---------------------------------------------------------
    async def open(
        self,
        task: SolverInput,
        environment: EnvironmentConfig,
        *,
        client_id: str | None = None,
        tool_budget: int | None = None,
        timeout_seconds: float | None = None,
    ) -> str:
        if tool_budget is not None and tool_budget < 0:
            raise BridgeError("tool_budget 不能为负数")
        instance = await self.environments.create_environment(environment)
        try:
            tools = await self.environments.get_tools(
                instance, [task.patient_id], decision_point=task.decision_point,
                tool_result_limit=environment.tool_result_limit,
            )
        except BaseException:
            await self._release(environment, instance)
            raise
        session = ToolSession(
            session_id=f"tool-session-{uuid4().hex[:12]}",
            client_id=client_id,
            task=task,
            environment=environment,
            instance=instance,
            tools=list(tools),
            native=tools if isinstance(tools, NativeTools) else None,
            tool_budget=tool_budget,
            timeout_seconds=timeout_seconds,
        )
        self._sessions[session.session_id] = session
        return session.session_id

    async def list(self, session_id: str, *, client_id: str | None = None) -> list[dict]:
        session = self._require(session_id, client_id)
        if session.native is not None:
            return [{
                "name": MEDAGENTBENCH_TOOL,
                "description": MEDAGENTBENCH_DESCRIPTION,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "description": "一条原生 GET、POST 或 FINISH 动作文本",
                        }
                    },
                    "required": ["action"],
                    "additionalProperties": False,
                },
                "native": MEDAGENTBENCH_NATIVE,
                "interaction_adapter": True,
            }]
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.params_json_schema,
            }
            for tool in session.tools
        ]

    async def invoke(
        self,
        session_id: str,
        name: str,
        arguments: dict,
        *,
        client_id: str | None = None,
    ) -> dict:
        session = self._require(session_id, client_id)
        if session.closed:
            raise BridgeError(f"工具会话已关闭：{session_id}")
        order = len(session.calls) + 1
        record = {
            "name": name,
            "arguments": dict(arguments or {}),
            "output": None,
            "status": "ok",
            "error": None,
            "write": False,
            "persisted": None,
            "kind": None,
            "answers": None,
            "order": order,
            "duration_ms": 0.0,
        }
        if session.tool_budget is not None and len(session.calls) >= session.tool_budget:
            record["status"] = "budget_exhausted"
            record["error"] = f"工具调用预算已用尽：{session.tool_budget}"
            session.events.append({"type": "tool_budget", "name": name, "order": order})
            session.calls.append(record)
            return record
        started = self._clock()
        session.events.append({
            "type": "tool_start", "name": name, "order": order,
            "arguments": record["arguments"],
        })
        try:
            if session.native is not None:
                record["output"] = await self._native_call(session, name, record)
            else:
                record["output"] = await self._tool_call(session, name, record)
        except ToolArgumentError as exc:
            record["status"] = "invalid_arguments"
            record["error"] = str(exc)
        except asyncio.CancelledError:
            record["status"] = "cancelled"
            record["error"] = "工具调用被取消"
            record["duration_ms"] = self._duration(started)
            session.events.append({
                "type": "tool_cancelled", "name": name, "order": order,
            })
            session.calls.append(record)
            await self._release(session.environment, session.instance)
            session.closed = True
            raise
        except asyncio.TimeoutError:
            record["status"] = "timeout"
            record["error"] = f"工具调用超时：{session.timeout_seconds} 秒"
        except Exception as exc:
            record["status"] = "error"
            record["error"] = f"{type(exc).__name__}: {exc}"
        if record["status"] != "ok" and record["persisted"] is True:
            record["persisted"] = False
        record["duration_ms"] = self._duration(started)
        session.calls.append(record)
        session.events.append({
            "type": "tool_output", "name": name, "order": order,
            "status": record["status"], "output": record["output"],
            "write": record["write"], "persisted": record["persisted"],
            "duration_ms": record["duration_ms"],
        })
        return record

    async def close(self, session_id: str, *, client_id: str | None = None) -> dict:
        session = self._require(session_id, client_id)
        if session.closed and session.summary is not None:
            return {**session.summary, "already_closed": True}
        state, state_error = None, None
        try:
            state = await self.environments.read_state(
                session.instance, self._state_task(session)
            )
        except Exception as exc:
            state_error = f"{type(exc).__name__}: {exc}"
        release_error = await self._release(session.environment, session.instance)
        session.closed = True
        session.summary = {
            "session_id": session.session_id,
            "patient_id": session.task.patient_id,
            "environment": session.environment.name,
            "tool_calls": len(session.calls),
            "calls": session.calls,
            "events": session.events,
            "state": state,
            "state_error": state_error,
            "release_error": release_error,
            "closed": True,
            "already_closed": False,
        }
        return dict(session.summary)

    async def aclose(self) -> None:
        for session_id in list(self._sessions):
            session = self._sessions[session_id]
            if not session.closed:
                await self.close(session_id)

    # -- internals ---------------------------------------------------------
    def _require(self, session_id: str, client_id: str | None) -> ToolSession:
        session = self._sessions.get(session_id)
        if session is None:
            raise UnknownSession(f"未知工具会话：{session_id}")
        if client_id is not None and client_id != session.client_id:
            raise SessionIdentityMismatch(
                f"工具会话 {session_id} 不属于调用容器 {client_id}"
            )
        return session

    def _duration(self, started: float) -> float:
        return round((self._clock() - started) * 1000, 3)

    async def _with_timeout(self, session: ToolSession, awaitable: Awaitable[Any]) -> Any:
        if session.timeout_seconds is None:
            return await awaitable
        return await asyncio.wait_for(awaitable, timeout=session.timeout_seconds)

    async def _tool_call(self, session: ToolSession, name: str, record: dict) -> Any:
        tool = next((item for item in session.tools if item.name == name), None)
        if tool is None:
            raise ToolArgumentError(f"未知工具：{name}")
        self._check_arguments(tool.params_json_schema, record["arguments"])
        payload = json.dumps(record["arguments"], ensure_ascii=False)
        context = ToolContext(
            context=None,
            tool_name=name,
            tool_call_id=f"{session.session_id}-{record['order']}",
            tool_arguments=payload,
            run_config=RunConfig(tracing_disabled=True),
        )
        raw = await self._with_timeout(session, tool.on_invoke_tool(context, payload))
        output, error = _parse_tool_output(raw)
        record["write"] = _is_write_tool(name)
        record["persisted"] = True if record["write"] else None
        if error is not None:
            record["status"] = "error"
            record["error"] = error
        return output

    async def _native_call(self, session: ToolSession, name: str, record: dict) -> Any:
        if name != MEDAGENTBENCH_TOOL:
            raise ToolArgumentError(f"未知工具：{name}")
        action = record["arguments"].get("action")
        if not isinstance(action, str) or not action.strip():
            raise ToolArgumentError("medical_action 需要非空的 action 文本")
        result = await self._with_timeout(session, session.native.execute(action))
        kind = result.get("kind")
        record["kind"] = kind
        record["write"] = kind == "POST"
        # POST is accepted by the native protocol but never persisted.
        record["persisted"] = False if kind == "POST" else None
        record["answers"] = result.get("answers")
        if kind == "error":
            record["status"] = "error"
            record["error"] = str(result.get("reply"))
        return result

    @staticmethod
    def _check_arguments(schema: dict, arguments: dict) -> None:
        if not isinstance(arguments, dict):
            raise ToolArgumentError("工具参数必须是 JSON 对象")
        properties = schema.get("properties", {})
        missing = [key for key in schema.get("required", []) if key not in arguments]
        if missing:
            raise ToolArgumentError(f"缺少必需参数：{', '.join(sorted(missing))}")
        if schema.get("additionalProperties") is False:
            unknown = [key for key in arguments if key not in properties]
            if unknown:
                raise ToolArgumentError(f"未知参数：{', '.join(sorted(unknown))}")

    @staticmethod
    def _state_task(session: ToolSession) -> Task:
        """A reference-free task: only the patient question reaches read_state."""
        return Task(
            id=session.session_id,
            patient_id=session.task.patient_id,
            environment=session.environment.name,
            round_index=0,
            source_method="tool_bridge",
            instruction=session.task.instruction,
            task_type="",
            reference_answer="",
            evidence=[],
            decision_point=session.task.decision_point,
        )

    async def _release(self, environment: EnvironmentConfig, instance: Any) -> str | None:
        try:
            await self.environments.release_environment(environment, instance)
        except Exception as exc:  # cleanup must not mask the original failure
            return f"{type(exc).__name__}: {exc}"
        return None


_SESSIONS = ToolSessions()


async def open_tool_session(
    task: SolverInput, environment: EnvironmentConfig, **kwargs
) -> str:
    return await _SESSIONS.open(task, environment, **kwargs)


async def list_tools(session_id: str, *, client_id: str | None = None) -> list[dict]:
    return await _SESSIONS.list(session_id, client_id=client_id)


async def invoke_tool(
    session_id: str, name: str, arguments: dict, *, client_id: str | None = None
) -> dict:
    return await _SESSIONS.invoke(session_id, name, arguments, client_id=client_id)


async def close_tool_session(session_id: str, *, client_id: str | None = None) -> dict:
    return await _SESSIONS.close(session_id, client_id=client_id)


def sessions() -> ToolSessions:
    return _SESSIONS


def create_bridge_app(registry: ToolSessions | None = None) -> FastAPI:
    """HTTP routes: GET /sessions/{id}/tools and POST /sessions/{id}/calls."""
    tool_sessions = registry if registry is not None else _SESSIONS
    app = FastAPI(title="MedEvolve Harness Medical Tool Bridge")

    def caller_id(client: str | None) -> str:
        if not client:
            raise HTTPException(403, f"缺少 {SESSION_HEADER} 调用容器标识")
        return client

    @app.get("/sessions/{session_id}/tools")
    async def list_session_tools(
        session_id: str,
        x_medevolve_client: str | None = Header(None, alias=SESSION_HEADER),
    ) -> list[dict]:
        try:
            return await tool_sessions.list(
                session_id, client_id=caller_id(x_medevolve_client)
            )
        except UnknownSession as exc:
            raise HTTPException(404, str(exc)) from exc
        except SessionIdentityMismatch as exc:
            raise HTTPException(403, str(exc)) from exc

    @app.post("/sessions/{session_id}/calls")
    async def call_tool(
        session_id: str,
        payload: dict = Body(...),
        x_medevolve_client: str | None = Header(None, alias=SESSION_HEADER),
    ):
        name, arguments = payload.get("name"), payload.get("arguments", {})
        if not isinstance(name, str) or not isinstance(arguments, dict):
            raise HTTPException(422, "calls 需要 name 与 arguments 字段")
        try:
            record = await tool_sessions.invoke(
                session_id, name, arguments, client_id=caller_id(x_medevolve_client)
            )
        except UnknownSession as exc:
            raise HTTPException(404, str(exc)) from exc
        except SessionIdentityMismatch as exc:
            raise HTTPException(403, str(exc)) from exc
        if record["status"] == "budget_exhausted":
            return JSONResponse(record, status_code=429)
        if record["status"] == "invalid_arguments":
            return JSONResponse(record, status_code=422)
        return record

    return app
