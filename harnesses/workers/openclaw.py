"""Container-side OpenClaw worker: Gateway, native agent turns, Skill Workshop.

Verified against the pinned upstream `v2026.9.2`
(`3928bad9badfcb6c7d140530435e806fb8092190`); see
`docker/harnesses/upstreams.lock.json` for the file/line evidence behind every
command and config key used here.

Flow per group:

- `start` renders `openclaw.json`, links the medical tool plugin and runs a
  persistent Gateway (`openclaw gateway run`), so every later turn goes through
  Gateway-owned background work instead of a one-shot embedded run;
- `execute` submits one task with `openclaw agent --agent medical
  --session-id ... --message-file ... --json` and keeps the Gateway session;
- `learn` feeds the public training review into that same training session and
  then waits for a definite learning outcome; the pinned version only schedules
  its detached experience review when the foreground turn used at least ten model
  iterations, so a turn below that bound is reported as `not_triggered` rather
  than as an update;
- `export` returns only Workshop skills and the workspace memory files the pinned
  version permits persisting, never session stores or transcripts.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path

from contracts import RunResult
from harnesses.contracts import HarnessLimits, ModelConfig
from harnesses.snapshots import collect_files, directory_digest
from harnesses.worker_protocol import WorkerError, WorkerRequest, WorkerResponse
from harnesses.workers.base import HarnessWorker, WorkerPaths

__all__ = ["MIN_LEARNING_ITERATIONS", "OpenClawWorker", "render_openclaw_config"]

AGENT_ID = "medical"
PROVIDER_ID = "medevolve"
PLUGIN_DIR = Path("/opt/medevolve/openclaw-tools")
EXPORT_INCLUDE = ("skills", "memory", "MEMORY.md")
FROZEN_DENIED_TOOLS = ("exec", "process", "write", "edit", "apply_patch")
MIN_LEARNING_ITERATIONS = 10
DEFAULT_GATEWAY_PORT = 18789
DEFAULT_QUIET_SECONDS = 30.0
DEFAULT_POLL_SECONDS = 5.0


def render_openclaw_config(
    *,
    workspace: Path,
    model: ModelConfig,
    limits: HarnessLimits,
    gateway_base_url: str,
    client_id: str,
    task_id: str,
    phase: str,
    frozen: bool,
) -> dict:
    """The experiment's own `openclaw.json`; never the operator's config.

    Keys are the ones the pinned schema accepts: `gateway.mode`,
    `agents.entries.<id>.{workspace,model,tools.deny}`,
    `models.providers.<id>.{baseUrl,api,apiKey,request.headers,models}` and
    `skills.workshop.autonomous.mode`. The provider headers are what let the
    model gateway attribute every OpenClaw model call - including the detached
    background review - to this group, task and phase.
    """
    entry: dict = {
        "name": "Medical",
        "workspace": str(workspace),
        "model": {"primary": f"{PROVIDER_ID}/{model.model}"},
    }
    if frozen:
        # Frozen runs keep reading skills but lose every tool that mutates state.
        entry["tools"] = {"deny": list(FROZEN_DENIED_TOOLS)}
    return {
        "gateway": {"mode": "local"},
        "agents": {"entries": {AGENT_ID: entry}},
        "models": {
            "mode": "merge",
            "providers": {
                PROVIDER_ID: {
                    "baseUrl": gateway_base_url,
                    "api": "openai-completions",
                    "apiKey": "${" + model.api_key_env + "}",
                    "request": {
                        "headers": {
                            "x-medevolve-client": client_id,
                            "x-medevolve-task": task_id or AGENT_ID,
                            "x-medevolve-phase": phase,
                        }
                    },
                    "models": [{
                        "id": model.model,
                        "name": model.model,
                        "contextWindow": model.context_window or limits.max_input_tokens,
                        "maxTokens": model.max_output_tokens or limits.max_output_tokens,
                    }],
                }
            },
        },
        "skills": {
            "workshop": {"autonomous": {"mode": "off" if frozen else "auto"}}
        },
    }


def _stat(envelope: dict, name: str):
    """Run stats live at the top level for `agent exec` and under `meta.agentMeta`."""
    if name in envelope:
        return envelope[name]
    meta = envelope.get("meta")
    if isinstance(meta, dict):
        agent_meta = meta.get("agentMeta")
        if isinstance(agent_meta, dict) and name in agent_meta:
            return agent_meta[name]
    return None


def _parse_envelope(text: str) -> dict:
    stripped = (text or "").strip()
    if not stripped:
        raise WorkerError("openclaw 没有输出 JSON 响应")
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    for line in reversed(stripped.splitlines()):
        if line.strip().startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    raise WorkerError("openclaw 的 stdout 不是合法 JSON 响应")


class OpenClawWorker(HarnessWorker):
    """Native OpenClaw execution, learning and experience export."""

    name = "openclaw"

    def __init__(self, paths: WorkerPaths, cli, *, sleeper=asyncio.sleep):
        super().__init__(paths, cli)
        self._sleep = sleeper
        self.config_path = paths.runtime_state / "openclaw.json"
        self.pid_path = paths.runtime_state / "gateway.pid"
        self.phase = "execution"

    @property
    def started(self) -> bool:
        """The worker is one process per request, so readiness lives on disk."""
        return self.config_path.is_file()

    # -- actions -----------------------------------------------------------
    async def health(self, request: WorkerRequest) -> WorkerResponse:
        if not self.cli.available():
            return WorkerResponse(
                action="health", status="failed",
                error="容器内找不到 openclaw 可执行文件，无法执行原生 Agent",
            )
        return WorkerResponse(action="health", payload={
            "workspace": str(self.paths.workspace),
            "durable_root": str(self.paths.durable_root),
            "runtime_state": str(self.paths.runtime_state),
            "frozen": self.paths.frozen,
            "gateway_running": self.started,
        })

    async def start(self, request: WorkerRequest) -> WorkerResponse:
        model = ModelConfig.model_validate(request.payload.get("model") or {})
        limits = HarnessLimits.model_validate(request.payload.get("limits") or {})
        client_id = request.payload.get("client_id") or request.group.run_id
        gateway_base_url = request.payload.get("gateway_base_url") or ""
        if not gateway_base_url:
            raise WorkerError("缺少模型网关地址，拒绝启动原生 Agent")
        session_id = request.payload.get("session_id") or f"medevolve-{client_id}"
        if not self.paths.frozen:
            self.paths.workspace.mkdir(parents=True, exist_ok=True)
            (self.paths.workspace / "skills").mkdir(parents=True, exist_ok=True)
        self.write_text(self.config_path, json.dumps(render_openclaw_config(
            workspace=self.paths.workspace,
            model=model,
            limits=limits,
            gateway_base_url=gateway_base_url,
            client_id=client_id,
            task_id=request.task_id,
            phase="auxiliary",
            frozen=self.paths.frozen,
        ), indent=2, ensure_ascii=False))
        await self._install_plugin()
        spawned = False
        if not await self._gateway_ready():
            await self._start_gateway(
                port=int(request.payload.get("gateway_port") or DEFAULT_GATEWAY_PORT),
                timeout=float(request.payload.get("start_timeout_seconds") or 120),
            )
            spawned = True
        return WorkerResponse(action="start", payload={
            "session_id": session_id,
            "config_path": str(self.config_path),
            "workspace": str(self.paths.workspace),
            "runtime_state": str(self.paths.runtime_state),
            "learning_mode": "off" if self.paths.frozen else "auto",
            "gateway_spawned": spawned,
        })

    async def execute(self, request: WorkerRequest) -> WorkerResponse:
        self._require_started()
        message = request.payload.get("message")
        if not isinstance(message, str) or not message.strip():
            raise WorkerError("执行请求缺少题目文本")
        session_id = self._session(request)
        message_file = self.write_text(
            self.paths.writable / "tasks" / f"{request.task_id or 'task'}.md", message
        )
        self._write_tool_session(request.payload.get("tool_session"))
        self._set_phase("execution", request.task_id)
        envelope = await self._agent_turn(
            session_id, message_file,
            timeout=float(request.payload.get("timeout_seconds") or 600),
        )
        run = self._to_run(envelope, session_id)
        self.record("execution", int(_stat(envelope, "assistantTurns") or 1))
        return WorkerResponse(action="execute", payload={
            "run": run.model_dump(mode="json"),
            "session_id": _stat(envelope, "sessionId") or session_id,
            "message_file": str(message_file),
            "tool_session": request.payload.get("tool_session"),
            "skills_read": self._skill_names(),
        })

    async def learn(self, request: WorkerRequest) -> WorkerResponse:
        self._require_started()
        message = request.payload.get("message")
        if not isinstance(message, str) or not message.strip():
            raise WorkerError("学习请求缺少训练评价文本")
        session_id = self._session(request)
        before = self._skills_digest()
        feedback_file = self.write_text(
            self.paths.writable / "feedback" / f"{request.task_id or 'task'}.md", message
        )
        self._set_phase("learning", request.task_id)
        envelope = await self._agent_turn(
            session_id, feedback_file,
            timeout=float(request.payload.get("learning_timeout_seconds") or 900),
        )
        run = self._to_run(envelope, session_id)
        iterations = int(_stat(envelope, "assistantTurns") or 0)
        self.record("learning", max(iterations, 1))
        if run.status != "completed":
            return WorkerResponse(action="learn", status="failed", payload={
                "learning_status": "failed",
                "run": run.model_dump(mode="json"),
                "learning_runs": [run.model_dump(mode="json")],
                "reason": f"训练反馈回合未完成：{run.error or run.status}",
            })
        if iterations < MIN_LEARNING_ITERATIONS:
            # Pinned trigger: the detached experience review is scheduled only
            # when the foreground turn used at least ten model iterations.
            return WorkerResponse(action="learn", status="ok", payload={
                "learning_status": "not_triggered",
                "run": run.model_dump(mode="json"),
                "learning_runs": [run.model_dump(mode="json")],
                "reason": (
                    f"前台回合只用了 {iterations} 个模型轮次，未达到锁定版本 "
                    f"后台经验复盘所需的 {MIN_LEARNING_ITERATIONS} 轮，学习未触发"
                ),
            })
        status, reason = await self._await_learning(
            before,
            quiet_seconds=float(request.payload.get("learning_quiet_seconds")
                                if request.payload.get("learning_quiet_seconds") is not None
                                else DEFAULT_QUIET_SECONDS),
            timeout=float(request.payload.get("learning_timeout_seconds") or 900),
            poll_seconds=float(request.payload.get("learning_poll_seconds") or DEFAULT_POLL_SECONDS),
        )
        return WorkerResponse(action="learn", status="ok", payload={
            "learning_status": status,
            "run": run.model_dump(mode="json"),
            "learning_runs": [run.model_dump(mode="json")],
            "reason": reason,
            "skills_digest_before": before,
            "skills_digest_after": self._skills_digest(),
        })

    async def export(self, request: WorkerRequest) -> WorkerResponse:
        files = collect_files(self.paths.workspace, include=EXPORT_INCLUDE)
        return WorkerResponse(action="export", payload={
            "root": str(self.paths.workspace),
            "include": list(EXPORT_INCLUDE),
            "files": files,
        })

    async def stop(self, request: WorkerRequest) -> WorkerResponse:
        errors: list[str] = []
        pid = self._gateway_pid()
        if pid is not None:
            self.cli.terminate(pid)
            self.pid_path.unlink(missing_ok=True)
        try:
            result = await self.cli.run(["gateway", "stop"], env=self._env(), timeout=60)
            if not result.ok and pid is None:
                errors.append(result.stderr.strip() or f"退出码 {result.returncode}")
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
        await self.cli.stop()
        return WorkerResponse(
            action="stop",
            status="ok" if not errors else "failed",
            payload={"gateway_stopped": not errors, "gateway_pid": pid},
            error="; ".join(errors) or None,
        )

    # -- internals ---------------------------------------------------------
    def _require_started(self) -> None:
        if not self.started:
            raise WorkerError("Gateway 尚未启动，拒绝提交任务")

    @staticmethod
    def _session(request: WorkerRequest) -> str:
        session_id = request.payload.get("session_id")
        if not isinstance(session_id, str) or not session_id.strip():
            raise WorkerError("请求缺少会话标识，拒绝提交到未知会话")
        return session_id

    def _env(self) -> dict[str, str]:
        return {
            "OPENCLAW_STATE_DIR": str(self.paths.runtime_state),
            "OPENCLAW_CONFIG_PATH": str(self.config_path),
            "MEDEVOLVE_FROZEN": "1" if self.paths.frozen else "0",
        }

    def _set_phase(self, phase: str, task_id: str) -> None:
        """Keep the gateway phase header aligned with the work in flight.

        The header is static config, so the worker rewrites it before each phase.
        It stays on `learning` until the next `execute`, which is what makes the
        detached background review's model calls count as learning rather than
        execution.
        """
        self.phase = phase
        config = json.loads(self.config_path.read_text(encoding="utf-8"))
        headers = config["models"]["providers"][PROVIDER_ID]["request"]["headers"]
        headers["x-medevolve-phase"] = phase
        headers["x-medevolve-task"] = task_id or AGENT_ID
        self.write_text(
            self.config_path, json.dumps(config, indent=2, ensure_ascii=False)
        )

    def _tool_session_path(self) -> Path:
        return self.paths.runtime_state / "tool-session.json"

    def _write_tool_session(self, tool_session: str | None) -> None:
        """Hand the attempt's medical tool session to the plugin.

        The Gateway outlives one request, so the plugin cannot be told the
        session through its environment; it reads this file on every tool call.
        The bridge URL is the container's own, and the client id is the same one
        the model gateway and the bridge already know.
        """
        config = json.loads(self.config_path.read_text(encoding="utf-8"))
        client_id = config["models"]["providers"][PROVIDER_ID]["request"]["headers"][
            "x-medevolve-client"
        ]
        self.write_text(self._tool_session_path(), json.dumps({
            "bridge": self.paths.tool_bridge,
            "session": tool_session,
            "client": client_id,
        }, indent=2, ensure_ascii=False))

    def _gateway_pid(self) -> int | None:
        try:
            return int(self.pid_path.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            return None

    async def _install_plugin(self) -> None:
        result = await self.cli.run(
            ["plugins", "install", "--link", str(PLUGIN_DIR), "--force"],
            env=self._env(), timeout=300,
        )
        if not result.ok:
            raise WorkerError(
                f"医疗工具插件安装失败：{result.stderr.strip() or result.returncode}"
            )

    async def _gateway_ready(self) -> bool:
        """Ready only when this attempt's own state holds a live Gateway.

        A frozen attempt gets a fresh scratch volume, so it never inherits a
        recorded pid from an earlier question; that is why the pid file, and not
        the health probe alone, decides whether a Gateway must be started.
        """
        if not self.pid_path.is_file():
            return False
        result = await self.cli.run(["health", "--json"], env=self._env(), timeout=30)
        if not result.ok:
            return False
        try:
            return bool(_parse_envelope(result.stdout).get("ok"))
        except WorkerError:
            return False

    async def _start_gateway(self, *, port: int, timeout: float) -> None:
        pid = await self.cli.spawn(
            ["gateway", "run", "--allow-unconfigured", "--bind", "loopback",
             "--port", str(port)],
            env=self._env(),
        )
        if pid:
            self.write_text(self.pid_path, str(pid))
        deadline = time.monotonic() + timeout
        last = ""
        while time.monotonic() < deadline:
            result = await self.cli.run(["health", "--json"], env=self._env(), timeout=30)
            if result.ok:
                try:
                    snapshot = _parse_envelope(result.stdout)
                except WorkerError:
                    snapshot = {}
                if snapshot.get("ok"):
                    return
                last = json.dumps(snapshot, ensure_ascii=False)[:400]
            else:
                last = result.stderr.strip() or f"退出码 {result.returncode}"
            await self._sleep(1.0)
        raise WorkerError(f"Gateway 在 {timeout} 秒内未就绪：{last}")

    async def _agent_turn(self, session_id: str, message_file: Path, *, timeout: float) -> dict:
        result = await self.cli.run(
            ["agent", "--agent", AGENT_ID, "--session-id", session_id,
             "--message-file", str(message_file), "--json"],
            env=self._env(), timeout=timeout,
        )
        envelope = _parse_envelope(result.stdout)
        if envelope.get("status") == "in_flight":
            raise WorkerError(f"会话 {session_id} 已有正在运行的回合，拒绝重复提交")
        if not result.ok and envelope.get("ok") is not False:
            raise WorkerError(
                f"openclaw agent 失败：{result.stderr.strip() or result.returncode}"
            )
        return envelope

    @staticmethod
    def _to_run(envelope: dict, session_id: str) -> RunResult:
        raw_status = envelope.get("status") or ("ok" if envelope.get("ok") else "error")
        error = envelope.get("error")
        message = error.get("message") if isinstance(error, dict) else (
            error if isinstance(error, str) else None
        )
        if raw_status == "ok":
            status = "completed"
        elif message and ("429" in message or "budget" in message.lower()):
            status = "budget_exhausted"
        else:
            status = "failed"
        payloads = envelope.get("payloads")
        trace = [{
            "type": "agent_turn",
            "session_id": _stat(envelope, "sessionId") or session_id,
            "status": raw_status,
            "assistant_turns": _stat(envelope, "assistantTurns"),
            "model": envelope.get("model"),
            "provider": envelope.get("provider"),
            "tool_summary": (envelope.get("meta") or {}).get("toolSummary")
            if isinstance(envelope.get("meta"), dict) else None,
            "run_id": envelope.get("runId"),
            "origin": envelope.get("origin"),
        }]
        usage = envelope.get("usage")
        if isinstance(usage, dict):
            trace[0]["usage"] = usage
        return RunResult(
            output={
                "final": envelope.get("final"),
                "payloads": payloads if isinstance(payloads, list) else [],
                "session_id": _stat(envelope, "sessionId") or session_id,
            },
            trace=trace,
            usage=usage if isinstance(usage, dict) else {},
            status=status,
            error=message,
        )

    def _skills_dir(self) -> Path:
        return self.paths.workspace / "skills"

    def _skills_digest(self) -> str:
        return directory_digest(self._skills_dir())

    def _skill_names(self) -> list[str]:
        directory = self._skills_dir()
        if not directory.is_dir():
            return []
        return sorted(
            path.parent.name for path in directory.glob("*/SKILL.md")
        )

    async def _workshop_records(self) -> list[dict]:
        result = await self.cli.run(
            ["skills", "workshop", "list", "--json"], env=self._env(), timeout=60
        )
        if not result.ok:
            return []
        try:
            payload = _parse_envelope(result.stdout)
        except WorkerError:
            return []
        for key in ("proposals", "items", "records"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
        return []

    async def _await_learning(
        self, before: str, *, quiet_seconds: float, timeout: float, poll_seconds: float
    ) -> tuple[str, str]:
        """Wait for a definite outcome instead of watching for a file to appear.

        The pinned version starts its detached review only after the system has
        been quiet for 30 seconds with no agent run active, so the wait begins
        with that quiet period. A Workshop record or a changed skill tree is the
        evidence; a timeout with neither is reported as `not_triggered`.
        """
        if quiet_seconds > 0:
            await self._sleep(quiet_seconds)
        deadline = time.monotonic() + timeout
        records: list[dict] = []
        while True:
            if self._skills_digest() != before:
                return "updated", "后台复盘已生成或修订 Workshop 技能"
            records = await self._workshop_records()
            if records:
                return "unchanged", (
                    f"后台复盘留下 {len(records)} 条 Workshop 记录，但技能内容未变"
                )
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            await self._sleep(min(poll_seconds, remaining))
        return "not_triggered", (
            f"等待 {timeout} 秒后既没有技能变更也没有 Workshop 终态记录，"
            "本次学习未触发"
        )
