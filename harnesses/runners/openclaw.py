"""OpenClaw 本机 Gateway runner。"""
from __future__ import annotations

import asyncio
import json
import os
import secrets
import hashlib
import shutil
import sqlite3
import time
from pathlib import Path

from contracts import RunResult, SolverInput
from harnesses.contracts import HarnessStep, Snapshot, TrainingFeedback
from harnesses.local import export_experience, run_command
from harnesses.snapshots import directory_digest

PLUGIN_SOURCE = Path(__file__).resolve().parents[2] / "docker/harnesses/openclaw-tools"


class OpenClawRunner:
    def __init__(self, *, config, group_dir, bridge_url):
        self.config = config
        self.group_dir = Path(group_dir)
        self.bridge_url = bridge_url
        self.state_dir = self.group_dir / "openclaw-state"
        self.workspace = self.group_dir / "workspace"
        self._command_runner = config.get("command_runner", run_command)
        self._process_factory = config.get("process_factory", asyncio.create_subprocess_exec)
        self._process = None
        self._session_id = None
        self._pending = {}
        self._frozen = False
        self._gateway_port = int(config.get("gateway_port", 18961))
        self._client_id = config.get("client_id") or secrets.token_hex(12)
        self._review_reader = config.get("review_reader", self._read_review_status)

    @property
    def current_session_id(self):
        return self._session_id

    def _env(self, model):
        key = model.get("api_key_env", "")
        env = {"OPENCLAW_CONFIG_PATH": str(self.state_dir / "openclaw.json"),
               "OPENCLAW_STATE_DIR": str(self.state_dir)}
        if key:
            env[key] = os.environ.get(key, "")
        return env

    async def start(self, group, initial: Snapshot):
        self.group = group
        self._client_id = self.config.get("client_id") or group.run_id
        self.group_dir.mkdir(parents=True, exist_ok=True)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.workspace.mkdir(parents=True, exist_ok=True)
        self._plugin_dir = self.state_dir / "plugins/medevolve-medical-tools"
        shutil.copytree(PLUGIN_SOURCE, self._plugin_dir, dirs_exist_ok=True)
        model = self.config.get("execution_model", {})
        self._frozen = bool(self.config.get("frozen", False))
        if not model.get("model") or not model.get("base_url"):
            return
        cfg = {
            "gateway": {"mode": "local", "port": self._gateway_port},
            "agents": {"defaults": {"workspace": str(self.workspace)}, "entries": {"medical": {
                "name": "Medical", "workspace": str(self.workspace),
                "model": {"primary": f"medevolve/{model['model']}"},
                **({"tools": {"deny": ["write", "edit", "apply_patch", "exec", "process", "skill_workshop"]}} if self._frozen else {}),
            }}},
            "models": {"mode": "merge", "providers": {"medevolve": {
                "baseUrl": model["base_url"], "api": "openai-completions",
                "apiKey": "${" + model["api_key_env"] + "}",
                "models": [{"id": model["model"], "name": model["model"]}],
            }}},
            "plugins": {"load": {"paths": [str(self._plugin_dir)]},
                        "entries": {"medevolve-medical-tools": {"enabled": True}}},
            "skills": {"workshop": {"autonomous": {"mode": "off" if self._frozen else "auto"}}},
        }
        initial_dir = Path(initial.path)
        if initial_dir.is_dir():
            export_experience(initial_dir, self.workspace, include=("skills", "memory", "MEMORY.md"))
        (self.state_dir / "openclaw.json").write_text(json.dumps(cfg), encoding="utf-8")
        for attempt in range(2):
            self._process = await self._process_factory(
                self.config.get("openclaw_bin", "openclaw"), "gateway", "run", "--port", str(self._gateway_port),
                cwd=str(self.group_dir), env={**os.environ, **self._env(model)},
                stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE,
            )
            try:
                await self._wait_gateway()
                break
            except RuntimeError:
                if self._process.returncode is None:
                    raise
                _, stderr = await self._process.communicate()
                detail = stderr.decode("utf-8", errors="replace")
                # 锁定版本首次迁移插件清单后要求重启，沿用迁移后的配置重启一次。
                if attempt or "plugin migration inputs changed during startup convergence" not in detail:
                    raise RuntimeError(f"OpenClaw Gateway 提前退出：{detail[-1200:]}")

    async def _wait_gateway(self):
        import httpx
        async with httpx.AsyncClient(timeout=1, trust_env=False) as client:
            for _ in range(80):
                if self._process and self._process.returncode is not None:
                    raise RuntimeError("OpenClaw Gateway 提前退出")
                try:
                    await client.get(f"http://127.0.0.1:{self._gateway_port}/health")
                    return
                except Exception:
                    await asyncio.sleep(.25)
        raise TimeoutError("等待本组 OpenClaw Gateway 超时")

    async def execute(self, task: SolverInput, tool_session: str) -> RunResult:
        model = self.config.get("execution_model", {})
        if not self._process:
            return RunResult(status="failed", output=None, error="OpenClaw Gateway 未启动", trace=[], usage={"known": False})
        self._session_id = f"medevolve-{self._client_id}-{secrets.token_hex(8)}"
        self._pending[self._session_id] = {"iterations": None,
            "review_before": self._review_reader(), "review_started_ms": int(time.time() * 1000)}
        message = f"请完成以下医疗任务：\n{task.instruction}\n患者标识：{task.patient_id}\n决策点：{task.decision_point or '无'}\n使用 medical_tools_list 查看工具，再调用 medical_tool_call 查询。给出最终答案。"
        msg_path = self.state_dir / f"{self._session_id}.md"
        msg_path.write_text(message, encoding="utf-8")
        (self.state_dir / "tool-session.json").write_text(json.dumps({"bridge": self.bridge_url, "session": tool_session, "client": self._client_id}), encoding="utf-8")
        code, stdout, stderr = await self._command_runner(
            [self.config.get("openclaw_bin", "openclaw"), "agent", "--agent", "medical", "--session-id", self._session_id, "--message-file", str(msg_path), "--json"],
            group_dir=self.group_dir, group_env=self._env(model), timeout=float(self.config.get("timeout_seconds", 900)))
        try:
            envelope = json.loads(stdout)
        except json.JSONDecodeError:
            return RunResult(status="failed", output=None, error=f"OpenClaw 未返回 JSON：{stderr[-300:]}", trace=[], usage={"known": False})
        answer = self._answer(envelope)
        meta = self._agent_meta(envelope)
        usage = meta.get("usage") or envelope.get("usage")
        count = meta.get("assistantTurns", envelope.get("assistantTurns"))
        self._pending[self._session_id]["iterations"] = count if isinstance(count, int) else None
        native_error = self._native_error(envelope)
        trace = [{"type": "native_response", "tool_calls": meta.get("toolCalls"), "error": native_error}]
        failed = self._native_status_failed(envelope) or native_error is not None
        if code or failed or not answer:
            return RunResult(status="failed", output={"answer": answer} if answer else None, error=stderr[-1000:] or native_error or ("OpenClaw 未提取到答案" if not answer else f"OpenClaw退出码 {code}"), trace=trace, usage={"known": isinstance(usage, dict), **(usage or {})})
        return RunResult(status="completed", output={"answer": answer}, error=None, trace=trace, usage={"known": bool(usage), **(usage or {})})

    @staticmethod
    def _answer(obj):
        result = obj.get("result") if isinstance(obj.get("result"), dict) else obj
        payloads = result.get("payloads", [])
        texts = [item.get("text") for item in payloads if isinstance(item, dict) and isinstance(item.get("text"), str)] if isinstance(payloads, list) else []
        for source in (result, obj):
            for key in ("final", "finalResponse", "text", "content"):
                value = source.get(key) if isinstance(source, dict) else None
                if isinstance(value, str) and value.strip():
                    texts.append(value)
        return next((value for value in reversed(texts) if value.strip()), "")

    @staticmethod
    def _agent_meta(envelope):
        result = envelope.get("result") if isinstance(envelope.get("result"), dict) else {}
        result_meta = result.get("meta") if isinstance(result.get("meta"), dict) else {}
        envelope_meta = envelope.get("meta") if isinstance(envelope.get("meta"), dict) else {}
        agent_meta = result_meta.get("agentMeta") or envelope_meta.get("agentMeta") or {}
        return agent_meta if isinstance(agent_meta, dict) else {}

    @staticmethod
    def _native_error(envelope):
        result = envelope.get("result") if isinstance(envelope.get("result"), dict) else {}
        error = envelope.get("error") or result.get("error")
        if not error and OpenClawRunner._native_status_failed(envelope):
            error = envelope.get("summary") or result.get("summary")
        payloads = result.get("payloads", [])
        if not error and isinstance(payloads, list):
            error_payload = next((item for item in payloads if isinstance(item, dict) and item.get("isError")), None)
            if error_payload:
                error = error_payload.get("text") or "OpenClaw 返回错误 payload"
        return error

    @staticmethod
    def _native_status_failed(envelope):
        result = envelope.get("result") if isinstance(envelope.get("result"), dict) else {}
        return envelope.get("status") in {"error", "failed", "timeout", "in_flight"} or result.get("status") in {"error", "failed", "timeout", "in_flight"}

    async def learn(self, feedback: TrainingFeedback) -> HarnessStep:
        if feedback.completed is None:
            run = RunResult(status="failed", output=None, error="完成评审状态未知，跳过学习", trace=[], usage={"known": False})
            return HarnessStep(run=run, learning_runs=[], learning_status="failed")
        if self._frozen:
            run = RunResult(status="failed", output=None, error="冻结经验只读，拒绝学习", trace=[], usage={"known": False})
            return HarnessStep(run=run, learning_runs=[], learning_status="failed")
        pending = self._pending.get(self._session_id or "")
        if not pending or not isinstance(pending["iterations"], int) or pending["iterations"] < 10:
            if pending:
                pending["review_waited"] = True
            run = RunResult(status="completed", output={"learning_status": "not_triggered"}, trace=[], usage={"known": False})
            return HarnessStep(run=run, learning_runs=[], learning_status="not_triggered")
        model = self.config.get("execution_model", {})
        feedback_file = self.state_dir / f"{self._session_id}-feedback.md"
        feedback_file.write_text(
            "本题评审反馈：\n"
            f"完成状态：{feedback.completed}\n失败类别：{feedback.failure_category or '无'}\n"
            f"评审意见：{feedback.reason}\n参考答案：{feedback.reference_answer}\n"
            f"证据摘要：{json.dumps(feedback.evidence, ensure_ascii=False)}\n"
            "请结合本次经验进行反思；不要再执行医疗工具。",
            encoding="utf-8",
        )
        code, stdout, stderr = await self._command_runner(
            [self.config.get("openclaw_bin", "openclaw"), "agent", "--agent", "medical",
             "--session-id", self._session_id, "--message-file", str(feedback_file), "--json"],
            group_dir=self.group_dir, group_env=self._env(model),
            timeout=float(self.config.get("learning_timeout_seconds", 300)))
        feedback_response = None
        try:
            feedback_response = json.loads(stdout)
        except json.JSONDecodeError:
            pass
        feedback_meta = self._agent_meta(feedback_response or {})
        feedback_usage = feedback_meta.get("usage") or (feedback_response or {}).get("usage")
        feedback_run = RunResult(
            status="failed" if code or feedback_response is None or self._native_status_failed(feedback_response) or self._native_error(feedback_response) is not None else "completed",
            output={"answer": self._answer(feedback_response)} if feedback_response else None,
            error=stderr[-1000:] if code else ("OpenClaw反馈回合无JSON响应" if feedback_response is None else None),
            trace=[{"type": "learning_feedback_turn", "tool_calls": feedback_meta.get("toolCalls")}],
            usage={"known": isinstance(feedback_usage, dict), **(feedback_usage or {}), "model": model.get("model")},
        )
        if feedback_run.status == "failed":
            return HarnessStep(run=feedback_run, learning_runs=[], learning_status="failed")
        deadline = asyncio.get_running_loop().time() + float(self.config.get("review_timeout_seconds", 155))
        while asyncio.get_running_loop().time() < deadline:
            status = self._review_reader()
            before = pending.get("review_before")
            if status and status.get("attemptedAtMs", 0) > max(
                pending["review_started_ms"], (before or {}).get("attemptedAtMs", 0)
            ):
                outcome = status.get("outcome")
                usage = status.get("usage")
                learning_run = RunResult(status="completed", output={
                    "learning_status": outcome,
                    "proposal_id": status.get("proposalId"),
                    "error": status.get("error"),
                }, trace=[{"type": "native_learning", "attempted_at_ms": status["attemptedAtMs"]}],
                    usage={"known": isinstance(usage, dict), "model": model.get("model"), **(usage or {})})
                state = "updated" if outcome in {"applied", "proposed"} else "failed" if outcome == "failed" else "unchanged"
                pending["review_waited"] = True
                return HarnessStep(run=feedback_run, learning_runs=[learning_run], learning_status=state)
            await asyncio.sleep(float(self.config.get("review_poll_seconds", 2)))
        run = RunResult(status="failed", output={"learning_status": "unknown"},
            error="本题已满足深度门槛，但超时未观察到新的 OpenClaw review 记录", trace=[], usage={"known": False})
        pending["review_waited"] = True
        return HarnessStep(run=feedback_run, learning_runs=[run], learning_status="failed")

    def _read_review_status(self):
        db = self.state_dir / "state/openclaw.sqlite"
        if not db.is_file():
            return None
        try:
            uri = f"file:{db}?mode=ro"
            with sqlite3.connect(uri, uri=True, timeout=1) as conn:
                row = conn.execute("SELECT value_json FROM config_machine_state WHERE state_key = ?", ("skills.curatorState",)).fetchone()
            if not row:
                return None
            state = json.loads(row[0]).get("lastResult", {})
            key = hashlib.sha256(str(self.workspace.resolve()).encode()).hexdigest()
            return state.get("experienceReviews", {}).get(key)
        except (sqlite3.Error, ValueError, OSError):
            return None

    async def freeze(self, destination: Path):
        pending = self._pending.get(self._session_id) if self._session_id else None
        if pending and isinstance(pending.get("iterations"), int) and pending["iterations"] >= 10:
            # The preceding learn call owns the bounded wait. Freeze is only valid after that turn.
            if not pending.get("review_waited"):
                raise RuntimeError("OpenClaw 本题 review 尚未确认；先调用 learn() 再冻结")
        await self.close()
        target = Path(destination)
        export_experience(self.workspace, target, include=("skills", "memory", "MEMORY.md"))
        return Snapshot(path=target, digest=directory_digest(target))

    async def close(self):
        if self._process and self._process.returncode is None:
            self._process.terminate()
            try:
                await asyncio.wait_for(self._process.wait(), 10)
            except asyncio.TimeoutError:
                self._process.kill()
                await self._process.wait()
        self._process = None
