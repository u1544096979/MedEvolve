"""Hermes 的隔离 Python 子进程 runner。"""
from __future__ import annotations
import asyncio
import hashlib
import json
import os
import secrets
import sys
from pathlib import Path

from contracts import RunResult, SolverInput
from harnesses.contracts import HarnessStep, Snapshot, TrainingFeedback
from harnesses.local import export_experience, run_command
from harnesses.snapshots import collect_files, directory_digest

WORKER = Path(__file__).with_name("hermes_worker.py")


class HermesRunner:
    def __init__(self, *, config, group_dir, bridge_url):
        self.config, self.group_dir, self.bridge_url = config, Path(group_dir), bridge_url
        self.home = self.group_dir / "hermes-home"
        self.home.mkdir(parents=True, exist_ok=True)
        self.python = config.get("python_path", sys.executable)
        self.source_dir = config.get("source_dir", "")
        self._command_runner = config.get("command_runner", run_command)
        self._history = {}
        self._session_id = None
        self._tool_session = None
        self._frozen = False

    @property
    def current_session_id(self): return self._session_id

    async def start(self, group, initial: Snapshot):
        self.group = group
        self.client_id = self.config.get("client_id") or group.run_id
        self._frozen = bool(self.config.get("frozen", False))
        self.home.mkdir(parents=True, exist_ok=True)
        (self.home / "skills").mkdir(exist_ok=True)
        initial_path = Path(initial.path)
        if initial_path.is_dir():
            export_experience(initial_path, self.home, include=("skills", "memory", "MEMORY.md", "USER.md"))
        (self.home / "skills").mkdir(parents=True, exist_ok=True)
        (self.home / "memory").mkdir(parents=True, exist_ok=True)

    def _model(self, phase):
        return self.config.get(f"{phase}_model", self.config.get("execution_model", {}))

    async def _call(self, payload, model, timeout):
        self.group_dir.mkdir(parents=True, exist_ok=True)
        env = {"HERMES_HOME": str(self.home), "PYTHONPATH": str(self.source_dir)}
        api_key_env = model.get("api_key_env")
        if api_key_env:
            env[api_key_env] = os.environ.get(api_key_env, "")
        # The worker protocol is a single JSON request/response file to keep stdout clean.
        request = self.group_dir / "hermes-request.json"
        response = self.group_dir / "hermes-response.json"
        response.unlink(missing_ok=True)
        request.write_text(json.dumps({**payload, "request_path": str(request), "response_path": str(response)}, ensure_ascii=False), encoding="utf-8")
        # Re-run with paths supplied as arguments; child sets HERMES_HOME before imports.
        code, stdout, stderr = await self._command_runner([self.python, str(WORKER), str(request), str(response)], group_dir=self.group_dir, group_env=env, timeout=timeout)
        if code or not response.is_file():
            return None, stderr[-1000:] or "Hermes worker 未返回结果"
        try: return json.loads(response.read_text(encoding="utf-8")), None
        except (ValueError, OSError) as exc: return None, f"Hermes 响应解析失败：{exc}"

    async def execute(self, task: SolverInput, tool_session: str):
        model = self._model("execution")
        self._session_id = secrets.token_hex(16)
        self._tool_session = tool_session
        self._history[self._session_id] = []
        result, error = await self._call({"action": "execute", "model": model, "session_id": self._session_id,
            "message": f"完成医疗任务：\n{task.instruction}\n患者标识：{task.patient_id}\n决策点：{task.decision_point or '无'}",
            "history": [], "bridge_url": self.bridge_url, "tool_session": tool_session,
            "client_id": self.client_id, "frozen": self._frozen,
            "max_iterations": self.config.get("execution_budget", {}).get("max_model_turns", 30)}, model, self.config.get("timeout_seconds", 900))
        if error:
            return RunResult(status="failed", output=None, error=error, trace=[], usage={"known": False})
        self._history[self._session_id] = result.get("messages", [])
        usage = result.get("usage")
        answer = result.get("final_response")
        if result.get("failed") or result.get("completed") is not True or not answer:
            return RunResult(status="failed", output={"answer": answer} if answer else None,
                error=result.get("error") or "Hermes 原生任务未成功完成", trace=result.get("trace", []), usage={"known": bool(usage), **(usage or {})})
        return RunResult(status="completed", output={"answer": answer}, error=None,
            trace=result.get("trace", []), usage={"known": isinstance(usage, dict), **(usage or {})})

    async def learn(self, feedback: TrainingFeedback):
        if feedback.completed is None:
            run=RunResult(status="failed", output=None, error="完成评审状态未知，跳过学习", trace=[], usage={"known": False})
            return HarnessStep(run=run, learning_runs=[], learning_status="failed")
        if not self._session_id:
            run=RunResult(status="failed", output=None, error="没有可延续的 Hermes 任务会话", trace=[], usage={"known": False})
            return HarnessStep(run=run, learning_runs=[], learning_status="failed")
        if self._frozen:
            run=RunResult(status="failed", output=None, error="冻结经验只读，拒绝学习", trace=[], usage={"known": False})
            return HarnessStep(run=run,learning_runs=[],learning_status="failed")
        model=self._model("learning")
        before = self._experience_digest()
        message={"task_id": feedback.task_id, "completed": feedback.completed,
                 "failure_category": feedback.failure_category, "reason": feedback.reason,
                 "reference_answer": feedback.reference_answer, "evidence": feedback.evidence}
        result,error=await self._call({"action":"learn", "model":model, "session_id":self._session_id,
            "message":"根据本题反馈改进后续经验：\n"+json.dumps(message,ensure_ascii=False),
            "history":self._history.get(self._session_id,[]), "bridge_url":self.bridge_url,
            "tool_session":self._tool_session, "client_id":self.client_id, "frozen":self._frozen,
            "max_iterations": self.config.get("learning_budget", {}).get("max_model_turns", 10)}, model,
            self.config.get("learning_timeout_seconds",300))
        if error:
            run=RunResult(status="failed",output=None,error=error,trace=[],usage={"known":False})
            return HarnessStep(run=run,learning_runs=[run],learning_status="failed")
        self._history[self._session_id]=result.get("messages",self._history.get(self._session_id,[]))
        usage=result.get("usage")
        if result.get("failed") or result.get("completed") is not True:
            run=RunResult(status="failed",output={"learning_status":"failed"},
                error=result.get("error") or "Hermes 反馈回合未成功完成",trace=result.get("trace",[]),
                usage={"known":isinstance(usage,dict),**(usage or {})})
            return HarnessStep(run=run,learning_runs=[run],learning_status="failed")
        changed = self._experience_digest() != before
        learning_status = "updated" if changed else "unchanged"
        run=RunResult(status="completed",output={"learning_status":learning_status},trace=result.get("trace",[]),usage={"known":bool(usage),**(usage or {})})
        return HarnessStep(run=run,learning_runs=[run],learning_status=learning_status)

    async def freeze(self, destination: Path):
        target=Path(destination)
        export_experience(self.home,target,include=("skills","memory","MEMORY.md","USER.md"))
        (target / "skills").mkdir(parents=True, exist_ok=True)
        (target / "memory").mkdir(parents=True, exist_ok=True)
        return Snapshot(path=target,digest=directory_digest(target))

    def _experience_digest(self):
        files = collect_files(self.home, include=("skills", "memory", "MEMORY.md", "USER.md"))
        payload = json.dumps(files, ensure_ascii=False, sort_keys=True).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    async def close(self):
        self._history.clear()
