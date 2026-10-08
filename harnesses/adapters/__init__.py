"""Host-side adapters: one per harness, all behind the same contract.

The host side owns isolation, budgets and the frozen comparison; the harness's
own runtime lives in its image. No module here imports an OpenClaw, Hermes, ACE
or Reflexion package: an adapter only speaks `harnesses.worker_protocol` to its
container-side worker.

`HarnessAdapterBase` implements the parts every harness shares:

- `start` materialises the group workspace, then asks the worker to become ready;
- `execute` sends only the solver prompt, and in frozen mode rotates the attempt
  first so every test question gets a fresh container, session directory and
  Gateway while the read-only experience mount stays the same;
- `learn` is refused while frozen, so a frozen run can never accumulate
  experience;
- `freeze` materialises the worker's export and digests it;
- `close` stops the worker and removes exactly this run's containers and volumes.

`create_adapter` is the registry the test fixtures and the experiment runner use.
"""

from __future__ import annotations

import hashlib
from importlib import import_module
from pathlib import Path
from typing import ClassVar, NamedTuple, Protocol

from contracts import ModelConfig, RunResult, SolverInput
from harnesses.contracts import (
    GroupKey, HarnessAdapter, HarnessLimits, HarnessName, HarnessStep, Snapshot,
    TrainingFeedback,
)
from harnesses.snapshots import directory_digest, materialize_files
from harnesses.worker_protocol import (
    WorkerAction, WorkerRequest, WorkerResponse, WorkerTransport,
)

__all__ = [
    "ADAPTER_REGISTRY", "AdapterError", "HarnessAdapterBase", "NullRecorder",
    "create_adapter",
]

DEFAULT_LEARNING_QUIET_SECONDS = 30.0


class AdapterError(RuntimeError):
    """A harness adapter could not start, execute, learn or freeze."""


class ModelCallRecorder(Protocol):
    """Receives the model calls a worker attributed to each phase."""

    def record_model_call(self, phase: str) -> None: ...


class NullRecorder:
    """Stand-in recorder for runs whose accounting lives elsewhere."""

    def record_model_call(self, phase: str) -> None:
        return None


class AdapterSpec(NamedTuple):
    module: str
    attribute: str


# Later sub-dispatches add one entry per harness; the adapter factory and the
# parametrized adapter test read this registry instead of hard-coding names.
ADAPTER_REGISTRY: dict[HarnessName, AdapterSpec] = {
    "openclaw": AdapterSpec("harnesses.adapters.openclaw", "OpenClawAdapter"),
}


def create_adapter(name: HarnessName, **kwargs) -> HarnessAdapterBase:
    """Build the adapter for `name`; raises `AdapterError` when unavailable."""
    spec = ADAPTER_REGISTRY.get(name)
    if spec is None:
        raise AdapterError(f"没有 {name} 的适配器实现")
    try:
        factory = getattr(import_module(spec.module), spec.attribute)
    except (ImportError, AttributeError) as exc:
        raise AdapterError(f"{name} 适配器不可用：{exc}") from exc
    return factory(**kwargs)


class HarnessAdapterBase:
    """Shared lifecycle for the four harness adapters."""

    name: ClassVar[HarnessName]

    def __init__(
        self,
        *,
        isolation,
        model: ModelConfig,
        limits: HarnessLimits,
        transport_factory,
        recorder: ModelCallRecorder | None = None,
        frozen: bool = False,
        gateway_path: str = "/v1",
        learning_quiet_seconds: float | None = None,
        extra_start_payload: dict | None = None,
    ):
        self.isolation = isolation
        self.model = model
        self.limits = limits
        self.transport_factory = transport_factory
        self.recorder: ModelCallRecorder = recorder or NullRecorder()
        self.frozen = frozen
        self.gateway_path = gateway_path
        self.learning_quiet_seconds = (
            DEFAULT_LEARNING_QUIET_SECONDS if learning_quiet_seconds is None
            else learning_quiet_seconds
        )
        self.extra_start_payload = dict(extra_start_payload or {})
        self.group: GroupKey | None = None
        self.workspace = None
        self.transport: WorkerTransport | None = None
        self.session_id = ""
        self.client_id = ""
        self.started = False

    # -- contract ----------------------------------------------------------
    async def start(self, group: GroupKey, initial: Snapshot) -> None:
        if self.started:
            raise AdapterError(f"{self.name} 适配器已经启动")
        self.group = group
        self.workspace = await self.isolation.start_group(
            group, initial, frozen=self.frozen
        )
        self.transport = self.transport_factory(self.workspace)
        self.client_id = self._client_id(group)
        health = await self._call("health")
        if not health.ok:
            raise AdapterError(f"{self.name} worker 健康检查失败：{health.error}")
        ready = await self._call("start", payload=self._start_payload())
        if not ready.ok:
            raise AdapterError(f"{self.name} worker 启动失败：{ready.error}")
        self.session_id = ready.payload.get("session_id") or self.session_id
        self.started = True

    async def execute(self, task: SolverInput, tool_session: str | None) -> RunResult:
        self._require_started()
        if self.frozen:
            # Every test question refreshes the container and session directory
            # while the read-only experience mount stays the same.
            await self.isolation.start_attempt(self.workspace)
            ready = await self._call("start", payload=self._start_payload())
            if not ready.ok:
                raise AdapterError(f"{self.name} worker 重启失败：{ready.error}")
        response = await self._call(
            "execute",
            payload=self._execute_payload(task, tool_session),
            task_id=self._task_id(task),
        )
        if response.status == "unsupported":
            raise AdapterError(f"{self.name} worker 不支持执行：{response.error}")
        return self._run_result(response)

    async def learn(self, feedback: TrainingFeedback) -> HarnessStep:
        self._require_started()
        if self.frozen:
            raise AdapterError(
                "冻结运行禁止学习：参考答案和训练反馈不能进入冻结阶段的经验状态"
            )
        response = await self._call(
            "learn",
            payload=self._learn_payload(feedback),
            task_id=feedback.task_id,
        )
        payload = response.payload or {}
        run = self._run_result(response) if payload.get("run") else None
        if run is None:
            raise AdapterError(f"{self.name} 学习响应缺少运行结果：{response.error}")
        learning_runs = [
            RunResult.model_validate(item) for item in payload.get("learning_runs") or []
        ]
        status = payload.get("learning_status")
        if status is None:
            status = "failed"
        return HarnessStep(run=run, learning_runs=learning_runs, learning_status=status)

    async def freeze(self, destination: Path) -> Snapshot:
        self._require_started()
        response = await self._call("export")
        if not response.ok:
            raise AdapterError(f"{self.name} 经验导出失败：{response.error}")
        materialize_files(destination, response.payload.get("files") or {})
        return Snapshot(path=Path(destination), digest=directory_digest(destination))

    async def close(self) -> None:
        if self.transport is not None and self.started:
            try:
                await self._call("stop")
            except Exception:
                pass  # cleanup must not hide the original failure
        self.started = False
        await self.isolation.close()

    # -- shared payload builders ------------------------------------------
    def solver_message(self, task: SolverInput) -> str:
        """The solver prompt: question, patient and decision point only.

        Reference answers and frozen evidence are never part of this text - the
        contract only receives `SolverInput`, so no learning material can leak
        into a later attempt.
        """
        decision_point = task.decision_point or "无"
        return (
            "你在一次医疗任务实验中执行一道题。\n"
            f"患者标识：{task.patient_id}\n"
            f"决策点：{decision_point}\n"
            f"题目：{task.instruction}\n"
            "请使用本容器工具桥提供的医疗工具完成题目，并在结束时给出明确结论。"
        )

    def feedback_message(self, feedback: TrainingFeedback) -> str:
        """The public training review, addressed to the learning role only."""
        evidence = "\n".join(
            f"- {item.get('source_id', '')}: {item.get('statement', '')}"
            for item in feedback.evidence
        ) or "- 无"
        completed = (
            "未知" if feedback.completed is None
            else ("已完成" if feedback.completed else "未完成")
        )
        return (
            "以下是公共评审对上一道训练题的评价，用于更新可复用的医疗工具操作经验。\n"
            f"任务：{feedback.task_id}\n"
            f"完成情况：{completed}\n"
            f"失败类别：{feedback.failure_category or '无'}\n"
            f"评价理由：{feedback.reason}\n"
            f"参考答案：{feedback.reference_answer}\n"
            "证据：\n"
            f"{evidence}\n"
            "只提炼可复用的操作经验，不要把本题答案当作下一题的回答。"
        )

    # -- internals ---------------------------------------------------------
    def _client_id(self, group: GroupKey) -> str:
        """Stable group-scoped identity for the gateway and the tool bridge."""
        return f"{group.run_id}:{self.isolation.group_name(group)}"

    @property
    def gateway_base_url(self) -> str:
        base = (self.isolation.gateway_url or "").rstrip("/")
        return f"{base}{self.gateway_path}"

    def _start_payload(self) -> dict:
        return {
            "model": self.model.model_dump(mode="json"),
            "limits": self.limits.model_dump(mode="json"),
            "gateway_base_url": self.gateway_base_url,
            "client_id": self.client_id,
            "session_id": self.session_id or f"medevolve-{self.client_id}",
            "gateway_port": self._gateway_port(),
            **self.extra_start_payload,
        }

    def _gateway_port(self) -> int:
        return 18789

    def _execute_payload(self, task: SolverInput, tool_session: str | None) -> dict:
        return {
            "message": self.solver_message(task),
            "session_id": self._execute_session_id(),
            "tool_session": tool_session,
            "timeout_seconds": self.limits.timeout_seconds,
        }

    def _execute_session_id(self) -> str:
        if not self.frozen:
            return self.session_id
        attempt = getattr(self.workspace, "attempt", 0)
        return f"{self.session_id}-attempt-{attempt}"

    def _learn_payload(self, feedback: TrainingFeedback) -> dict:
        return {
            "message": self.feedback_message(feedback),
            "session_id": self.session_id,
            "learning_timeout_seconds": self.limits.learning_timeout_seconds,
            "learning_quiet_seconds": self.learning_quiet_seconds,
        }

    @staticmethod
    def _task_id(task: SolverInput) -> str:
        digest = hashlib.sha1(
            f"{task.patient_id}|{task.instruction}".encode()
        ).hexdigest()[:10]
        return f"{task.patient_id}-{digest}"

    @staticmethod
    def _run_result(response: WorkerResponse) -> RunResult:
        run = (response.payload or {}).get("run")
        if not isinstance(run, dict):
            raise AdapterError(
                f"worker 响应缺少运行结果：{response.error or response.status}"
            )
        return RunResult.model_validate(run)

    def _require_started(self) -> None:
        if not self.started or self.transport is None:
            raise AdapterError(f"{self.name} 适配器尚未启动")

    async def _call(
        self, action: WorkerAction, *, payload: dict | None = None, task_id: str = ""
    ) -> WorkerResponse:
        if self.group is None:
            raise AdapterError(f"{self.name} 适配器缺少分组")
        request = WorkerRequest(
            action=action,
            harness=self.name,
            group=self.group,
            frozen=self.frozen,
            task_id=task_id,
            payload=payload or {},
        )
        response = await self.transport.call(request)
        for phase, count in (response.model_calls or {}).items():
            for _ in range(int(count)):
                self.recorder.record_model_call(phase)
        return response
