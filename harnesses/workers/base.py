"""Container-side worker plumbing shared by the four harness workers.

A worker runs inside its harness image, where that harness's own runtime and
dependencies exist. It receives one `WorkerRequest` and answers one
`WorkerResponse`; it never imports MedEvolve's medical environment and never
creates its own FHIR client. Medical tools reach the attempt-scoped session
through the tool bridge URL only.

Two path rules are enforced here rather than in each harness worker:

- the durable state directory is the only place experience survives a task, and
  in frozen mode it is a read-only mount;
- every write outside `MEDEVOLVE_WRITABLE_DIR` is refused while frozen, so a
  frozen run cannot turn a writable copy into cross-test experience.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import signal
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from harnesses.contracts import HarnessName
from harnesses.worker_protocol import (
    MODEL_PHASES, WorkerError, WorkerRequest, WorkerResponse,
)

__all__ = [
    "CommandResult", "CliRunner", "HarnessWorker", "SubprocessCli", "WorkerPaths",
]


@dataclass
class CommandResult:
    args: tuple[str, ...]
    returncode: int
    stdout: str = ""
    stderr: str = ""

    @property
    def ok(self) -> bool:
        return self.returncode == 0


class CliRunner:
    """How a worker reaches its harness's own command line."""

    def available(self) -> bool:  # pragma: no cover - overridden
        raise NotImplementedError

    async def run(
        self,
        args: Sequence[str],
        *,
        stdin: str | None = None,
        env: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> CommandResult:  # pragma: no cover - overridden
        raise NotImplementedError

    async def spawn(
        self, args: Sequence[str], *, env: dict[str, str] | None = None
    ) -> int | None:  # pragma: no cover - overridden
        raise NotImplementedError

    async def stop(self) -> None:  # pragma: no cover - overridden
        return None

    def terminate(self, pid: int) -> None:
        """Stop a daemon this worker previously spawned."""
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass


class SubprocessCli(CliRunner):
    """The real harness CLI, run as a subprocess with an explicit env."""

    def __init__(self, executable: str = "openclaw"):
        self.executable = executable
        self._daemons: list[asyncio.subprocess.Process] = []

    def available(self) -> bool:
        return shutil.which(self.executable) is not None

    async def run(self, args, *, stdin=None, env=None, timeout=None) -> CommandResult:
        process = await asyncio.create_subprocess_exec(
            self.executable, *args,
            stdin=asyncio.subprocess.PIPE if stdin is not None else None,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={**os.environ, **(env or {})},
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(stdin.encode() if stdin is not None else None),
                timeout=timeout,
            )
        except (asyncio.TimeoutError, asyncio.CancelledError):
            process.kill()
            await process.communicate()
            raise
        return CommandResult(
            args=tuple(args), returncode=process.returncode or 0,
            stdout=stdout.decode(), stderr=stderr.decode(),
        )

    async def spawn(self, args, *, env=None) -> int | None:
        process = await asyncio.create_subprocess_exec(
            self.executable, *args,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
            env={**os.environ, **(env or {})},
        )
        self._daemons.append(process)
        return process.pid

    async def stop(self) -> None:
        for process in self._daemons:
            if process.returncode is None:
                process.terminate()
        for process in self._daemons:
            if process.returncode is None:
                try:
                    await asyncio.wait_for(process.wait(), timeout=10)
                except asyncio.TimeoutError:
                    process.kill()
                    await process.wait()
        self._daemons.clear()


@dataclass
class WorkerPaths:
    """Container paths a worker may use, split by write permission."""

    harness: HarnessName
    durable_root: Path
    workspace: Path
    writable: Path
    frozen: bool = False
    tool_bridge: str = ""

    @classmethod
    def from_env(cls, harness: HarnessName) -> "WorkerPaths":
        state_root = os.environ.get("MEDEVOLVE_STATE_DIR", "/state")
        writable = os.environ.get("MEDEVOLVE_WRITABLE_DIR", "/scratch")
        frozen = os.environ.get("MEDEVOLVE_FROZEN") == "1"
        durable_root = Path(state_root) / harness
        return cls(
            harness=harness,
            durable_root=durable_root,
            workspace=durable_root / "workspace",
            writable=Path(writable),
            frozen=frozen,
            tool_bridge=os.environ.get("MEDEVOLVE_TOOL_BRIDGE", ""),
        )

    @property
    def runtime_state(self) -> Path:
        """Where the harness keeps its own runtime state (never the snapshot)."""
        if self.frozen:
            return self.writable / f"{self.harness}-state"
        return self.durable_root

    def assert_writable(self, path: Path) -> None:
        """Refuse writes outside the scratch directory while frozen."""
        if not self.frozen:
            return
        target = Path(path).resolve()
        scratch = self.writable.resolve()
        if target != scratch and scratch not in target.parents:
            raise WorkerError(
                f"冻结运行禁止写入 {target}：所有需要写状态的组件必须重定向到 "
                f"{scratch}"
            )


class HarnessWorker:
    """Action dispatch, phase attribution and write guards for one harness."""

    name: HarnessName

    def __init__(self, paths: WorkerPaths, cli: CliRunner):
        self.paths = paths
        self.cli = cli
        self.model_calls: dict[str, int] = {}

    # -- dispatch ----------------------------------------------------------
    async def handle(self, request: WorkerRequest) -> WorkerResponse:
        if request.harness != self.name:
            return WorkerResponse(
                action=request.action, status="failed",
                error=f"worker 是 {self.name}，收到 {request.harness} 的请求",
            )
        handler = {
            "health": self.health,
            "start": self.start,
            "execute": self.execute,
            "learn": self.learn,
            "export": self.export,
            "stop": self.stop,
        }[request.action]
        try:
            response = await handler(request)
        except WorkerError as exc:
            return WorkerResponse(action=request.action, status="failed", error=str(exc))
        except Exception as exc:  # a worker fault must not crash the host adapter
            return WorkerResponse(
                action=request.action, status="failed",
                error=f"{type(exc).__name__}: {exc}",
            )
        response.model_calls = dict(self.model_calls)
        return response

    def record(self, phase: str, count: int = 1) -> None:
        if phase not in MODEL_PHASES:
            raise WorkerError(f"未知模型阶段：{phase}")
        if count <= 0:
            return
        self.model_calls[phase] = self.model_calls.get(phase, 0) + count

    # -- actions -----------------------------------------------------------
    async def health(self, request: WorkerRequest) -> WorkerResponse:
        raise NotImplementedError

    async def start(self, request: WorkerRequest) -> WorkerResponse:
        raise NotImplementedError

    async def execute(self, request: WorkerRequest) -> WorkerResponse:
        raise NotImplementedError

    async def learn(self, request: WorkerRequest) -> WorkerResponse:
        raise NotImplementedError

    async def export(self, request: WorkerRequest) -> WorkerResponse:
        raise NotImplementedError

    async def stop(self, request: WorkerRequest) -> WorkerResponse:
        raise NotImplementedError

    # -- helpers -----------------------------------------------------------
    def write_text(self, path: Path, text: str) -> Path:
        self.paths.assert_writable(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def read_text(self, path: Path) -> str:
        return Path(path).read_text(encoding="utf-8")
