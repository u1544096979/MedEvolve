"""Worker protocol shared by host-side adapters and container-side workers.

The four harnesses' own dependencies live in their own images and are not merged
into the MedEvolve environment. The host side therefore never imports a harness
package: `harnesses/adapters/*.py` only speaks this request/response protocol to a
container-side entry point under `harnesses/workers/`.

One exchange is one JSON document in each direction over a single stdin/stdout
pipe:

    python3 -m harnesses.workers.entry   # reads one WorkerRequest on stdin,
                                         # writes one WorkerResponse on stdout

Diagnostics go to stderr so stdout stays parseable. `model_calls` carries the
model calls the worker attributed to each phase; the host adapter forwards them to
its recorder, which is the same accounting path the model gateway uses.
"""

from __future__ import annotations

import asyncio
import json
from typing import Literal, Protocol

from pydantic import BaseModel, Field

from harnesses.contracts import GroupKey, HarnessName

__all__ = [
    "MODEL_PHASES", "ContainerWorkerTransport", "WorkerAction", "WorkerError",
    "WorkerRequest", "WorkerResponse", "WorkerStatus", "WorkerTransport",
    "decode_request", "decode_response", "encode_request", "encode_response",
    "parse_worker_stdout",
]

WorkerAction = Literal["health", "start", "execute", "learn", "export", "stop"]
WorkerStatus = Literal["ok", "failed", "unsupported", "not_triggered", "not_ready"]
MODEL_PHASES = ("execution", "auxiliary", "learning")
WORKER_ENTRYPOINT = ("python3", "-m", "harnesses.workers.entry")


class WorkerError(RuntimeError):
    """The container-side worker could not be reached or answered invalidly."""


class WorkerRequest(BaseModel):
    """One host-to-container instruction."""

    action: WorkerAction
    harness: HarnessName
    group: GroupKey
    frozen: bool = False
    task_id: str = ""
    payload: dict = Field(default_factory=dict)


class WorkerResponse(BaseModel):
    """One container-to-host answer; `payload` is action-specific."""

    action: WorkerAction
    status: WorkerStatus = "ok"
    payload: dict = Field(default_factory=dict)
    model_calls: dict[str, int] = Field(default_factory=dict)
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.status == "ok"


class WorkerTransport(Protocol):
    """How the host adapter reaches its container-side worker."""

    async def call(self, request: WorkerRequest) -> WorkerResponse: ...


def encode_request(request: WorkerRequest) -> str:
    return json.dumps(request.model_dump(mode="json"), ensure_ascii=False)


def decode_request(text: str) -> WorkerRequest:
    try:
        return WorkerRequest.model_validate_json(text)
    except Exception as exc:
        raise WorkerError(f"worker 请求不是合法 JSON：{exc}") from exc


def encode_response(response: WorkerResponse) -> str:
    return json.dumps(response.model_dump(mode="json"), ensure_ascii=False)


def decode_response(text: str) -> WorkerResponse:
    try:
        return WorkerResponse.model_validate_json(text)
    except Exception as exc:
        raise WorkerError(f"worker 响应不是合法 JSON：{exc}") from exc


def parse_worker_stdout(text: str) -> WorkerResponse:
    """Read the response from the last non-empty stdout line."""
    for line in reversed((text or "").splitlines()):
        if line.strip():
            return decode_response(line)
    raise WorkerError("worker 没有输出任何响应")


class ContainerWorkerTransport:
    """Drive the container-side entry point through the isolation layer's exec."""

    def __init__(
        self,
        *,
        isolation,
        workspace,
        entrypoint: tuple[str, ...] = WORKER_ENTRYPOINT,
        timeout: float = 900.0,
    ):
        self.isolation = isolation
        self.workspace = workspace
        self.entrypoint = tuple(entrypoint)
        self.timeout = timeout

    async def call(self, request: WorkerRequest) -> WorkerResponse:
        stdout = await asyncio.wait_for(
            self.isolation.exec(
                self.workspace, *self.entrypoint, stdin=encode_request(request)
            ),
            timeout=self.timeout,
        )
        response = parse_worker_stdout(stdout)
        if response.action != request.action:
            raise WorkerError(
                f"worker 响应动作不匹配：请求 {request.action}，响应 {response.action}"
            )
        return response
