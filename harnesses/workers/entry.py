"""Container-side entry point: one JSON request in, one JSON response out.

    python3 -m harnesses.workers.entry < request.json > response.json

Used by `harnesses.worker_protocol.ContainerWorkerTransport` through the
isolation layer's `docker exec`. Diagnostics go to stderr; stdout carries only
the response line.
"""

from __future__ import annotations

import asyncio
import sys

from harnesses.worker_protocol import (
    WorkerRequest, WorkerResponse, decode_request, encode_response,
)
from harnesses.workers import build_worker
from harnesses.workers.base import SubprocessCli, WorkerPaths

__all__ = ["dispatch", "main"]


async def dispatch(
    request: WorkerRequest, *, paths: WorkerPaths | None = None, cli=None
) -> WorkerResponse:
    """Run one request against the worker for its harness.

    `paths` and `cli` are injectable so the same code path can be exercised
    without Docker and without the harness binary.
    """
    try:
        worker = build_worker(
            request.harness,
            paths or WorkerPaths.from_env(request.harness),
            cli if cli is not None else SubprocessCli(request.harness),
        )
    except Exception as exc:
        return WorkerResponse(
            action=request.action, status="unsupported",
            error=f"{type(exc).__name__}: {exc}",
        )
    return await worker.handle(request)


async def main() -> int:
    raw = sys.stdin.read()
    try:
        request = decode_request(raw)
    except Exception as exc:
        print(encode_response(WorkerResponse(
            action="health", status="failed", error=str(exc),
        )))
        return 2
    response = await dispatch(request)
    print(encode_response(response))
    return 0 if response.ok else 1


if __name__ == "__main__":  # pragma: no cover - exercised through `main`
    raise SystemExit(asyncio.run(main()))
