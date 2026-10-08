"""Container-side workers: one entry point per harness image.

Host-side adapters never import a harness package; they send a `WorkerRequest` to
this package's entry point and read one `WorkerResponse` back. Each image ships
only its own worker module, so the registry resolves the worker lazily.
"""

from __future__ import annotations

from importlib import import_module
from typing import NamedTuple

from harnesses.contracts import HarnessName
from harnesses.worker_protocol import WorkerError

__all__ = ["WORKER_REGISTRY", "build_worker"]


class WorkerSpec(NamedTuple):
    module: str
    attribute: str


WORKER_REGISTRY: dict[HarnessName, WorkerSpec] = {
    "openclaw": WorkerSpec("harnesses.workers.openclaw", "OpenClawWorker"),
}


def build_worker(harness: HarnessName, paths, cli):
    """Instantiate the worker that belongs to this image."""
    spec = WORKER_REGISTRY.get(harness)
    if spec is None:
        raise WorkerError(f"本镜像没有 {harness} 的容器端 worker")
    return getattr(import_module(spec.module), spec.attribute)(paths, cli)
