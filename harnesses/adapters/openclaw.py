"""Host-side OpenClaw adapter.

Everything OpenClaw-specific lives in the image: this module only drives
`harnesses.workers.openclaw` through the worker protocol and therefore imports no
OpenClaw package. Its defaults are the container-side entry point and the
container transport, so the experiment runner gets a real container worker while
tests inject a substitute transport and a substitute CLI.
"""

from __future__ import annotations

from harnesses.adapters import HarnessAdapterBase
from harnesses.contracts import HarnessName
from harnesses.worker_protocol import ContainerWorkerTransport

__all__ = ["OpenClawAdapter", "default_transport_factory"]

DEFAULT_GATEWAY_PORT = 18789


def default_transport_factory(isolation, *, timeout: float = 900.0):
    """Build the container transport for one group workspace."""

    def factory(workspace):
        return ContainerWorkerTransport(
            isolation=isolation, workspace=workspace, timeout=timeout
        )

    return factory


class OpenClawAdapter(HarnessAdapterBase):
    """Native OpenClaw execution with Skill Workshop learning."""

    name: HarnessName = "openclaw"

    def __init__(self, *, isolation, model, limits, transport_factory=None,
                 recorder=None, frozen=False, learning_quiet_seconds=None,
                 extra_start_payload=None):
        super().__init__(
            isolation=isolation,
            model=model,
            limits=limits,
            transport_factory=transport_factory
            or default_transport_factory(isolation),
            recorder=recorder,
            frozen=frozen,
            learning_quiet_seconds=learning_quiet_seconds,
            extra_start_payload=extra_start_payload,
        )

    def _gateway_port(self) -> int:
        # The Gateway binds inside the container's own network namespace, so this
        # never collides with an operator's service on the host.
        return DEFAULT_GATEWAY_PORT
