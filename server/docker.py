"""Docker lifecycle for supported benchmark environments.

seed_data is a locally loaded, pre-populated image (upstream: fhir-full:v1).
Docker creates an independent writable layer and independent anonymous volumes
from that image for every instance. No host or named data volumes are mounted.
"""

import asyncio
import json
import os
import uuid

import httpx

from contracts import EnvironmentConfig, EnvironmentInstance
from environments.errors import EnvironmentUnavailable

SUPPORTED_DOCKER_ENVIRONMENTS = {"PhysicianBench", "MedAgentBench"}


class DockerCommandError(EnvironmentUnavailable):
    pass


async def docker_command(*args: str) -> str:
    try:
        process = await asyncio.create_subprocess_exec(
            "docker",
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        raise DockerCommandError("Docker CLI is not installed") from exc
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=60)
    except (asyncio.TimeoutError, asyncio.CancelledError):
        process.kill()
        await process.communicate()
        raise
    if process.returncode:
        raise DockerCommandError(stderr.decode().strip())
    return stdout.decode().strip()


class DockerManager:
    def __init__(
        self, public_host: str, bind_host: str = "0.0.0.0", startup_timeout: float = 180
    ):
        if not public_host or any(c in public_host for c in "/?#@"):
            raise ValueError("public_host must be a client-reachable hostname or IP")
        self.public_host = public_host
        self.bind_host = bind_host
        self.startup_timeout = startup_timeout

    @staticmethod
    def _name(instance_id: str) -> str:
        if str(uuid.UUID(instance_id)) != instance_id:
            raise ValueError("instance_id must be a canonical UUID")
        return f"medevolve-{instance_id}"

    async def _inspect(self, name: str) -> dict | None:
        try:
            result = await docker_command("container", "inspect", name)
        except DockerCommandError as exc:
            if "No such container" in str(exc) or "No such object" in str(exc):
                return None
            raise
        return json.loads(result)[0]

    async def _remove(self, name: str) -> None:
        try:
            await docker_command("container", "rm", "--force", "--volumes", name)
        except DockerCommandError as exc:
            if "No such container" not in str(exc) and "No such object" not in str(exc):
                raise

    async def _wait_ready(self, port: int) -> None:
        host = "127.0.0.1" if self.bind_host == "0.0.0.0" else self.bind_host
        url = f"http://{host}:{port}/fhir/metadata"
        async with httpx.AsyncClient(timeout=5, trust_env=False) as client:

            async def poll() -> None:
                while True:
                    try:
                        response = await client.get(url)
                        response.raise_for_status()
                        if response.json().get("resourceType") == "CapabilityStatement":
                            return
                    except (httpx.HTTPError, ValueError):
                        pass
                    await asyncio.sleep(1)

            await asyncio.wait_for(poll(), timeout=self.startup_timeout)

    async def create(self, config: EnvironmentConfig) -> EnvironmentInstance:
        if config.name not in SUPPORTED_DOCKER_ENVIRONMENTS:
            raise EnvironmentUnavailable(
                f"{config.name} is unavailable"
            )
        if not config.seed_data or config.seed_data.startswith("-"):
            raise ValueError(
                "seed_data must identify a locally loaded environment image"
            )
        image = json.loads(await docker_command("image", "inspect", config.seed_data))[
            0
        ]
        # Pin this request to the inspected image, even if the tag changes later.
        image_id = image["Id"]
        if not image_id.startswith("sha256:"):
            raise EnvironmentUnavailable("Docker did not return an immutable image ID")
        if any(
            key.startswith(("SPRING_DATASOURCE_URL=", "spring.datasource.url="))
            and not key.split("=", 1)[1].startswith("jdbc:h2:")
            for key in image.get("Config", {}).get("Env", []) or []
        ):
            raise EnvironmentUnavailable(
                "Image uses an external database; isolation is unavailable"
            )
        instance_id = str(uuid.uuid4())
        name = self._name(instance_id)
        # H2 2.3.232 compaction can invalidate index pages during startup.
        java_options = "-Dh2.autoCompactFillRate=0"
        if os.environ.get("MEDEVOLVE_CONTAINER_MEMORY"):
            java_options += " -XX:MaxRAMPercentage=60.0"
        try:
            await docker_command(
                "run",
                "--detach",
                "--pull=never",
                "--name",
                name,
                "--label",
                "medevolve.managed=true",
                "--label",
                f"medevolve.environment={config.name}",
                "--publish",
                f"{self.bind_host}::8080",
                "--env",
                f"JAVA_TOOL_OPTIONS={java_options}",
                *(["--memory", os.environ["MEDEVOLVE_CONTAINER_MEMORY"]]
                  if os.environ.get("MEDEVOLVE_CONTAINER_MEMORY") else []),
                image_id,
            )
            container = await self._inspect(name)
            if container is None:
                raise EnvironmentUnavailable("Created container disappeared")
            ports = container["NetworkSettings"]["Ports"]["8080/tcp"]
            port = int(ports[0]["HostPort"])
            await self._wait_ready(port)
            return EnvironmentInstance(
                instance_id=instance_id,
                endpoint=f"http://{self.public_host}:{port}/fhir/",
                environment=config.name,
            )
        except BaseException:
            await asyncio.shield(self._remove(name))
            raise

    async def release(self, instance_id: str) -> None:
        name = self._name(instance_id)
        container = await self._inspect(name)
        if container is None:
            return
        labels = container.get("Config", {}).get("Labels", {}) or {}
        if labels.get("medevolve.managed") != "true":
            raise ValueError("Container is not managed by MedEvolve")
        await self._remove(name)
