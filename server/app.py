"""FastAPI factory: MEDEVOLVE_PUBLIC_HOST must be reachable from the local client.

Start with: uvicorn server.app:create_app --factory --host 0.0.0.0 --port 8000
MEDEVOLVE_BIND_HOST defaults to 0.0.0.0 for dynamically published FHIR ports.
Only images already loaded on the Docker host are accepted; no images are pulled.
"""

import asyncio
import os

from fastapi import FastAPI, HTTPException, Response

from contracts import EnvironmentConfig, EnvironmentInstance
from environments.errors import EnvironmentUnavailable
from server.docker import DockerManager


def create_app(manager: DockerManager | None = None) -> FastAPI:
    if manager is None:
        public_host = os.environ.get("MEDEVOLVE_PUBLIC_HOST", "")
        manager = DockerManager(
            public_host, os.environ.get("MEDEVOLVE_BIND_HOST", "0.0.0.0")
        )
    app = FastAPI(title="MedEvolve Environment Manager")

    @app.post("/instances", response_model=EnvironmentInstance, status_code=201)
    async def create_instance(config: EnvironmentConfig) -> EnvironmentInstance:
        try:
            return await manager.create(config)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except (EnvironmentUnavailable, asyncio.TimeoutError) as exc:
            raise HTTPException(
                503, str(exc) or "Environment startup timed out"
            ) from exc

    @app.delete("/instances/{instance_id}", status_code=204)
    async def release_instance(instance_id: str) -> Response:
        try:
            await manager.release(instance_id)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except EnvironmentUnavailable as exc:
            raise HTTPException(503, str(exc)) from exc
        return Response(status_code=204)

    return app
