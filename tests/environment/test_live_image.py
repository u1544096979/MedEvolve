"""Opt-in local Docker check against an already loaded PhysicianBench image.

MEDEVOLVE_TEST_IMAGE=sha256:<image-id> python -m pytest tests/environment/test_live_image.py -q
No image is pulled and no model is called. Missing opt-in configuration skips the test.
"""

import json
import os
import uuid

import httpx
import pytest

from contracts import EnvironmentConfig
from environments.fhir import FHIRClient
from environments.physicianbench import fhir_communication_create_message
from server.docker import DockerCommandError, DockerManager, docker_command


@pytest.mark.asyncio
@pytest.mark.skipif(
    not os.environ.get("MEDEVOLVE_TEST_IMAGE"),
    reason="A locally loaded PhysicianBench image is required",
)
async def test_real_container_writes_and_release_are_isolated():
    manager = DockerManager("127.0.0.1", bind_host="127.0.0.1")
    config = EnvironmentConfig(
        name="PhysicianBench",
        manager_url="http://unused",
        seed_data=os.environ["MEDEVOLVE_TEST_IMAGE"],
    )
    instances = []
    volumes = set()
    try:
        for _ in range(2):
            instances.append(await manager.create(config))
        first, second = instances
        mounts = []
        for instance in instances:
            container = json.loads(
                await docker_command(
                    "container", "inspect", f"medevolve-{instance.instance_id}"
                )
            )[0]
            names = {m["Name"] for m in container["Mounts"] if m["Type"] == "volume"}
            assert all(m["Type"] != "bind" for m in container["Mounts"])
            mounts.append(names)
            volumes.update(names)
        assert not mounts[0] & mounts[1]
        assert first.endpoint != second.endpoint
        async with httpx.AsyncClient(trust_env=False) as client:
            response = await client.get(
                first.endpoint + "Patient", params={"_count": 1}
            )
            response.raise_for_status()
            identifier = response.json()["entry"][0]["resource"]["id"]
        first_client = FHIRClient(first.endpoint, [identifier])
        second_client = FHIRClient(second.endpoint, [identifier])
        marker = "isolation-check-" + uuid.uuid4().hex
        created = await fhir_communication_create_message(
            first_client, f"Patient/{identifier}", marker
        )
        first_state = await first_client.snapshot(identifier)
        second_state = await second_client.snapshot(identifier)
        assert any(
            r.get("id") == created["id"] and r.get("resourceType") == "Communication"
            for r in first_state
        )
        assert marker not in json.dumps(second_state)
        await manager.release(first.instance_id)
        assert await manager._inspect(f"medevolve-{first.instance_id}") is None
        assert await second_client.snapshot(identifier)
        await manager.release(second.instance_id)
        assert await manager._inspect(f"medevolve-{second.instance_id}") is None
    finally:
        for instance in instances:
            await manager.release(instance.instance_id)
    for volume in volumes:
        with pytest.raises(DockerCommandError, match="[Nn]o such volume"):
            await docker_command("volume", "inspect", volume)
