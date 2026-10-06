"""Opt-in serial checks against the pinned MedAgentBench image.

MEDEVOLVE_MEDAGENTBENCH_TEST_IMAGE=jyxsu6/medagentbench@sha256:<digest> \
  python -m pytest tests/environment/test_medagentbench_live.py -q
"""

import json
import os

import pytest

from contracts import EnvironmentConfig, EnvironmentInstance, Task
from environments import create_environment, get_tools, read_state, release_environment
from environments.medagentbench import NativeTools


IMAGE = "jyxsu6/medagentbench@sha256:3fb83d7ed71c5476f9eb6212bd440a909ef7505922bbc757dc488a8fc0701966"
MANAGER_URL = "http://127.0.0.1:18084"
TASK5_INSTRUCTION = (
    "Check patient S6315806's last serum magnesium level within last 24 hours. "
    "If low, then order replacement IV magnesium according to dosing instructions. "
    "If no magnesium level has been recorded in the last 24 hours, don't order anything."
)


@pytest.mark.asyncio
@pytest.mark.skipif(
    not os.environ.get("MEDEVOLVE_MEDAGENTBENCH_TEST_IMAGE"),
    reason="Set MEDEVOLVE_MEDAGENTBENCH_TEST_IMAGE to the locally loaded pinned image",
)
async def test_official_interaction_post_semantics_and_serial_cleanup():
    config = EnvironmentConfig(
        name="MedAgentBench",
        manager_url=MANAGER_URL,
        seed_data=os.environ["MEDEVOLVE_MEDAGENTBENCH_TEST_IMAGE"],
    )
    first = await create_environment(config)
    first_state = None
    try:
        tools = await get_tools(first, ["S6534835", "S6315806"])
        assert isinstance(tools, NativeTools)

        patient_search = await tools.execute(
            "GET " + first.endpoint
            + "Patient?name=Peter&family=Stafford&birthdate=1932-12-29"
        )
        assert patient_search["reply"].startswith(
            "Here is the response from the GET request:"
        )
        assert patient_search["reply"].endswith(
            "Please call FINISH if you have got answers for all the questions and finished all the requested tasks"
        )
        patient_data = patient_search["data"]
        if isinstance(patient_data, str):
            patient_data = json.loads(patient_data)
        matches = patient_data["entry"]
        assert [entry["resource"]["id"] for entry in matches] == ["S6534835"]
        assert (await tools.execute('FINISH(["S6534835"])'))["answers"] == ["S6534835"]

        magnesium_search = await tools.execute(
            "GET " + first.endpoint
            + "Observation?patient=Patient%2FS6315806&code=MG"
        )
        assert magnesium_search["kind"] == "GET"
        assert magnesium_search["reply"].startswith(
            "Here is the response from the GET request:"
        )
        assert magnesium_search["reply"].endswith(
            "Please call FINISH if you have got answers for all the questions and finished all the requested tasks"
        )
        magnesium_data = magnesium_search["data"]
        if isinstance(magnesium_data, str):
            magnesium_data = json.loads(magnesium_data)
        assert magnesium_data["resourceType"] == "Bundle"
        task5_finish = await tools.execute(
            'FINISH(["No serum magnesium result with code MG was found; no order was placed."])'
        )
        assert len(task5_finish["answers"]) == 1

        task = Task(
            id="task5_1",
            patient_id="S6315806",
            environment="MedAgentBench",
            round_index=0,
            source_method="official",
            instruction=TASK5_INSTRUCTION,
            task_type="治疗规划",
            reference_answer="",
            evidence=[],
        )
        first_state = await read_state(first, task)
        payload = {
            "resourceType": "MedicationRequest",
            "status": "active",
            "intent": "order",
            "subject": {"reference": "Patient/S6315806"},
            "medicationCodeableConcept": {
                "coding": [{"system": "http://hl7.org/fhir/sid/ndc", "code": "0338-1715-40"}]
            },
            "dosageInstruction": [
                {"doseAndRate": [{"doseQuantity": {"value": 1, "unit": "g"}}]}
            ],
        }
        post = await tools.execute(
            "POST " + first.endpoint + "MedicationRequest\n" + json.dumps(payload)
        )
        assert post["kind"] == "POST"
        assert post["request"]["payload"] == payload
        assert "POST request accepted and executed successfully" in post["reply"]
        after_post = await read_state(first, task)
        assert after_post["resources"] == first_state["resources"]
        assert after_post["operation_semantics"]["acceptance_proves_database_write"] is False
    finally:
        await release_environment(config, first)

    # The next instance is not created until the first instance has been released.
    second = await create_environment(config)
    try:
        assert second.instance_id != first.instance_id
        assert second.endpoint != first.endpoint
        task = Task(
            id="task5_1",
            patient_id="S6315806",
            environment="MedAgentBench",
            round_index=0,
            source_method="official",
            instruction=TASK5_INSTRUCTION,
            task_type="治疗规划",
            reference_answer="",
            evidence=[],
        )
        second_state = await read_state(second, task)
        assert second_state["resources"] == first_state["resources"]
    finally:
        await release_environment(config, second)


@pytest.mark.asyncio
@pytest.mark.skipif(
    not os.environ.get("MEDEVOLVE_MEDAGENTBENCH_TEST_IMAGE"),
    reason="Set MEDEVOLVE_MEDAGENTBENCH_TEST_IMAGE to the locally loaded pinned image",
)
async def test_native_fhir_next_link_is_followed_as_an_explicit_root_get():
    config = EnvironmentConfig(
        name="MedAgentBench",
        manager_url=MANAGER_URL,
        seed_data=os.environ["MEDEVOLVE_MEDAGENTBENCH_TEST_IMAGE"],
    )
    instance = await create_environment(config)
    try:
        tools = await get_tools(instance, ["S6315806"])
        first = await tools.execute(
            "GET " + instance.endpoint
            + "Observation?patient=Patient%2FS6315806&_count=20"
        )
        first_data = first["data"]
        if isinstance(first_data, str):
            first_data = json.loads(first_data)
        assert first["kind"] == "GET"
        assert first_data["resourceType"] == "Bundle"
        next_link = next(
            link["url"] for link in first_data.get("link", [])
            if link.get("relation") == "next"
        )

        next_page = await tools.execute("GET " + next_link)
        next_data = next_page["data"]
        if isinstance(next_data, str):
            next_data = json.loads(next_data)
        assert next_page["kind"] == "GET"
        assert next_page["request"]["url"].startswith(instance.endpoint.rstrip("/"))
        assert next_data["resourceType"] == "Bundle"
        assert next_data.get("entry")
        for entry in next_data["entry"]:
            resource = entry["resource"]
            assert resource.get("subject", {}).get("reference") == "Patient/S6315806"
    finally:
        await release_environment(config, instance)
