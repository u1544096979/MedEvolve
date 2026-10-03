"""MedAgentBench native protocol and patient-scoped raw data access."""

import hashlib
import json
from pathlib import Path

import httpx
import pytest

from contracts import EnvironmentInstance, Task
from environments import get_tools, prepare_trajectory, read_evidence, read_state
from environments.errors import PatientAccessError
from environments.medagentbench import NativeTools, load_functions


@pytest.fixture
def fhir(monkeypatch):
    requests = []
    resources = {
        "Patient/p1": {"resourceType": "Patient", "id": "p1", "identifier": [{"value": "MRN-77"}]},
        "Patient/p2": {"resourceType": "Patient", "id": "p2"},
        "Encounter/e1": {"resourceType": "Encounter", "id": "e1", "subject": {"reference": "Patient/p1"}, "period": {"start": "2024-02-01"}},
        "Observation/o1": {"resourceType": "Observation", "id": "o1", "subject": {"reference": "Patient/p1"}, "encounter": {"reference": "Encounter/e1"}, "effectiveDateTime": "2024-02-02", "valueString": "raw"},
    }

    def handler(request):
        requests.append(request)
        path = request.url.path.removeprefix("/fhir/")
        if request.method == "POST":
            pytest.fail("MedAgentBench POST must not send an HTTP request")
        if path.endswith("/$everything"):
            patient_resources = [r for r in resources.values() if r.get("id") == "p1" or r.get("subject", {}).get("reference") == "Patient/p1"]
            return httpx.Response(200, json={"resourceType": "Bundle", "entry": [{"resource": r} for r in patient_resources]})
        if path in resources:
            return httpx.Response(200, json=resources[path])
        patient_filter = request.url.params.get("patient", "").removeprefix("Patient/")
        ids = request.url.params.get("_id", "").split(",")
        if path == "Patient":
            matched = [r for r in resources.values() if r["resourceType"] == "Patient" and (not ids or ids == [""] or r["id"] in ids)]
        else:
            matched = [r for r in resources.values() if r["resourceType"] == path and r.get("subject", {}).get("reference", "").removeprefix("Patient/") in patient_filter.split(",")]
        return httpx.Response(200, json={"resourceType": "Bundle", "entry": [{"resource": r} for r in matched]})

    real_client = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: real_client(transport=httpx.MockTransport(handler), **kwargs))
    return requests, resources


def instance():
    return EnvironmentInstance(instance_id="instance", endpoint="http://fixture/fhir/", environment="MedAgentBench")


def task():
    return Task(id="task", patient_id="p1", environment="MedAgentBench", round_index=0, source_method="medevolve", instruction="Review", task_type="诊断与解释", reference_answer="", evidence=[])


@pytest.mark.asyncio
async def test_functions_are_unmodified_official_declaration():
    declaration = (Path(__file__).parents[2] / "environments/medagentbench-functions.json").read_bytes()
    assert hashlib.sha256(declaration).hexdigest() == (
        "c977266db5eca75182c0d12cac1962b33ae6b5a54b9029ceb104e1a93b4c38c1"
    )
    assert len(load_functions()) == 9


@pytest.mark.asyncio
async def test_native_get_post_finish_and_instance_boundary(fhir):
    requests, _ = fhir
    tools = await get_tools(instance(), ["p1"])
    assert isinstance(tools, NativeTools)
    assert tools.functions == load_functions()
    get = await tools.execute("GET " + tools.endpoint + "Observation?patient=Patient%2Fp1&category=laboratory")
    assert get["kind"] == "GET"
    assert "_format=json" in get["request"]["url"]
    assert "Here is the response from the GET request:\n" in get["reply"]
    assert get["data"]["resourceType"] == "Bundle"
    assert requests[-1].url.params["category"] == "laboratory"
    payload = {"resourceType": "MedicationRequest", "subject": {"reference": "Patient/p1"}}
    result = await tools.execute("POST " + tools.endpoint + "MedicationRequest\n" + json.dumps(payload))
    assert result["kind"] == "POST" and result["request"]["payload"] == payload
    assert "accepted and executed successfully" in result["reply"]
    mismatched = await tools.execute("POST " + tools.endpoint + "Observation\n" + json.dumps(payload))
    assert "accepted and executed successfully" in mismatched["reply"]
    assert len(requests) == 1
    assert (await tools.execute("POST " + tools.endpoint + "MedicationRequest\n{"))["reply"] == "Invalid POST request"
    assert (await tools.execute("FINISH([\"ok\", 2])"))["answers"] == ["ok", 2]
    with pytest.raises(PatientAccessError):
        await tools.execute("GET http://elsewhere/fhir/Observation?patient=p1")
    with pytest.raises(PatientAccessError):
        await tools.execute("GET " + tools.endpoint + "Observation?patient=p2")
    with pytest.raises(PatientAccessError):
        await tools.execute("POST " + tools.endpoint + "MedicationRequest\n" + json.dumps({"resourceType": "MedicationRequest", "subject": {"reference": "Patient/p2"}}))
    with pytest.raises(PatientAccessError):
        await tools.execute("POST " + tools.endpoint + "Observation\n" + json.dumps({"resourceType": "Observation", "valueString": "no owner"}))
    with pytest.raises(PatientAccessError):
        await tools.execute("POST " + tools.endpoint + "Observation\n[1, 2]")
    with pytest.raises(PatientAccessError):
        await tools.execute("POST " + tools.endpoint + "Observation\n" + json.dumps({
            "resourceType": "Observation", "subject": {"reference": "Patient/p1"},
            "patient": {"reference": "Patient/p2"},
        }))
    both_allowed = NativeTools(tools.endpoint, ["p1", "p2"], load_functions())
    with pytest.raises(PatientAccessError):
        await both_allowed.execute("POST " + both_allowed.endpoint + "Observation\n" + json.dumps({
            "resourceType": "Observation", "subject": {"reference": "Patient/p1"},
            "patient": {"reference": "Patient/p2"},
        }))
    with pytest.raises(PatientAccessError):
        await tools.execute("POST " + tools.endpoint + "Observation\n" + json.dumps({
            "resourceType": "Observation", "subject": {"reference": "Patient/p1"},
            "extension": [{"valueReference": {"reference": "Patient/p2"}}],
        }))
    assert len(requests) == 1


@pytest.mark.asyncio
async def test_get_preserves_explicit_scope_query_and_returns_official_http_error(monkeypatch):
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(503, text="service unavailable")

    real_client = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: real_client(
        transport=httpx.MockTransport(handler), **kwargs))
    tools = NativeTools("http://fixture/fhir/", ["p1"], load_functions())
    action = "GET " + tools.endpoint + "Observation?subject=Patient%2Fp1&category=laboratory"
    result = await tools.execute(action)
    assert seen[0].url.params.get_list("subject") == ["Patient/p1"]
    assert seen[0].url.params["category"] == "laboratory"
    assert result["reply"].startswith("Error in sending the GET request: ")
    assert "503" in result["reply"]
    assert result["failure_category"] == "environment_error"


@pytest.mark.asyncio
async def test_patient_search_filters_results_and_trajectory_keeps_raw_resources(fhir):
    requests, resources = fhir
    tools = await get_tools(instance(), ["p1"])
    patient_result = (
        await tools.execute(
            "GET " + tools.endpoint + "Patient?name=Smith&given=Jane&given=Mary"
        )
    )["data"]
    assert [entry["resource"]["id"] for entry in patient_result["entry"]] == ["p1"]
    assert requests[-1].url.params["name"] == "Smith"
    assert requests[-1].url.params.get_list("given") == ["Jane", "Mary"]
    trajectory = await prepare_trajectory(instance(), "p1")
    assert trajectory.patient_id == "p1"
    assert [source.source_id for source in trajectory.sources] == ["Encounter/e1", "Observation/o1", "Patient/p1"]
    assert json.loads(trajectory.sources[-1].statement) == resources["Patient/p1"]
    assert "MRN-77" in trajectory.text
    assert "Patient/p1" in trajectory.text and "Encounter/e1" in trajectory.text
    assert "p2" not in json.dumps([source.model_dump() for source in trajectory.sources])
    assert await read_evidence(instance(), "p1", "Observation/o1") == resources["Observation/o1"]
    state = await read_state(instance(), task())
    assert {resource["id"] for resource in state["resources"]} == {"p1", "e1", "o1"}
    assert state["operation_semantics"] == {"post": "accept_request_without_persistence", "acceptance_proves_database_write": False}
