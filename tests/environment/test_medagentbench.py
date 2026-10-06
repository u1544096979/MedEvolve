"""MedAgentBench native protocol and patient-scoped raw data access."""

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


def test_function_declarations_keep_upstream_structure_and_provenance():
    functions = load_functions()
    assert len(functions) == 9
    assert [function["name"] for function in functions] == [
        "GET {api_base}/Condition",
        "GET {api_base}/Observation",
        "GET {api_base}/Observation",
        "POST {api_base}/Observation",
        "GET {api_base}/MedicationRequest",
        "POST {api_base}/MedicationRequest",
        "GET {api_base}/Procedure",
        "POST {api_base}/ServiceRequest",
        "GET {api_base}/Patient",
    ]
    assert set(functions[0]["parameters"]["properties"]) == {"category", "patient"}
    assert functions[0]["parameters"]["required"] == ["patient"]
    assert set(functions[4]["parameters"]["properties"]) == {"category", "date", "patient"}
    assert functions[4]["parameters"]["required"] == ["patient"]
    assert {
        index: (set(functions[index]["parameters"]["properties"]),
                functions[index]["parameters"]["required"])
        for index in (3, 5, 7)
    } == {
        3: ({"resourceType", "category", "code", "effectiveDateTime", "status",
             "valueString", "subject"},
            ["resourceType", "category", "code", "effectiveDateTime", "status",
             "valueString", "subject"]),
        5: ({"resourceType", "medicationCodeableConcept", "authoredOn",
             "dosageInstruction", "status", "intent", "subject"},
            ["resourceType", "medicationCodeableConcept", "authoredOn",
             "dosageInstruction", "status", "intent", "subject"]),
        7: ({"resourceType", "code", "authoredOn", "status", "intent", "priority",
             "subject", "note", "occurrenceDateTime"},
            ["resourceType", "code", "authoredOn", "status", "intent", "priority",
             "subject"]),
    }
    assert "category" in functions[0]["description"]
    assert "status" not in functions[0]["description"]
    assert "status" not in functions[4]["description"]
    provenance = (
        Path(__file__).parents[2] / "environments/medagentbench.py"
    ).read_text(encoding="utf-8")
    assert "Stanford Machine Learning Group" in provenance
    assert "99260117137b09f04837a8c18d18a1107efa55ae" in provenance
    assert "MIT" in provenance
    assert "locally adapted" in provenance


PARAMETER_ERROR_BODY = {
    "resourceType": "OperationOutcome",
    "issue": [{"severity": "error", "code": "invalid",
               "diagnostics": "Unknown search parameter status"}],
}


def mock_fhir(monkeypatch, handler):
    real_client = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: real_client(
        transport=httpx.MockTransport(handler), **kwargs))
    return NativeTools("http://fixture/fhir/", ["p1"], load_functions())


@pytest.mark.asyncio
@pytest.mark.parametrize(("status", "body", "expected_category"), [
    (400, PARAMETER_ERROR_BODY, "tool_error"),
    (400, {"resourceType": "OperationOutcome", "issue": [{
        "severity": "error", "code": "invalid",
        "details": {"text": "Invalid value in search parameter status"}}]}, "tool_error"),
    (400, "Bad Request", "tool_error"),
    (401, "unauthorized", "environment_error"),
    (403, "forbidden", "environment_error"),
    (404, "not found", "environment_error"),
    (503, "service unavailable", "environment_error"),
])
async def test_get_failure_classification(monkeypatch, status, body, expected_category):
    def handler(request):
        if isinstance(body, dict):
            return httpx.Response(status, json=body)
        return httpx.Response(status, text=body)

    tools = mock_fhir(monkeypatch, handler)
    result = await tools.execute("GET " + tools.endpoint + "Condition?patient=Patient%2Fp1")
    assert result["kind"] == "GET"
    assert result["failure_category"] == expected_category
    if expected_category == "tool_error":
        assert result["status_code"] == 400
        assert result["reply"].startswith("HTTP 400: ")
        assert "Correct the query parameters and retry." in result["reply"]
    else:
        assert result["reply"].startswith("Error in sending the GET request: ")
        assert "status_code" not in result


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [
    httpx.ConnectError("connect failed"),
    httpx.ReadTimeout("read timed out"),
])
async def test_get_transport_failure_is_environment_error(monkeypatch, error):
    def handler(request):
        raise error

    tools = mock_fhir(monkeypatch, handler)
    result = await tools.execute("GET " + tools.endpoint + "Condition?patient=Patient%2Fp1")
    assert result["kind"] == "GET"
    assert result["failure_category"] == "environment_error"
    assert result["reply"].startswith("Error in sending the GET request: ")
    assert "status_code" not in result


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
async def test_model_can_follow_scoped_official_root_pagination_link(monkeypatch):
    seen = []
    next_url = (
        "http://fixture/fhir?_getpages=opaque%2Ftoken&_getpagesoffset=20&_count=20"
    )

    def handler(request):
        seen.append(request)
        if request.url.path == "/fhir/Observation":
            return httpx.Response(200, json={
                "resourceType": "Bundle", "entry": [],
                "link": [{"relation": "next", "url": next_url}],
            })
        return httpx.Response(200, json={
            "resourceType": "Bundle", "entry": [{"resource": {
                "resourceType": "Observation", "id": "o2",
                "subject": {"reference": "Patient/p1"},
            }}],
        })

    real_client = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: real_client(
        transport=httpx.MockTransport(handler), **kwargs))
    tools = NativeTools("http://fixture/fhir/", ["p1"], load_functions())

    first = await tools.execute(
        "GET " + tools.endpoint + "Observation?patient=Patient%2Fp1"
    )
    assert first["data"]["link"][0]["url"] == next_url
    assert len(seen) == 1

    second = await tools.execute("GET " + next_url)
    assert second["data"]["entry"][0]["resource"]["id"] == "o2"
    assert len(seen) == 2
    assert seen[1].url.path == "/fhir"
    assert "_getpages=opaque%2Ftoken&_getpagesoffset=20&_count=20" in seen[1].url.query.decode()
    assert "patient=" not in seen[1].url.query.decode()


@pytest.mark.asyncio
async def test_root_pagination_checks_patient_scope_and_instance(monkeypatch):
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json={
            "resourceType": "Bundle", "entry": [{"resource": {
                "resourceType": "Observation", "id": "leak",
                "subject": {"reference": "Patient/p2"},
            }}],
        })

    real_client = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: real_client(
        transport=httpx.MockTransport(handler), **kwargs))
    tools = NativeTools("http://fixture/fhir/", ["p1"], load_functions())
    with pytest.raises(PatientAccessError):
        await tools.execute(
            "GET http://fixture/fhir?_getpages=token&_getpagesoffset=20"
        )
    assert len(seen) == 1
    with pytest.raises(PatientAccessError):
        await tools.execute(
            "GET http://other/fhir?_getpages=token&_getpagesoffset=20"
        )
    with pytest.raises(PatientAccessError):
        await tools.execute("GET http://fixture/fhir")
    assert len(seen) == 1


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
