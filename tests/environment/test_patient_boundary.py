"""Exercise SDK tools through a native-protocol HTTP fixture, without a model."""

import base64
import copy
import json

import httpx
import pytest
from agents.tool_context import ToolContext

from contracts import EnvironmentInstance, Task
from environments import get_tools, prepare_trajectory, read_evidence, read_state
from environments.errors import PatientAccessError, UnsupportedDecisionPoint


def bundle(resources, next_url=None):
    result = {"resourceType": "Bundle", "entry": [{"resource": r} for r in resources]}
    if next_url:
        result["link"] = [{"relation": "next", "url": next_url}]
    return result


@pytest.fixture
def native_fhir(monkeypatch):
    initial = {
        "Patient/a": {"resourceType": "Patient", "id": "a"},
        "Patient/b": {"resourceType": "Patient", "id": "b"},
        "Encounter/ea": {
            "resourceType": "Encounter",
            "id": "ea",
            "subject": {"reference": "Patient/a"},
        },
        "Encounter/eb": {
            "resourceType": "Encounter",
            "id": "eb",
            "subject": {"reference": "Patient/b"},
        },
        "DocumentReference/note": {
            "resourceType": "DocumentReference",
            "id": "note",
            "subject": {"reference": "Patient/a"},
            "date": "2025-01-02T01:00:00+08:00",
            "content": [
                {
                    "attachment": {
                        "contentType": "text/plain",
                        "data": base64.b64encode(b"original note").decode(),
                    }
                }
            ],
        },
        "Observation/lab": {
            "resourceType": "Observation",
            "id": "lab",
            "subject": {"reference": "Patient/a"},
            "effectiveDateTime": "2025-01-01T20:00:00Z",
            "valueString": "original lab",
        },
    }
    stores = {
        "instance-one": copy.deepcopy(initial),
        "instance-two": copy.deepcopy(initial),
    }
    requests = []
    override = {"response": None}

    def handle(request):
        requests.append(request)
        if override["response"]:
            response = override["response"]
            if callable(response):
                response = response(request)
            if isinstance(response, httpx.Response):
                return response
            return httpx.Response(200, json=response)
        store = stores[request.url.host]
        path = request.url.path.removeprefix("/fhir/")
        if request.method == "POST":
            resource = json.loads(request.content)
            resource["id"] = f"written-{len(store)}"
            store[f"{resource['resourceType']}/{resource['id']}"] = resource
            return httpx.Response(201, json=resource)
        if path.endswith("/$everything"):
            identifier = path.split("/")[1]
            resources = [
                r
                for r in store.values()
                if r.get("id") == identifier
                or r.get("subject", {}).get("reference") == f"Patient/{identifier}"
            ]
            return httpx.Response(200, json=bundle(resources))
        if "/" in path:
            if path not in store:
                return httpx.Response(404, json={"resourceType": "OperationOutcome"})
            return httpx.Response(200, json=store[path])
        identifiers = request.url.params.get(
            "patient", request.url.params.get("_id", "")
        ).split(",")
        resources = [
            r
            for r in store.values()
            if r["resourceType"] == path
            and (
                r.get("subject", {}).get("reference", "").removeprefix("Patient/")
                in identifiers
                or (path == "Patient" and r["id"] in identifiers)
            )
        ]
        return httpx.Response(200, json=bundle(resources))

    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kwargs: real_client(transport=httpx.MockTransport(handle), **kwargs),
    )
    return stores, requests, override


def instance(host="instance-one"):
    return EnvironmentInstance(
        instance_id=host, endpoint=f"http://{host}/fhir/", environment="PhysicianBench"
    )


async def invoke(tools, name, **params):
    tool = next(t for t in tools if t.name == name)
    arguments = json.dumps(params)
    context = ToolContext(
        context=None, tool_name=name, tool_call_id="test", tool_arguments=arguments
    )
    result = await tool.on_invoke_tool(context, arguments)
    return json.loads(result) if isinstance(result, str) else result


def task(**extra):
    return Task(
        id="task",
        patient_id="a",
        environment="PhysicianBench",
        round_index=0,
        source_method="medevolve",
        instruction="Review",
        task_type="诊断与解释",
        reference_answer="reference",
        evidence=[],
        **extra,
    )


@pytest.mark.asyncio
async def test_evidence_read_returns_exact_record_and_rejects_other_patient(native_fhir):
    record = await read_evidence(instance(), "a", "Observation/lab")
    assert record["valueString"] == "original lab"
    with pytest.raises(PatientAccessError):
        await read_evidence(instance(), "a", "Encounter/eb")


@pytest.mark.asyncio
async def test_malformed_native_arguments_return_error_then_allow_retry(native_fhir):
    tools = await get_tools(instance(), ["a"])
    tool = next(t for t in tools if t.name == "fhir_observation_search_labs")
    arguments = '{"patient": a}'
    context = ToolContext(
        context=None, tool_name=tool.name, tool_call_id="malformed", tool_arguments=arguments
    )
    result = json.loads(await tool.on_invoke_tool(context, arguments))
    assert "error" in result
    assert not native_fhir[1]
    corrected = await invoke(tools, tool.name, patient="a")
    assert corrected["entries"][0]["id"] == "lab"


@pytest.mark.asyncio
async def test_tool_result_limit_truncates_fhir_results(native_fhir):
    _, _, override = native_fhir
    records = [
        {"resourceType": "Observation", "id": f"obs-{n}",
         "subject": {"reference": "Patient/a"}, "valueString": "x" * 200}
        for n in range(80)
    ]
    override["response"] = bundle(records)

    async def raw(limit=None):
        tools = (await get_tools(instance(), ["a"]) if limit is None else
                 await get_tools(instance(), ["a"], tool_result_limit=limit))
        tool = next(t for t in tools if t.name == "fhir_observation_search_labs")
        arguments = json.dumps({"patient": "a"})
        context = ToolContext(
            context=None, tool_name=tool.name, tool_call_id="limit", tool_arguments=arguments
        )
        return await tool.on_invoke_tool(context, arguments)

    for limit, delivered in ((10000, await raw()), (500, await raw(500))):
        result = json.loads(delivered)
        returned = result["entries"]
        assert len(delivered) <= limit
        assert result["truncation"]["truncated"] is True
        assert result["truncation"]["total_records"] == 80
        assert result["truncation"]["returned_records"] == len(returned)
        assert result["truncation"]["message"]
        # 返回记录必须与原始记录逐字段一致，且不完整交付。
        assert returned == records[:len(returned)]
        assert len(returned) < 80
    whole_raw = await raw(200000)
    whole = json.loads(whole_raw)
    assert len(whole_raw) > 10000
    assert "truncation" not in whole
    assert whole["entries"] == records


@pytest.mark.asyncio
async def test_tool_result_limit_below_minimum_is_rejected_before_http(native_fhir):
    with pytest.raises(ValueError):
        await get_tools(instance(), ["a"], tool_result_limit=1)
    assert not native_fhir[1]


@pytest.mark.asyncio
async def test_oversized_tool_error_returns_valid_limited_json(native_fhir):
    native_fhir[2]["response"] = lambda request: httpx.Response(
        400, json={"resourceType": "OperationOutcome"}
    )
    tools = await get_tools(instance(), ["a"], tool_result_limit=256)
    tool = next(t for t in tools if t.name == "fhir_observation_search_labs")
    arguments = json.dumps({"patient": "a", "code": "x" * 600})
    context = ToolContext(
        context=None, tool_name=tool.name, tool_call_id="error-limit",
        tool_arguments=arguments,
    )
    raw = await tool.on_invoke_tool(context, arguments)
    result = json.loads(raw)
    assert len(raw) <= 256
    assert "error" in result
    assert result["truncation"]["truncated"] is True
    assert "succeeded" not in raw
    assert "exceeds the limit" in raw
    assert "developer.mozilla.org" not in raw


@pytest.mark.asyncio
async def test_oversized_rejected_write_reports_error_not_success(native_fhir):
    tools = await get_tools(instance(), ["a"], tool_result_limit=256)
    result = await invoke(
        tools, "fhir_appointment_create",
        patient_reference="Patient/a", practitioner_reference="Practitioner/x",
        description="Cardiology consult",
    )
    assert "succeeded" not in json.dumps(result)
    assert result["error"] == "Tool call failed; error message exceeds the limit."
    assert all(r.method != "POST" for r in native_fhir[1])


@pytest.mark.asyncio
async def test_native_reads_writes_state_and_instance_isolation(native_fhir):
    tools = await get_tools(instance(), ["a"])
    assert len(tools) == 13
    assert all("base_url" not in t.params_json_schema["properties"] for t in tools)
    patients = await invoke(tools, "fhir_patient_search_demographics")
    assert [r["id"] for r in patients["entries"]] == ["a"]
    notes = await invoke(tools, "fhir_document_reference_search_clinical_notes")
    assert notes["entries"][0]["content"][0]["attachment"]["data"] == "original note"
    created = await invoke(
        tools,
        "fhir_medication_request_create",
        patient_reference="Patient/a",
        medication_display="example",
        requester_reference="Practitioner/p",
        dose_value=5,
        dose_unit="mg",
    )
    assert created["dosageInstruction"][0]["doseAndRate"][0]["doseQuantity"] == {
        "value": 5,
        "unit": "mg",
    }
    first = await read_state(instance(), task())
    second = await read_state(instance("instance-two"), task())
    assert first["patient_id"] == "a"
    assert first["environment"] == "PhysicianBench"
    assert {r["resourceType"] for r in first["resources"]} <= {
        "Communication",
        "ServiceRequest",
        "MedicationRequest",
        "Appointment",
    }
    assert any(r.get("id") == created["id"] for r in first["resources"])
    assert not any(
        r["resourceType"] == "MedicationRequest" for r in second["resources"]
    )
    assert native_fhir[0]["instance-one"]["Patient/b"] == {
        "resourceType": "Patient",
        "id": "b",
    }


@pytest.mark.asyncio
async def test_disallowed_patient_queries_and_writes_never_reach_http(native_fhir):
    tools = await get_tools(instance(), ["a"])
    for name, args in [
        ("fhir_observation_search_labs", {"patient": "b"}),
        ("fhir_condition_search_problems", {"subject": "Patient/b"}),
        ("fhir_observation_search_labs", {"patient": "a,b"}),
        (
            "fhir_medication_request_create",
            {
                "patient_reference": "Patient/b",
                "medication_display": "x",
                "requester_reference": "Practitioner/p",
            },
        ),
        (
            "fhir_communication_create_message",
            {
                "patient_reference": "Patient/a",
                "message_text": "x",
                "recipient_reference": "Patient/b",
            },
        ),
    ]:
        denied = await invoke(tools, name, **args)
        assert "PatientAccessError" in denied["error"]
        assert "entries" not in denied
    assert not native_fhir[1]
    corrected = await invoke(tools, "fhir_observation_search_labs", patient="a")
    assert corrected["entries"][0]["id"] == "lab"


@pytest.mark.asyncio
async def test_foreign_linked_resources_block_queries_and_writes(native_fhir):
    tools = await get_tools(instance(), ["a"])
    denied = await invoke(
        tools,
        "fhir_document_reference_search_clinical_notes",
        encounter="Encounter/eb",
    )
    assert "PatientAccessError" in denied["error"]
    denied = await invoke(
        tools,
        "fhir_service_request_create",
        patient_reference="Patient/a",
        code_code="x",
        code_display="x",
        requester_reference="Practitioner/p",
        encounter_reference="Encounter/eb",
    )
    assert "PatientAccessError" in denied["error"]
    assert all(r.method != "POST" for r in native_fhir[1])
    tools = await get_tools(instance(), ["a", "b"])
    denied = await invoke(
        tools,
        "fhir_service_request_create",
        patient_reference="Patient/a",
        code_code="x",
        code_display="x",
        requester_reference="Practitioner/p",
        encounter_reference="Encounter/eb",
    )
    assert "PatientAccessError" in denied["error"]


@pytest.mark.asyncio
async def test_response_scope_and_pagination_are_checked(native_fhir):
    tools = await get_tools(instance(), ["a"])
    native_fhir[2]["response"] = bundle([native_fhir[0]["instance-one"]["Patient/b"]])
    denied = await invoke(tools, "fhir_patient_search_demographics")
    assert "PatientAccessError" in denied["error"]
    assert "entries" not in denied
    native_fhir[2]["response"] = bundle(
        [
            {
                "resourceType": "Patient",
                "id": "a",
                "contained": [{"resourceType": "Patient", "id": "b"}],
            }
        ]
    )
    denied = await invoke(tools, "fhir_patient_search_demographics")
    assert "PatientAccessError" in denied["error"]
    assert "entries" not in denied
    native_fhir[2]["response"] = bundle([], "http://another-instance/fhir/Observation")
    denied = await invoke(tools, "fhir_observation_search_labs")
    assert "PatientAccessError" in denied["error"] and "pagination" in denied["error"]


@pytest.mark.asyncio
async def test_trajectory_retains_raw_sources_and_orders_by_actual_time(native_fhir):
    trajectory = await prepare_trajectory(instance(), "a")
    # The note timestamp is earlier than the lab after timezone normalization.
    assert trajectory.text.index("DocumentReference/note") < trajectory.text.index(
        "Observation/lab"
    )
    assert "original note" in trajectory.text
    assert "[2025-01-01T20:00:00Z] Observation/lab" in trajectory.text
    assert "[undated] Patient/a" in trajectory.text
    assert "\"subject\"" not in trajectory.text
    assert "\"resourceType\"" not in trajectory.text
    evidence = next(
        e for e in trajectory.sources if e.source_id == "DocumentReference/note"
    )
    assert (
        json.loads(evidence.statement)
        == native_fhir[0]["instance-one"]["DocumentReference/note"]
    )
    assert all("Patient/b" != e.source_id for e in trajectory.sources)


@pytest.mark.asyncio
async def test_trajectory_collects_all_native_pages(native_fhir):
    store = native_fhir[0]["instance-one"]

    def page(request):
        if request.url.params.get("page") == "2":
            return bundle([store["Observation/lab"]])
        return bundle(
            [store["Patient/a"], store["DocumentReference/note"]],
            "http://instance-one/fhir/?page=2",
        )

    native_fhir[2]["response"] = page
    trajectory = await prepare_trajectory(instance(), "a")
    assert {e.source_id for e in trajectory.sources} == {
        "Patient/a",
        "DocumentReference/note",
        "Observation/lab",
    }
    assert len(native_fhir[1]) == 2


@pytest.mark.asyncio
async def test_decision_point_refused_before_access(native_fhir):
    with pytest.raises(UnsupportedDecisionPoint):
        await get_tools(instance(), ["a"], "2025-01-01T00:00:00Z")
    with pytest.raises(UnsupportedDecisionPoint):
        await read_state(instance(), task(decision_point="2025-01-01T00:00:00Z"))
    assert not native_fhir[1]
