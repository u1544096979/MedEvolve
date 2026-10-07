"""FHIR-AgentBench 原生检索工具机制测试。

全部使用 httpx.MockTransport，不连接 Docker、PostgreSQL 或 http://127.0.0.1:18086。
"""

import copy
import json

import httpx
import pytest
from agents.tool_context import ToolContext

from contracts import EnvironmentConfig, EnvironmentInstance
from environments import (
    fhir_agentbench,
    get_tools,
    read_evidence,
    release_environment,
)
from environments.errors import (
    EnvironmentUnavailable,
    PatientAccessError,
    UnsupportedDecisionPoint,
)
from environments.fhir import FHIRClient
from environments.fhir_agentbench import (
    CORRECTABLE_ERRORS,
    REFERENCE_ROUNDS,
    SUPPORTED_TYPES,
    PatientOutOfScope,
    UnsupportedResourceType,
    _check_resource_type,
    get_native_tools,
    read_patient_resources,
    read_source,
)

# 官方 tools/resource_tools.py 在提交 bbb42909a5a7eb907d1cd91f72a560729e7037ea 注册的 schema 原文。
OFFICIAL_TOOL_SCHEMAS = [
    (
        "get_resources_by_resource_id",
        "Get FHIR resources by resource ID",
        {
            "type": "object",
            "properties": {
                "resource_type": {"type": "string", "description": "FHIR resource type"},
                "resource_id": {"type": "string", "description": "FHIR resource ID"},
            },
            "required": ["resource_type", "resource_id"],
        },
    ),
    (
        "get_resources_by_patient_fhir_id",
        "Get FHIR resources by patient FHIR ID",
        {
            "type": "object",
            "properties": {
                "resource_type": {"type": "string", "description": "FHIR resource type"},
                "patient_fhir_id": {"type": "string", "description": "Patient FHIR ID"},
            },
            "required": ["resource_type", "patient_fhir_id"],
        },
    ),
]


def record(resource_type, identifier, **fields):
    """带 text 与 meta 的原始记录：工具输出要移除，原始证据要保留。"""
    resource = {
        "resourceType": resource_type,
        "id": identifier,
        "text": {"status": "generated", "div": f"<div>{resource_type}/{identifier}</div>"},
        "meta": {"versionId": "1", "lastUpdated": "2136-04-22T18:47:00-04:00"},
    }
    resource.update(fields)
    return resource


def initial_store():
    return {
        "Patient/a": record("Patient", "a"),
        "Patient/b": record("Patient", "b"),
        "Encounter/ea": record(
            "Encounter",
            "ea",
            subject={"reference": "Patient/a"},
            location=[{"location": {"reference": "Location/loc-a"}}],
        ),
        "Encounter/eb": record(
            "Encounter",
            "eb",
            subject={"reference": "Patient/b"},
            location=[{"location": {"reference": "Location/loc-b"}}],
        ),
        "Observation/o1": record(
            "Observation",
            "o1",
            subject={"reference": "Patient/a"},
            effectiveDateTime="2136-04-22T18:47:00-04:00",
            valueQuantity={"value": 80, "unit": "beats/min"},
        ),
        "Observation/o2": record(
            "Observation",
            "o2",
            subject={"reference": "Patient/a"},
            effectiveDateTime="2136-04-23T08:00:00-04:00",
            valueQuantity={"value": 76, "unit": "beats/min"},
        ),
        "Observation/ob": record(
            "Observation",
            "ob",
            subject={"reference": "Patient/b"},
            effectiveDateTime="2136-04-22T19:00:00-04:00",
        ),
        "MedicationRequest/mr1": record(
            "MedicationRequest",
            "mr1",
            subject={"reference": "Patient/a"},
            medicationReference={"reference": "Medication/m1"},
        ),
        "MedicationRequest/mr2": record(
            "MedicationRequest",
            "mr2",
            subject={"reference": "Patient/b"},
            medicationReference={"reference": "Medication/m2"},
        ),
        "Medication/m1": record(
            "Medication",
            "m1",
            code={"text": "patient-a drug"},
            ingredient=[{"itemReference": {"reference": "Medication/mi"}}],
        ),
        "Medication/mi": record(
            "Medication",
            "mi",
            code={"text": "patient-a ingredient"},
            ingredient=[{"itemReference": {"reference": "Medication/mii"}}],
        ),
        "Medication/mii": record("Medication", "mii", code={"text": "second hop"}),
        "Medication/m2": record("Medication", "m2", code={"text": "patient-b drug"}),
        "Location/loc-a": record("Location", "loc-a", name="ward a"),
        "Location/loc-b": record("Location", "loc-b", name="ward b"),
    }


def bundle(resources, next_url=None):
    result = {"resourceType": "Bundle", "entry": [{"resource": r} for r in resources]}
    if next_url:
        result["link"] = [{"relation": "next", "url": next_url}]
    return result


def partial_bundle(resources, total):
    """服务端声明 total 条匹配，却提前结束分页且不给 next 链接的 Bundle。"""
    return {
        "resourceType": "Bundle",
        "total": total,
        "entry": [{"resource": r} for r in resources],
    }


def owner(resource):
    return resource.get("subject", {}).get("reference")


def search(store, resource_type, params):
    resources = sorted(
        (r for r in store.values() if r["resourceType"] == resource_type),
        key=lambda r: r["id"],
    )
    if params.get("_id"):
        wanted = set(params["_id"].split(","))
        return [r for r in resources if r["id"] in wanted]
    if params.get("patient"):
        wanted = set(params["patient"].split(","))
        return [r for r in resources if owner(r) in {f"Patient/{p}" for p in wanted}]
    # 未带患者范围约束的检索：真实服务会返回全库记录，测试用它的存在与否判定是否越界。
    return resources


class FakeFHIR:
    """按主机名隔离的 fake FHIR 服务，记录每个请求。"""

    def __init__(self):
        self.stores = {}
        self.requests = []
        self.override = None

    def store_for(self, host):
        return self.stores.setdefault(host, copy.deepcopy(initial_store()))

    def handle(self, request):
        self.requests.append(request)
        if self.override is not None:
            return self.override(request)
        return self.respond(request)

    def respond(self, request):
        store = self.store_for(request.url.host)
        path = request.url.path.removeprefix("/fhir/")
        if path.endswith("/$everything"):
            identifier = path.split("/")[1]
            resources = [
                r
                for r in store.values()
                if (r["resourceType"] == "Patient" and r["id"] == identifier)
                or owner(r) == f"Patient/{identifier}"
            ]
            return httpx.Response(
                200,
                json=bundle(sorted(resources, key=lambda r: (r["resourceType"], r["id"]))),
            )
        if "/" in path:
            if path not in store:
                return httpx.Response(404, json={"resourceType": "OperationOutcome"})
            return httpx.Response(200, json=store[path])
        params = request.url.params
        resources = search(store, path, params)
        if path == "Observation" and "_id" not in params:
            if params.get("page") == "2":
                return httpx.Response(200, json=bundle(search(store, path, params)[1:]))
            if len(resources) > 1:
                next_url = str(request.url.copy_merge_params({"page": "2"}))
                return httpx.Response(200, json=bundle(resources[:1], next_url))
        return httpx.Response(200, json=bundle(resources))


@pytest.fixture
def fake(monkeypatch):
    fake = FakeFHIR()
    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kwargs: real_client(transport=httpx.MockTransport(fake.handle), **kwargs),
    )
    monkeypatch.setattr(fhir_agentbench, "_READERS", {})
    return fake


def instance(host="instance-one"):
    return EnvironmentInstance(
        instance_id=host, endpoint=f"http://{host}/fhir/", environment="FHIR-AgentBench"
    )


async def invoke(tools, name, **params):
    tool = next(t for t in tools if t.name == name)
    arguments = json.dumps(params)
    context = ToolContext(
        context=None, tool_name=name, tool_call_id="test", tool_arguments=arguments
    )
    result = await tool.on_invoke_tool(context, arguments)
    return json.loads(result) if isinstance(result, str) else result


async def invoke_raw(tools, name, **params):
    """返回工具原始返回值，用于钉住字符串契约。"""
    tool = next(t for t in tools if t.name == name)
    arguments = json.dumps(params)
    context = ToolContext(
        context=None, tool_name=name, tool_call_id="test", tool_arguments=arguments
    )
    return await tool.on_invoke_tool(context, arguments)


@pytest.mark.asyncio
async def test_taskcraft_binds_fhir_tools_with_patient_scope(fake):
    from methods.taskcraft.environment import bind_generation_tools

    original = get_native_tools(instance(), ["a"])
    tools = bind_generation_tools(original)
    assert [tool.name for tool in tools] == [tool.name for tool in original]
    assert all(first is second for first, second in zip(tools, original))
    by_patient = await invoke(
        tools, "get_resources_by_patient_fhir_id",
        resource_type="Observation", patient_fhir_id="a",
    )
    assert [resource["id"] for resource in by_patient["Observation"]] == ["o1", "o2"]
    by_id = await invoke(
        tools, "get_resources_by_resource_id",
        resource_type="Observation", resource_id="o1",
    )
    assert by_id["Observation"][0]["id"] == "o1"
    denied = await invoke(
        tools, "get_resources_by_patient_fhir_id",
        resource_type="Observation", patient_fhir_id="b",
    )
    assert "outside the allowed patient scope" in denied["error"]


def paths(fake):
    return [request.url.path for request in fake.requests]


@pytest.mark.asyncio
async def test_two_pages_returned_and_text_meta_removed_from_tool_output(fake):
    tools = get_native_tools(instance(), ["a"])
    result = await invoke(
        tools,
        "get_resources_by_patient_fhir_id",
        resource_type="Observation",
        patient_fhir_id="a",
    )
    assert [r["id"] for r in result["Observation"]] == ["o1", "o2"]
    assert all("text" not in r and "meta" not in r for r in result["Observation"])
    assert result["Observation"][0]["valueQuantity"] == {"value": 80, "unit": "beats/min"}
    # 分页链接被跟随
    assert sum(1 for path in paths(fake) if path.endswith("Observation")) == 2


@pytest.mark.asyncio
async def test_raw_evidence_keeps_text_and_meta(fake):
    evidence = await read_source(instance(), "a", "Observation/o1")
    assert evidence["meta"]["versionId"] == "1"
    assert evidence["text"]["status"] == "generated"
    assert evidence["valueQuantity"] == {"value": 80, "unit": "beats/min"}


@pytest.mark.asyncio
async def test_empty_result_is_an_empty_object(fake):
    tools = get_native_tools(instance(), ["a"])
    result = await invoke(
        tools,
        "get_resources_by_patient_fhir_id",
        resource_type="Procedure",
        patient_fhir_id="a",
    )
    assert result == {}
    assert not isinstance(result, list)


@pytest.mark.asyncio
async def test_out_of_scope_patient_is_reported_as_a_correctable_error(fake):
    """越界患者既要报错，也不能终止整轮运行：模型换一个开放患者就能继续。"""
    tools = get_native_tools(instance(), ["a"])
    for patient_fhir_id in ("b", "nonexistent-patient-id"):
        raw = await invoke_raw(
            tools,
            "get_resources_by_patient_fhir_id",
            resource_type="Observation",
            patient_fhir_id=patient_fhir_id,
        )
        assert isinstance(raw, str)
        payload = json.loads(raw)
        assert set(payload) == {"error"}
        assert (
            f"Patient {patient_fhir_id} is outside the allowed patient scope"
            in payload["error"]
        )
    assert not fake.requests


@pytest.mark.asyncio
async def test_record_outside_the_patient_scope_is_still_raised(fake):
    """服务端交回的患者外记录是数据完整性问题，仍走运行错误通道。"""
    tools = get_native_tools(instance(), ["a"])
    with pytest.raises(PatientAccessError) as failure:
        await invoke(
            tools, "get_resources_by_resource_id", resource_type="Observation", resource_id="ob"
        )
    assert "patient scope" in str(failure.value)
    assert not isinstance(failure.value, PatientOutOfScope)
    assert paths(fake) == ["/fhir/Observation"]


@pytest.mark.asyncio
async def test_unreferenced_shared_resources_are_reported_as_a_correctable_error(fake):
    """与开放患者无关的共享记录：只报错，且不按被拒 ID 取数。"""
    tools = get_native_tools(instance(), ["a"])
    for resource_type, identifier in (("Medication", "m2"), ("Location", "loc-b")):
        raw = await invoke_raw(
            tools,
            "get_resources_by_resource_id",
            resource_type=resource_type,
            resource_id=identifier,
        )
        assert isinstance(raw, str)
        payload = json.loads(raw)
        assert set(payload) == {"error"}
        assert (
            f"{resource_type}/{identifier} is not referenced by an allowed patient"
            in payload["error"]
        )
    # 展开可达闭包只读取开放患者可达的记录，从不按被拒 ID 取数。
    requested = {
        identifier
        for request in fake.requests
        for identifier in (request.url.params.get("_id") or "").split(",")
        if identifier
    }
    assert "m2" not in requested
    assert "loc-b" not in requested
    assert requested <= {"m1", "mi", "mii", "loc-a"}
    reachable = await invoke(
        tools, "get_resources_by_resource_id", resource_type="Medication", resource_id="m1"
    )
    assert [r["id"] for r in reachable["Medication"]] == ["m1"]
    assert reachable["Medication"][0]["code"] == {"text": "patient-a drug"}


@pytest.mark.asyncio
async def test_shared_resource_reached_through_patient_search(fake):
    tools = get_native_tools(instance(), ["a"])
    medications = await invoke(
        tools,
        "get_resources_by_patient_fhir_id",
        resource_type="Medication",
        patient_fhir_id="a",
    )
    # 共享记录按可达闭包返回：直接引用的 m1 与被它 ingredient 引用的复方成分。
    assert [r["id"] for r in medications["Medication"]] == ["m1", "mi", "mii"]
    lookups = [r for r in fake.requests if r.url.params.get("_id") == "m1"]
    assert lookups and "patient" not in lookups[-1].url.params
    locations = await invoke(
        tools,
        "get_resources_by_patient_fhir_id",
        resource_type="Location",
        patient_fhir_id="a",
    )
    assert [r["id"] for r in locations["Location"]] == ["loc-a"]


@pytest.mark.asyncio
async def test_patient_type_is_queried_by_id_without_patient_param(fake):
    tools = get_native_tools(instance(), ["a"])
    result = await invoke(
        tools,
        "get_resources_by_patient_fhir_id",
        resource_type="Patient",
        patient_fhir_id="a",
    )
    assert [r["id"] for r in result["Patient"]] == ["a"]
    request = fake.requests[-1]
    assert request.url.params.get("_id") == "a"
    assert "patient" not in request.url.params
    assert "subject" not in request.url.params


@pytest.mark.asyncio
async def test_shared_resource_reachability_follows_shared_references(fake):
    """只被另一条 Medication 的 ingredient 引用的复方成分也必须可按同一 ID 回查。"""
    direct = await read_source(instance(), "a", "Medication/m1")
    assert direct["id"] == "m1"
    ingredient = await read_source(instance(), "a", "Medication/mi")
    assert ingredient["code"] == {"text": "patient-a ingredient"}
    second_hop = await read_source(instance(), "a", "Medication/mii")
    assert second_hop["code"] == {"text": "second hop"}
    tools = get_native_tools(instance(), ["a"])
    resolved = await invoke(
        tools, "get_resources_by_resource_id", resource_type="Medication", resource_id="mii"
    )
    assert [r["id"] for r in resolved["Medication"]] == ["mii"]
    with pytest.raises(PatientAccessError):
        await read_source(instance(), "a", "Medication/m2")


@pytest.mark.asyncio
async def test_shared_reference_chain_beyond_the_round_limit_is_reported(fake):
    """可达性超过迭代上限时报错，而不是静默返回被截断的子集。"""
    store = fake.store_for("instance-one")
    chain = [f"chain-{index}" for index in range(REFERENCE_ROUNDS + 2)]
    store["MedicationRequest/mr1"]["medicationReference"] = {
        "reference": f"Medication/{chain[0]}"
    }
    for index, identifier in enumerate(chain):
        linked = chain[index + 1] if index + 1 < len(chain) else None
        fields = (
            {"ingredient": [{"itemReference": {"reference": f"Medication/{linked}"}}]}
            if linked
            else {}
        )
        store[f"Medication/{identifier}"] = record("Medication", identifier, **fields)
    with pytest.raises(EnvironmentUnavailable, match="converge"):
        await read_source(instance(), "a", f"Medication/{chain[0]}")


@pytest.mark.asyncio
async def test_shared_type_read_by_id_has_no_patient_param(fake):
    tools = get_native_tools(instance(), ["a"])
    await invoke(
        tools, "get_resources_by_resource_id", resource_type="Location", resource_id="loc-a"
    )
    lookups = [r for r in fake.requests if r.url.params.get("_id") == "loc-a"]
    assert lookups
    assert "patient" not in lookups[-1].url.params
    assert "subject" not in lookups[-1].url.params


@pytest.mark.asyncio
async def test_unsupported_resource_type_is_reported_as_a_correctable_error(fake):
    """参数错误交回模型：官方 description 固定，模型无法预知允许的类型。"""
    tools = get_native_tools(instance(), ["a"])
    raw = await invoke_raw(
        tools,
        "get_resources_by_patient_fhir_id",
        resource_type="AllergyIntolerance",
        patient_fhir_id="a",
    )
    assert isinstance(raw, str)
    payload = json.loads(raw)
    assert set(payload) == {"error"}
    assert "Unsupported FHIR resource type: AllergyIntolerance" in payload["error"]
    for resource_type in SUPPORTED_TYPES:
        assert resource_type in payload["error"]
    assert not fake.requests
    assert "MedicationAdministration" in SUPPORTED_TYPES
    assert "MedicationStatement" not in SUPPORTED_TYPES


@pytest.mark.asyncio
async def test_unsupported_resource_type_by_id_is_reported_the_same_way(fake):
    tools = get_native_tools(instance(), ["a"])
    result = await invoke(
        tools,
        "get_resources_by_resource_id",
        resource_type="AllergyIntolerance",
        resource_id="x",
    )
    assert set(result) == {"error"}
    assert "Unsupported FHIR resource type: AllergyIntolerance" in result["error"]
    for resource_type in SUPPORTED_TYPES:
        assert resource_type in result["error"]
    assert not fake.requests


def test_check_resource_type_raises_a_dedicated_value_error():
    with pytest.raises(UnsupportedResourceType) as failure:
        _check_resource_type("AllergyIntolerance")
    assert isinstance(failure.value, ValueError)
    assert "Expected one of" in str(failure.value)


def test_only_correctable_errors_are_handed_back_to_the_model():
    """可纠正的错误自成一支：分页越界、记录不匹配等仍抛给运行错误通道。"""
    assert issubclass(PatientOutOfScope, PatientAccessError)
    assert not issubclass(UnsupportedResourceType, PatientAccessError)
    assert set(CORRECTABLE_ERRORS) == {UnsupportedResourceType, PatientOutOfScope}


@pytest.mark.asyncio
async def test_tool_refuses_a_shared_record_returned_for_an_unrequested_id(fake):
    """服务端返回的记录 ID 与请求不符：不得当成完整结果交回模型。"""
    store = fake.store_for("instance-one")
    requested = "m1,mi,mii"

    def overreaching(request):
        if request.url.params.get("_id") == requested:
            return httpx.Response(
                200,
                json=bundle(
                    [
                        store["Medication/m1"],
                        store["Medication/mi"],
                        store["Medication/mii"],
                        store["Medication/m2"],
                    ]
                ),
            )
        return fake.respond(request)

    fake.override = overreaching
    tools = get_native_tools(instance(), ["a"])
    with pytest.raises(PatientAccessError) as failure:
        await invoke(
            tools,
            "get_resources_by_patient_fhir_id",
            resource_type="Medication",
            patient_fhir_id="a",
        )
    assert "unreachable Medication record" in str(failure.value)
    assert not isinstance(failure.value, PatientOutOfScope)


@pytest.mark.asyncio
async def test_pagination_link_leaving_the_instance_is_refused(fake):
    fake.override = lambda request: httpx.Response(
        200, json=bundle([], "http://another-instance/fhir/Observation?page=2")
    )
    tools = get_native_tools(instance(), ["a"])
    with pytest.raises(PatientAccessError, match="pagination") as failure:
        await invoke(
            tools,
            "get_resources_by_patient_fhir_id",
            resource_type="Observation",
            patient_fhir_id="a",
        )
    assert not isinstance(failure.value, PatientOutOfScope)


@pytest.mark.asyncio
async def test_empty_ids_stay_inside_the_open_patient_scope(fake):
    tools = get_native_tools(instance(), ["a"])
    observations = await invoke(
        tools, "get_resources_by_resource_id", resource_type="Observation", resource_id=""
    )
    assert [r["id"] for r in observations["Observation"]] == ["o1", "o2"]
    scoped = [r for r in fake.requests if r.url.path.endswith("Observation")]
    assert scoped
    assert all(request.url.params.get("patient") == "a" for request in scoped)

    fake.requests.clear()
    patients = await invoke(
        tools, "get_resources_by_patient_fhir_id", resource_type="Patient", patient_fhir_id=""
    )
    assert [r["id"] for r in patients["Patient"]] == ["a"]
    assert fake.requests[-1].url.params.get("_id") == "a"
    assert "patient" not in fake.requests[-1].url.params


@pytest.mark.asyncio
async def test_server_error_and_non_bundle_are_raised(fake):
    tools = get_native_tools(instance(), ["a"])
    fake.override = lambda request: httpx.Response(
        500, json={"resourceType": "OperationOutcome"}
    )
    with pytest.raises(httpx.HTTPStatusError):
        await invoke(
            tools,
            "get_resources_by_patient_fhir_id",
            resource_type="Observation",
            patient_fhir_id="a",
        )
    fake.override = lambda request: httpx.Response(
        200, json={"resourceType": "OperationOutcome"}
    )
    with pytest.raises(ValueError, match="Bundle"):
        await invoke(
            tools,
            "get_resources_by_patient_fhir_id",
            resource_type="Observation",
            patient_fhir_id="a",
        )


@pytest.mark.asyncio
async def test_failing_second_page_yields_no_partial_result(fake):
    store = fake.store_for("instance-one")
    next_url = "http://instance-one/fhir/Observation?_count=200&patient=a&page=2"

    def partial(request):
        if request.url.params.get("page") == "2":
            return httpx.Response(500, json={"resourceType": "OperationOutcome"})
        return httpx.Response(200, json=bundle([store["Observation/o1"]], next_url))

    fake.override = partial
    tools = get_native_tools(instance(), ["a"])
    with pytest.raises(httpx.HTTPStatusError):
        await invoke(
            tools,
            "get_resources_by_patient_fhir_id",
            resource_type="Observation",
            patient_fhir_id="a",
        )


@pytest.mark.asyncio
async def test_search_all_rejects_a_total_it_did_not_deliver(fake):
    store = fake.store_for("instance-one")
    fake.override = lambda request: httpx.Response(
        200,
        json=partial_bundle([store["Observation/o1"], store["Observation/o2"]], 3),
    )
    client = FHIRClient(instance().endpoint, ["a"])
    with pytest.raises(ValueError, match="total=3"):
        await client.search_all("Observation", {"_count": 200, "patient": "a"}, {"a"})


@pytest.mark.asyncio
async def test_tool_refuses_a_bundle_that_delivers_fewer_records_than_its_total(fake):
    store = fake.store_for("instance-one")
    fake.override = lambda request: httpx.Response(
        200,
        json=partial_bundle([store["Observation/o1"], store["Observation/o2"]], 3),
    )
    tools = get_native_tools(instance(), ["a"])
    with pytest.raises(ValueError, match="total=3"):
        await invoke(
            tools,
            "get_resources_by_patient_fhir_id",
            resource_type="Observation",
            patient_fhir_id="a",
        )


@pytest.mark.asyncio
async def test_tool_output_is_json_text_and_keeps_non_ascii_records(fake):
    fake.override = lambda request: httpx.Response(
        200,
        json=bundle(
            [
                record(
                    "Observation",
                    "o-cn",
                    subject={"reference": "Patient/a"},
                    code={"text": "体温"},
                    valueString="36.5 摄氏度",
                )
            ]
        ),
    )
    tools = get_native_tools(instance(), ["a"])
    raw = await invoke_raw(
        tools,
        "get_resources_by_patient_fhir_id",
        resource_type="Observation",
        patient_fhir_id="a",
    )
    assert isinstance(raw, str)
    assert "体温" in raw
    assert "36.5 摄氏度" in raw
    assert "\\u" not in raw
    decoded = json.loads(raw)
    assert list(decoded) == ["Observation"]
    assert decoded["Observation"][0]["code"]["text"] == "体温"
    assert decoded["Observation"][0]["valueString"] == "36.5 摄氏度"
    assert all("text" not in r and "meta" not in r for r in decoded["Observation"])


def test_tool_names_descriptions_and_schemas_match_official(fake):
    tools = get_native_tools(instance(), ["a"])
    assert [tool.name for tool in tools] == [schema[0] for schema in OFFICIAL_TOOL_SCHEMAS]
    for tool, (name, description, parameters) in zip(tools, OFFICIAL_TOOL_SCHEMAS):
        assert tool.description == description
        assert json.dumps(tool.params_json_schema, ensure_ascii=False) == json.dumps(
            parameters, ensure_ascii=False
        )
    assert not any(tool.name == "get_all_resources_by_patient_fhir_id" for tool in tools)


@pytest.mark.asyncio
async def test_public_entry_points_dispatch_to_the_native_tools(fake):
    with pytest.raises(UnsupportedDecisionPoint):
        await get_tools(instance(), ["a"], "2136-01-01T00:00:00Z")
    assert not fake.requests
    tools = await get_tools(instance(), ["a"])
    assert [tool.name for tool in tools] == [schema[0] for schema in OFFICIAL_TOOL_SCHEMAS]
    evidence = await read_evidence(instance(), "a", "Observation/o1")
    assert evidence["meta"]["versionId"] == "1"
    with pytest.raises(PatientAccessError):
        await read_evidence(instance(), "a", "Observation/ob")


@pytest.mark.asyncio
async def test_patient_resources_keep_raw_records(fake):
    resources = await read_patient_resources(instance(), "a")
    by_source = {f"{r['resourceType']}/{r['id']}": r for r in resources}
    assert set(by_source) == {
        "Patient/a",
        "Encounter/ea",
        "Observation/o1",
        "Observation/o2",
        "MedicationRequest/mr1",
    }
    assert by_source["Observation/o1"]["meta"]["versionId"] == "1"


@pytest.mark.asyncio
async def test_shared_read_is_blocked_for_a_different_patient_scope(fake):
    with pytest.raises(PatientAccessError):
        await read_source(instance(), "b", "Medication/m1")
    requested = {
        identifier
        for request in fake.requests
        for identifier in (request.url.params.get("_id") or "").split(",")
        if identifier
    }
    assert "m1" not in requested


@pytest.mark.asyncio
async def test_reachability_scan_is_cached_per_instance_and_dropped(fake):
    def scans():
        return [r for r in fake.requests if r.url.path.endswith("MedicationRequest")]

    await read_source(instance(), "a", "Medication/m1")
    assert len(scans()) == 1
    await read_source(instance(), "a", "Medication/m1")
    assert len(scans()) == 1
    await read_source(instance("instance-two"), "a", "Medication/m1")
    assert len(scans()) == 2
    fhir_agentbench.forget_instance("instance-one")
    await read_source(instance(), "a", "Medication/m1")
    assert len(scans()) == 3


@pytest.mark.asyncio
async def test_release_drops_the_instance_reader_cache(fake, monkeypatch):
    await read_source(instance(), "a", "Medication/m1")
    dropped = []
    monkeypatch.setattr(fhir_agentbench, "forget_instance", dropped.append)
    fake.override = lambda request: httpx.Response(200, json={})
    config = EnvironmentConfig(
        name="FHIR-AgentBench", manager_url="http://manager", seed_data="/seed"
    )
    await release_environment(config, instance())
    assert dropped == ["instance-one"]
