"""FHIR-AgentBench 患者轨迹整理测试。

全部使用 httpx.MockTransport，不连接 Docker、PostgreSQL 或 http://127.0.0.1:18086。
"""

import copy
import json
import re

import httpx
import pytest

from contracts import EnvironmentInstance
from environments import fhir_agentbench, prepare_trajectory
from environments.errors import EnvironmentUnavailable, PatientAccessError
from environments.fhir_agentbench import (
    FALLBACK_RESOURCE_TYPES,
    REFERENCE_ROUNDS,
    read_patient_material,
    read_source,
)
from environments.fhir_agentbench_trajectory import (
    build_trajectory,
    encounter_kind,
    prepare,
    time_fields,
)

HOSP = "http://mimic.mit.edu/fhir/mimic/identifier/encounter-hosp"
ED = "http://mimic.mit.edu/fhir/mimic/identifier/encounter-ed"
ICU = "http://mimic.mit.edu/fhir/mimic/identifier/encounter-icu"

PATIENT = {"reference": "Patient/a"}


def fhir(resource_type, resource_id, **fields):
    """带 text 与 meta 的原始记录：原始证据保留这两个字段。"""
    resource = {
        "resourceType": resource_type,
        "id": resource_id,
        "meta": {"versionId": "1", "lastUpdated": "2136-04-22T18:47:00-04:00"},
        "text": {"status": "generated", "div": f"<div>{resource_type}/{resource_id}</div>"},
    }
    resource.update(fields)
    return resource


def initial_store():
    return {
        "Patient/a": fhir("Patient", "a", birthDate="2100-01-01", gender="female"),
        "Patient/b": fhir("Patient", "b", birthDate="2101-01-01", gender="male"),
        "Encounter/eh": fhir(
            "Encounter",
            "eh",
            subject=PATIENT,
            **{
                "class": {"code": "AMB"},
                "identifier": [{"system": HOSP}],
                "period": {
                    "start": "2136-04-10T20:33:00-04:00",
                    "end": "2136-05-02T16:35:00-04:00",
                },
                "serviceProvider": {"reference": "Organization/org-1"},
                "location": [{"location": {"reference": "Location/loc-a"}}],
            },
        ),
        "Encounter/ei": fhir(
            "Encounter",
            "ei",
            subject=PATIENT,
            partOf={"reference": "Encounter/eh"},
            **{
                "class": {"code": "ICU"},
                "identifier": [{"system": ICU}],
                "period": {"start": "2136-04-22T18:01:00-04:00"},
            },
        ),
        "Condition/c1": fhir(
            "Condition", "c1", subject=PATIENT, encounter={"reference": "Encounter/eh"}
        ),
        "Observation/o1": fhir(
            "Observation",
            "o1",
            subject=PATIENT,
            encounter={"reference": "Encounter/ei"},
            effectiveDateTime="2136-04-22T18:47:00-04:00",
            issued="2136-04-22T18:48:00-04:00",
            valueDateTime="2136-03-27T18:47:00-04:00",
            valueQuantity={"value": 80, "unit": "beats/min"},
        ),
        "Observation/mb": fhir(
            "Observation",
            "mb",
            subject=PATIENT,
            effectiveDateTime="2136-04-23T09:00:00-04:00",
            specimen={"reference": "Specimen/s1"},
            derivedFrom=[{"reference": "Observation/d1"}],
            hasMember=[{"reference": "Observation/h1"}],
        ),
        "Observation/d1": fhir(
            "Observation",
            "d1",
            subject=PATIENT,
            effectiveDateTime="2136-04-23T08:00:00-04:00",
        ),
        "Observation/h1": fhir(
            "Observation",
            "h1",
            subject=PATIENT,
            effectiveDateTime="2136-04-23T08:30:00-04:00",
        ),
        "Specimen/s1": fhir(
            "Specimen",
            "s1",
            subject=PATIENT,
            collection={"collectedDateTime": "2136-04-23T07:00:00-04:00"},
        ),
        "MedicationRequest/mr1": fhir(
            "MedicationRequest",
            "mr1",
            subject=PATIENT,
            encounter={"reference": "Encounter/eh"},
            authoredOn="2136-04-10T21:00:00-04:00",
            medicationCodeableConcept={
                "coding": [
                    {
                        "system": "http://www.nlm.nih.gov/research/umls/rxnorm",
                        "code": "866924",
                        "display": "metoprolol succinate",
                    }
                ],
                "text": "metoprolol",
            },
        ),
        "MedicationRequest/mr2": fhir(
            "MedicationRequest",
            "mr2",
            subject=PATIENT,
            authoredOn="2136-04-24T10:00:00-04:00",
            medicationReference={"reference": "Medication/m1"},
        ),
        "MedicationAdministration/ma1": fhir(
            "MedicationAdministration",
            "ma1",
            subject=PATIENT,
            encounter={"reference": "Encounter/ei"},
            effectiveDateTime="2136-04-24T11:00:00-04:00",
            medicationReference={"reference": "Medication/m1"},
        ),
        "MedicationStatement/ms1": fhir(
            "MedicationStatement",
            "ms1",
            subject=PATIENT,
            encounter={"reference": "Encounter/eh"},
            dateAsserted="2136-04-05T00:00:00-04:00",
        ),
        "MedicationDispense/md1": fhir(
            "MedicationDispense",
            "md1",
            subject=PATIENT,
            encounter={"reference": "Encounter/eh"},
            medicationReference={"reference": "Medication/m1"},
        ),
        "Procedure/p1": fhir(
            "Procedure",
            "p1",
            subject=PATIENT,
            encounter={"reference": "Encounter/ei"},
            performedDateTime="2136-04-25T06:00:00-04:00",
        ),
        "Observation/ob": fhir(
            "Observation",
            "ob",
            subject={"reference": "Patient/b"},
            effectiveDateTime="2136-04-22T19:00:00-04:00",
        ),
        "Medication/m1": fhir(
            "Medication",
            "m1",
            code={
                "coding": [
                    {
                        "system": "http://www.nlm.nih.gov/research/umls/rxnorm",
                        "code": "866924",
                        "display": "metoprolol succinate",
                    }
                ]
            },
        ),
        "Location/loc-a": fhir("Location", "loc-a", name="ward a"),
        "Organization/org-1": fhir("Organization", "org-1", name="hospital one"),
    }


def material(store=None):
    """患者 a 的材料：不含另一名患者的记录。"""
    store = store or initial_store()
    for other in ("Patient/b", "Observation/ob"):
        store.pop(other, None)
    return list(copy.deepcopy(store).values())


def owner(resource):
    return resource.get("subject", {}).get("reference")


def bundle(resources, next_url=None):
    result = {"resourceType": "Bundle", "entry": [{"resource": r} for r in resources]}
    if next_url:
        result["link"] = [{"relation": "next", "url": next_url}]
    return result


def search(store, resource_type, params):
    resources = sorted(
        (r for r in store.values() if r["resourceType"] == resource_type),
        key=lambda r: r["id"],
    )
    if params.get("_id"):
        wanted = set(params["_id"].split(","))
        return [r for r in resources if r["id"] in wanted]
    for key in ("patient", "subject"):
        if params.get(key):
            wanted = {f"Patient/{p}" for p in params[key].split(",")}
            return [r for r in resources if owner(r) in wanted]
    return resources


SEARCH_PARAMS = {
    "Patient": ["_id", "patient"],
    "Encounter": ["_id", "patient"],
    "Condition": ["_id", "patient"],
    "Observation": ["_id", "patient"],
    "MedicationRequest": ["_id", "patient"],
    "MedicationAdministration": ["_id", "patient"],
    "MedicationStatement": ["_id", "patient"],
    "MedicationDispense": ["_id", "patient"],
    "Procedure": ["_id", "patient"],
    "Specimen": ["_id", "patient"],
    "Medication": ["_id"],
    "Location": ["_id"],
    "Organization": ["_id"],
}


class FakeFHIR:
    """按主机名隔离的 fake FHIR 服务，记录每个请求。"""

    def __init__(self):
        self.stores = {}
        self.requests = []
        self.override = None
        self.search_params = None

    def store_for(self, host):
        return self.stores.setdefault(host, copy.deepcopy(initial_store()))

    def capability(self):
        params = self.search_params or SEARCH_PARAMS
        return {
            "resourceType": "CapabilityStatement",
            "rest": [
                {
                    "resource": [
                        {
                            "type": resource_type,
                            "searchParam": [{"name": name} for name in names],
                        }
                        for resource_type, names in params.items()
                    ]
                }
            ],
        }

    def handle(self, request):
        self.requests.append(request)
        if self.override is not None:
            response = self.override(request)
            if response is not None:
                return response
        store = self.store_for(request.url.host)
        path = request.url.path.removeprefix("/fhir/")
        if path == "metadata":
            return httpx.Response(200, json=self.capability())
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
        return httpx.Response(200, json=bundle(search(store, path, request.url.params)))


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


def instance(host="trajectory-one"):
    return EnvironmentInstance(
        instance_id=host, endpoint=f"http://{host}/fhir/", environment="FHIR-AgentBench"
    )


def unsupported_everything(status):
    def override(request):
        if request.url.path.endswith("/$everything"):
            return httpx.Response(status, json={"resourceType": "OperationOutcome"})
        return None

    return override


def paths(fake):
    return [request.url.path for request in fake.requests]


RECORD_LINE = re.compile(r"(?m)^\s*(?:\[[^\]]*\]\s+)?([A-Za-z]+/[A-Za-z0-9.\-]+)")


def line(text, source_id):
    """返回某条记录所在的唯一一行，避免断言落到别的记录上。"""
    matches = [
        row
        for row in text.splitlines()
        if re.match(rf"^\s*(?:\[[^\]]*\]\s+)?{re.escape(source_id)}\b", row)
    ]
    assert matches, f"{source_id} is not presented in the trajectory text"
    assert len(matches) == 1, f"{source_id} is presented {len(matches)} times"
    return matches[0]


def recorded_sources(text):
    return RECORD_LINE.findall(text)


def test_nested_hospital_encounters_are_each_rendered_once():
    """医院就诊 partOf 另一个医院就诊时，父级只作为顶层渲染一次。"""
    resources = [
        fhir("Patient", "a", birthDate="2100-01-01"),
        fhir(
            "Encounter",
            "hb",
            subject=PATIENT,
            identifier=[{"system": HOSP}],
            period={
                "start": "2136-04-01T00:00:00-04:00",
                "end": "2136-04-10T00:00:00-04:00",
            },
        ),
        fhir(
            "Encounter",
            "ha",
            subject=PATIENT,
            identifier=[{"system": HOSP}],
            partOf={"reference": "Encounter/hb"},
            period={
                "start": "2136-04-11T00:00:00-04:00",
                "end": "2136-04-20T00:00:00-04:00",
            },
        ),
    ]
    text = build_trajectory("a", resources).text
    outer = line(text, "Encounter/hb")
    inner = line(text, "Encounter/ha")
    assert [source for source in recorded_sources(text) if source.startswith("Encounter/")] == [
        "Encounter/hb",
        "Encounter/ha",
    ]
    assert "[encounter-hosp]" in outer and "[encounter-hosp]" in inner
    assert "partOf=" not in outer
    assert "partOf=Encounter/hb" in inner
    assert "2136-04-01T00:00:00-04:00" in outer and "2136-04-10T00:00:00-04:00" in outer
    assert "2136-04-11T00:00:00-04:00" in inner and "2136-04-20T00:00:00-04:00" in inner
    assert "医院 2" in text


def test_hospital_kind_comes_from_the_identifier_not_the_class():
    for code in ("AMB", "EMER", "OBSENC"):
        encounter = fhir("Encounter", "eh", **{"class": {"code": code}}, identifier=[{"system": HOSP}])
        assert encounter_kind(encounter) == "encounter-hosp"
    assert encounter_kind(fhir("Encounter", "ee", identifier=[{"system": ED}])) == "encounter-ed"
    assert encounter_kind(fhir("Encounter", "ei", identifier=[{"system": ICU}])) == "encounter-icu"
    # class 是 AMB/EMER/OBSENC 也不能用来判断就诊来源。
    assert encounter_kind(fhir("Encounter", "eu", **{"class": {"code": "AMB"}})) is None
    assert (
        encounter_kind(fhir("Encounter", "ex", identifier=[{"system": "http://example.org/x"}]))
        is None
    )


def test_observation_times_stay_distinct_and_undated_diagnosis_gets_nothing():
    observation = fhir(
        "Observation",
        "o",
        subject=PATIENT,
        encounter={"reference": "Encounter/ei"},
        effectiveDateTime="2136-04-22T18:47:00-04:00",
        valueDateTime="2136-03-27T18:47:00-04:00",
    )
    diagnosis = fhir("Condition", "c", subject=PATIENT, encounter={"reference": "Encounter/eh"})
    fields = time_fields(observation)
    assert fields["effectiveDateTime"] == observation["effectiveDateTime"]
    assert fields["valueDateTime"] == observation["valueDateTime"]
    assert fields["effectiveDateTime"] != fields["valueDateTime"]
    assert time_fields(diagnosis) == {}


def test_specimen_and_period_times_keep_their_source_field_names():
    specimen = fhir(
        "Specimen", "s", subject=PATIENT, collection={"collectedDateTime": "2136-04-23T07:00:00-04:00"}
    )
    assert time_fields(specimen) == {"collection.collectedDateTime": "2136-04-23T07:00:00-04:00"}
    procedure = fhir(
        "Procedure",
        "p",
        subject=PATIENT,
        performedPeriod={"start": "2136-04-25T06:00:00-04:00", "end": "2136-04-25T07:00:00-04:00"},
    )
    assert time_fields(procedure) == {
        "performedPeriod": {"start": "2136-04-25T06:00:00-04:00", "end": "2136-04-25T07:00:00-04:00"}
    }
    assert time_fields(fhir("MedicationDispense", "md", subject=PATIENT)) == {}
    statement = fhir("MedicationStatement", "ms", subject=PATIENT, dateAsserted="2136-04-05T00:00:00-04:00")
    assert time_fields(statement) == {"dateAsserted": "2136-04-05T00:00:00-04:00"}


def test_evidence_statement_round_trips_the_input_json():
    resources = material()
    trajectory = build_trajectory("a", resources)
    by_source = {f"{r['resourceType']}/{r['id']}": r for r in resources}
    assert len(trajectory.sources) == len(by_source)
    assert trajectory.patient_id == "a"
    assert trajectory.environment == "FHIR-AgentBench"
    for evidence in trajectory.sources:
        assert json.loads(evidence.statement) == by_source[evidence.source_id]


def test_hospital_count_and_icu_parent_reference():
    trajectory = build_trajectory("a", material())
    assert (
        "资源 18 条；就诊 2 条（医院 1、急诊 0、ICU 1、其他 0）；"
        "事件记录 9 条；无事件时间记录 6 条；时间解析失败 0 条"
    ) in trajectory.text
    icu = line(trajectory.text, "Encounter/ei")
    assert "partOf=Encounter/eh" in icu
    assert "[encounter-icu]" in icu
    assert "[encounter-hosp]" in line(trajectory.text, "Encounter/eh")


def test_every_record_line_is_a_source_id_in_sources():
    trajectory = build_trajectory("a", material())
    recorded = recorded_sources(trajectory.text)
    assert set(recorded) == {evidence.source_id for evidence in trajectory.sources}
    assert len(recorded) == len(trajectory.sources)


def test_child_encounter_earlier_than_the_parent_keeps_its_own_times():
    resources = material()
    by_source = {f"{r['resourceType']}/{r['id']}": r for r in resources}
    by_source["Encounter/ei"]["period"] = {"start": "2136-04-01T08:00:00-04:00"}
    by_source["Observation/o1"]["effectiveDateTime"] = "2136-04-01T09:00:00-04:00"
    trajectory = build_trajectory("a", list(by_source.values()))
    icu = line(trajectory.text, "Encounter/ei")
    assert "2136-04-01T08:00:00-04:00" in icu
    assert "2136-04-10T20:33:00-04:00" not in icu
    assert "partOf=Encounter/eh" in icu
    assert "2136-04-01T09:00:00-04:00" in line(trajectory.text, "Observation/o1")


def test_dated_measurement_does_not_replace_the_observation_time():
    observation = line(build_trajectory("a", material()).text, "Observation/o1")
    assert observation.startswith("[2136-04-22T18:47:00-04:00]")
    assert "effectiveDateTime=2136-04-22T18:47:00-04:00" in observation
    assert "issued=2136-04-22T18:48:00-04:00 (发布时间)" in observation
    assert "valueDateTime=2136-03-27T18:47:00-04:00 (测量值日期)" in observation
    assert "encounter=Encounter/ei" in observation


def test_condition_without_times_is_not_given_a_date():
    trajectory = build_trajectory("a", material())
    condition = line(trajectory.text, "Condition/c1")
    assert condition.startswith("[无事件时间]")
    assert not re.match(r"^\[2136", condition)
    assert "onset" not in condition and "recordedDate" not in condition
    assert "background=Encounter/eh 2136-04-10T20:33:00-04:00→2136-05-02T16:35:00-04:00" in condition


def test_condition_without_encounter_has_no_background_period():
    resources = material() + [fhir("Condition", "c2", subject=PATIENT)]
    condition = line(build_trajectory("a", resources).text, "Condition/c2")
    assert condition.startswith("[无事件时间]")
    assert "encounter=none" in condition
    assert "background=" not in condition


def test_medication_codeable_concept_and_reference_are_both_kept():
    trajectory = build_trajectory("a", material())
    direct = line(trajectory.text, "MedicationRequest/mr1")
    assert "medicationCodeableConcept" in direct
    assert "metoprolol" in direct
    referenced = line(trajectory.text, "MedicationRequest/mr2")
    assert "medicationReference" in referenced
    assert "Medication/m1" in referenced
    assert "encounter=none" in referenced
    medication = line(trajectory.text, "Medication/m1")
    # 药名读取原始记录：编码与 display 都在，不额外造可读药名字段。
    assert "866924" in medication and "metoprolol succinate" in medication
    assert "Medication/m1" in {evidence.source_id for evidence in trajectory.sources}


def test_microbiology_chain_is_kept_without_guessing_an_encounter():
    trajectory = build_trajectory("a", material())
    mb = line(trajectory.text, "Observation/mb")
    assert "encounter=none" in mb
    assert "Observation/d1" in mb and "Observation/h1" in mb and "Specimen/s1" in mb
    assert "Encounter/" not in mb
    assert "encounter=Encounter/ei" in line(trajectory.text, "Observation/o1")
    assert {"Observation/mb", "Observation/d1", "Observation/h1", "Specimen/s1"} <= {
        evidence.source_id for evidence in trajectory.sources
    }


def test_medication_statement_date_asserted_is_a_statement_time():
    statement = line(build_trajectory("a", material()).text, "MedicationStatement/ms1")
    assert statement.startswith("[无事件时间]")
    assert "dateAsserted=2136-04-05T00:00:00-04:00 (陈述时间)" in statement
    assert "background=Encounter/eh" in statement
    assert "effectiveDateTime" not in statement


def test_dispense_without_handover_time_stays_in_the_undated_group():
    dispense = line(build_trajectory("a", material()).text, "MedicationDispense/md1")
    assert dispense.startswith("[无事件时间]")
    assert "whenHandedOver" not in dispense
    assert "background=Encounter/eh" in dispense


def test_context_reference_attributes_a_medication_request():
    resources = material()
    by_source = {f"{r['resourceType']}/{r['id']}": r for r in resources}
    by_source["MedicationRequest/mr1"].pop("encounter")
    by_source["MedicationRequest/mr1"]["context"] = {"reference": "Encounter/ei"}
    request = line(build_trajectory("a", list(by_source.values())).text, "MedicationRequest/mr1")
    assert request.startswith("[2136-04-10T21:00:00-04:00]")
    assert "encounter=Encounter/ei" in request


def test_unparsable_time_is_kept_and_counted():
    resources = [
        fhir("Patient", "a", birthDate="2100-01-01"),
        fhir(
            "Observation",
            "bad",
            subject=PATIENT,
            effectiveDateTime="2136-13-45T99:99:99",
        ),
    ]
    trajectory = build_trajectory("a", resources)
    bad = line(trajectory.text, "Observation/bad")
    assert bad.startswith("[2136-13-45T99:99:99]")
    assert "effectiveDateTime=2136-13-45T99:99:99" in bad
    assert "时间解析失败 1 条" in trajectory.text


def test_same_time_is_sorted_stably_by_resource_type_and_id():
    resources = [
        fhir("Patient", "a"),
        fhir("Observation", "oz", subject=PATIENT, effectiveDateTime="2136-04-25T00:00:00-04:00"),
        fhir("Observation", "oa", subject=PATIENT, effectiveDateTime="2136-04-25T00:00:00-04:00"),
    ]
    text = build_trajectory("a", resources).text
    assert text.index("Observation/oa") < text.index("Observation/oz")


def test_material_requires_the_patient_record():
    with pytest.raises(ValueError, match="Patient/a"):
        build_trajectory("a", [fhir("Encounter", "eh", subject=PATIENT)])


@pytest.mark.asyncio
async def test_everything_is_preferred_and_shared_records_are_completed(fake):
    resources = await read_patient_material(instance(), "a")
    assert "/fhir/Patient/a/$everything" in paths(fake)
    assert not any(path.endswith(("Observation", "Encounter", "Condition")) for path in paths(fake))
    identifiers = {f"{r['resourceType']}/{r['id']}" for r in resources}
    assert identifiers == set(initial_store()) - {"Patient/b", "Observation/ob"}
    assert len(identifiers) == len(resources)
    assert all(r["meta"]["versionId"] == "1" for r in resources)


@pytest.mark.asyncio
async def test_everything_fallback_reads_every_type_and_dedupes(fake):
    fake.override = unsupported_everything(405)
    resources = await read_patient_material(instance(), "a")
    assert "Patient/a" in {f"{r['resourceType']}/{r['id']}" for r in resources}
    assert {f"{r['resourceType']}/{r['id']}" for r in resources} == set(
        initial_store()
    ) - {"Patient/b", "Observation/ob"}
    assert len(resources) == len({f"{r['resourceType']}/{r['id']}" for r in resources})
    assert "/fhir/metadata" in paths(fake)
    for resource_type in FALLBACK_RESOURCE_TYPES:
        assert f"/fhir/{resource_type}" in paths(fake)
    observations = [
        request for request in fake.requests if request.url.path.endswith("Observation")
    ]
    assert all(request.url.params.get("patient") == "a" for request in observations)


@pytest.mark.asyncio
async def test_reference_completion_reaches_a_fixed_point_in_both_paths(fake):
    """第二跳引用在 $everything 与逐类型回退两条路径上补齐到同一个闭包。"""
    for host in ("trajectory-one", "trajectory-two"):
        store = fake.store_for(host)
        store["Medication/m1"]["ingredient"] = [
            {"itemReference": {"reference": "Medication/mi"}}
        ]
        store["Medication/mi"] = fhir(
            "Medication",
            "mi",
            ingredient=[{"itemReference": {"reference": "Medication/mii"}}],
        )
        store["Medication/mii"] = fhir("Medication", "mii", code={"text": "second hop"})
    everything = await read_patient_material(instance(), "a")
    fake.override = unsupported_everything(405)
    fallback = await read_patient_material(instance("trajectory-two"), "a")

    def identifiers(resources):
        return {f"{r['resourceType']}/{r['id']}" for r in resources}

    assert "Medication/mii" in identifiers(everything)
    assert identifiers(fallback) == identifiers(everything)


@pytest.mark.asyncio
async def test_reference_completion_reports_a_chain_beyond_the_round_limit(fake):
    """引用链超过迭代上限时报错，而不是把一个截断的子集当成完整材料。"""
    store = fake.store_for("trajectory-one")
    chain = [f"chain-{index}" for index in range(REFERENCE_ROUNDS + 2)]
    store["MedicationRequest/mr2"]["medicationReference"] = {
        "reference": f"Medication/{chain[0]}"
    }
    for index, identifier in enumerate(chain):
        linked = chain[index + 1] if index + 1 < len(chain) else None
        fields = (
            {"ingredient": [{"itemReference": {"reference": f"Medication/{linked}"}}]}
            if linked
            else {}
        )
        store[f"Medication/{identifier}"] = fhir("Medication", identifier, **fields)
    with pytest.raises(ValueError, match="converge"):
        await read_patient_material(instance(), "a")


@pytest.mark.asyncio
async def test_only_unsupported_statuses_fall_back(fake):
    fake.override = unsupported_everything(500)
    with pytest.raises(httpx.HTTPStatusError):
        await read_patient_material(instance(), "a")
    assert paths(fake) == ["/fhir/Patient/a/$everything"]


@pytest.mark.asyncio
async def test_search_parameters_come_from_the_capability_statement(fake):
    fake.search_params = {
        **SEARCH_PARAMS,
        "MedicationStatement": ["_id", "subject"],
        "Procedure": ["_id"],
    }
    fake.override = unsupported_everything(404)
    with pytest.raises(EnvironmentUnavailable, match="Procedure"):
        await read_patient_material(instance(), "a")
    statements = [
        request for request in fake.requests if request.url.path.endswith("MedicationStatement")
    ]
    assert statements
    assert statements[-1].url.params.get("subject") == "a"
    assert "patient" not in statements[-1].url.params


@pytest.mark.asyncio
async def test_missing_referenced_record_is_reported(fake):
    store = fake.store_for("trajectory-one")
    store["MedicationRequest/mr2"]["medicationReference"] = {"reference": "Medication/missing"}
    with pytest.raises(ValueError, match="Medication/missing"):
        await read_patient_material(instance(), "a")


@pytest.mark.asyncio
async def test_another_patients_referenced_record_is_refused(fake):
    store = fake.store_for("trajectory-one")
    store["Observation/o1"]["hasMember"] = [{"reference": "Observation/ob"}]
    with pytest.raises(PatientAccessError):
        await read_patient_material(instance(), "a")


@pytest.mark.asyncio
async def test_prepare_returns_a_trajectory_whose_sources_are_readable(fake):
    trajectory = await prepare(instance(), "a")
    assert trajectory.patient_id == "a"
    assert trajectory.environment == "FHIR-AgentBench"
    by_source = {evidence.source_id: json.loads(evidence.statement) for evidence in trajectory.sources}
    assert set(recorded_sources(trajectory.text)) == set(by_source)
    for source_id, statement in sorted(by_source.items()):
        assert await read_source(instance(), "a", source_id) == statement


@pytest.mark.asyncio
async def test_prepare_trajectory_dispatches_to_this_environment(fake):
    dispatched = await prepare_trajectory(instance(), "a")
    assert dispatched.environment == "FHIR-AgentBench"
    assert dispatched.patient_id == "a"
    assert dispatched.text == (await prepare(instance(), "a")).text
