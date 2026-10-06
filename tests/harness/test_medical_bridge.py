"""Medical tool bridge: native tools, patient scope, write state and budget.

PhysicianBench and MedAgentBench tools are the real ones; only the FHIR backend
and the environment manager are simulated.
"""

import asyncio
import json

import httpx
import pytest
from agents import function_tool

from contracts import EnvironmentConfig, SolverInput
from harnesses.medical_bridge import (
    MEDAGENTBENCH_TOOL, SESSION_HEADER, SessionIdentityMismatch, ToolSessions,
    UnknownSession, create_bridge_app,
)

PHYSICIANBENCH_WRITE = "fhir_communication_create_message"
PHYSICIANBENCH_READ = "fhir_condition_search_problems"


@pytest.mark.asyncio
async def test_physicianbench_tools_keep_native_names_and_schemas(tool_sessions, fake_environments):
    session_id = await tool_sessions.open(
        SolverInput(instruction="核对患者用药", patient_id="p1"),
        EnvironmentConfig(name="fixture", manager_url="http://unused", seed_data="seed"),
    )
    tools = await tool_sessions.list(session_id)
    assert [tool["name"] for tool in tools] == ["write_message", "read_messages"]
    assert set(tools[0]) == {"name", "description", "parameters"}
    assert tools[0]["parameters"]["required"] == ["patient", "text"]
    assert fake_environments.tool_sessions[0][1] == ("p1",)
    assert fake_environments.tool_sessions[0][2] is None


@pytest.mark.asyncio
async def test_cross_patient_tool_call_is_refused_without_any_fhir_request(real_physicianbench):
    rig = real_physicianbench
    registry = ToolSessions(environments=rig.environments)
    session_id = await registry.open(
        SolverInput(instruction="核对患者用药", patient_id="p1"), rig.config
    )
    record = await registry.invoke(
        session_id, PHYSICIANBENCH_READ, {"patient": "p2"}
    )
    assert record["status"] == "error"
    assert "outside the allowed patient scope" in record["error"]
    assert rig.fhir.requests == []


@pytest.mark.asyncio
async def test_physicianbench_read_and_write_are_recorded_with_scope(real_physicianbench):
    rig = real_physicianbench
    registry = ToolSessions(environments=rig.environments)
    session_id = await registry.open(
        SolverInput(instruction="核对患者用药", patient_id="p1"), rig.config
    )
    read = await registry.invoke(
        session_id, PHYSICIANBENCH_READ, {"patient": "p1"}
    )
    assert read["status"] == "ok"
    assert read["write"] is False
    assert read["output"]["total"] == 1
    assert all("patient=p1" in url or "_id=p1" in url for url in rig.fhir.urls())
    write = await registry.invoke(
        session_id, PHYSICIANBENCH_WRITE,
        {"patient_reference": "Patient/p1", "message_text": "复查已完成"},
    )
    assert write["status"] == "ok"
    assert write["write"] is True
    assert write["persisted"] is True
    assert rig.fhir.methods()[-1] == "POST"
    assert rig.fhir.methods().count("POST") == 1


@pytest.mark.asyncio
async def test_tool_events_keep_order_and_duration(real_physicianbench):
    rig = real_physicianbench
    registry = ToolSessions(environments=rig.environments)
    session_id = await registry.open(
        SolverInput(instruction="核对患者用药", patient_id="p1"), rig.config
    )
    for arguments in ({"patient": "p1"}, {"patient": "p1"}):
        await registry.invoke(session_id, PHYSICIANBENCH_READ, arguments)
    summary = await registry.close(session_id)
    assert [event["type"] for event in summary["events"]] == [
        "tool_start", "tool_output", "tool_start", "tool_output"]
    assert [call["order"] for call in summary["calls"]] == [1, 2]
    assert all(call["duration_ms"] >= 0 for call in summary["calls"])
    assert summary["tool_calls"] == 2
    assert summary["state"]["patient_id"] == "p1"


@pytest.mark.asyncio
async def test_medagentbench_keeps_native_get_post_and_finish(real_medagentbench):
    rig = real_medagentbench
    registry = ToolSessions(environments=rig.environments)
    session_id = await registry.open(
        SolverInput(instruction="查询患者情况", patient_id="p1"), rig.config
    )
    tools = await registry.list(session_id)
    assert [tool["name"] for tool in tools] == [MEDAGENTBENCH_TOOL]
    assert tools[0]["parameters"]["required"] == ["action"]
    assert tools[0]["parameters"]["properties"]["action"]["type"] == "string"

    fetch = await registry.invoke(
        session_id, MEDAGENTBENCH_TOOL, {"action": "GET Condition?patient=p1"}
    )
    assert fetch["status"] == "ok"
    assert fetch["output"]["kind"] == "GET"
    assert "Here is the response from the GET request" in fetch["output"]["reply"]

    finish = await registry.invoke(
        session_id, MEDAGENTBENCH_TOOL, {"action": 'FINISH(["患者有高血压"])'}
    )
    assert finish["output"]["kind"] == "FINISH"
    assert finish["output"]["answers"] == ["患者有高血压"]
    assert finish["write"] is False


@pytest.mark.asyncio
async def test_medagentbench_post_is_accepted_without_a_database_write(real_medagentbench):
    rig = real_medagentbench
    registry = ToolSessions(environments=rig.environments)
    session_id = await registry.open(
        SolverInput(instruction="创建消息", patient_id="p1"), rig.config
    )
    payload = json.dumps({
        "resourceType": "Communication",
        "status": "in-progress",
        "subject": {"reference": "Patient/p1"},
        "payload": [{"contentString": "复查已完成"}],
    })
    record = await registry.invoke(
        session_id, MEDAGENTBENCH_TOOL,
        {"action": f"POST http://fhir-1/fhir/Communication\n{payload}"},
    )
    assert record["status"] == "ok"
    assert record["output"]["kind"] == "POST"
    assert "accepted and executed successfully" in record["output"]["reply"]
    assert record["write"] is True
    assert record["persisted"] is False
    assert "POST" not in rig.fhir.methods()
    summary = await registry.close(session_id)
    assert summary["calls"][0]["persisted"] is False
    assert summary["state"]["patient_id"] == "p1"


@pytest.mark.asyncio
async def test_tool_budget_is_deducted_at_the_bridge(tool_sessions, fake_environments):
    session_id = await tool_sessions.open(
        SolverInput(instruction="核对患者用药", patient_id="p1"),
        EnvironmentConfig(name="fixture", manager_url="http://unused", seed_data="seed"),
        tool_budget=1,
    )
    first = await tool_sessions.invoke(session_id, "read_messages", {"patient": "p1"})
    assert first["status"] == "ok"
    second = await tool_sessions.invoke(session_id, "read_messages", {"patient": "p1"})
    assert second["status"] == "budget_exhausted"
    assert second["output"] is None
    assert len(fake_environments.tool_calls) == 1
    summary = await tool_sessions.close(session_id)
    assert summary["events"][-1]["type"] == "tool_budget"


@pytest.mark.asyncio
async def test_invalid_arguments_are_rejected_before_the_environment(
    tool_sessions, fake_environments
):
    session_id = await tool_sessions.open(
        SolverInput(instruction="核对患者用药", patient_id="p1"),
        EnvironmentConfig(name="fixture", manager_url="http://unused", seed_data="seed"),
    )
    record = await tool_sessions.invoke(session_id, "read_messages", {"text": "x"})
    assert record["status"] == "invalid_arguments"
    assert "patient" in record["error"]
    unknown = await tool_sessions.invoke(session_id, "not_a_tool", {"patient": "p1"})
    assert unknown["status"] == "invalid_arguments"
    assert fake_environments.tool_calls == []


@pytest.mark.asyncio
async def test_session_identity_is_bound_to_the_calling_container(tool_sessions, fake_environments):
    session_id = await tool_sessions.open(
        SolverInput(instruction="核对患者用药", patient_id="p1"),
        EnvironmentConfig(name="fixture", manager_url="http://unused", seed_data="seed"),
        client_id="container-1",
    )
    with pytest.raises(SessionIdentityMismatch):
        await tool_sessions.invoke(
            session_id, "read_messages", {"patient": "p1"}, client_id="container-2"
        )
    with pytest.raises(SessionIdentityMismatch):
        await tool_sessions.list(session_id, client_id="container-2")
    assert (await tool_sessions.invoke(
        session_id, "read_messages", {"patient": "p1"}, client_id="container-1"
    ))["status"] == "ok"
    with pytest.raises(UnknownSession):
        await tool_sessions.list("does-not-exist")


@pytest.mark.asyncio
async def test_close_saves_state_before_releasing_and_is_idempotent(
    tool_sessions, fake_environments
):
    session_id = await tool_sessions.open(
        SolverInput(instruction="核对患者用药", patient_id="p1"),
        EnvironmentConfig(name="fixture", manager_url="http://unused", seed_data="seed"),
    )
    await tool_sessions.invoke(session_id, "write_message", {"patient": "p1", "text": "记录"})
    summary = await tool_sessions.close(session_id)
    assert summary["state"]["messages"] == ["记录"]
    assert fake_environments.active == 0
    assert len(fake_environments.released) == 1
    again = await tool_sessions.close(session_id)
    assert again["already_closed"] is True
    assert again["state"] == summary["state"]
    assert len(fake_environments.released) == 1


@pytest.mark.asyncio
async def test_startup_failure_releases_the_medical_instance(fake_environments):
    fake_environments.fail_get_tools = True
    registry = ToolSessions(environments=fake_environments)
    with pytest.raises(RuntimeError, match="tool setup failed"):
        await registry.open(
            SolverInput(instruction="核对患者用药", patient_id="p1"),
            EnvironmentConfig(name="fixture", manager_url="http://unused", seed_data="seed"),
        )
    assert fake_environments.active == 0
    assert fake_environments.released == fake_environments.created


@pytest.mark.asyncio
async def test_cancellation_releases_the_medical_instance(fake_environments_factory):
    started = asyncio.Event()

    @function_tool
    async def slow_tool(patient: str) -> str:
        """Block until cancelled."""
        started.set()
        await asyncio.Event().wait()
        return "{}"

    environments = fake_environments_factory(tool_factory=lambda *args: [slow_tool])
    registry = ToolSessions(environments=environments)
    session_id = await registry.open(
        SolverInput(instruction="核对患者用药", patient_id="p1"),
        EnvironmentConfig(name="fixture", manager_url="http://unused", seed_data="seed"),
    )
    task = asyncio.create_task(registry.invoke(session_id, "slow_tool", {"patient": "p1"}))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert environments.active == 0
    assert len(environments.released) == 1


@pytest.mark.asyncio
async def test_call_timeout_is_recorded_and_close_still_releases(fake_environments_factory):
    @function_tool
    async def slow_tool(patient: str) -> str:
        """Sleep longer than the bridge timeout."""
        await asyncio.sleep(1)
        return "{}"

    environments = fake_environments_factory(tool_factory=lambda *args: [slow_tool])
    registry = ToolSessions(environments=environments)
    session_id = await registry.open(
        SolverInput(instruction="核对患者用药", patient_id="p1"),
        EnvironmentConfig(name="fixture", manager_url="http://unused", seed_data="seed"),
        timeout_seconds=0.01,
    )
    record = await registry.invoke(session_id, "slow_tool", {"patient": "p1"})
    assert record["status"] == "timeout"
    summary = await registry.close(session_id)
    assert summary["state"]["patient_id"] == "p1"
    assert environments.active == 0


@pytest.mark.asyncio
async def test_bridge_never_carries_the_reference_answer(tool_sessions, fake_environments):
    task = SolverInput(
        instruction="查询患者已有用药并说明依据", patient_id="p1", decision_point="2026-01-01"
    )
    session_id = await tool_sessions.open(
        task,
        EnvironmentConfig(name="fixture", manager_url="http://unused", seed_data="seed"),
    )
    summary = await tool_sessions.close(session_id)
    instance, checked_task = fake_environments.state_reads[0]
    # The bridge builds a reference-free task for read_state: only the patient
    # question and the decision point reach the environment.
    assert checked_task.patient_id == "p1"
    assert checked_task.decision_point == "2026-01-01"
    assert checked_task.reference_answer == ""
    assert checked_task.evidence == []
    assert instance.environment == "fixture"
    assert summary["state"]["reference_answer"] == ""


@pytest.mark.asyncio
async def test_http_bridge_routes_bind_sessions_to_callers(tool_sessions, fake_environments):
    session_id = await tool_sessions.open(
        SolverInput(instruction="核对患者用药", patient_id="p1"),
        EnvironmentConfig(name="fixture", manager_url="http://unused", seed_data="seed"),
        client_id="container-1", tool_budget=1,
    )
    app = create_bridge_app(tool_sessions)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://bridge") as client:
        assert (await client.get(f"/sessions/{session_id}/tools")).status_code == 403
        response = await client.get(
            f"/sessions/{session_id}/tools", headers={SESSION_HEADER: "container-1"}
        )
        assert response.status_code == 200
        assert [tool["name"] for tool in response.json()] == ["write_message", "read_messages"]
        assert (await client.post(
            f"/sessions/{session_id}/calls", json={"name": "read_messages", "arguments": {}}
        )).status_code == 403
        call = await client.post(
            f"/sessions/{session_id}/calls",
            json={"name": "read_messages", "arguments": {"patient": "p1"}},
            headers={SESSION_HEADER: "container-1"},
        )
        assert call.status_code == 200
        assert call.json()["status"] == "ok"
        exhausted = await client.post(
            f"/sessions/{session_id}/calls",
            json={"name": "read_messages", "arguments": {"patient": "p1"}},
            headers={SESSION_HEADER: "container-1"},
        )
        assert exhausted.status_code == 429
        assert exhausted.json()["status"] == "budget_exhausted"
        assert (await client.get(
            "/sessions/unknown/tools", headers={SESSION_HEADER: "container-1"}
        )).status_code == 404


@pytest.mark.asyncio
async def test_http_route_reports_unknown_tool_and_argument_errors(tool_sessions):
    session_id = await tool_sessions.open(
        SolverInput(instruction="核对患者用药", patient_id="p1"),
        EnvironmentConfig(name="fixture", manager_url="http://unused", seed_data="seed"),
        client_id="container-1",
    )
    app = create_bridge_app(tool_sessions)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://bridge") as client:
        response = await client.post(
            f"/sessions/{session_id}/calls",
            json={"name": "read_messages", "arguments": {"text": "x"}},
            headers={SESSION_HEADER: "container-1"},
        )
        assert response.status_code == 422
        assert response.json()["status"] == "invalid_arguments"
