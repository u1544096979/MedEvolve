"""长轨迹、阶段材料和工具历史的公共输入预算。"""

import json

import pytest

from contracts import Evidence, ModelConfig, PatientTrajectory, TaskBatch
from environments.fitting import fit_trajectory, payload_chars
from runtime.context import (ContextBudgetError, count_tokens, dumps, fit_history,
                             fit_patient, input_limit, prepare_input)


MODEL = ModelConfig(model="gpt-4o-mini", api_key_env="UNUSED",
                    context_window=12000, max_output_tokens=512)


def patient(count=80):
    records = [("Patient/p", '{"id":"p"}')]
    records += [(f"Observation/o{index}", dumps({"value": index,
                  "note": '患者病历记录，日期、数值与单位。"\\' * 80})) for index in range(count)]
    return PatientTrajectory(patient_id="p", environment="PhysicianBench",
        text="\n\n".join(f"[2020-01-01] {source}\n{body}" for source, body in records),
        sources=[Evidence(source_id=source, statement=body) for source, body in records])


def test_small_material_is_unchanged_and_unknown_models_use_bytes():
    trajectory = patient(1)
    body = dumps({"trajectory": trajectory.model_dump(mode="json")})
    assert prepare_input(body, MODEL, instructions="i", schema={})[0] == body
    unknown = MODEL.model_copy(update={"model": "custom-model"})
    assert count_tokens('中文"\\', unknown) == len('中文"\\'.encode("utf-8"))


def test_token_fitting_keeps_original_records_and_cache():
    original = patient()
    before = original.model_dump()
    fitted = fit_patient(original, MODEL)
    assert fitted.input_truncated
    assert count_tokens(dumps(fitted.model_dump(mode="json")), MODEL) <= input_limit(MODEL) // 2
    source_map = {source.source_id: source.statement for source in original.sources}
    assert all(source.statement == source_map[source.source_id] for source in fitted.sources)
    assert len(fitted.text.split("\n\n")) == len(fitted.sources)
    assert original.model_dump() == before


def test_review_growth_preserves_cited_old_record_and_json():
    original = patient()
    candidate = {"evidence": [{"source_id": "Observation/o0", "statement": "原始事实"}],
                 "reference_answer": "候选答案" * 300}
    body = {"trajectory": original.model_dump(mode="json"), "current": candidate,
            "feedback": "修订意见" * 300}
    text, entry = prepare_input(dumps(body), MODEL, instructions="review",
                                schema=TaskBatch.model_json_schema())
    fitted = json.loads(text)["trajectory"]
    assert fitted["input_truncated"]
    assert any(source["source_id"] == "Observation/o0" for source in fitted["sources"])
    assert json.loads(text)["current"] == candidate
    assert entry["input_tokens_estimate"] <= entry["input_limit"]


def test_multi_patient_feedback_fits_joint_budget():
    body = {"trajectories": [patient().model_dump(mode="json") for _ in range(3)],
            "tasks": [{"evidence": [{"source_id": "Observation/o0"}]}]}
    text, entry = prepare_input(dumps(body), MODEL, instructions="feedback", schema={})
    assert entry["input_tokens_estimate"] <= entry["input_limit"]
    assert len(json.loads(text)["trajectories"]) == 3
    assert all(any(source["source_id"] == "Observation/o0" for source in value["sources"])
               for value in json.loads(text)["trajectories"])


@pytest.mark.parametrize("unlocated", [False, True])
def test_impossible_record_or_unlocated_text_fails_without_cutting(unlocated):
    original = PatientTrajectory(patient_id="p", environment="PhysicianBench",
        text=("" if unlocated else "Patient/p\n") + "x" * 10000,
        sources=[Evidence(source_id="Patient/p", statement="x" * 10000)])
    with pytest.raises(ValueError, match="exceed input budget"):
        fit_trajectory(original, 1000)
    assert len(original.text) >= 10000


def test_line_mode_handles_unrendered_sources_before_rendered_records():
    original = PatientTrajectory(patient_id="p", environment="FHIR-AgentBench",
        text="Patient/p patient\n" + "\n".join(f"Observation/o{i} " + "x" * 500 for i in range(8)),
        sources=[Evidence(source_id="Condition/unrendered", statement="hidden"),
                 Evidence(source_id="Patient/p", statement="patient"),
                 *[Evidence(source_id=f"Observation/o{i}", statement="x" * 500) for i in range(8)]])
    fitted = fit_trajectory(original, 1800, protected_source_ids={"Observation/o0"})
    assert payload_chars(fitted) <= 1800
    assert "Condition/unrendered" not in {source.source_id for source in fitted.sources}
    assert "Observation/o0" in {source.source_id for source in fitted.sources}


def test_fixed_material_is_rejected_before_model_call():
    with pytest.raises(ContextBudgetError, match="Fixed stage material"):
        prepare_input("plain text", MODEL, instructions="病历" * 20000, schema={})
    with pytest.raises(ContextBudgetError, match="no input budget"):
        input_limit(MODEL.model_copy(update={"context_window": 500}))


def test_history_drops_complete_pairs_and_keeps_task_and_latest_result():
    model = MODEL.model_copy(update={"context_window": 3000})
    items = [{"role": "system", "content": "instructions"},
             {"role": "user", "content": "patient task"}]
    for index in range(3):
        items.extend([{"role": "assistant", "content": f"GET record-{index}"},
                      {"role": "user", "content": f"record-{index}:" + "数值单位" * 250}])
    selected, dropped = fit_history(items, model, fixed_tokens=0, prefix_count=2,
                                   group_start=lambda item: item["role"] == "assistant")
    assert dropped == 4
    assert selected[:2] == items[:2]
    assert selected[-2:] == items[-2:]
    assert count_tokens(dumps(selected), model) <= input_limit(model)
    assert len(items) == 8


def test_latest_record_is_never_silently_cut():
    items = [{"role": "user", "content": "task"},
             {"role": "assistant", "content": "GET"},
             {"role": "user", "content": "数值单位" * 20000}]
    with pytest.raises(ContextBudgetError, match="latest complete interaction"):
        fit_history(items, MODEL, fixed_tokens=0, prefix_count=1,
                    group_start=lambda item: item["role"] == "assistant")


@pytest.mark.asyncio
async def test_sdk_loop_bounds_history_and_keeps_tool_call_ids_paired(monkeypatch):
    from agents import function_tool
    from contracts import Budget, SolverAnswer
    from runtime import run_agent
    from tests.runtime.test_execution import FINAL_MESSAGE, TOOL_MESSAGE, install_transport

    model = MODEL.model_copy(update={"context_window": 3200, "base_url": "http://model.test/v1"})
    requests = install_transport(monkeypatch, [TOOL_MESSAGE] * 3 + [FINAL_MESSAGE])

    @function_tool
    async def lookup() -> str:
        """Read patient facts."""
        return "字段数值单位" * 200

    result = await run_agent(instructions="Read and answer", input_text="patient task", tools=[lookup],
        output_type=SolverAnswer, model=model, budget=Budget(max_model_turns=5, max_tool_calls=4))
    assert result.status == "completed", result.error
    assert result.usage["model_turns"] == len(requests) == 4
    assert result.usage["tool_calls"] == 3
    assert any(item.get("history_items_dropped", 0) > 0 for item in result.trace)
    for request in requests:
        calls = {call["id"] for message in request["messages"] for call in message.get("tool_calls", [])}
        outputs = {message["tool_call_id"] for message in request["messages"] if message["role"] == "tool"}
        assert calls == outputs
        tokens = sum(count_tokens(message.get("content") or "", model) + 8 for message in request["messages"])
        tokens += count_tokens(dumps(request.get("tools", [])) + dumps(request["response_format"]), model)
        assert tokens + request["max_tokens"] < model.context_window


@pytest.mark.asyncio
async def test_native_loop_bounds_history_without_losing_latest_result(monkeypatch):
    from contracts import Budget, SolverAnswer
    from runtime import run_agent
    from tests.runtime.test_native import install_transport, native_tools

    model = MODEL.model_copy(update={"context_window": 3600, "base_url": "http://model.test/v1"})
    responses = [{"role": "assistant", "content": f"GET Observation?_id={index}"} for index in range(3)]
    requests = install_transport(monkeypatch, responses + [{"role": "assistant", "content": 'FINISH(["result"])'}])
    tools = native_tools()

    async def execute(action):
        if action.startswith("FINISH"):
            return {"kind": "FINISH", "answers": ["result"], "reply": "FINISH"}
        return {"kind": "GET", "reply": action + "字段数值单位" * 200}

    monkeypatch.setattr(tools, "execute", execute)
    result = await run_agent(instructions="Read and answer", input_text="patient task", tools=tools,
        output_type=SolverAnswer, model=model, budget=Budget(max_model_turns=5, max_tool_calls=4))
    assert result.status == "completed", result.error
    assert result.usage["model_turns"] == len(requests) == 4
    assert any(item.get("history_items_dropped", 0) > 0 for item in result.trace)
    assert "_id=2" in requests[-1]["messages"][-1]["content"]
    assert all(sum(count_tokens(message["content"], model) + 8 for message in request["messages"])
               + request["max_tokens"] < model.context_window for request in requests)


@pytest.mark.asyncio
async def test_unfittable_request_uses_no_model_turns(monkeypatch):
    from runtime import call_llm, execution

    def unexpected_client(model):
        raise AssertionError("request must be rejected before client creation")

    monkeypatch.setattr(execution, "_client", unexpected_client)
    result = await call_llm(instructions="fixed" * 20000, input_text="patient task",
                            output_type=TaskBatch, model=MODEL)
    assert result.status == "failed"
    assert "ContextBudgetError" in result.error
    assert result.usage["model_turns"] == 0


@pytest.mark.asyncio
async def test_taskcraft_request_keeps_task_current_plan_and_latest_complete_step(monkeypatch):
    from contracts import Budget, RuntimeConfig
    from methods.common import GenerationBudget
    from methods.taskcraft.agent import _SearchRun, _memory_messages
    from methods.taskcraft.ledger import Ledger
    from methods.taskcraft.schemas import AgentThought, AnswerList
    from tests.runtime.test_execution import install_transport

    model = MODEL.model_copy(update={"base_url": "http://model.test/v1"})
    ledger = Ledger(GenerationBudget(RuntimeConfig(models={"generator": model},
        budgets={"generator": Budget(max_model_turns=10, max_tool_calls=0)})))
    search = _SearchRun(ledger=ledger, stage="search", instructions="read", task="患者任务",
                        tools=[], output_type=AnswerList, max_action_steps=20)
    search.memory.append({"kind": "planning", "facts": "当前事实", "plan": "当前计划"})
    search.memory.extend({"kind": "action", "step": index,
                           "observations": f"record-{index}:" + "患者病历数值" * 500}
                          for index in range(20))
    original = dumps(search.memory)
    requests = install_transport(monkeypatch, [{"role": "assistant", "content": dumps(
        {"think": "依据材料", "answer": "结论"})}])
    result = await search.request("reflection", kind="reflection", step=20, instructions="review",
                                  input_text=_memory_messages(search.memory), output_type=AgentThought)
    assert result.status == "completed", result.error
    sent = requests[0]["messages"][-1]["content"]
    assert "患者任务" in sent and "当前计划" in sent and "record-19:" in sent
    assert "record-0:" not in sent
    assert any(item["type"] == "context_history" for item in search.trace)
    assert dumps(search.memory) == original
    assert len(requests) == search.usage["model_turns"] == 1
