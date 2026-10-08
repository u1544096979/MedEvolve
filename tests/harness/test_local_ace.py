"""ACE 适配在锁定上游源码边界使用假的模型与医疗工具。"""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from contracts import SolverInput
from harnesses.contracts import GroupKey, Snapshot, TrainingFeedback
from harnesses.runners import ace_worker
from harnesses.runners.ace import ACERunner

UPSTREAM = Path(__file__).parents[2] / ".superpowers/sdd/implementation-plan/upstreams/ace"


class FakeResponse:
    def __init__(self, content=None, tool_calls=None, prompt_tokens=11, completion_tokens=7):
        self.choices=[SimpleNamespace(message=FakeMessage(content,tool_calls))]
        self.usage=SimpleNamespace(prompt_tokens=prompt_tokens,completion_tokens=completion_tokens)

class FakeMessage:
    def __init__(self, content, tool_calls): self.content=content; self.tool_calls=tool_calls
    def model_dump(self,exclude_none=True):
        return {"role":"assistant","content":self.content,"tool_calls":[{"id":c.id,"type":"function","function":{"name":c.function.name,"arguments":c.function.arguments}} for c in self.tool_calls]}

class FakeCompletions:
    def __init__(self, queue): self.queue=queue; self.requests=[]
    def create(self, **kwargs): self.requests.append(kwargs); return self.queue.pop(0)


def test_locked_ace_generator_reflector_curator_and_tool_client(monkeypatch,tmp_path):
    import openai
    import sys, types
    dotenv=types.ModuleType("dotenv"); dotenv.load_dotenv=lambda *a,**k:None
    monkeypatch.setitem(sys.modules,"dotenv",dotenv)
    tool=SimpleNamespace(id="call-1",function=SimpleNamespace(name="read_record",arguments='{"patient":"p1"}'))
    completions=FakeCompletions([
        FakeResponse(tool_calls=[tool]),
        FakeResponse('{"reasoning":"review evidence","bullet_ids":[],"final_answer":"answer-one"}'),
        FakeResponse('{"reasoning":"check evidence","error_identification":"missing context","root_cause_analysis":"short review","correct_approach":"query needed evidence","key_insight":"use evidence","bullet_tags":[]}'),
        FakeResponse('{"reasoning":"add safe workflow","operations":[{"type":"ADD","section":"workflow","content":"Cross-check evidence before concluding."}]}',prompt_tokens=22,completion_tokens=9),
        FakeResponse('{"reasoning":"review another feedback","error_identification":"missed a source","root_cause_analysis":"stopped early","correct_approach":"check all relevant records","key_insight":"verify scope","bullet_tags":[]}'),
        FakeResponse('{"reasoning":"add verification step","operations":[{"type":"ADD","section":"workflow","content":"Confirm the requested scope before finalizing."}]}'),
    ])
    monkeypatch.setattr(openai,"OpenAI",lambda **kwargs:SimpleNamespace(chat=SimpleNamespace(completions=completions)))
    monkeypatch.setattr(ace_worker,"_get",lambda *a:[{"name":"read_record","description":"read", "parameters":{"type":"object","properties":{},"additionalProperties":False}}])
    calls=[]
    monkeypatch.setattr(ace_worker,"_post",lambda *a:(calls.append(a) or {"output":"record"}))
    monkeypatch.setenv("FAKE_KEY","not-a-secret")
    req={"source_dir":str(UPSTREAM),"state_dir":str(tmp_path),"model":{"model":"fake","api_key_env":"FAKE_KEY"},
        "bridge_url":"http://bridge","client_id":"test","tool_session":"session","max_tool_calls":3,
        "max_model_turns":10,"max_output_tokens":100,"instruction":"read record and answer",
        "patient_id":"p1","decision_point":"medication review"}
    executed=ace_worker.main({**req,"action":"execute"})
    assert executed["tool_calls"] == 1 and calls
    assert json.loads(executed["raw"])["final_answer"] == "answer-one"
    assert "GENERATOR_PROMPT" not in completions.requests[0]["messages"][1]["content"]
    assert "playbook" in completions.requests[0]["messages"][1]["content"].lower()
    assert completions.requests[0]["tools"][0]["function"]["parameters"]["additionalProperties"] is False
    assert completions.requests[0]["response_format"]=={"type":"json_object"}
    assert completions.requests[0]["max_completion_tokens"]==100
    assert "p1" in completions.requests[0]["messages"][1]["content"]
    assert "medication review" in completions.requests[0]["messages"][1]["content"]

    learned=ace_worker.main({**req,"action":"learn","tool_session":"","answer":"answer-one",
        "trajectory":"task-local trajectory","bullets_used":[],"feedback":{"completed":False,
        "reference_answer":"ground truth only for training","reason":"clinical feedback"},
        "question_context":"reusable task type"})
    learned_again=ace_worker.main({**req,"action":"learn","tool_session":"","answer":"answer-one",
        "trajectory":"task-local trajectory","bullets_used":[],"feedback":{"completed":False,
        "reference_answer":"ground truth only for training","reason":"another clinical feedback"},
        "question_context":"reusable task type","current_step":2})
    playbook=(tmp_path/"playbook.md").read_text()
    assert learned["learning_status"] == learned_again["learning_status"] == "updated"
    assert learned["usage"]["prompt_tokens"]==33 and learned["usage"]["completion_tokens"]==16
    assert [item["usage"]["calls"] for item in learned["learning_runs"]]==[1,1]
    assert [item["usage"]["prompt_tokens"] for item in learned["learning_runs"]]==[11,22]
    assert "[misc-00001] helpful=0 harmful=0 :: Cross-check evidence before concluding." in playbook
    assert "[misc-00002] helpful=0 harmful=0 :: Confirm the requested scope before finalizing." in playbook
    assert len(completions.requests) == 6
    assert all(item["response_format"]=={"type":"json_object"} and item["max_completion_tokens"]==100 for item in completions.requests)


@pytest.mark.asyncio
async def test_ace_runner_passes_full_solver_context_and_separate_budgets(tmp_path):
    runner=ACERunner(config={"execution_model":{"model":"m","max_output_tokens":37},
        "execution_budget":{"max_model_turns":7,"max_tool_calls":23}},group_dir=tmp_path,bridge_url="http://bridge")
    captured={}
    async def call(payload,phase):
        captured.update(payload)
        return {"raw":json.dumps({"final_answer":"ok","reasoning":""}),"bullet_ids":[],"tool_calls":0,
            "tool_trace":[],"usage":{"known":True,"calls":1}},None
    runner._call=call
    await runner.execute(SolverInput(instruction="task",patient_id="patient-a",decision_point="dose adjustment"),"session")
    assert captured["patient_id"]=="patient-a" and captured["decision_point"]=="dose adjustment"
    assert captured["max_model_turns"]==7 and captured["max_tool_calls"]==23 and captured["max_output_tokens"]==37


def test_ace_client_shim_preserves_json_mode_and_clamps_official_token_parameter(monkeypatch):
    import openai
    completions=FakeCompletions([FakeResponse('{"final_answer":"ok"}')])
    monkeypatch.setattr(openai,"OpenAI",lambda **kwargs:SimpleNamespace(chat=SimpleNamespace(completions=completions)))
    monkeypatch.setattr(ace_worker,"_get",lambda *a:[])
    shim=ace_worker.ACEClientShim("fake","key",None,"http://bridge","","client",2,100,23)
    shim.create(model="fake",messages=[{"role":"user","content":"prompt"}],max_completion_tokens=700,
        response_format={"type":"json_object"},temperature=0)
    assert completions.requests[0]["response_format"]=={"type":"json_object"}
    assert completions.requests[0]["max_completion_tokens"]==23


def test_ace_client_shim_model_turn_limit_is_separate_from_tool_limit(monkeypatch):
    import openai
    tool=SimpleNamespace(id="call",function=SimpleNamespace(name="read_record",arguments="{}"))
    completions=FakeCompletions([FakeResponse(tool_calls=[tool]),FakeResponse(tool_calls=[tool])])
    monkeypatch.setattr(openai,"OpenAI",lambda **kwargs:SimpleNamespace(chat=SimpleNamespace(completions=completions)))
    monkeypatch.setattr(ace_worker,"_get",lambda *a:[{"name":"read_record","description":"read","parameters":{}}])
    calls=[];monkeypatch.setattr(ace_worker,"_post",lambda *a:(calls.append(a) or {"ok":True}))
    shim=ace_worker.ACEClientShim("fake","key",None,"http://bridge","session","client",2,50,100)
    with pytest.raises(RuntimeError,match="ACE_MODEL_TURN_BUDGET_EXHAUSTED"):
        shim.create(model="fake",messages=[{"role":"user","content":"prompt"}],max_completion_tokens=100)
    assert len(completions.requests)==2 and len(calls)==1


def test_ace_client_shim_counts_failed_requests_before_upstream_retries(monkeypatch):
    import openai

    class TimeoutCompletions:
        def __init__(self): self.requests=[]
        def create(self, **kwargs):
            self.requests.append(kwargs)
            raise TimeoutError("request timed out")

    completions=TimeoutCompletions()
    monkeypatch.setattr(openai,"OpenAI",lambda **kwargs:SimpleNamespace(chat=SimpleNamespace(completions=completions)))
    monkeypatch.setattr(ace_worker,"_get",lambda *a:[])
    shim=ace_worker.ACEClientShim("fake","key",None,"http://bridge","","client",3,20,100)
    for _ in range(3):
        with pytest.raises(TimeoutError,match="timed out"):
            shim.create(model="fake",messages=[{"role":"user","content":"prompt"}],max_tokens=100)
    with pytest.raises(RuntimeError,match="ACE_MODEL_TURN_BUDGET_EXHAUSTED"):
        shim.create(model="fake",messages=[{"role":"user","content":"prompt"}],max_tokens=100)
    assert len(completions.requests)==shim.calls==3
    assert shim.usage_known is False


@pytest.mark.asyncio
async def test_ace_frozen_learning_does_not_call_native_worker(tmp_path):
    runner=ACERunner(config={"frozen":True,"command_runner":lambda *a,**k:pytest.fail("worker called")},
        group_dir=tmp_path/"group",bridge_url="http://bridge")
    initial=tmp_path/"initial"; initial.mkdir(); (initial/"playbook.md").write_text("## baseline\n",encoding="utf-8")
    await runner.start(GroupKey(run_id="r",environment="PhysicianBench",harness="ace",source_method="medevolve",seed=42),
        Snapshot(path=initial,digest="before"))
    before=(runner.state/"playbook.md").read_bytes()
    result=await runner.learn(TrainingFeedback(task_id="t",completed=False,failure_category="clinical",reason="r",reference_answer="gold",evidence=[]))
    assert result.learning_status == "failed"
    assert (runner.state/"playbook.md").read_bytes()==before
