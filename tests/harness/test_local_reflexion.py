"""Reflexion actor/reflector 真实上游代码的无付费 transport 测试。"""
import json, os, sys, types
from pathlib import Path

import pytest
from contracts import SolverInput
from harnesses.contracts import GroupKey, Snapshot, TrainingFeedback
from harnesses.runners import reflexion_worker
from harnesses.runners.reflexion import ReflexionRunner

UPSTREAM=Path(__file__).parents[2]/".superpowers/sdd/implementation-plan/upstreams/reflexion"

class PromptTemplate:
    def __init__(self,input_variables,template):self.input_variables=input_variables;self.template=template
    def format(self,**kwargs):return self.template.format(**kwargs)
class DummyDocstore: pass
class DummyExplorer:
    def __init__(self,docstore):pass


def install_legacy_langchain(monkeypatch):
    langchain=types.ModuleType("langchain"); langchain.__path__=[]
    langchain.OpenAI=lambda *a,**k:object(); langchain.Wikipedia=lambda *a,**k:DummyDocstore()
    chat=types.ModuleType("langchain.chat_models"); chat.__path__=[]; chat.ChatOpenAI=lambda *a,**k:object()
    chatbase=types.ModuleType("langchain.chat_models.base"); chatbase.BaseChatModel=type("BaseChatModel",(),{})
    schema=types.ModuleType("langchain.schema")
    for name in ("SystemMessage","HumanMessage","AIMessage"): setattr(schema,name,type(name,(),{}))
    agents=types.ModuleType("langchain.agents"); agents.__path__=[]
    react=types.ModuleType("langchain.agents.react"); react.__path__=[]
    reactbase=types.ModuleType("langchain.agents.react.base"); reactbase.DocstoreExplorer=DummyExplorer
    docstore=types.ModuleType("langchain.docstore"); docstore.__path__=[]
    docbase=types.ModuleType("langchain.docstore.base"); docbase.Docstore=DummyDocstore
    prompts=types.ModuleType("langchain.prompts"); prompts.PromptTemplate=PromptTemplate
    llms=types.ModuleType("langchain.llms"); llms.__path__=[]
    llmbase=types.ModuleType("langchain.llms.base"); llmbase.BaseLLM=type("BaseLLM",(),{})
    for name,module in {"langchain":langchain,"langchain.chat_models":chat,"langchain.chat_models.base":chatbase,
        "langchain.schema":schema,"langchain.agents":agents,"langchain.agents.react":react,
        "langchain.agents.react.base":reactbase,"langchain.docstore":docstore,"langchain.docstore.base":docbase,
        "langchain.prompts":prompts,"langchain.llms":llms,"langchain.llms.base":llmbase}.items():
        monkeypatch.setitem(sys.modules,name,module)
    monkeypatch.setattr("tiktoken.encoding_for_model",lambda name:types.SimpleNamespace(encode=lambda text:text.split()))


@pytest.mark.parametrize("separator", ["|", ","])
@pytest.mark.parametrize("fenced_experience", [False, True])
def test_locked_reflexion_actor_runs_medical_tool_and_official_reflection(monkeypatch,tmp_path,separator,fenced_experience):
    install_legacy_langchain(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY","fake")
    for name in ("llm","prompts","fewshots","utils","logger","reflexion_locked_agents"):
        if name in sys.modules: monkeypatch.delitem(sys.modules,name)
    prompts_seen=[]
    experience_json='{"condition":"diagnostic task","procedure":"Review relevant records","pitfall":"Avoid relying on one finding"}'
    outputs=iter(["review the task " * 900,f"Tool[read_record{separator}{{}}]","use evidence " * 900,"Finish[secret-answer]",
        "Review evidence before concluding.",
        f"```json\n{experience_json}\n```" if fenced_experience else experience_json])
    monkeypatch.setattr(reflexion_worker,"_chat_completion",lambda model,key,prompt,tokens,stop=None:(prompts_seen.append(prompt) or {
        "choices":[{"message":{"content":next(outputs)}}],"usage":{"prompt_tokens":4,"completion_tokens":3}}))
    tool_schema=[{"name":"read_record","description":"read scoped record","parameters":{"type":"object","properties":{}}}]
    tool_calls=[]
    def fake_request(url,payload=None,client_id=""):
        if url.endswith("/tools"):return tool_schema
        tool_calls.append(payload);return {"status":"ok","output":"record"}
    monkeypatch.setattr(reflexion_worker,"_request",fake_request)
    state=tmp_path/"state"; model={"model":"fake","api_key_env":"FAKE_KEY","base_url":"http://model/v1"}
    monkeypatch.setenv("FAKE_KEY","fake")
    req={"action":"execute","source_dir":str(UPSTREAM),"state_dir":str(state),"bridge_url":"http://bridge",
        "tool_session":"fresh-session","client_id":"test","model":model,"instruction":"Review the medication plan " + "medical context " * 3000,
        "patient_id":"patient-007","decision_point":"medication","reset_reflections":True,"max_steps":8,
        "metadata":{"task_type":"药物处方","tool_names":["read_record"]}}
    executed=reflexion_worker.main(req)
    assert executed["native"]=="ReactReflectAgent.run"
    assert executed["answer"]=="secret-answer" and tool_calls
    assert any("read_record" in prompt and '"parameters"' in prompt for prompt in prompts_seen)
    assert any("Decision point: medication" in prompt for prompt in prompts_seen)
    assert tool_calls[0]=={"name":"read_record","arguments":{}}
    assert not any("gold-answer" in prompt for prompt in prompts_seen)

    learned=reflexion_worker.main({**req,"action":"learn","tool_session":"","scratchpad":executed["scratchpad"],
        "feedback":{"task_id":"public-task-1","completed":False,"failure_category":"clinical",
            "reason":"The completion review found the evidence was not reconciled.","reference_answer":"gold-answer"}})
    assert learned["native"]=="ReactReflectAgent.reflect" and learned["experience"]
    experience_file=state/"experiences.json"
    content=experience_file.read_text()
    assert "patient-007" not in content and "secret-answer" not in content and "gold-answer" not in content
    assert json.loads(content)[0]["tool_names"]==["read_record"]
    next_task=SolverInput(instruction="review a different medication plan",patient_id="patient-008")
    runner=ReflexionRunner(config={"task_metadata":{ReflexionRunner.stable_task_key(next_task):
        {"task_type":"药物处方","tool_names":["read_record"]}}},group_dir=tmp_path/"next-group",bridge_url="http://bridge")
    runner.state=state
    assert runner._select_experiences(next_task)[0]["source_task_id"]=="public-task-1"


@pytest.mark.asyncio
async def test_reflexion_retry_uses_host_new_session_and_cross_task_selection(tmp_path):
    task=SolverInput(instruction="review medication",patient_id="p1",decision_point="dose adjustment")
    runner=ReflexionRunner(config={"reflexion_max_retries":2,"experience_limit":{"max_entries":3,"max_context_tokens":1000}},
        group_dir=tmp_path/"group",bridge_url="http://bridge")
    runner.config["task_metadata"]={runner.stable_task_key(task):{"task_type":"药物处方","tool_names":["a"]}}
    initial=tmp_path/"initial";initial.mkdir();(initial/"experiences.json").write_text(json.dumps([
        {"id":"z","task_type":"治疗规划","tool_names":["a"],"condition":"c","procedure":"p","pitfall":"q","source_task_id":"s2"},
        {"id":"b","task_type":"药物处方","tool_names":[],"condition":"c","procedure":"p","pitfall":"q","source_task_id":"s1"},
        {"id":"a","task_type":"药物处方","tool_names":["a"],"condition":"c","procedure":"p","pitfall":"q","source_task_id":"s0"}],ensure_ascii=False))
    await runner.start(GroupKey(run_id="r",environment="PhysicianBench",harness="reflexion",source_method="medevolve",seed=42),Snapshot(path=initial,digest="d"))
    selected=runner._select_experiences(task)
    assert [x["id"] for x in selected]==["a","b","z"]
    calls=[]
    async def fake_call(payload,phase):
        calls.append((payload,phase))
        if payload["action"]=="execute":return {"answer":"answer","scratchpad":"trace","usage":{"calls":2}},None
        return {"learning_status":"updated","reflection":"reflect","usage":{"calls":1},"experience":{"id":"e"},"native":"ReactReflectAgent.reflect"},None
    runner._call=fake_call
    await runner.execute(task,"instance-1")
    assert calls[-1][0]["decision_point"]=="dose adjustment"
    first=await runner.learn(TrainingFeedback(task_id="t",completed=False,failure_category="clinical",reason="why",reference_answer="gold",evidence=[]))
    assert first.learning_status=="updated" and runner.retry_state["ready"]
    await runner.execute_retry(task,"instance-2")
    assert calls[-1][0]["tool_session"]=="instance-2" and calls[-1][0]["reset_reflections"] is False
    assert runner.retry_state["attempts"]==2


@pytest.mark.asyncio
async def test_reflexion_frozen_mode_never_reflects_or_retries(tmp_path):
    task=SolverInput(instruction="task",patient_id="p")
    runner=ReflexionRunner(config={"frozen":True},group_dir=tmp_path/"g",bridge_url="http://bridge")
    calls=[]
    async def call(payload,phase):
        calls.append(payload["action"])
        if payload["action"]=="learn":pytest.fail("frozen learning invoked worker")
        return {"answer":"done","scratchpad":"trace","usage":{"calls":1,"tokens_known":True}},None
    runner._call=call
    initial=tmp_path/"initial";initial.mkdir();(initial/"experiences.json").write_text("[]")
    await runner.start(GroupKey(run_id="r",environment="PhysicianBench",harness="reflexion",source_method="medevolve",seed=42),Snapshot(path=initial,digest="d"))
    before=(runner.state/"experiences.json").read_bytes()
    await runner.execute(task,"readonly-session")
    result=await runner.learn(TrainingFeedback(task_id="t",completed=False,failure_category="clinical",reason="why",reference_answer="gold",evidence=[]))
    assert result.learning_status=="not_triggered" and runner.retry_state["retries_remaining"]==0
    assert calls==["execute"] and (runner.state/"experiences.json").read_bytes()==before
