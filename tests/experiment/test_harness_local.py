"""Harness orchestration uses only local fake runners, bridges and environments."""
from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from contracts import EnvironmentConfig, PatientTrajectory, RunResult, Task
from experiments import harness_test, harness_train
from harnesses.contracts import HarnessStep, Snapshot, TrainingFeedback
from harnesses.datasets import freeze_datasets
from harnesses.snapshots import directory_digest


class FakeSessions:
    def __init__(self):
        self.events=[]; self.counter=0; self.active={}; self.environments=self
    async def open(self, task, environment, *, client_id, read_only, tool_budget, timeout_seconds):
        self.counter+=1; sid=f"session-{self.counter}"; instance=SimpleNamespace(id=sid)
        self.events.append(("open",sid,client_id,read_only)); self.active[sid]=instance; return sid
    def inspect(self, sid, *, client_id):
        return SimpleNamespace(instance=self.active[sid],events=[],calls=[])
    async def list(self,sid,*,client_id):
        return [{"name":"fhir_condition_search_problems"},{"name":"medical_action"}]
    async def read_state(self,instance,task):
        return {"patient_id":task.patient_id,"observed":True}
    def _state_task(self,context):
        return SimpleNamespace(patient_id="fixture")
    async def close(self,sid,*,client_id):
        instance=self.active.pop(sid); self.events.append(("release",sid,client_id))
        return {"session_id":sid,"closed":True,"calls":[],"events":[],"state":{},"release_error":None}
    async def aclose(self):
        for sid in list(self.active): await self.close(sid,client_id="cleanup")


class FakeRunner:
    def __init__(self,harness,config,group_dir):
        self.harness=harness; self.config=config; self.group_dir=Path(group_dir); self.events=config["events"]
        self.bridge_url=""; self.retry_ready=False; self.current_task=None; self.retry_used=False; self.last_session=None
        self.frozen=bool(config.get("frozen")); self.started_snapshot=None
    async def start(self,group,initial):
        self.started_snapshot=initial; self.events.append(("start",self.harness,str(initial.path),self.frozen,str(self.group_dir)))
    async def execute(self,task,session):
        self.current_task=task; self.retry_used=False; self.last_session=session
        self.events.append(("execute",self.harness,session))
        return RunResult(status="completed",output={"answer":"fixture answer"},trace=[],
                         usage={"known":True,"input_tokens":10,"output_tokens":4},error=None)
    async def execute_retry(self,task,new_tool_session):
        assert self.harness=="reflexion" and self.retry_ready
        assert any(event[0]=="release" and event[1]==self.last_session for event in self.events)
        self.retry_ready=False; self.retry_used=True; self.last_session=new_tool_session
        self.events.append(("retry",new_tool_session))
        return RunResult(status="completed",output={"answer":"fixture retry"},trace=[],
                         usage={"known":True,"input_tokens":5,"output_tokens":2},error=None)
    @property
    def retry_state(self): return {"ready":self.retry_ready}
    async def learn(self,feedback):
        assert any(event[0]=="release" and event[1]==self.last_session for event in self.events)
        self.events.append(("learn",self.harness,feedback.task_id))
        if self.harness=="reflexion" and not self.retry_used:
            self.retry_ready=True
        summary=RunResult(status="completed",output={"learning_status":"updated"},trace=[{"type":"learning_feedback_turn"}],
                          usage={"known":True,"calls":2,"input_tokens":8,"output_tokens":2})
        if self.harness=="hermes": detail=[summary]
        elif self.harness=="openclaw":
            detail=[RunResult(status="completed",output={"component":"native"},trace=[],usage={"known":True,"input_tokens":2,"output_tokens":1})]
        else:
            detail=[RunResult(status="completed",output={"component":"reflect"},trace=[],usage={"known":True,"input_tokens":3,"output_tokens":1}),
                    RunResult(status="completed",output={"component":"curate"},trace=[],usage={"known":True,"input_tokens":2,"output_tokens":1})]
        return HarnessStep(run=summary,learning_runs=detail,learning_status="updated")
    async def freeze(self,destination):
        path=Path(destination); path.mkdir(parents=True,exist_ok=True)
        (path/"experience.json").write_text("[]")
        self.events.append(("freeze",self.harness))
        return Snapshot(path=path,digest=directory_digest(path))
    async def close(self): self.events.append(("runner_close",self.harness))


@pytest.fixture
def rig(tmp_path,monkeypatch):
    tasks=[]; evaluations=[]
    for i in range(10):
        task={"id":f"t{i}","instruction":f"题目 {i}","task_type":"治疗规划",
              "reference_answer":f"gold {i}","evidence":[{"source_id":"Condition/c1","statement":"证据"}],
              "decision_point":None,"patient_id":f"p{i%5}","environment":"PhysicianBench",
              "round_index":0,"source_method":"medevolve"}
        tasks.append(task)
        evaluations.append({"task_id":f"t{i}","verification":{"valid":True,"reason":"ok","read_only_task":True},
            "completion":{"completed":True,"reason":"baseline"},"observed_state":{},"status":"completed","runs":[]})
    source=tmp_path/"source"; source.mkdir()
    (source/"tasks.json").write_text(json.dumps(tasks)); (source/"evaluations.json").write_text(json.dumps(evaluations))
    frozen=tmp_path/"frozen"/"PhysicianBench"/"medevolve"
    freeze_datasets({"medevolve":source/"tasks.json"},count=10,seed=42,destination=frozen)
    template=tmp_path/"template"; (template/"skills").mkdir(parents=True); (template/"memory").mkdir()
    events=[]
    config={"run_id":"test","execute":True,"output_root":str(tmp_path/"out"),
        "run_dir_template":str(tmp_path/"out/<run_id>/<environment>/<harness>/<source_method>/<seed>"),
        "frozen_root":str(tmp_path/"frozen"),"tasks_per_group":10,"evolution_rounds":1,"seeds":[42],
        "harnesses":["openclaw","hermes","ace","reflexion"],"source_methods":["medevolve"],
        "models":{"execution":{"model":"fake","api_key_env":"FAKE_EXEC"},
          "learning":{"model":"fake-learning","api_key_env":"FAKE_LEARN"},
          "completion_review":{"model":"fake-review","api_key_env":"FAKE_REVIEW"}},
        "limits":{"execution":{"max_model_turns":5,"max_tool_calls":10},"learning":{"max_model_turns":3,"max_tool_calls":0},
          "timeout_seconds":10,"learning_timeout_seconds":10},
        "environments":{"PhysicianBench":{"name":"PhysicianBench","manager_url":"unused","seed_data":"fake",
            "sources":{"medevolve":{"tasks":str(source/"tasks.json")}}}},
        "local_harnesses":{name:{"source_dir":str(tmp_path/"src"),"program_path":str(tmp_path/"prog"),
             "python_path":str(Path(os.sys.executable)),"upstream_commit":"x","initial_state_template":str(template)}
             for name in ("openclaw","hermes","ace","reflexion")},
        "upstreams_lock":"docker/harnesses/upstreams.lock.json","events":events}
    monkeypatch.setenv("MEDEVOLVE_HARNESS_LIVE","1")
    monkeypatch.setenv("FAKE_EXEC","fake"); monkeypatch.setenv("FAKE_LEARN","fake"); monkeypatch.setenv("FAKE_REVIEW","fake")
    sessions=FakeSessions(); sessions.events=events
    monkeypatch.setattr("harnesses.medical_bridge.ToolSessions",lambda:sessions)
    async def fake_prepare(instance,patient): return PatientTrajectory(patient_id=patient,environment="PhysicianBench",text="frozen",sources=[])
    monkeypatch.setattr("environments.prepare_trajectory",fake_prepare)
    @asynccontextmanager
    async def no_server(_sessions): yield "http://127.0.0.1:12345"
    monkeypatch.setattr(harness_train,"bridge_server",no_server)
    monkeypatch.setattr(harness_test,"bridge_server",no_server)
    monkeypatch.setattr(harness_train,"create_runner",lambda name,config,group_dir,bridge_url:FakeRunner(name,{**config,"events":events},group_dir))
    monkeypatch.setattr(harness_test,"create_runner",lambda name,config,group_dir,bridge_url:FakeRunner(name,{**config,"events":events},group_dir))
    async def fake_feedback(task,trajectory,run,state,config,*,record=None):
        if record: record(RunResult(status="completed",output={"completed":False,"reason":"fake"},trace=[],usage={"known":True,"input_tokens":2,"output_tokens":1}))
        return TrainingFeedback(task_id=task.id,completed=False,failure_category="clinical",reason="fake review",
                                reference_answer=task.reference_answer,evidence=[])
    monkeypatch.setattr(harness_train,"make_training_feedback",fake_feedback)
    monkeypatch.setattr(harness_test,"make_training_feedback",fake_feedback)
    return config,events,sessions


@pytest.mark.asyncio
@pytest.mark.parametrize("harness",["openclaw","hermes","ace","reflexion"])
async def test_train_and_frozen_test_run_four_harnesses_with_shared_7_3_split(rig,harness,tmp_path,monkeypatch):
    config,events,sessions=rig
    monkeypatch.setattr(harness_train,"preflight",lambda config,combos,mode="train":[{"ok":True}])
    monkeypatch.setattr(harness_test,"preflight",lambda config,combos,mode="test":[{"ok":True}])
    trained=await harness_train.run_training(config,"PhysicianBench",harness,"medevolve")
    assert len(trained["train_task_ids"])==7 and len(trained["test_task_ids"])==3
    assert len({row["task_id"] for row in trained["feedback"]})==7
    assert len({a["tool_session"]["session_id"] for a in trained["attempts"]})==len(trained["attempts"])
    assert all(a["tool_session"]["closed"] for a in trained["attempts"])
    assert all(a["feedback"] is None or a["feedback"]["completed"] is False for a in trained["attempts"])
    assert all(a["retry"] is False for a in trained["attempts"] if harness!="reflexion")
    assert len([e for e in events if e[0]=="learn" and e[1]==harness])==7 + (7 if harness=="reflexion" else 0)
    expected_learning_runs={"openclaw":14,"hermes":7,"ace":14,"reflexion":28}[harness]
    assert len(trained["learning_runs"])==expected_learning_runs
    if harness=="reflexion":
        assert len([e for e in events if e[0]=="retry"])==7
    train_path=Path(config["run_dir_template"].replace("<run_id>","test").replace("<environment>","PhysicianBench").replace("<harness>",harness).replace("<source_method>","medevolve").replace("<seed>","42"))
    (train_path/"train.json").write_text(json.dumps(trained))
    learn_before=sum(1 for e in events if e[0]=="learn" and e[1]==harness)
    result=await harness_test.run_test(config,"PhysicianBench",harness,"medevolve")
    assert result["planned_denominator"]==3
    assert result["valid_denominators"]=={"initial":3,"final":3}
    assert result["snapshots_unchanged"] is True
    assert sum(1 for e in events if e[0]=="learn" and e[1]==harness)==learn_before
    assert len([e for e in events if e[0]=="start" and e[1]==harness and e[3]])==6
    assert len({e[4] for e in events if e[0]=="start" and e[1]==harness and e[3]})==6
    learn_before=sum(1 for e in events if e[0]=="learn" and e[1]==harness)
    assert learn_before==7 + (7 if harness=="reflexion" else 0)
    assert not sessions.active


def test_preflight_checks_only_selected_source_and_does_not_require_execute(rig):
    config,_,_=rig
    config["execute"]=False
    config["environments"]["PhysicianBench"]["sources"]={"medevolve":config["environments"]["PhysicianBench"]["sources"]["medevolve"],
                                                                  "workflow":{"tasks":None,"missing_reason":"not supplied"}}
    report=harness_train.preflight(config,[("PhysicianBench","ace","medevolve")])
    assert len(report)==1 and report[0]["source"]=="medevolve"
    assert report[0]["data"]["status"]=="ok"


def test_official_import_probe_uses_locked_loader_without_network_or_real_credentials(monkeypatch,tmp_path):
    seen={}
    def fake_run(command,**kwargs):
        seen["command"]=command; seen.update(kwargs)
        return SimpleNamespace(stdout='{"official_import_ok": true}',stderr="",returncode=0)
    monkeypatch.setattr(harness_train.subprocess,"run",fake_run)
    result=harness_train._official_import_probe("reflexion",tmp_path,Path("/fake/python"))
    assert result=={"official_import_ok":True}
    probe=seen["command"][2]
    assert "_official(sys.argv[1], 'probe-only-invalid')" in probe
    assert "socket.socket.connect=deny" in probe and "urllib.request.urlopen=deny" in probe
    assert seen["env"]["PYTHON_DOTENV_DISABLED"]=="1"
    assert seen["env"]["OPENAI_API_KEY"]=="probe-only-invalid"


def test_freeze_manifest_isolated_by_environment_and_source(rig,tmp_path):
    config,_,_=rig
    med_source=tmp_path/"medagent"; med_source.mkdir()
    task_rows=json.loads((tmp_path/"source"/"tasks.json").read_text())
    evaluation_rows=json.loads((tmp_path/"source"/"evaluations.json").read_text())
    for row in task_rows: row["environment"]="MedAgentBench"
    (med_source/"tasks.json").write_text(json.dumps(task_rows))
    (med_source/"evaluations.json").write_text(json.dumps(evaluation_rows))
    config["environments"]["MedAgentBench"]={"name":"MedAgentBench","manager_url":"unused","seed_data":"fake",
        "sources":{"medevolve":{"tasks":str(med_source/"tasks.json")}}}
    harness_train.ensure_frozen(config,"PhysicianBench","medevolve")
    harness_train.ensure_frozen(config,"MedAgentBench","medevolve")
    p1=Path(config["frozen_root"])/"PhysicianBench"/"medevolve"/"manifest.json"
    p2=Path(config["frozen_root"])/"MedAgentBench"/"medevolve"/"manifest.json"
    assert p1.is_file() and p2.is_file() and p1!=p2
    assert json.loads(p1.read_text())["environment"]=="PhysicianBench"
    assert json.loads(p2.read_text())["environment"]=="MedAgentBench"


@pytest.mark.asyncio
async def test_scoring_failure_stays_separate_from_runner_failure(rig,tmp_path,monkeypatch):
    config,_,_=rig
    monkeypatch.setattr(harness_train,"preflight",lambda config,combos,mode="train":[{"ok":True}])
    monkeypatch.setattr(harness_test,"preflight",lambda config,combos,mode="test":[{"ok":True}])
    trained=await harness_train.run_training(config,"PhysicianBench","ace","medevolve")
    group_path=Path(config["run_dir_template"].replace("<run_id>","test").replace("<environment>","PhysicianBench").replace("<harness>","ace").replace("<source_method>","medevolve").replace("<seed>","42"))
    (group_path/"train.json").write_text(json.dumps(trained))
    async def failed_review(task,trajectory,run,state,review_config,*,record=None):
        if record: record(RunResult(status="failed",output=None,error="review unavailable",trace=[],usage={"known":False}))
        return TrainingFeedback(task_id=task.id,completed=None,failure_category="model_error",reason="review unavailable",
            reference_answer=task.reference_answer,evidence=[])
    monkeypatch.setattr(harness_test,"make_training_feedback",failed_review)
    result=await harness_test.run_test(config,"PhysicianBench","ace","medevolve")
    assert len(result["records"])==6
    assert all(row["status"]=="scoring_error" and row["failure_category"]=="model_error" for row in result["records"])
    assert result["valid_denominators"]=={"initial":0,"final":0}


@pytest.mark.asyncio
async def test_partial_runner_startup_is_closed(rig,monkeypatch):
    config,events,sessions=rig
    monkeypatch.setattr(harness_train,"preflight",lambda config,combos,mode="train":[{"ok":True}])
    class BrokenRunner(FakeRunner):
        async def start(self,group,initial): raise RuntimeError("startup failed")
    monkeypatch.setattr(harness_train,"create_runner",lambda name,config,group_dir,bridge_url:BrokenRunner(name,{**config,"events":events},group_dir))
    with pytest.raises(RuntimeError,match="startup failed"):
        await harness_train.run_training(config,"PhysicianBench","openclaw","medevolve")
    assert any(event[0]=="runner_close" and event[1]=="openclaw" for event in events)
    assert not any(event[0]=="open" for event in events)
    assert not sessions.active


def test_preflight_and_runner_preserve_virtualenv_interpreter_symlink(rig, monkeypatch, tmp_path):
    from harnesses.runners import create_runner
    config, _, _ = rig
    interpreter = tmp_path / "venv" / "bin" / "python"
    interpreter.parent.mkdir(parents=True)
    interpreter.symlink_to(os.sys.executable)
    config["local_harnesses"]["ace"]["python_path"] = str(interpreter)
    seen = {}
    def probe(harness, source, program, python, lock):
        seen["python"] = python
        return {"source_dir": True, "program_path": True, "source_sha": "x",
                "components": {}, "runtime": {"official_import_ok": True}}
    monkeypatch.setattr(harness_train, "_source_probe", probe)
    harness_train.preflight(config, [("PhysicianBench", "ace", "medevolve")])
    runner = create_runner("ace", config=config, group_dir=tmp_path / "group", bridge_url="")
    assert seen["python"] == interpreter
    assert runner.python == str(interpreter)
