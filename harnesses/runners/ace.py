"""ACE 官方策略组件的本机隔离 runner。"""
from __future__ import annotations
import hashlib, json, os, sys
from pathlib import Path
from contracts import RunResult, SolverInput
from harnesses.contracts import HarnessStep, Snapshot, TrainingFeedback
from harnesses.local import run_command
from harnesses.snapshots import collect_files, directory_digest, materialize_files

WORKER = Path(__file__).with_name("ace_worker.py")

class ACERunner:
    def __init__(self, *, config, group_dir, bridge_url):
        self.config, self.group_dir, self.bridge_url = config, Path(group_dir), bridge_url
        self.state = self.group_dir / "ace-state"
        self.python = config.get("python_path", sys.executable)
        self.source_dir = config.get("source_dir", "")
        self._command_runner = config.get("command_runner", run_command)
        self._client_id = config.get("client_id", "medevolve-ace")
        self._frozen = False; self._session = None; self._last = None; self._step = 0

    async def start(self, group, initial: Snapshot):
        self.group, self._frozen = group, bool(self.config.get("frozen", False))
        self._client_id = self.config.get("client_id") or group.run_id
        self.state.mkdir(parents=True, exist_ok=True)
        if Path(initial.path).is_dir():
            materialize_files(self.state, collect_files(initial.path, include=("playbook.md", "experiences.json")))

    @staticmethod
    def stable_task_key(task: SolverInput) -> str:
        raw = "\0".join((task.patient_id, task.instruction, task.decision_point or ""))
        return hashlib.sha256(raw.encode()).hexdigest()

    def _metadata(self, task):
        return self.config.get("task_metadata", {}).get(self.stable_task_key(task), {})

    async def _call(self, payload, phase):
        self.group_dir.mkdir(parents=True, exist_ok=True)
        request, response = self.group_dir / "ace-request.json", self.group_dir / "ace-response.json"
        response.unlink(missing_ok=True)
        request.write_text(json.dumps({**payload, "source_dir":str(Path(self.source_dir).expanduser().resolve()),
            "state_dir":str(self.state), "client_id":self._client_id, "bridge_url":self.bridge_url}, ensure_ascii=False), encoding="utf-8")
        model = self.config.get(f"{phase}_model", self.config.get("execution_model", {}))
        env={"PYTHONPATH":str(Path(__file__).resolve().parents[2]),"PYTHON_DOTENV_DISABLED":"1"}
        keyname=model.get("api_key_env")
        if keyname: env[keyname]=os.environ.get(keyname, "")
        code, _, stderr = await self._command_runner([self.python, str(WORKER), str(request), str(response)],
            group_dir=self.group_dir, group_env=env, timeout=float(self.config.get("learning_timeout_seconds" if phase=="learning" else "timeout_seconds",900)))
        if code or not response.is_file(): return None, stderr[-1200:] or "ACE worker 未返回结果"
        try:
            envelope=json.loads(response.read_text(encoding="utf-8"))
            if not envelope.get("ok"): return None, envelope.get("error", "ACE worker 执行失败")
            return envelope["result"], None
        except (ValueError,OSError,KeyError) as exc: return None,f"ACE worker 响应解析失败：{exc}"

    async def execute(self, task: SolverInput, tool_session: str) -> RunResult:
        self._session, self._task = tool_session, task
        model=self.config.get("execution_model",{})
        token_limit=model.get("max_output_tokens") or self.config.get("max_output_tokens") or 2048
        result,error=await self._call({"action":"execute","instruction":task.instruction,
            "patient_id":task.patient_id,"decision_point":task.decision_point,"tool_session":tool_session,
            "model":model,"metadata":self._metadata(task),
            "max_model_turns":self.config.get("execution_budget",{}).get("max_model_turns",30),
            "max_tool_calls":self.config.get("execution_budget",{}).get("max_tool_calls",30),
            "max_output_tokens":token_limit},"execution")
        if error:
            exhausted="ACE_MODEL_TURN_BUDGET_EXHAUSTED" in error
            return RunResult(status="budget_exhausted" if exhausted else "failed",output=None,error=error,trace=[{"type":"model_budget","calls":self.config.get("execution_budget",{}).get("max_model_turns",30)}] if exhausted else [],
                usage={"known":False,"calls":self.config.get("execution_budget",{}).get("max_model_turns",30) if exhausted else 0})
        try: parsed=json.loads(result["raw"])
        except (ValueError,TypeError): parsed={"final_answer":result.get("raw", ""),"reasoning":""}
        answer=str(parsed.get("final_answer", ""))
        self._last={"task":task,"answer":answer,"reasoning":str(parsed.get("reasoning", "")),"bullets":result.get("bullet_ids",[])}
        self._last["tool_trace"]=result.get("tool_trace",[])
        trace=[{"type":"ace_generator","native":"ace.core.Generator.generate","tool_calls":result.get("tool_calls",0),"reasoning":self._last["reasoning"],"tool_trace":self._last["tool_trace"]}]
        if not answer: return RunResult(status="failed",output=None,error="ACE Generator 未返回 final_answer",trace=trace,usage={"known":False})
        return RunResult(status="completed",output={"answer":answer},error=None,trace=trace,
            usage={**result.get("usage",{}),"known":result.get("usage",{}).get("known",False)})

    async def learn(self, feedback: TrainingFeedback) -> HarnessStep:
        if feedback.completed is None or self._frozen or not self._last:
            message="完成评审状态未知，跳过ACE学习" if feedback.completed is None else "冻结ACE经验只读，拒绝学习" if self._frozen else "缺少本题Generator轨迹，跳过ACE学习"
            run=RunResult(status="failed",output=None,error=message,trace=[],usage={"known":False})
            return HarnessStep(run=run,learning_runs=[],learning_status="failed")
        task=self._last["task"]
        model=self.config.get("learning_model",self.config.get("execution_model",{}))
        token_limit=model.get("max_output_tokens") or self.config.get("max_output_tokens") or 2048
        result,error=await self._call({"action":"learn","instruction":task.instruction,"tool_session":"",
            "patient_id":task.patient_id,"decision_point":task.decision_point,"model":model,
            "feedback":feedback.model_dump(mode="json"),"answer":self._last["answer"],
            "trajectory":json.dumps(self._last,ensure_ascii=False,default=str),"bullets_used":self._last["bullets"],
            "metadata":self._metadata(task),"question_context":json.dumps(self._metadata(task),ensure_ascii=False),
            "max_model_turns":self.config.get("learning_budget",{}).get("max_model_turns",10),
            "max_tool_calls":0,"max_output_tokens":token_limit,
            "current_step":self._step+1,"total_samples":self.config.get("total_samples",1),
            "playbook_token_budget":self.config.get("playbook_token_budget",2000)},"learning")
        self._step+=1
        if error:
            exhausted="ACE_MODEL_TURN_BUDGET_EXHAUSTED" in error
            run=RunResult(status="budget_exhausted" if exhausted else "failed",output=None,error=error,
                trace=[{"type":"model_budget","calls":self.config.get("learning_budget",{}).get("max_model_turns",10)}] if exhausted else [],
                usage={"known":False,"calls":self.config.get("learning_budget",{}).get("max_model_turns",10) if exhausted else 0})
            return HarnessStep(run=run,learning_runs=[run],learning_status="budget_exhausted" if exhausted else "failed")
        state=result.get("learning_status","unchanged")
        details=[]
        for item in result.get("learning_runs",[]):
            usage=item.get("usage",{})
            details.append(RunResult(status="completed",output={"component":item.get("kind")},
                trace=[{"type":"ace_learning","component":item.get("kind")}],usage={**usage,"known":usage.get("known",False)}))
        summary=RunResult(status="completed",output={"learning_status":state},
            trace=[{"type":"ace_learning","components":["ace.core.Reflector.reflect","ace.core.Curator.curate"]}],
            usage={"known":False,"calls":result.get("usage",{}).get("calls",0)})
        return HarnessStep(run=summary,learning_runs=details,learning_status=state)

    async def freeze(self, destination: Path):
        target=Path(destination); target.mkdir(parents=True,exist_ok=True)
        materialize_files(target,collect_files(self.state,include=("playbook.md","experiences.json")))
        return Snapshot(path=target,digest=directory_digest(target))

    async def close(self): self._last=None; self._session=None
