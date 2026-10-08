"""锁定 Reflexion ReactReflectAgent 的本机医疗适配。"""
from __future__ import annotations
import hashlib, json, os, sys
from pathlib import Path
from contracts import CrossTaskExperience, RunResult, SolverInput
from harnesses.contracts import HarnessStep, Snapshot, TrainingFeedback
from harnesses.local import run_command
from harnesses.snapshots import collect_files, directory_digest, materialize_files

WORKER=Path(__file__).with_name("reflexion_worker.py")

class ReflexionRunner:
    def __init__(self, *, config, group_dir, bridge_url):
        self.config,self.group_dir,self.bridge_url=config,Path(group_dir),bridge_url
        self.state=self.group_dir/"reflexion-state"; self.python=config.get("python_path",sys.executable)
        self.source_dir=config.get("source_dir",""); self._command_runner=config.get("command_runner",run_command)
        self._client_id=config.get("client_id","medevolve-reflexion"); self._frozen=False
        self._task=None; self._session=None; self._task_key=None; self._attempts=0; self._retry_ready=False
        self._last_scratchpad=""

    @property
    def retry_state(self):
        limit=0 if self._frozen else int(self.config.get("reflexion_max_retries",2))
        retries_used=max(0,self._attempts-1)
        return {"ready":self._retry_ready and retries_used<limit,
                "retries_remaining":max(0,limit-retries_used),"attempts":self._attempts,
                "task_id":self._task_key}

    @staticmethod
    def stable_task_key(task: SolverInput)->str:
        raw="\0".join((task.patient_id,task.instruction,task.decision_point or ""))
        return hashlib.sha256(raw.encode()).hexdigest()

    def _metadata(self,task): return self.config.get("task_metadata",{}).get(self.stable_task_key(task),{})

    async def start(self,group,initial:Snapshot):
        self.group,self._frozen=group,bool(self.config.get("frozen",False))
        self._client_id=self.config.get("client_id") or group.run_id
        self.state.mkdir(parents=True,exist_ok=True)
        src=Path(initial.path)
        if src.is_dir(): materialize_files(self.state,collect_files(src,include=("experiences.json",)))

    def _select_experiences(self,task):
        path=self.state/"experiences.json"
        if not path.is_file(): return []
        try: entries=[CrossTaskExperience.model_validate(x) for x in json.loads(path.read_text(encoding="utf-8"))]
        except (ValueError,OSError,TypeError): return []
        metadata=self._metadata(task); task_type=metadata.get("task_type",""); tools=set(metadata.get("tool_names",[]))
        entries.sort(key=lambda e:(0 if e.task_type==task_type else 1,-len(tools.intersection(e.tool_names)),e.id))
        limit=int(self.config.get("experience_limit",{}).get("max_entries",3))
        token_limit=int(self.config.get("experience_limit",{}).get("max_context_tokens",2000))
        selected=[]; used=0
        for entry in entries:
            if len(selected)>=min(3,limit): break
            rendered=f"{entry.condition}: {entry.procedure} 注意：{entry.pitfall}"
            try:
                import tiktoken
                count=len(tiktoken.get_encoding("cl100k_base").encode(rendered))
            except Exception: count=max(1,len(rendered)//4)
            if used+count>token_limit: continue
            selected.append(entry.model_dump(mode="json")); used+=count
        return selected

    async def _call(self,payload,phase):
        self.group_dir.mkdir(parents=True,exist_ok=True)
        request,response=self.group_dir/"reflexion-request.json",self.group_dir/"reflexion-response.json"
        response.unlink(missing_ok=True)
        request.write_text(json.dumps({**payload,"source_dir":str(Path(self.source_dir).expanduser().resolve()),
            "state_dir":str(self.state),"bridge_url":self.bridge_url,"client_id":self._client_id},ensure_ascii=False),encoding="utf-8")
        model=self.config.get(f"{phase}_model",self.config.get("execution_model",{})); env={"PYTHONPATH":str(Path(__file__).resolve().parents[2])}
        key=model.get("api_key_env")
        if key:
            env[key]=os.environ.get(key,"")
            env["OPENAI_API_KEY"]=env[key]
        code,_,stderr=await self._command_runner([self.python,str(WORKER),str(request),str(response)],group_dir=self.group_dir,
            group_env=env,timeout=float(self.config.get("learning_timeout_seconds" if phase=="learning" else "timeout_seconds",900)))
        if code or not response.is_file(): return None,stderr[-1200:] or "Reflexion worker 未返回结果"
        try:
            envelope=json.loads(response.read_text(encoding="utf-8"))
            if not envelope.get("ok"): return None,envelope.get("error","Reflexion worker 执行失败")
            return envelope["result"],None
        except (ValueError,OSError,KeyError) as exc:return None,f"Reflexion worker 响应解析失败：{exc}"

    def _usage(self,data):
        usage=data.get("usage") or {}
        return {**usage,"known":bool(usage.get("calls",0)) and usage.get("tokens_known",False)}

    async def execute(self,task:SolverInput,tool_session:str)->RunResult:
        self._task,self._session=task,tool_session
        self._task_key=self.stable_task_key(task); self._attempts=0; self._retry_ready=False
        return await self._execute(task,tool_session,reset_reflections=True)

    async def _execute(self,task,tool_session,*,reset_reflections):
        model=self.config.get("execution_model",{})
        result,error=await self._call({"action":"execute","instruction":task.instruction,"patient_id":task.patient_id,
            "tool_session":tool_session,"model":model,"max_steps":self.config.get("execution_budget",{}).get("max_model_turns",30),
            "reset_reflections":reset_reflections,"metadata":self._metadata(task),"experiences":self._select_experiences(task)},"execution")
        if error:return RunResult(status="failed",output=None,error=error,trace=[],usage={"known":False})
        self._attempts+=1; self._retry_ready=False; self._last_scratchpad=result.get("scratchpad","")
        trace=[{"type":"reflexion_actor","native":"ReactReflectAgent.run","scratchpad":self._last_scratchpad}]
        if not result.get("answer"):
            return RunResult(status="failed",output=None,error="Reflexion Actor 未返回 Finish 答案",trace=trace,usage=self._usage(result))
        return RunResult(status="completed",output={"answer":result["answer"]},error=None,trace=trace,usage=self._usage(result))

    async def learn(self,feedback:TrainingFeedback)->HarnessStep:
        if feedback.completed is not False or feedback.failure_category!="clinical" or self._frozen:
            status="failed" if feedback.completed is None else "not_triggered"
            message="评审状态未知，跳过反思" if status=="failed" else "Reflexion 仅对可评审的临床失败学习"
            run=RunResult(status="completed" if status=="not_triggered" else "failed",output={"learning_status":status},error=None if status=="not_triggered" else message,trace=[],usage={"known":False})
            self._retry_ready=False
            return HarnessStep(run=run,learning_runs=[],learning_status=status)
        if self._task is None:
            run=RunResult(status="failed",output=None,error="没有可反思的 Reflexion 任务",trace=[],usage={"known":False})
            return HarnessStep(run=run,learning_runs=[run],learning_status="failed")
        model=self.config.get("learning_model",self.config.get("execution_model",{}))
        result,error=await self._call({"action":"learn","instruction":self._task.instruction,"patient_id":self._task.patient_id,
            "tool_session":"","model":model,"feedback":feedback.model_dump(mode="json"),"scratchpad":self._last_scratchpad,
            "metadata":self._metadata(self._task)},"learning")
        if error:
            run=RunResult(status="failed",output=None,error=error,trace=[],usage={"known":False})
            return HarnessStep(run=run,learning_runs=[run],learning_status="failed")
        self._retry_ready=True
        usage=self._usage(result)
        details=[]
        for kind,bucket in (result.get("usage",{}).get("by_call_type",{})).items():
            details.append(RunResult(status="completed",output={"component":kind},trace=[{"type":"reflexion_learning","component":kind}],
                usage={**bucket,"known":bool(bucket.get("calls")) and bucket.get("tokens_known",False)}))
        lr=RunResult(status="completed",output={"learning_status":result.get("learning_status"),"experience_saved":bool(result.get("experience"))},
            trace=[{"type":"reflexion_learning","native":result.get("native"),"experience_extraction_calls":result.get("experience_extraction_calls",0)}],
            usage={"known":False,"calls":usage.get("calls",0)})
        limit=int(self.config.get("reflexion_max_retries",2))
        if max(0,self._attempts-1)>=limit: self._retry_ready=False
        return HarnessStep(run=lr,learning_runs=details,learning_status=result.get("learning_status","updated"))

    async def execute_retry(self,task:SolverInput,new_tool_session:str)->RunResult:
        if self._frozen: raise RuntimeError("冻结 Reflexion runner 不支持同题重试")
        if not self.retry_state["ready"]: raise RuntimeError("Reflexion 尚未完成失败反思，或重试次数已达上限")
        if self._task is None or self.stable_task_key(task)!=self._task_key: raise ValueError("Reflexion retry 必须使用同一道任务")
        self._session=new_tool_session
        return await self._execute(task,new_tool_session,reset_reflections=False)

    async def freeze(self,destination:Path):
        target=Path(destination); target.mkdir(parents=True,exist_ok=True)
        path=self.state/"reflexion-state.json"
        if path.is_file():
            try: experiences=json.loads(path.read_text(encoding="utf-8")).get("experiences",[])
            except (ValueError,OSError): experiences=[]
            (target/"experiences.json").write_text(json.dumps(experiences,ensure_ascii=False,indent=2),encoding="utf-8")
        else: materialize_files(target,collect_files(self.state,include=("experiences.json",)))
        return Snapshot(path=target,digest=directory_digest(target))

    async def close(self): self._task=None; self._session=None; self._last_scratchpad=""
