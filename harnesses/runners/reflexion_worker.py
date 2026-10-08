"""在锁定 Reflexion checkout 中执行官方 ReactReflectAgent。"""
from __future__ import annotations
import importlib, json, os, re, sys, urllib.request
from pathlib import Path


def _request(url, payload=None, client_id=""):
    data=None if payload is None else json.dumps(payload,ensure_ascii=False).encode()
    req=urllib.request.Request(url,data=data,headers={"x-medevolve-client":client_id,**({"Content-Type":"application/json"} if data else {})})
    with urllib.request.urlopen(req,timeout=60) as resp: return json.loads(resp.read())


def _chat_completion(model, api_key, prompt, max_tokens):
    base=(model.get("base_url") or "https://api.openai.com/v1").rstrip("/")
    body={"model":model["model"],"messages":[{"role":"user","content":prompt}],
          "temperature":0,"max_tokens":max_tokens}
    request=urllib.request.Request(base+"/chat/completions",data=json.dumps(body).encode(),
        headers={"Authorization":f"Bearer {api_key}","Content-Type":"application/json"})
    with urllib.request.urlopen(request,timeout=180) as response: return json.loads(response.read())


def _official(source_dir, api_key):
    source=Path(source_dir).resolve(); agents=source/"hotpotqa_runs"/"agents.py"
    if not agents.is_file(): raise RuntimeError(f"Reflexion 官方源码缺失：{agents}；请配置锁定 checkout 的 source_dir")
    os.environ.setdefault("OPENAI_API_KEY",api_key or "missing-api-key")
    sys.path.insert(0,str(source/"hotpotqa_runs"))
    try:
        spec=importlib.util.spec_from_file_location("reflexion_locked_agents", agents)
        module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        prompts=importlib.import_module("prompts")
    except Exception as exc:
        raise RuntimeError(f"无法加载锁定 Reflexion ReactReflectAgent；请在 python_path 环境安装 hotpotqa_runs/requirements.txt 兼容依赖：{type(exc).__name__}: {exc}") from exc
    if not hasattr(module,"ReactReflectAgent") or not hasattr(module,"ReflexionStrategy"):
        raise RuntimeError("锁定 Reflexion checkout 缺少 ReactReflectAgent/ReflexionStrategy")
    return module,prompts


class EmptyDocstore:
    def search(self, search): return "Medical tools are available through the action adapter."
    def lookup(self, search): raise ValueError(search)


class MedicalReactReflectAgent:
    """只覆写上游动作执行边界，保留官方 Actor、run/reflect/prompt 实现。"""
    def __new__(cls, *args, **kwargs):
        raise TypeError("use build_agent")


def build_agent(module,prompts,req,model,tools,usage):
    from langchain.prompts import PromptTemplate
    source=Path(req["source_dir"]).resolve(); sys.path.insert(0,str(source/"hotpotqa_runs"))
    agent_base=module.ReactReflectAgent
    strategy=module.ReflexionStrategy
    bridge,session,client=req["bridge_url"],req["tool_session"],req["client_id"]
    functions={t["name"]:t for t in tools}
    key=os.environ.get(model.get("api_key_env",""),"")
    def call(prompt, phase, call_type):
        response=_chat_completion(model,key,prompt,800 if phase=="execution" else 600)
        bucket=usage["by_call_type"].setdefault(call_type,{"calls":0,"prompt_tokens":0,"completion_tokens":0,"tokens_known":True})
        bucket["calls"]+=1; usage["calls"]+=1
        if response.get("usage"):
            bucket["prompt_tokens"]+=response["usage"].get("prompt_tokens",0) or 0
            bucket["completion_tokens"]+=response["usage"].get("completion_tokens",0) or 0
            usage["prompt_tokens"]+=response["usage"].get("prompt_tokens",0) or 0
            usage["completion_tokens"]+=response["usage"].get("completion_tokens",0) or 0
        else:
            bucket["tokens_known"]=False; usage["tokens_known"]=False
        choices=response.get("choices") or []
        return ((choices[0].get("message") or {}).get("content") or "") if choices else ""
    tool_catalog=json.dumps([{"name":t.get("name"),"description":t.get("description"),"parameters":t.get("parameters")} for t in tools],ensure_ascii=False)
    medical_prompt=PromptTemplate(input_variables=["examples","reflections","question","scratchpad"],
        template="""Solve a medical task using Thought, Action, Observation. Available actions are Tool[name|JSON arguments] and Finish[answer]. Use only listed tools, cite observed evidence, and do not write patient records.\nMedical tool names and JSON schemas:\n"""+tool_catalog.replace("{","{{").replace("}","}}")+"""\n{examples}\n{reflections}\nQuestion: {question}{scratchpad}""")
    class Adapter(agent_base):
        def step(self):
            self.scratchpad += f"\nThought {self.step_n}: " + self.prompt_agent()
            self.scratchpad += f"\nAction {self.step_n}: "
            action=self.prompt_agent(); self.scratchpad += action
            parsed=module.parse_action(action)
            self.scratchpad += f"\nObservation {self.step_n}: "
            if not parsed:
                self.scratchpad += "Invalid action. Use Tool[name|JSON] or Finish[answer]."; self.step_n+=1; return
            kind,arg=parsed
            if kind=="Finish": self.answer=arg; self.finished=True; self.scratchpad+="Final answer submitted."; self.step_n+=1; return
            if kind!="Tool" or "|" not in arg:
                self.scratchpad += "Invalid action. Use Tool[name|JSON arguments]."; self.step_n+=1; return
            name,raw=arg.split("|",1)
            try:
                arguments=json.loads(raw.strip()); tool=functions.get(name.strip())
                if not tool: raise ValueError("tool name is not available")
                result=_request(f"{bridge}/sessions/{session}/calls",{"name":name.strip(),"arguments":arguments},client)
                self.scratchpad += json.dumps(result,ensure_ascii=False)
            except Exception as exc: self.scratchpad += f"Tool error: {type(exc).__name__}: {exc}"
            self.step_n+=1
    actor=Adapter(question=req["instruction"],key="__clinical_review_owned__",max_steps=int(req.get("max_steps",12)),
        agent_prompt=medical_prompt,reflect_prompt=prompts.reflect_prompt,docstore=EmptyDocstore(),
        react_llm=lambda prompt:call(prompt,"execution","actor"),reflect_llm=lambda prompt:call(prompt,"learning","reflexion_reflector"))
    return actor,strategy,lambda prompt:call(prompt,"learning","experience_extraction")


def main(req):
    model=req["model"]; api_key=os.environ.get(model.get("api_key_env",""),"")
    if not api_key: raise RuntimeError(f"缺少模型 API key 环境变量：{model.get('api_key_env')}")
    module,prompts=_official(req["source_dir"],api_key)
    state_dir=Path(req["state_dir"]); state_dir.mkdir(parents=True,exist_ok=True)
    state_path=state_dir/"reflexion-state.json"
    state=json.loads(state_path.read_text()) if state_path.exists() else {"reflections":[],"experiences":[],"attempts":0}
    if not state_path.exists() and (state_dir/"experiences.json").is_file():
        state["experiences"]=json.loads((state_dir/"experiences.json").read_text(encoding="utf-8"))
    if req["action"]=="execute" and req.get("reset_reflections"):
        state["reflections"]=[]; state["reflection_text"]=""
    tools=(_request(f"{req['bridge_url']}/sessions/{req['tool_session']}/tools",None,req["client_id"])
           if req["action"]=="execute" else [])
    usage={"prompt_tokens":0,"completion_tokens":0,"calls":0,"tokens_known":True,"by_call_type":{}}
    experiences=req.get("experiences",[]) if req["action"]=="execute" else []
    if experiences:
        req["instruction"] += "\n\nTransferable workflow experience:\n" + "\n".join(
            f"- {item.get('condition')}: {item.get('procedure')} Pitfall: {item.get('pitfall')}"
            for item in experiences)
    if req["action"]=="execute":
        req["instruction"] += f"\nPatient identifier for scoped tool access: {req.get('patient_id','')}\nDecision point: {req.get('decision_point') or 'none'}"
    actor,strategy,extract_experience=build_agent(module,prompts,req,model,tools,usage)
    actor.reflections=list(state.get("reflections",[])); actor.reflections_str="\n".join(actor.reflections)
    if req["action"]=="execute":
        actor.run(reset=True,reflect_strategy=strategy.REFLEXION)
        state.update({"scratchpad":actor.scratchpad,"answer":actor.answer,"finished":actor.finished,"attempts":state.get("attempts",0)+1})
        state_path.write_text(json.dumps(state,ensure_ascii=False),encoding="utf-8")
        return {"answer":actor.answer,"scratchpad":actor.scratchpad,"finished":actor.finished,"usage":usage,"native":"ReactReflectAgent.run"}
    if req["action"]=="learn":
        feedback=req["feedback"]
        if feedback.get("completed") is None or feedback.get("completed") is True:
            return {"learning_status":"not_triggered","usage":usage,"reflection":None}
        actor.scratchpad=req.get("scratchpad") or state.get("scratchpad","")
        reason=str(feedback.get("reason", ""))
        if reason: actor.scratchpad += "\nClinical completion review: " + reason
        actor.finished=True; actor.answer=state.get("answer","")
        actor.reflect(strategy.REFLEXION)
        state["reflections"]=list(actor.reflections); state["reflection_text"]=actor.reflections[-1] if actor.reflections else ""
        experience=None
        metadata=req.get("metadata") or {}
        if metadata.get("task_type"):
            extraction_prompt=("Convert the following Reflexion into one reusable clinical workflow experience. "
                "Return JSON with condition, procedure, pitfall only. Do not include patient identifiers, "
                "individual clinical values, or the task's answer. Use general clinical process language.\n"
                f"Task type: {metadata.get('task_type')}\nReflection: {state['reflection_text']}\n"
                f"Metadata procedure: {metadata.get('procedure','')}\nMetadata pitfall: {metadata.get('pitfall','')}")
            extracted=extract_experience(extraction_prompt)
            try:
                candidate=json.loads(extracted)
                values=[str(candidate.get(k,"")) for k in ("condition","procedure","pitfall")]
                patient_id=str(req.get("patient_id", ""))
                answer=str(state.get("answer", ""))
                if all(values) and not any(re.search(r"\d",value) for value in values) and not any(
                    secret and secret in " ".join(values) for secret in (patient_id,answer)):
                    entry={"id":f"experience-{feedback.get('task_id','unknown')}","task_type":metadata["task_type"],
                        "tool_names":sorted(set(metadata.get("tool_names",[]))),"condition":values[0],
                        "procedure":values[1],"pitfall":values[2],"source_task_id":feedback.get("task_id","unknown")}
                    state["experiences"]=[e for e in state.get("experiences",[]) if e.get("source_task_id")!=entry["source_task_id"]]
                    state["experiences"].append(entry); experience=entry
            except (ValueError,TypeError,KeyError):
                experience=None
        state_path.write_text(json.dumps(state,ensure_ascii=False),encoding="utf-8")
        (state_dir/"experiences.json").write_text(json.dumps(state.get("experiences",[]),ensure_ascii=False,indent=2),encoding="utf-8")
        return {"learning_status":"updated","usage":usage,"reflection":state["reflection_text"],"experience":experience,
            "native":"ReactReflectAgent.reflect","experience_extraction_calls":1 if metadata.get("task_type") else 0}
    raise RuntimeError(f"未知 Reflexion worker action: {req['action']}")

if __name__=="__main__":
    request,response=map(Path,sys.argv[1:3])
    try: result=main(json.loads(request.read_text(encoding="utf-8"))); envelope={"ok":True,"result":result}
    except Exception as exc: envelope={"ok":False,"error":f"{type(exc).__name__}: {exc}"}
    response.write_text(json.dumps(envelope,ensure_ascii=False),encoding="utf-8")
