"""在共同冻结任务集上比较初始与训练后的 Harness 经验。"""
from __future__ import annotations

import argparse
import asyncio
import json
import tempfile
import sys
from pathlib import Path

from contracts import EnvironmentConfig, SolverInput
from harnesses.contracts import GroupKey, Snapshot
from harnesses.feedback import make_training_feedback
from harnesses.runners import create_runner
from harnesses.snapshots import directory_digest
from experiments.harness_train import (_require_execution, _task_metadata,
    bridge_server, ensure_frozen, evaluation_config, freeze_root, group_id, load_config, preflight, rooted, selections,
    runner_config, update_task_tool_metadata, write_json)


async def run_test(config, environment, harness, source):
    checks=preflight(config,[(environment,harness,source)],mode="test")
    if not checks[0]["ok"]:
        raise ValueError("本地预检未通过："+json.dumps(checks[0],ensure_ascii=False))
    _require_execution(config)
    output_file=rooted(group_id(config,environment,harness,source,config["seeds"][0]))/"test.json"
    if output_file.exists():
        raise FileExistsError(f"测试结果目录已存在：{output_file}")
    frozen = ensure_frozen(config, environment, source)
    tasks = frozen[source]
    manifest = json.loads((freeze_root(config,environment,source) / "manifest.json").read_text(encoding="utf-8"))
    split = manifest["sources"][source]["split"]
    test_ids = set(split["test_task_ids"])
    tasks = [task for task in tasks if task.id in test_ids]
    settings = config["local_harnesses"][harness]
    envcfg = EnvironmentConfig(**config["environments"][environment])
    review_cfg = evaluation_config(config)
    from environments import prepare_trajectory
    from harnesses.medical_bridge import ToolSessions
    sessions = ToolSessions()
    group_path = rooted(group_id(config, environment, harness, source, config["seeds"][0]))
    seed_group = GroupKey(run_id=f"{config['run_id']}-{environment}-{harness}-{source}",
        environment=environment,harness=harness,source_method=source,seed=config["seeds"][0])
    snapshots = {"initial": group_path / "initial-experience",
                 "final": group_path / "final-experience"}
    if not snapshots["initial"].is_dir():
        from harnesses.local import prepare_group_directory
        prepare_group_directory(rooted(settings["initial_state_template"]), snapshots["initial"])
    before = {key: directory_digest(path) for key,path in snapshots.items()}
    train_path=group_path/"train.json"
    if train_path.is_file():
        train_record=json.loads(train_path.read_text(encoding="utf-8"))
        expected_digest=train_record.get("initial_snapshot",{}).get("digest")
        if expected_digest and before["initial"] != expected_digest:
            raise ValueError("测试初始经验副本摘要与训练记录不一致")
    records, cost_runs = [], []
    try:
      async with bridge_server(sessions) as bridge_url:
        for state_name, snapshot_path in snapshots.items():
              if state_name == "final" and not snapshot_path.is_dir():
                  records.append({"state":state_name,"status":"unavailable","reason":"训练最终经验快照缺失"})
                  continue
              for index, task in enumerate(tasks):
                  with tempfile.TemporaryDirectory(prefix=f"medevolve-{harness}-{state_name}-") as tmp:
                      group_dir = Path(tmp) / "group"
                      group_dir.mkdir()
                      task_group = seed_group.model_copy(update={"run_id":f"{seed_group.run_id}-{state_name}-{task.id}"})
                      rconfig=runner_config(config,harness,_task_metadata([task]),client_id=task_group.run_id,frozen=True)
                      runner = create_runner(harness, config=rconfig, group_dir=group_dir,bridge_url="")
                      session_id = None
                      run = None
                      trajectory = None
                      score_runs = []
                      session_summary = None
                      try:
                          snapshot = Snapshot(path=snapshot_path,digest=directory_digest(snapshot_path))
                          runner.bridge_url=bridge_url
                          await runner.start(task_group,snapshot)
                          solver_input = SolverInput(instruction=task.instruction,patient_id=task.patient_id,
                                                     decision_point=task.decision_point)
                          session_id = await sessions.open(solver_input,envcfg,client_id=task_group.run_id,
                              read_only=True,tool_budget=config["limits"]["execution"].get("max_tool_calls"),
                              timeout_seconds=config["limits"]["timeout_seconds"])
                          context=sessions.inspect(session_id,client_id=task_group.run_id)
                          tool_catalog=await sessions.list(session_id,client_id=task_group.run_id)
                          update_task_tool_metadata(rconfig,task,tool_catalog)
                          trajectory=await prepare_trajectory(context.instance,task.patient_id)
                          run=await runner.execute(solver_input,session_id)
                          run=run.model_copy(update={"trace":[*run.trace,*context.events,*context.calls]})
                          observed=await sessions.environments.read_state(context.instance,sessions._state_task(context))
                          try:
                              if run.status != "completed" or run.output is None:
                                  feedback=None; scoring_error=False; score=None
                                  reason=run.error or run.status
                                  failure=next((item.get("failure_category") for item in reversed(run.trace)
                                                if isinstance(item,dict) and item.get("failure_category")),None)
                                  session_summary=await sessions.close(session_id,client_id=task_group.run_id)
                                  cost_runs.append({"task_id":task.id,"state":state_name,"stage":"testing","role":"execution","run":run.model_dump(mode="json")})
                                  records.append({"state":state_name,"task_id":task.id,"patient_id":task.patient_id,
                                      "status":"execution_failed","completed":None,"failure_category":failure,
                                      "reason":reason,"scoring_error":False,"run":run.model_dump(mode="json"),
                                      "trajectory":trajectory.model_dump(mode="json"),"observed_state":observed})
                                  continue
                              feedback=await make_training_feedback(task,trajectory,run,observed or {},review_cfg,
                                  record=lambda rr: score_runs.append(rr))
                              scoring_error=feedback.completed is None
                              score=feedback.completed
                              reason=feedback.reason
                              failure=feedback.failure_category
                          except Exception as exc:
                              feedback=None; scoring_error=True; score=None; reason=f"{type(exc).__name__}: {exc}"; failure=None
                          session_summary=await sessions.close(session_id,client_id=task_group.run_id)
                          cost_runs.append({"task_id":task.id,"state":state_name,"stage":"testing","role":"execution","run":run.model_dump(mode="json")})
                          for review_run in score_runs:
                              cost_runs.append({"task_id":task.id,"state":state_name,"stage":"testing_review","role":"completion","run":review_run.model_dump(mode="json")})
                          records.append({"state":state_name,"task_id":task.id,"patient_id":task.patient_id,
                              "status":"scoring_error" if scoring_error else "scored",
                              "completed":score,"failure_category":failure,"reason":reason,
                              "scoring_error":scoring_error,"run":run.model_dump(mode="json"),
                              "trajectory":trajectory.model_dump(mode="json"),"observed_state":observed,
                              "tool_session":session_summary,"learning_feedback":None})
                      except Exception as exc:
                          records.append({"state":state_name,"task_id":task.id,"patient_id":task.patient_id,
                              "status":"execution_error","completed":None,"failure_category":None,
                              "scoring_error":False,"error":f"{type(exc).__name__}: {exc}",
                              "run":run.model_dump(mode="json") if run else None,
                              "trajectory":trajectory.model_dump(mode="json") if trajectory else None})
                      finally:
                          try:
                              if session_id and session_summary is None:
                                  session_summary=await sessions.close(session_id,client_id=task_group.run_id)
                          finally:
                              await runner.close()
    finally:
        await sessions.aclose()
    after = {key: directory_digest(path) for key,path in snapshots.items()}
    return {"environment":environment,"harness":harness,"source":source,"seed":config["seeds"][0],
        "planned_denominator":len(tasks),"valid_denominators":{
            state:sum(1 for row in records if row.get("state")==state and row.get("status")=="scored")
            for state in snapshots},"records":records,"runs":cost_runs,
        "snapshot_digests_before":before,"snapshot_digests_after":after,
        "snapshots_unchanged":before==after}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",default="configs/harness-evolution.json")
    parser.add_argument("--environment"); parser.add_argument("--harness"); parser.add_argument("--source")
    parser.add_argument("--check",action="store_true")
    args=parser.parse_args(argv); config=load_config(args.config)
    combos=selections(config,args.environment,args.harness,args.source)
    if not combos: parser.error("所选环境、Harness 或来源未配置")
    checks=preflight(config,combos,mode="test")
    if args.check:
        print(json.dumps(checks,ensure_ascii=False,indent=2))
        return 0 if all(item["ok"] for item in checks) else 2
    if not all(item["ok"] for item in checks):
        print(json.dumps(checks,ensure_ascii=False,indent=2),file=sys.stderr)
        return 2
    _require_execution(config)
    for env,harness,source in combos:
        result=asyncio.run(run_test(config,env,harness,source))
        path=rooted(group_id(config,env,harness,source,config["seeds"][0]))/"test.json"
        write_json(path,result,config); print(path)
    return 0


if __name__=="__main__": raise SystemExit(main())
