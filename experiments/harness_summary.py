"""汇总 Harness 测试成绩、患者配对区间与阶段成本。"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from experiments.harness_train import ROOT, freeze_root, load_config, rooted, write_json

USAGE_ALIASES = {
    "input_tokens": ("input_tokens", "prompt_tokens", "input", "inputTokens"),
    "output_tokens": ("output_tokens", "completion_tokens", "output", "outputTokens"),
    "cache_read_tokens": ("cache_read_tokens", "cacheRead", "cache_read", "cacheReadTokens"),
    "cache_write_tokens": ("cache_write_tokens", "cacheWrite", "cache_write", "cacheWriteTokens"),
}


def _metric(usage, aliases):
    for key in aliases:
        value = usage.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return value
    return None


def _paired_interval(pairs, seed=42, samples=1000):
    """按患者聚类重采样，保留同一患者的全部配对题目。"""
    if len(pairs) < 2:
        return None
    rng = random.Random(seed)
    patients = sorted(pairs)
    estimate = sum(sum(pairs[p]) for p in patients) / sum(len(pairs[p]) for p in patients)
    means = []
    for _ in range(samples):
        sampled = [rng.choice(patients) for _ in patients]
        values = [value for patient in sampled for value in pairs[patient]]
        means.append(sum(values)/len(values))
    means.sort()
    return {"estimate":estimate*100,"low":means[int(.025*samples)]*100,
            "high":means[min(samples-1,int(.975*samples))]*100,
            "method":"patient_cluster_bootstrap","samples":samples,"seed":seed}


def _score(records, state):
    valid = [row for row in records if row.get("state")==state and row.get("status")=="scored"
             and isinstance(row.get("completed"),bool)]
    denominator = len(valid)
    return {"numerator":sum(row["completed"] for row in valid),"denominator":denominator,
            "rate":sum(row["completed"] for row in valid)/denominator if denominator else None,
            "valid_denominator":denominator,
            "failed_categories":_counts(row.get("failure_category") for row in valid),
            "scoring_errors":sum(row.get("status")=="scoring_error" for row in records if row.get("state")==state)}


def _counts(values):
    counts={}
    for value in values:
        key=value or "none"
        counts[key]=counts.get(key,0)+1
    return counts


def _model_for(item, stage, config):
    run=item.get("run",{}); usage=run.get("usage",{})
    model_name=usage.get("model")
    for entry in run.get("trace",[]):
        identity=entry.get("model_identity",{}) if isinstance(entry,dict) else {}
        model_name=model_name or identity.get("model")
    for name, model in config.get("models",{}).items():
        if model.get("model")==model_name:
            return model
    role=item.get("role")
    if role in config.get("models",{}):
        return config["models"][role]
    return config.get("models",{}).get("execution" if stage in {"execution","testing"} else "completion_review" if stage=="review" else "learning",{})


def _cost(run_rows, config, generation=None):
    result={}
    for stage in ("generation","execution","learning","review","testing"):
        records=[]
        if stage=="generation":
            records=[{"role":"generation","usage":generation,"known":None}]
        else:
            for row in run_rows:
                normalized = "review" if row.get("stage") in {"review","testing_review"} else row.get("stage")
                if normalized != stage:
                    continue
                run=row.get("run",{}); usage=run.get("usage",{})
                explicit=usage.get("known")
                input_value=_metric(usage,USAGE_ALIASES["input_tokens"])
                output_value=_metric(usage,USAGE_ALIASES["output_tokens"])
                known=(explicit is not False and input_value is not None and output_value is not None)
                records.append({"task_id":row.get("task_id"),"role":row.get("role"),"usage":usage,
                    "known":known,"status":run.get("status"),"model":_model_for(row,stage,config).get("model")})
        totals={key:0 for key in USAGE_ALIASES}
        observed={key:False for key in USAGE_ALIASES}
        known=bool(records)
        raw=[]
        for item in records:
            usage=item.get("usage") or {}; raw.append(item)
            if item.get("known") is not True:
                known=False
            for name,aliases in USAGE_ALIASES.items():
                value=_metric(usage,aliases)
                if value is not None:
                    totals[name]+=value; observed[name]=True
        normalized_usage={key:(totals[key] if observed[key] else None) for key in totals}
        cost_values=[]
        for item in raw:
            model=_model_for(item,stage,config)
            price=model.get("price_per_million_tokens") if isinstance(model,dict) else None
            usage=item.get("usage") or {}
            incoming=_metric(usage,USAGE_ALIASES["input_tokens"])
            outgoing=_metric(usage,USAGE_ALIASES["output_tokens"])
            if price is None or incoming is None or outgoing is None or item.get("known") is not True:
                cost_values.append(None)
            else:
                cost_values.append((incoming+outgoing)*price/1_000_000)
        amount=sum(cost_values) if cost_values and all(x is not None for x in cost_values) else None
        result[stage]={"known":known,"usage":normalized_usage,"raw_usage":raw,
                       "cost_usd":amount,"cost_status":"known" if amount is not None else "unknown",
                       "unknown_records":sum(item.get("known") is not True for item in raw),
                       "deduplicated":stage=="generation" and generation is None}
    return result


def summarize(test_results, config, manifest=None):
    output=[]
    generation_seen=set()
    for result in test_results:
        environment,harness,source=result["environment"],result["harness"],result["source"]
        current_manifest = manifest or json.loads((freeze_root(config,environment,source)/"manifest.json").read_text(encoding="utf-8"))
        records=result.get("records",[])
        initial=_score(records,"initial"); final=_score(records,"final")
        frozen_source=current_manifest["sources"][source]
        # Pair same tasks first, then retain each patient's full task cluster.
        initial_by={row["task_id"]:row for row in records if row.get("state")=="initial" and row.get("status")=="scored" and isinstance(row.get("completed"),bool)}
        final_by={row["task_id"]:row for row in records if row.get("state")=="final" and row.get("status")=="scored" and isinstance(row.get("completed"),bool)}
        paired_by_patient={}
        for task_id in sorted(initial_by.keys() & final_by.keys()):
            patient=initial_by[task_id].get("patient_id")
            if patient is not None and patient==final_by[task_id].get("patient_id"):
                paired_by_patient.setdefault(patient,[]).append(int(final_by[task_id]["completed"])-int(initial_by[task_id]["completed"]))
        paired_estimate=sum(sum(items) for items in paired_by_patient.values())/sum(map(len,paired_by_patient.values())) if paired_by_patient else None
        final["lift_percentage_points"]=(final["rate"]-initial["rate"])*100 if final["rate"] is not None and initial["rate"] is not None else None
        final["paired_lift_percentage_points"]={"estimate":paired_estimate*100 if paired_estimate is not None else None,
            "confidence_interval":_paired_interval(paired_by_patient)}
        train_path=rooted(config["run_dir_template"].replace("<run_id>",config["run_id"]).replace("<environment>",environment).replace("<harness>",harness).replace("<source_method>",source).replace("<seed>",str(config["seeds"][0])))/"train.json"
        train=json.loads(train_path.read_text(encoding="utf-8")) if train_path.is_file() else {}
        generation=frozen_source.get("generation_cost")
        generation_key=(environment,source)
        if generation_key in generation_seen:
            generation=None
        else:
            generation_seen.add(generation_key)
        runs=[*train.get("runs",[]),*train.get("learning_runs",[]),*result.get("runs",[])]
        output.append({"environment":environment,"harness":harness,"source":source,
            "planned_denominator":result.get("planned_denominator"),
            "generation_cost_source":frozen_source.get("generation_cost_source"),
            "generation_cost_reason":frozen_source.get("generation_cost_reason"),"baseline":initial,
            "initial":initial,"final":final,"paired_patients":len(paired_by_patient),
            "failure_types":_counts(row.get("failure_category") for row in records),
            "costs":_cost(runs,config,generation)})
    return {"version":1,"seed":42,"results":output}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",default="configs/harness-evolution.json")
    parser.add_argument("--run-root")
    parser.add_argument("--output")
    args=parser.parse_args(argv); config=load_config(args.config)
    root=rooted(args.run_root or config["output_root"])
    results=[]
    for path in sorted(root.rglob("test.json")):
        results.append(json.loads(path.read_text(encoding="utf-8")))
    summary=summarize(results,config)
    out=Path(args.output) if args.output else root/"summary.json"
    write_json(out,summary,config); print(out)


if __name__=="__main__":main()
