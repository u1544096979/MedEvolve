"""Result aggregation boundaries: unknown usage, source costs and paired intervals."""
import json

from experiments.harness_summary import _paired_interval, summarize


def test_cost_usage_aliases_and_patient_cluster_bootstrap_are_stable():
    result={"environment":"PhysicianBench","harness":"ace","source":"medevolve","planned_denominator":2,
      "runs":[{"task_id":"a","state":"initial","stage":"testing","role":"execution","run":{"status":"completed","usage":{"input":8,"output":3}}},
              {"task_id":"a","state":"initial","stage":"testing_review","role":"completion","run":{"status":"completed","usage":{"inputTokens":2,"outputTokens":1}}}],
      "records":[{"state":"initial","task_id":"a","patient_id":"p1","status":"scored","completed":False},
                 {"state":"final","task_id":"a","patient_id":"p1","status":"scored","completed":True},
                 {"state":"initial","task_id":"b","patient_id":"p1","status":"scored","completed":False},
                 {"state":"final","task_id":"b","patient_id":"p1","status":"scored","completed":True},
                 {"state":"initial","task_id":"c","patient_id":"p2","status":"scored","completed":False},
                 {"state":"final","task_id":"c","patient_id":"p2","status":"scored","completed":True}],
      "valid_denominators":{"initial":3,"final":3}}
    config={"seeds":[42],"run_dir_template":"/nonexistent/<run_id>/<environment>/<harness>/<source_method>/<seed>","run_id":"x","models":{}}
    manifest={"sources":{"medevolve":{"generation_cost":None,"split":{"test_task_ids":["a","b","c"]},
      "evaluation_metadata":{"a":{"completion":{"completed":False}},"b":{"completion":{"completed":False}},"c":{"completion":{"completed":False}}}}}}
    summary=summarize([result],config,manifest)["results"][0]
    assert summary["baseline"] == summary["initial"]
    assert summary["final"]["lift_percentage_points"]==100
    ci=summary["final"]["paired_lift_percentage_points"]["confidence_interval"]
    assert ci["samples"]==1000 and ci["seed"]==42 and ci["estimate"]==100
    assert summary["costs"]["testing"]["known"] is True
    assert summary["costs"]["review"]["known"] is True
    partial={"environment":"PhysicianBench","harness":"ace","source":"medevolve","planned_denominator":1,
        "records":[],"runs":[{"task_id":"x","stage":"testing","role":"execution",
            "run":{"status":"failed","usage":{"input_tokens":7}}}]}
    cost_summary=summarize([partial],config,manifest)["results"][0]["costs"]["testing"]
    assert cost_summary["known"] is False and cost_summary["usage"]["input_tokens"]==7
    assert cost_summary["usage"]["output_tokens"] is None and cost_summary["cost_status"]=="unknown"
    assert _paired_interval({"p1":[1.0]}) is None



def test_generation_cost_is_counted_once_per_environment_and_source():
    result={"environment":"PhysicianBench","harness":"openclaw","source":"s","planned_denominator":0,
        "records":[],"runs":[]}
    second={**result,"harness":"hermes"}
    manifest={"sources":{"s":{"generation_cost":{"input_tokens":3,"output_tokens":2},
        "split":{"test_task_ids":[]},"evaluation_metadata":{}}}}
    config={"seeds":[42],"run_id":"x","run_dir_template":"/absent/<run_id>/<environment>/<harness>/<source_method>/<seed>","models":{}}
    output=summarize([result,second],config,manifest)["results"]
    assert output[0]["costs"]["generation"]["deduplicated"] is False
    assert output[1]["costs"]["generation"]["deduplicated"] is True


def test_generation_usage_prices_when_complete_and_deduplicates_across_four_harnesses():
    results=[{"environment":"PhysicianBench","harness":h,"source":"s","planned_denominator":0,
        "records":[],"runs":[]} for h in ("openclaw","hermes","ace","reflexion")]
    manifest={"sources":{"s":{"generation_cost":{"input_tokens":1_000_000,"output_tokens":500_000},
        "split":{"test_task_ids":[]},"evaluation_metadata":{}}}}
    config={"seeds":[42],"run_id":"x","run_dir_template":"/absent/<run_id>/<environment>/<harness>/<source_method>/<seed>",
        "models":{"generation":{"model":"generator","price_per_million_tokens":2}}}
    output=summarize(results,config,manifest)["results"]
    first=output[0]["costs"]["generation"]
    assert first["known"] is True and first["cost_usd"]==3
    assert sum(not row["costs"]["generation"]["deduplicated"] for row in output)==1


def test_openclaw_learning_uses_observed_execution_model_price():
    result={"environment":"PhysicianBench","harness":"openclaw","source":"s","planned_denominator":0,
        "records":[],"runs":[{"task_id":"a","stage":"learning","role":"learning","run":{
            "status":"completed","usage":{"model":"execution-model","input_tokens":1_000_000,"output_tokens":0}}}]}
    manifest={"sources":{"s":{"generation_cost":None,"split":{"test_task_ids":[]},"evaluation_metadata":{}}}}
    config={"seeds":[42],"run_id":"x","run_dir_template":"/absent/<run_id>/<environment>/<harness>/<source_method>/<seed>",
        "models":{"execution":{"model":"execution-model","price_per_million_tokens":4},
                 "learning":{"model":"learning-model","price_per_million_tokens":99}}}
    costs=summarize([result],config,manifest)["results"][0]["costs"]["learning"]
    assert costs["cost_usd"]==4
    assert costs["raw_usage"][0]["resolved_model"]["model"]=="execution-model"
