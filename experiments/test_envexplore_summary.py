import json
from experiments.summary import summarize_run

def test_method_cost_includes_failed_exploration_and_generation_only(tmp_path):
    config = dict(run_id="cost", method="envexplore", evolution_patients=["train"],
        evaluation_patients=[], round_patients=[], runtime={"budgets": {}},
        evaluation={"runtime": {"budgets": {}}})
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "manifest.json").write_text(json.dumps({"status": "failed", "errors": []}))
    records = []
    for stage, turns, tools, tokens in [
        ("explore_environment", 2, 1, 12),
        ("evaluation/generate/0", 1, 0, 10),
        ("evaluation/evaluate/task1", 3, 1, 30),
    ]:
        records.append(dict(stage=stage, result=dict(status="failed", trace=[],
            usage=dict(model_turns=turns, tool_calls=tools,
                input_tokens=tokens, output_tokens=0, total_tokens=tokens))))
    (tmp_path / "runs.json").write_text(json.dumps(records))
    summary = summarize_run(str(tmp_path))
    costs = summary["costs"]
    assert costs["exploration"]["total_tokens"] == 12
    assert costs["generation"]["total_tokens"] == 10
    assert costs["evaluation"]["total_tokens"] == 30
    assert costs["method"]["total_tokens"] == 22
    assert costs["method"]["model_turns"] == 3
    assert costs["method"]["tool_calls"] == 1 and costs["method"]["runs"] == 2
