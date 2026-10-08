import asyncio
import csv
import json
from pathlib import Path

import pytest

from contracts import ExperimentConfig, ModelConfig
from experiments.matrix import (
    MatrixConfig,
    apply_matrix_overrides,
    classify_run,
    model_key,
    run_matrix,
    summarize_costs,
    _write_cost_csv,
    _run_cost_rows,
)


def _model(name: str) -> ModelConfig:
    return ModelConfig(model=name, api_key_env=f"{name.upper().replace('-', '_')}_KEY")


def _matrix(tmp_path: Path, *, models=None, judges=None, methods=None) -> MatrixConfig:
    return MatrixConfig(
        models=models or [_model("model-a"), _model("model-b")],
        judges=judges or [_model("judge-a"), _model("judge-b"), _model("judge-c")],
        slugs=["physicianbench"],
        methods=methods or ["singleshot"],
        cohort_root=tmp_path / "cohorts",
        output_root=tmp_path / "out",
        manager_url="http://manager:18084",
        template_dir=Path("configs"),
    )


def _payload() -> dict:
    return json.loads(Path("configs/physicianbench-comparison.json").read_text())


def test_model_groups_have_unique_run_ids_and_solver_tracks_generator(tmp_path):
    matrix = _matrix(tmp_path)
    payload = _payload()
    first = apply_matrix_overrides(
        payload, matrix.models[0], matrix.judges, tmp_path / "a", "run-a",
        matrix.manager_url, "physicianbench", None, 12, 12,
    )
    second = apply_matrix_overrides(
        payload, matrix.models[1], matrix.judges, tmp_path / "b", "run-b",
        matrix.manager_url, "physicianbench", None, 12, 12,
    )
    assert first["run_id"] != second["run_id"]
    assert first["output_dir"] != second["output_dir"]
    assert first["evaluation"]["runtime"]["models"]["solver"] == first["runtime"]["models"]["generator"]
    assert first["evaluation"]["judges"] == [judge.model_dump(mode="json") for judge in matrix.judges]


def test_matrix_accepts_two_judges_and_emits_valid_experiment_config(tmp_path):
    judges = [_model("judge-a"), _model("judge-b")]
    matrix = _matrix(tmp_path, judges=judges)
    payload = apply_matrix_overrides(
        _payload(), matrix.models[0], matrix.judges, tmp_path / "run", "run-two-judges",
        matrix.manager_url, "physicianbench", None, 12, 12,
    )

    validated = ExperimentConfig.model_validate(payload)

    assert validated.evaluation.judges == judges


@pytest.mark.parametrize("count", [1, 4])
def test_matrix_rejects_judge_counts_outside_two_or_three(tmp_path, count):
    with pytest.raises(ValueError):
        _matrix(tmp_path, judges=[_model(f"judge-{index}") for index in range(count)])


def test_matrix_rejects_duplicate_judge_names_with_different_routes(tmp_path):
    judges = [ModelConfig(model="judge", base_url=f"https://provider-{index}.test",
                          api_key_env="UNUSED") for index in range(2)]
    with pytest.raises(ValueError, match="distinct"):
        _matrix(tmp_path, judges=judges)


def test_medevolve_config_uses_only_its_group_frozen_strategy(tmp_path):
    matrix = _matrix(tmp_path)
    payload = _payload()
    group_strategy = tmp_path / "model-a" / "physicianbench" / "medevolve" / "final_strategy.json"
    overridden = apply_matrix_overrides(
        payload, matrix.models[0], matrix.judges, tmp_path / "run", "run-a",
        matrix.manager_url, "physicianbench", None, 12, 12,
        guidance_path=str(group_strategy.parent / "guidance.json"),
    )
    assert overridden["guidance_path"] == str(group_strategy.parent / "guidance.json")


def test_completed_manifest_with_review_failures_is_not_clean_completion(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({"status": "completed", "errors": []}))
    (tmp_path / "evaluation").mkdir()
    (tmp_path / "evaluation" / "tasks.json").write_text(json.dumps([{"id": "t1"}, {"id": "t2"}]))
    (tmp_path / "evaluation" / "evaluations.json").write_text(json.dumps([
        {"task_id": "t1", "status": "completed", "verification": {"valid": True},
         "quality": {}, "difficulty": {}, "completion": {}},
        {"task_id": "t2", "status": "failed", "verification": {"valid": True}},
    ]))
    result = classify_run(tmp_path)
    assert result["status"] == "completed_with_errors"
    assert result["missing_reviews"] == ["t2"]
    assert result["failed_reviews"] == ["t2"]


@pytest.mark.asyncio
async def test_completed_batch_skips_and_incomplete_batch_uses_new_attempt(tmp_path, monkeypatch):
    matrix = _matrix(tmp_path, models=[_model("model-a")])
    calls = []

    def emit_configs(*, slug, method, output_root, configs_dir, **kwargs):
        root = Path(output_root)
        config_dir = Path(configs_dir)
        config_dir.mkdir(parents=True, exist_ok=True)
        outputs = []
        for batch in ("batch-01", "batch-02"):
            payload = _payload()
            payload.update({"run_id": batch, "method": method, "output_dir": str(root / batch)})
            path = config_dir / f"{batch}.json"
            path.write_text(json.dumps(payload))
            outputs.append(path)
        return outputs

    model_root = matrix.output_root / model_key(matrix.models[0])
    completed = model_root / "physicianbench" / "singleshot" / "batch-01" / "attempt-01"
    incomplete = model_root / "physicianbench" / "singleshot" / "batch-02" / "attempt-01"
    completed.mkdir(parents=True)
    (completed / "manifest.json").write_text(json.dumps({"status": "completed", "errors": []}))
    incomplete.mkdir(parents=True)
    (incomplete / "manifest.json").write_text(json.dumps({"status": "running", "errors": []}))
    old_config = _payload()
    old_config["method"] = "singleshot"
    (incomplete / "config.json").write_text(json.dumps(old_config))
    (incomplete / "runs.json").write_text(json.dumps([
        {"stage": "evaluation/generate/0", "result": {
            "status": "failed", "usage": {"input_tokens": 8, "output_tokens": 2, "total_tokens": 10},
            "trace": [],
        }},
    ]))

    async def fake_run(config):
        calls.append(Path(config.output_dir))
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        (Path(config.output_dir) / "manifest.json").write_text(json.dumps({"status": "completed", "errors": []}))
        (Path(config.output_dir) / "config.json").write_text(json.dumps(config.model_dump(mode="json")))
        (Path(config.output_dir) / "runs.json").write_text(json.dumps([
            {"stage": "evaluation/generate/0", "result": {
                "status": "completed", "usage": {"input_tokens": 9, "output_tokens": 3, "total_tokens": 12},
                "trace": [],
            }},
        ]))
        (Path(config.output_dir) / "evaluation").mkdir(exist_ok=True)
        (Path(config.output_dir) / "evaluation" / "tasks.json").write_text(json.dumps([{"id": "t-new"}]))
        (Path(config.output_dir) / "evaluation" / "evaluations.json").write_text(json.dumps([
            {"task_id": "t-new", "status": "invalid", "verification": {"valid": False}},
        ]))

    state = await run_matrix(matrix, emit_configs=emit_configs, experiment_runner=fake_run)
    assert len(calls) == 1
    assert calls[0].name == "attempt-02"
    assert state["cost_summary"]["totals"]["total_tokens"] == 22
    assert state["selected_totals"][0]["generated_questions"] == 1


@pytest.mark.asyncio
async def test_one_group_failure_does_not_block_another(tmp_path):
    matrix = _matrix(tmp_path, models=[_model("bad-model"), _model("good-model")])
    seen = []

    def emit_configs(*, slug, method, output_root, configs_dir, **kwargs):
        config_dir = Path(configs_dir)
        config_dir.mkdir(parents=True, exist_ok=True)
        payload = _payload()
        payload["output_dir"] = str(Path(output_root) / "batch-01")
        path = config_dir / "batch-01.json"
        path.write_text(json.dumps(payload))
        return [path]

    async def fake_run(config):
        seen.append(config.run_id)
        if "bad-model" in config.output_dir:
            raise RuntimeError("mock group failure")
        path = Path(config.output_dir)
        path.mkdir(parents=True, exist_ok=True)
        (path / "manifest.json").write_text(json.dumps({"status": "completed", "errors": []}))
        (path / "runs.json").write_text("[]")
        (path / "evaluation").mkdir(exist_ok=True)
        (path / "evaluation" / "tasks.json").write_text("[]")
        (path / "evaluation" / "evaluations.json").write_text("[]")

    await run_matrix(matrix, emit_configs=emit_configs, experiment_runner=fake_run)
    assert len(seen) == 2
    state = json.loads((matrix.output_root / "matrix-status.json").read_text())
    assert any(job["status"] == "failed" for job in state["jobs"])
    assert any(job["status"] in {"completed", "completed_with_errors"} for job in state["jobs"])


@pytest.mark.asyncio
async def test_later_models_wait_for_both_priority_model_lines(tmp_path):
    matrix = _matrix(tmp_path, models=[_model("priority-a"), _model("priority-b"), _model("later-c")])
    started = []
    releases = {name: asyncio.Event() for name in ("priority-a", "priority-b")}

    def emit_configs(*, slug, method, output_root, configs_dir, **kwargs):
        config_dir = Path(configs_dir)
        config_dir.mkdir(parents=True, exist_ok=True)
        payload = _payload()
        payload["output_dir"] = str(Path(output_root) / "batch-01")
        path = config_dir / "batch-01.json"
        path.write_text(json.dumps(payload))
        return [path]

    async def fake_run(config):
        path = Path(config.output_dir)
        model_name = next((name for name in releases if name in str(path)), "later-c")
        started.append(model_name)
        if model_name in releases:
            await releases[model_name].wait()
        path.mkdir(parents=True, exist_ok=True)
        (path / "manifest.json").write_text(json.dumps({"status": "completed", "errors": []}))
        (path / "runs.json").write_text("[]")
        (path / "evaluation").mkdir(exist_ok=True)
        (path / "evaluation" / "tasks.json").write_text("[]")
        (path / "evaluation" / "evaluations.json").write_text("[]")

    running = asyncio.create_task(run_matrix(matrix, emit_configs=emit_configs, experiment_runner=fake_run))
    for _ in range(100):
        if len(started) == 2:
            break
        await asyncio.sleep(0.01)
    assert set(started) == {"priority-a", "priority-b"}
    await asyncio.sleep(0.03)
    assert "later-c" not in started
    releases["priority-a"].set()
    await asyncio.sleep(0.03)
    assert "later-c" not in started
    releases["priority-b"].set()
    await running


def test_unknown_usage_is_missing_not_known_zero(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "config.json").write_text(json.dumps({
        "method": "singleshot", "environment": {"name": "PhysicianBench"},
        "runtime": {"models": {"generator": {"model": "gen"}}},
        "evaluation": {"runtime": {"models": {"solver": {"model": "solver"}}}},
    }))
    (run_dir / "runs.json").write_text(json.dumps([
        {"stage": "evaluation/generate/0", "result": {"usage": {}, "trace": []}},
    ]))
    summary = summarize_costs([run_dir])
    assert summary["missing_usage_runs"] == 1
    assert summary["totals"]["input_tokens"] is None
    output = tmp_path / "cost.csv"
    _write_cost_csv(output, summary)
    with output.open(newline="", encoding="utf-8") as f:
        row = next(csv.DictReader(f))
    assert row["input_tokens"] == ""


def test_provider_attempt_costs_are_counted_once_and_mark_partial_usage(tmp_path):
    run_dir = tmp_path / "fallback-run"
    run_dir.mkdir()
    (run_dir / "config.json").write_text(json.dumps({
        "method": "singleshot", "environment": {"name": "PhysicianBench"},
        "runtime": {"models": {"generator": {"model": "primary", "base_url": "https://primary"}}},
        "evaluation": {"runtime": {"models": {"solver": {"model": "primary"}}}},
    }))
    (run_dir / "runs.json").write_text(json.dumps([{
        "stage": "evaluation/generate/0",
        "result": {
            "status": "completed",
            "usage": {"input_tokens": 4, "output_tokens": 3, "total_tokens": 7,
                      "runs_without_known_tokens": 1},
            "trace": [
                {"type": "provider_attempt", "provider_identity": {
                    "model": "primary", "base_url": "https://primary"},
                 "result": {"status": "failed", "usage": {}, "error": "quota"}},
                {"type": "provider_switch"},
                {"type": "provider_attempt", "provider_identity": {
                    "model": "fallback", "base_url": "https://fallback"},
                 "result": {"status": "completed", "usage": {
                     "input_tokens": 4, "output_tokens": 3, "total_tokens": 7}}},
                {"type": "model_identity", "model_identity": {"role": "generator", "model": "primary"}},
            ],
        },
    }]))
    summary = summarize_costs([run_dir])
    assert summary["totals"]["total_tokens"] == 7
    assert summary["known_usage_runs"] == 1
    assert summary["missing_usage_runs"] == 1
    assert summary["partial_usage_runs"] == 1
    assert [row["base_url"] for row in summary["records"]] == ["https://primary", "https://fallback"]
    assert [row["model"] for row in summary["records"]] == ["primary", "fallback"]
    assert all(row["phase"] == "generation" for row in summary["records"])


def test_nested_generation_stages_export_actual_primary_and_cached_provider_costs(tmp_path):
    run_dir = tmp_path / "nested-run"
    run_dir.mkdir()
    (run_dir / "config.json").write_text(json.dumps({
        "method": "singleshot", "environment": {"name": "PhysicianBench"},
        "runtime": {"models": {"generator": {"model": "configured", "base_url": "https://configured"}}},
        "evaluation": {"runtime": {"models": {}}},
    }))
    usage_primary = {"input_tokens": 6, "output_tokens": 4, "total_tokens": 10}
    usage_fallback = {"input_tokens": 8, "output_tokens": 4, "total_tokens": 12}
    (run_dir / "runs.json").write_text(json.dumps([{
        "stage": "evaluation/generate/0",
        "result": {
            "status": "completed", "usage": {"input_tokens": 14, "output_tokens": 8, "total_tokens": 22},
            "trace": [
                {"stage": "candidate_generation", "result": {
                    "status": "completed", "usage": usage_primary,
                    "trace": [{"type": "provider_identity", "model": "opencode-model",
                               "base_url": "https://opencode.example/v1"}],
                }},
                {"stage": "cached_fallback", "result": {
                    "status": "completed", "usage": usage_fallback,
                    "trace": [{"type": "provider_switch"}, {
                        "type": "provider_attempt", "provider_identity": {
                            "model": "openlux-model", "base_url": "https://openlux.example/v1"},
                        "result": {"status": "completed", "usage": usage_fallback, "trace": []},
                    }],
                }},
            ],
        },
    }]))
    summary = summarize_costs([run_dir])
    assert summary["totals"]["total_tokens"] == 22
    assert len(summary["records"]) == 2
    assert [(item["model"], item["base_url"], item["total_tokens"]) for item in summary["records"]] == [
        ("opencode-model", "https://opencode.example/v1", 10),
        ("openlux-model", "https://openlux.example/v1", 12),
    ]
    assert all(item["phase"] == "generation" for item in summary["records"])


def test_patient_feedback_costs_flatten_each_success_and_failed_call_once(tmp_path):
    run_dir = tmp_path / "feedback-run"
    run_dir.mkdir()
    (run_dir / "config.json").write_text(json.dumps({
        "method": "medevolve", "environment": {"name": "PhysicianBench"},
        "runtime": {"models": {"feedback": {"model": "configured-feedback"}}},
        "evaluation": {"runtime": {"models": {}}},
    }))
    child_one = {"status": "completed", "usage": {
        "input_tokens": 10, "output_tokens": 2, "total_tokens": 12},
        "trace": [{"type": "provider_identity", "role": "feedback",
                   "model": "feedback-a", "base_url": "https://a.example"}]}
    child_failed = {"status": "failed", "usage": {"model_turns": 1, "tool_calls": 0},
        "error": "provider unavailable",
        "trace": [{"type": "provider_identity", "role": "feedback",
                   "model": "feedback-b", "base_url": "https://b.example"}]}
    child_three = {"status": "completed", "usage": {
        "input_tokens": 20, "output_tokens": 3, "total_tokens": 23},
        "trace": [{"type": "provider_identity", "role": "feedback",
                   "model": "feedback-c", "base_url": "https://c.example"}]}
    (run_dir / "runs.json").write_text(json.dumps([{
        "stage": "round/0001/feedback",
        "result": {"status": "completed", "usage": {
            "model_turns": 3, "tool_calls": 0, "input_tokens": 30, "output_tokens": 5,
            "total_tokens": 35, "runs_without_known_tokens": 1},
            "trace": [
                {"type": "patient_feedback", "stage": "feedback", "patient_id": "p1",
                 "result": child_one},
                {"type": "patient_feedback", "stage": "feedback", "patient_id": "p2",
                 "result": child_failed},
                {"type": "patient_feedback", "stage": "feedback", "patient_id": "p3",
                 "result": child_three},
            ]},
    }]))

    rows = _run_cost_rows(run_dir)
    summary = summarize_costs([run_dir])

    assert len(rows) == 3
    assert [row["model"] for row in rows] == ["feedback-a", "feedback-b", "feedback-c"]
    assert [row["phase"] for row in rows] == ["feedback"] * 3
    assert [row["total_tokens"] for row in rows] == [12, None, 23]
    assert [row["missing_usage_calls"] for row in rows] == [0, 1, 0]
    assert summary["totals"]["total_tokens"] == 35
    assert summary["missing_usage_runs"] == 1
    assert summary["partial_usage_runs"] == 1


def test_budget_exhaustion_with_full_batch_does_not_hide_task_counts(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({"status": "completed", "errors": []}))
    (tmp_path / "evaluation").mkdir()
    (tmp_path / "evaluation" / "tasks.json").write_text(json.dumps([{"id": "t1"}]))
    (tmp_path / "evaluation" / "attempts.json").write_text(json.dumps([
        {"requested": 1, "produced": 1, "status": "budget_exhausted"},
    ]))
    (tmp_path / "evaluation" / "evaluations.json").write_text(json.dumps([
        {"task_id": "t1", "status": "invalid", "verification": {"valid": False}},
    ]))
    result = classify_run(tmp_path)
    assert result["status"] == "completed"
    assert result["task_count"] == 1


@pytest.mark.asyncio
async def test_selected_matrix_summary_aggregates_public_reviews(tmp_path):
    matrix = _matrix(tmp_path, models=[_model("model-a")])

    def emit_configs(*, output_root, configs_dir, **kwargs):
        configs_dir.mkdir(parents=True, exist_ok=True)
        payload = _payload()
        payload["output_dir"] = str(output_root / "batch-01")
        path = configs_dir / "batch-01.json"
        path.write_text(json.dumps(payload))
        return [path]

    async def fake_run(config):
        root = Path(config.output_dir)
        root.mkdir(parents=True, exist_ok=True)
        (root / "config.json").write_text(json.dumps(config.model_dump(mode="json")))
        (root / "manifest.json").write_text(json.dumps({"status": "completed", "errors": []}))
        evaluation = root / "evaluation"
        evaluation.mkdir()
        (evaluation / "tasks.json").write_text(json.dumps([{"id": "task-1"}]))
        quality = {name: {"score": 4, "scores": [3, 4, 5], "reason": "ok"}
                   for name in ("factual_accuracy", "clinical_soundness",
                                "evidence_sufficiency", "task_clarity")}
        result = {
            "task_id": "task-1", "status": "completed",
            "verification": {"valid": True}, "quality": quality,
            "difficulty": {"reasoning_depth": 3, "reasoning_depths": [2, 3, 4],
                           "reason": "ok", "complexity_metrics": {
                               "evidence_count": 2, "resource_type_count": 1,
                           }},
            "completion": {"completed": True, "reason": "ok"},
        }
        (evaluation / "evaluations.json").write_text(json.dumps([result]))
        (root / "runs.json").write_text("[]")

    state = await run_matrix(matrix, emit_configs=emit_configs, experiment_runner=fake_run)
    summary = state["selected_totals"][0]
    assert summary["quality_summary"]["factual_accuracy"]["mean"] == 4.0
    assert summary["difficulty_summary"]["reasoning_depth"]["mean"] == 3.0
    assert summary["verification"] == {"reviews": 1, "valid": 1, "invalid": 0}
    assert summary["completion"] == {"reviews": 1, "completed": 1, "missing": 0}


@pytest.mark.asyncio
async def test_medevolve_uses_only_each_model_groups_frozen_strategy(tmp_path):
    matrix = _matrix(tmp_path, models=[_model("model-a"), _model("model-b")], methods=["medevolve"])
    strategies = []

    def emit_configs(*, method, output_root, configs_dir, strategy_path=None, evolution=False, **kwargs):
        configs_dir.mkdir(parents=True, exist_ok=True)
        payload = _payload()
        payload["method"] = "medevolve"
        payload["run_id"] = "evolution" if evolution else "batch-01"
        payload["output_dir"] = str(output_root / ("evolution" if evolution else "batch-01"))
        if strategy_path:
            strategies.append(strategy_path)
            payload["guidance_path"] = str(strategy_path.parent / "guidance.json")
        path = configs_dir / f"{payload['run_id']}.json"
        path.write_text(json.dumps(payload))
        return [path]

    async def fake_run(config):
        output = Path(config.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        (output / "config.json").write_text(json.dumps(config.model_dump(mode="json")))
        (output / "manifest.json").write_text(json.dumps({"status": "completed", "errors": []}))
        (output / "runs.json").write_text("[]")
        evaluation = output / "evaluation"
        evaluation.mkdir()
        (evaluation / "tasks.json").write_text("[]")
        (evaluation / "evaluations.json").write_text("[]")
        if output.parent.name == "evolution":
            (output / "final_strategy.json").write_text("{}")
            (output / "guidance.json").write_text("{}")

    await run_matrix(matrix, emit_configs=emit_configs, experiment_runner=fake_run)
    assert len(strategies) == 2
    assert strategies[0] != strategies[1]
    assert {path.parents[4].name for path in strategies} == {
        model_key(model) for model in matrix.models
    }


def test_failed_strategy_update_cannot_be_frozen_as_completed_evolution(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({"status": "completed", "errors": []}))
    (tmp_path / "config.json").write_text(json.dumps({"method": "medevolve", "round_patients": [["p1"], ["p2"]]}))
    for index in (1, 2):
        folder = tmp_path / "rounds" / f"{index:04d}"
        folder.mkdir(parents=True)
        (folder / "feedback_run.json").write_text(json.dumps({"status": "completed", "output": {}}))
        (folder / "update_run.json").write_text(json.dumps({"status": "failed" if index == 2 else "completed", "output": None if index == 2 else {}, "error": "input budget exceeded" if index == 2 else None}))
    assessment = classify_run(tmp_path)
    assert assessment["status"] == "failed"
    assert assessment["evolution_errors"] == [{"round": 2, "phase": "update", "error": "input budget exceeded"}]
