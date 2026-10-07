import asyncio
import json
import sys
from collections import Counter
from pathlib import Path

import pytest

from contracts import (
    ComplexityMetrics, CompletionReview, DifficultyScore, EnvironmentInstance,
    EvaluationResult, ExperimentConfig, Feedback, PatientTrajectory, QualityDimension,
    QualityReview, RunResult, Strategy, Verification,
)
from experiments import runner, summarize_run
from experiments.run import main
from experiments.summarize import main as summarize_main


def config_for(path):
    sample = Path(__file__).parents[2] / "configs" / "experiment.json"
    values = json.loads(sample.read_text())
    values.update(run_id="test", output_dir=str(path), evolution_patients=["e1", "e2"],
                  evaluation_patients=["h1"], round_patients=[["e1", "e2"], ["e1", "e2"]])
    values["environment"]["name"] = "fake"
    values["initial_strategy"]["generation_guidance"] = "s0"
    return ExperimentConfig.model_validate(values)


def result(output, usage=None):
    return RunResult(output=output, trace=[{"response": "saved"}],
                     usage=usage if usage is not None else {"input_tokens": 3, "output_tokens": 2},
                     status="completed")


def install_dependencies(monkeypatch, failure=False, preparation_failure=False):
    calls = Counter()
    generated = []
    trajectories = {}

    async def create(config):
        calls["create"] += 1
        return EnvironmentInstance(instance_id="cache", endpoint="fake", environment="fake")

    async def prepare(instance, patient):
        calls[f"prepare:{patient}"] += 1
        if preparation_failure:
            raise RuntimeError("trajectory read failed")
        trajectories[patient] = PatientTrajectory(patient_id=patient, environment="fake", text=patient, sources=[])
        return trajectories[patient]

    async def release(config, instance):
        calls["release"] += 1

    async def data(items, runtime):
        calls["data"] += 1
        assert len(items) == 1 and items[0].patient_id in {"e1", "e2"}
        return result({"guidance": "data"})

    async def environment(config, patients, runtime, environments=None):
        calls["environment"] += 1
        assert patients == ["e1", "e2"]
        return result({"guidance": "environment"})

    async def generate(trajectory, guidance, strategy, directions, runtime):
        assert trajectory is trajectories[trajectory.patient_id]
        assert len(directions) == 2
        assert guidance.data_guidance == "data" and guidance.environment_guidance == "environment"
        generated.append((trajectory.patient_id, strategy.generation_guidance,
                          [d.model_dump() for d in directions]))
        calls["generate"] += 1
        if failure and calls["generate"] == 3:
            return RunResult(trace=[{"raw_response": "unparsed"}], usage={}, status="failed", error="parse failure")
        count = 1 if failure and calls["generate"] == 4 else 2
        return result({"questions": [{"instruction": "question", "task_type": d.task_type,
                                      "reference_answer": "answer", "evidence": [],
                                      "execution_requirement": "read_only"}
                                     for d in directions[:count]]})

    async def evaluate(task, trajectory, environment, config, environments=None):
        calls["evaluate"] += 1
        assert task.patient_id == trajectory.patient_id
        assert task.source_method == "medevolve" and task.environment == "fake"
        if failure and calls["evaluate"] == 1:
            raise RuntimeError("evaluation transport failed")
        solver = result({"answer": "answer"}, usage={})
        return EvaluationResult(task_id=task.id, verification=Verification(valid=True, reason="supported"),
                                quality=QualityReview(**{name: QualityDimension(score=4, scores=[4, 4, 4], reason="clear")
                                                         for name in ("factual_accuracy", "clinical_soundness",
                                                                      "evidence_sufficiency", "task_clarity")}),
                                difficulty=DifficultyScore(
                                    reasoning_depth=2, reasoning_depths=[2, 2, 2], reason="整合",
                                    complexity_metrics=ComplexityMetrics(evidence_count=3, resource_type_count=2,
                                                                        time_span_days=5.0)),
                                completion=CompletionReview(completed=True, reason="done"),
                                solver_run=solver, runs=[solver], observed_state={}, status="completed")

    async def feedback(tasks, items, evaluations, runtime):
        calls["feedback"] += 1
        assert {t.patient_id for t in tasks} <= {"e1", "e2"}
        assert [t.patient_id for t in items] == ["e1", "e2"]
        assert len(evaluations) == len(tasks)
        return result(Feedback(quality_findings=[], difficulty_findings=[], failure_findings=[], suggestions=[]).model_dump())

    async def update(strategy, feedback, runtime):
        calls["update"] += 1
        if failure:
            return RunResult(trace=[{"raw_response": "budget"}], usage={}, status="budget_exhausted")
        return result(Strategy(generation_guidance=f"s{calls['update']}", allocation=strategy.allocation).model_dump())

    for name, function in dict(create_environment=create, prepare_trajectory=prepare, release_environment=release,
                               explore_data=data, explore_environment=environment, generate_tasks=generate,
                               evaluate_task=evaluate, analyze_feedback=feedback, update_strategy=update).items():
        monkeypatch.setattr(runner, name, function)
    return calls, generated


@pytest.mark.parametrize("failure", [False, True])
def test_two_rounds_and_read_only_summary(tmp_path, monkeypatch, capsys, failure):
    calls, generated = install_dependencies(monkeypatch, failure)
    config = config_for(tmp_path / "run")
    config_path = tmp_path / "config.json"
    config_path.write_text(config.model_dump_json())
    monkeypatch.setattr(sys, "argv", ["run", "--config", str(config_path)])
    main()
    assert calls["data"] == calls["environment"] == calls["create"] == calls["release"] == 1
    assert [calls[f"prepare:{p}"] for p in ["e1", "e2", "h1"]] == [1, 1, 1]
    assert calls["feedback"] == calls["update"] == 2
    assert [g[:2] for g in generated] == [("e1", "s0"), ("e2", "s0"),
                                          ("e1", "s0" if failure else "s1"),
                                          ("e2", "s0" if failure else "s1"),
                                          ("h1", "s0" if failure else "s2")]
    files = {p: p.read_bytes() for p in Path(config.output_dir).rglob("*.json")}
    summary = summarize_run(config.output_dir)
    monkeypatch.setattr(sys, "argv", ["summarize", "--run-dir", config.output_dir])
    summarize_main()
    assert json.loads(capsys.readouterr().out) == summary
    assert all(p.read_bytes() == content for p, content in files.items())
    assert summary["evolution"]["production"] == {"numerator": 5 if failure else 8, "denominator": 8,
                                                   "rate": 5 / 8 if failure else 1}
    assert summary["evaluation"]["completion_among_generated"] == {"numerator": 2, "denominator": 2, "rate": 1}
    reviews = 4 if failure else 8
    assert summary["evolution"]["quality_summary"]["task_clarity"] == {
        "mean": 4.0, "median": 4.0, "distribution": {"4": reviews}, "high_quality_rate": 1.0}
    assert summary["evolution"]["difficulty_summary"]["reasoning_depth"] == {
        "mean": 2.0, "distribution": {"2": reviews}}
    assert summary["evolution"]["difficulty_summary"]["complexity_metrics"] == {
        "avg_evidence_count": 3.0, "avg_resource_types": 2.0}
    assert summary["evaluation"]["quality_summary"]["factual_accuracy"]["high_quality_rate"] == 1.0
    assert summary["model_runs"]["runs_without_known_tokens"] > 0
    assert summary["model_runs"]["known_token_totals"]["input_tokens"] > 0
    assert summary["model_runs"]["count"] == (17 if failure else 21)
    if failure:
        assert summary["evolution"]["completion_among_generated"]["numerator"] == 4
        assert summary["model_runs"]["statuses"]["budget_exhausted"] == 2
        assert len(summary["errors"]) == 4
    else:
        other_calls, other_generated = install_dependencies(monkeypatch)
        asyncio.run(runner.run_experiment(config_for(tmp_path / "repeat")))
        assert generated == other_generated
    with pytest.raises(ValueError, match="already contains"):
        asyncio.run(runner.run_experiment(config))


def test_patient_boundaries_and_failed_preparation_release(tmp_path, monkeypatch):
    calls, _ = install_dependencies(monkeypatch, preparation_failure=True)
    config = config_for(tmp_path / "run")
    overlap = config.model_copy(update={"evaluation_patients": ["e1"]})
    with pytest.raises(ValueError, match="disjoint"):
        asyncio.run(runner.run_experiment(overlap))
    outside = config.model_copy(update={"round_patients": [["h1"]]})
    with pytest.raises(ValueError, match="evolution group"):
        asyncio.run(runner.run_experiment(outside))
    assert calls["create"] == 0
    with pytest.raises(RuntimeError, match="trajectory read"):
        asyncio.run(runner.run_experiment(config))
    assert calls["release"] == 1 and calls["data"] == 0
    summary = summarize_run(config.output_dir)
    assert summary["status"] == "failed"
    assert summary["evolution"]["completion_among_generated"]["rate"] is None
    assert summary["model_runs"]["known_token_totals"] == {}


def test_summary_ignores_legacy_quality_records():
    from experiments.summary import summarize_difficulty, summarize_quality
    legacy = {"quality": {"dimensions": {"clarity": {"judgment": "good", "reason": "clear"}}}}
    dimensions = ("factual_accuracy", "clinical_soundness",
                  "evidence_sufficiency", "task_clarity")
    current = {
        "quality": {name: {"score": 4, "scores": [4, 4, 4], "reason": "supported"}
                    for name in dimensions},
        "difficulty": {"reasoning_depth": 2, "reasoning_depths": [2, 2, 2], "reason": "整合",
                       "complexity_metrics": {"evidence_count": 3, "resource_type_count": 2,
                                              "time_span_days": 5.0}},
    }
    assert summarize_quality([legacy]) == {}
    assert summarize_difficulty([legacy]) == {}
    assert summarize_quality([legacy, current])["task_clarity"]["mean"] == 4.0
    assert summarize_difficulty([legacy, current])["reasoning_depth"]["distribution"] == {"2": 1}
def test_taskcraft_generation_receives_the_experiment_seed(tmp_path, monkeypatch):
    """公共分派把实验种子绑定到 TaskCraft 入口，其他方法入口不变。"""
    released, generated, trajectories = [], [], {}

    async def create(config):
        return EnvironmentInstance(instance_id="cache", endpoint="fake", environment="fake")

    async def prepare(instance, patient):
        trajectories[patient] = PatientTrajectory(patient_id=patient, environment="fake",
                                                  text=patient, sources=[])
        return trajectories[patient]

    async def release(config, instance):
        released.append(instance.instance_id)

    async def generate(trajectory, environment, directions, runtime, *, random_seed=None,
                       environments=None):
        assert trajectory is trajectories[trajectory.patient_id]
        assert len(directions) == 2 and environment.name == "fake"
        generated.append((trajectory.patient_id, random_seed))
        return result({"questions": [{"instruction": "问题", "task_type": directions[0].task_type,
                                      "reference_answer": "答案", "evidence": [],
                                      "execution_requirement": "read_only"}]})

    async def evaluate(task, trajectory, environment, config, environments=None):
        return EvaluationResult(task_id=task.id, observed_state={}, status="invalid", runs=[])

    for name, function in dict(create_environment=create, prepare_trajectory=prepare,
                               release_environment=release, evaluate_task=evaluate).items():
        monkeypatch.setattr(runner, name, function)
    monkeypatch.setattr("methods.taskcraft.generate_tasks", generate)

    config = config_for(tmp_path / "taskcraft")
    config = config.model_copy(update={"method": "taskcraft", "evolution_patients": [],
                                       "round_patients": []})
    asyncio.run(runner.run_experiment(config))

    assert generated == [("h1", config.random_seed)]
    assert released == ["cache"]
    assert (Path(config.output_dir) / "evaluation" / "tasks.json").exists()


@pytest.mark.parametrize("over_budget", [False, True])
def test_exhausted_generation_is_evaluated_only_within_budget(tmp_path, monkeypatch, over_budget):
    calls, _ = install_dependencies(monkeypatch)
    config = config_for(tmp_path / "partial").model_copy(update={
        "method": "taskcraft", "evolution_patients": [], "round_patients": []
    })

    async def generate(trajectory, environment, directions, runtime, *, random_seed=None,
                       environments=None):
        return RunResult(
            output={"questions": [{
                "instruction": "患者诊断是什么", "task_type": directions[0].task_type,
                "reference_answer": "答案", "evidence": [{
                    "source_id": "Condition/1", "statement": "记录中的诊断"
                }], "decision_point": None, "execution_requirement": "read_only",
            }]},
            trace=[], status="budget_exhausted", error="Cumulative generator budget exhausted",
            usage={"model_turns": runtime.budgets["generator"].max_model_turns + int(over_budget),
                   "tool_calls": 0, "input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        )

    async def evaluate(task, trajectory, environment, config, environments=None):
        calls["partial_evaluate"] += 1
        return EvaluationResult(task_id=task.id, observed_state={}, status="invalid", runs=[])

    monkeypatch.setattr("methods.taskcraft.generate_tasks", generate)
    monkeypatch.setattr(runner, "evaluate_task", evaluate)
    asyncio.run(runner.run_experiment(config))
    root = Path(config.output_dir) / "evaluation"
    attempts = json.loads((root / "attempts.json").read_text())
    tasks = json.loads((root / "tasks.json").read_text())
    assert attempts[0]["status"] == "budget_exhausted"
    assert attempts[0]["usage"]["total_tokens"] == 15
    assert attempts[0]["produced"] == len(tasks) == (0 if over_budget else 1)
    assert calls["partial_evaluate"] == (0 if over_budget else 1)
    summary = summarize_run(config.output_dir)
    assert summary["evaluation"]["generation_statuses"] == {"budget_exhausted": 1}


@pytest.mark.parametrize("name,source", [
    ("taskcraft-physicianbench", "physicianbench-comparison"),
    ("taskcraft-medagentbench", "medagentbench-flash"),
])
def test_taskcraft_configs_only_change_the_method_and_run_identity(name, source):
    """两份比较配置只改 method、run_id、output_dir，其余字段与公共配置完全相等。"""
    root = Path(__file__).parents[2] / "configs"
    derived = json.loads((root / f"{name}.json").read_text())
    original = json.loads((root / f"{source}.json").read_text())

    ExperimentConfig.model_validate_json((root / f"{name}.json").read_text())
    assert derived["method"] == "taskcraft"
    assert set(derived) == set(original)
    for field in ("method", "run_id", "output_dir"):
        assert derived[field] != original[field]
        derived.pop(field)
        original.pop(field)
    assert derived == original
