import asyncio
import json
import sys
from collections import Counter
from pathlib import Path

import pytest

from contracts import (
    CompletionReview, EnvironmentInstance, EvaluationResult, ExperimentConfig,
    Feedback, PatientTrajectory, QualityDimension, QualityReview, RunResult, Strategy,
    Verification,
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

    async def environment(config, patients, runtime):
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
                                      "reference_answer": "answer", "evidence": []}
                                     for d in directions[:count]]})

    async def evaluate(task, trajectory, environment, config):
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
