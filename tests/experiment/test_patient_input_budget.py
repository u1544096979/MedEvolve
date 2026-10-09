"""患者材料无法容纳时只记录该患者失败，原始缓存保持完整。"""

import json

import pytest

from contracts import Evidence, PatientTrajectory, RunResult
from experiments import runner
from tests.experiment.test_closed_loop import config_for, install_dependencies
from tests.runtime.test_context import MODEL, patient


@pytest.mark.asyncio
async def test_oversized_patient_does_not_block_other_patients(tmp_path, monkeypatch):
    install_dependencies(monkeypatch)
    originals = {}
    generated = []

    async def prepare(instance, patient_id):
        if patient_id == "bad":
            trajectory = PatientTrajectory(patient_id="bad", environment="fake",
                text="Patient/bad\n" + "病历" * 20000,
                sources=[Evidence(source_id="Patient/bad", statement="病历" * 20000)])
        else:
            trajectory = patient().model_copy(update={"patient_id": patient_id, "environment": "fake"})
        originals[patient_id] = trajectory.model_dump(mode="json")
        return trajectory

    async def generate(trajectory, environment, task_types, runtime):
        generated.append(trajectory)
        return RunResult(output={"questions": []}, trace=[], status="completed",
                         usage={"model_turns": 1, "tool_calls": 0})

    monkeypatch.setattr(runner, "prepare_trajectory", prepare)
    monkeypatch.setattr(runner, "resolve_generator", lambda method: generate)
    config = config_for(tmp_path / "run").model_copy(update={"method": "singleshot",
        "evolution_patients": [], "round_patients": [], "evaluation_patients": ["bad", "good"],
        "compare_initial_strategy": False})
    config.runtime.models["generator"] = MODEL
    await runner.run_experiment(config)
    attempts = json.loads((tmp_path / "run/evaluation/attempts.json").read_text())
    assert [attempt["status"] for attempt in attempts] == ["failed", "completed"]
    assert attempts[0]["usage"]["model_turns"] == 0
    assert "required records" in attempts[0]["error"]
    assert [trajectory.patient_id for trajectory in generated] == ["good"]
    assert generated[0].input_truncated
    for index, patient_id in enumerate(("bad", "good")):
        saved = json.loads((tmp_path / f"run/trajectories/evaluation/{index:04d}.json").read_text())
        assert saved == originals[patient_id]
    manifest = json.loads((tmp_path / "run/manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["generation"]["failed"] == 1
    assert manifest["evaluation"]["failed"] == 0
