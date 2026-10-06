"""Frozen task inputs: verified sources, single environment, shared sampling."""

import hashlib
import json
import re
from pathlib import Path

import pytest
from contracts import Budget, RunResult
from harnesses.contracts import (GroupKey, HarnessAdapter, HarnessLimits, HarnessStep,
                                 Snapshot, TrainingFeedback)
from harnesses.datasets import (DatasetFreezeError, freeze_datasets, load_frozen,
                                preflight_datasets)
from pydantic import ValidationError

ROOT = Path(__file__).resolve().parents[2]
SOURCE_FIXTURE = Path(__file__).parent / "fixtures" / "frozen_source"
SOURCE_METHODS = ("medevolve", "taskcraft", "workflow", "singleshot", "selfgen", "envexplore")
HARNESSES = ("openclaw", "hermes", "ace", "reflexion")


def _task(task_id, *, patient_id="P1", environment="PhysicianBench", source_method="medevolve"):
    return {
        "instruction": f"{task_id} 的题面",
        "task_type": "治疗规划",
        "reference_answer": f"{task_id} 的参考答案",
        "evidence": [{"source_id": "MedicationRequest/m1", "statement": "记录存在"}],
        "decision_point": None,
        "id": task_id,
        "patient_id": patient_id,
        "environment": environment,
        "round_index": 0,
        "source_method": source_method,
    }


def _evaluation(task_id, valid):
    return {
        "task_id": task_id,
        "verification": None if valid is None else {"valid": valid, "reason": ""},
        "observed_state": {},
        "status": "completed",
        "runs": [],
    }


def _write_source(directory, tasks, evaluations, patients=None):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "tasks.json").write_text(json.dumps(tasks, ensure_ascii=False))
    (directory / "evaluations.json").write_text(json.dumps(evaluations, ensure_ascii=False))
    if patients is not None:
        (directory / "patients.json").write_text(json.dumps(patients, ensure_ascii=False))
    return directory / "tasks.json"


@pytest.fixture
def source_files(tmp_path_factory):
    root = tmp_path_factory.mktemp("sources")
    files = {}
    for name in SOURCE_METHODS:
        directory = root / name
        directory.mkdir()
        for filename in ("tasks.json", "evaluations.json", "patients.json"):
            (directory / filename).write_text((SOURCE_FIXTURE / filename).read_text())
        files[name] = directory / "tasks.json"
    return files


def test_task_sampling_is_shared_across_harnesses(tmp_path, source_files):
    first = freeze_datasets(source_files, count=1, seed=42,
                            destination=tmp_path / "a")
    second = freeze_datasets(source_files, count=1, seed=42,
                             destination=tmp_path / "b")
    assert first == second
    assert all(len(tasks) == 1 for tasks in first.values())
    assert all("invalid-task" not in {t.id for t in tasks}
               for tasks in first.values())


def test_frozen_manifest_records_digests_counts_and_order(tmp_path, source_files):
    frozen = freeze_datasets(source_files, count=1, seed=42, destination=tmp_path / "frozen")
    manifest = json.loads((tmp_path / "frozen" / "manifest.json").read_text())
    assert manifest["seed"] == 42 and manifest["count"] == 1
    assert manifest["environment"] == "PhysicianBench"
    assert sorted(manifest["source_methods"]) == sorted(SOURCE_METHODS)
    detail = manifest["sources"]["taskcraft"]
    tasks_digest = hashlib.sha256((SOURCE_FIXTURE / "tasks.json").read_bytes()).hexdigest()
    evaluations_digest = hashlib.sha256((SOURCE_FIXTURE / "evaluations.json").read_bytes()).hexdigest()
    assert detail["tasks_sha256"] == tasks_digest
    assert detail["evaluations_sha256"] == evaluations_digest
    assert detail["task_ids"] == ["invalid-task", "valid-task"]
    assert detail["sampling_order"] == ["valid-task"]
    assert detail["sampled_task_ids"] == ["valid-task"]
    assert detail["counts"] == {"total": 2, "verified_valid": 1, "valid": 1,
                                "invalid": 1, "unverified": 0, "patient_missing": 0}
    assert detail["patient_roster"] == "checked"
    assert load_frozen(tmp_path / "frozen") == frozen


def test_same_seed_picks_same_tasks_from_larger_source(tmp_path):
    tasks = [_task(f"task-{index}") for index in range(6)]
    files = {"medevolve": _write_source(
        tmp_path / "medevolve", tasks, [_evaluation(t["id"], True) for t in tasks])}
    first = freeze_datasets(files, count=2, seed=42, destination=tmp_path / "a")
    second = freeze_datasets(files, count=2, seed=42, destination=tmp_path / "b")
    other = freeze_datasets(files, count=2, seed=7, destination=tmp_path / "c")
    assert [t.id for t in first["medevolve"]] == [t.id for t in second["medevolve"]]
    assert len(first["medevolve"]) == 2
    assert [t.id for t in first["medevolve"]] != [t.id for t in other["medevolve"]]
    assert (tmp_path / "a" / "manifest.json").read_text() == (tmp_path / "b" / "manifest.json").read_text()


def test_missing_source_is_reported(tmp_path, source_files):
    source_files["singleshot"].unlink()
    with pytest.raises(DatasetFreezeError) as failure:
        freeze_datasets(source_files, count=1, seed=42, destination=tmp_path / "frozen")
    message = str(failure.value)
    assert "singleshot" in message and "缺失" in message
    assert "taskcraft" in message
    assert not (tmp_path / "frozen" / "manifest.json").exists()


def test_missing_verification_file_is_reported(tmp_path, source_files):
    (source_files["workflow"].parent / "evaluations.json").unlink()
    with pytest.raises(DatasetFreezeError, match="核验结果"):
        freeze_datasets(source_files, count=1, seed=42, destination=tmp_path / "frozen")


def test_insufficient_valid_yield_lists_every_source(tmp_path):
    files = {}
    for name in SOURCE_METHODS:
        tasks = [_task(f"{name}-{index}", source_method=name)
                 for index in range(2 if name == "medevolve" else 1)]
        files[name] = _write_source(
            tmp_path / name, tasks, [_evaluation(t["id"], True) for t in tasks])
    with pytest.raises(DatasetFreezeError) as failure:
        freeze_datasets(files, count=2, seed=42, destination=tmp_path / "frozen")
    message = str(failure.value)
    assert "有效量不足" in message
    for name in SOURCE_METHODS:
        assert name in message
    assert "medevolve" in message and "有效 2/2" in message
    assert not (tmp_path / "frozen" / "manifest.json").exists()


def test_mixed_environments_in_one_source_are_rejected(tmp_path):
    tasks = [_task("pb-task"), _task("mab-task", patient_id="S1", environment="MedAgentBench")]
    files = {"medevolve": _write_source(
        tmp_path / "medevolve", tasks,
        [_evaluation("pb-task", True), _evaluation("mab-task", True)])}
    with pytest.raises(DatasetFreezeError, match="环境不一致"):
        freeze_datasets(files, count=1, seed=42, destination=tmp_path / "frozen")


def test_mixed_environments_across_sources_are_rejected(tmp_path):
    files = {
        "medevolve": _write_source(tmp_path / "medevolve", [_task("pb-task")],
                                   [_evaluation("pb-task", True)]),
        "taskcraft": _write_source(
            tmp_path / "taskcraft",
            [_task("mab-task", patient_id="S1", environment="MedAgentBench", source_method="taskcraft")],
            [_evaluation("mab-task", True)]),
    }
    with pytest.raises(DatasetFreezeError, match="环境不一致"):
        freeze_datasets(files, count=1, seed=42, destination=tmp_path / "frozen")


def test_unverified_and_unknown_patients_are_excluded(tmp_path):
    tasks = [_task("keep"), _task("rejected-verification"),
             _task("rejected-patient", patient_id=""), _task("unknown-patient", patient_id="P9")]
    evaluations = [_evaluation("keep", True), _evaluation("rejected-verification", False),
                   _evaluation("rejected-patient", True), _evaluation("unknown-patient", True)]
    files = {"medevolve": _write_source(
        tmp_path / "medevolve", tasks, evaluations,
        patients={"environment": "PhysicianBench", "patients": ["P1"]})}
    frozen = freeze_datasets(files, count=1, seed=42, destination=tmp_path / "frozen")
    assert [t.id for t in frozen["medevolve"]] == ["keep"]
    manifest = json.loads((tmp_path / "frozen" / "manifest.json").read_text())
    assert manifest["sources"]["medevolve"]["counts"] == {
        "total": 4, "verified_valid": 3, "valid": 1, "invalid": 3,
        "unverified": 0, "patient_missing": 2}


def test_patient_roster_environment_mismatch_is_rejected(tmp_path):
    tasks = [_task("keep")]
    files = {"medevolve": _write_source(
        tmp_path / "medevolve", tasks, [_evaluation("keep", True)],
        patients={"environment": "MedAgentBench", "patients": ["P1"]})}
    with pytest.raises(DatasetFreezeError, match="环境不一致"):
        freeze_datasets(files, count=1, seed=42, destination=tmp_path / "frozen")


def test_source_without_any_verification_is_rejected(tmp_path):
    tasks = [_task("task-a"), _task("task-b", patient_id="P2")]
    files = {"medevolve": _write_source(
        tmp_path / "medevolve", tasks,
        [_evaluation("task-a", None), _evaluation("task-b", None)])}
    with pytest.raises(DatasetFreezeError, match="核验结果"):
        freeze_datasets(files, count=1, seed=42, destination=tmp_path / "frozen")


def test_invalid_requested_count_is_rejected(tmp_path, source_files):
    with pytest.raises(DatasetFreezeError, match="任务量"):
        freeze_datasets(source_files, count=0, seed=42, destination=tmp_path / "frozen")


def test_harness_contracts_keep_frozen_field_names():
    assert set(GroupKey.model_fields) == {"run_id", "environment", "harness", "source_method", "seed"}
    assert set(HarnessStep.model_fields) == {"run", "learning_runs", "learning_status"}
    assert set(TrainingFeedback.model_fields) == {
        "task_id", "completed", "failure_category", "reason", "reference_answer", "evidence"}
    assert set(Snapshot.model_fields) == {"path", "digest"}
    assert GroupKey(run_id="run", environment="PhysicianBench", harness="ace",
                    source_method="selfgen", seed=42).harness == "ace"
    for attribute in ("start", "execute", "learn", "freeze", "close"):
        assert callable(getattr(HarnessAdapter, attribute))
    with pytest.raises(ValidationError):
        HarnessStep(run=RunResult(trace=[], usage={}, status="completed"),
                    learning_runs=[], learning_status="learned")
    for name in ("max_input_tokens", "max_output_tokens", "max_total_tokens",
                 "timeout_seconds", "learning_timeout_seconds"):
        assert name in HarnessLimits.model_fields


def test_harness_limits_reject_non_positive_values():
    limits = dict(execution=Budget(max_model_turns=1, max_tool_calls=0),
                  learning=Budget(max_model_turns=1, max_tool_calls=0),
                  max_input_tokens=1000, max_output_tokens=100,
                  max_total_tokens=2000, timeout_seconds=60, learning_timeout_seconds=30)
    assert HarnessLimits(**limits).max_total_tokens == 2000
    for name in ("max_input_tokens", "max_output_tokens", "max_total_tokens",
                 "timeout_seconds", "learning_timeout_seconds"):
        with pytest.raises(ValidationError):
            HarnessLimits(**{**limits, name: 0})


def test_duplicate_task_and_evaluation_ids_are_rejected(tmp_path):
    duplicate_tasks = [_task("same"), _task("same", patient_id="P2")]
    files = {"medevolve": _write_source(tmp_path / "medevolve", duplicate_tasks,
                                      [_evaluation("same", True)])}
    with pytest.raises(DatasetFreezeError, match="重复"):
        freeze_datasets(files, count=1, seed=42, destination=tmp_path / "a")
    duplicate_evaluations = [_evaluation("one", True), _evaluation("one", False)]
    files = {"medevolve": _write_source(tmp_path / "medevolve", [_task("one")],
                                      duplicate_evaluations)}
    with pytest.raises(DatasetFreezeError, match="重复"):
        freeze_datasets(files, count=1, seed=42, destination=tmp_path / "b")


def test_tampered_frozen_set_is_rejected(tmp_path, source_files):
    frozen = freeze_datasets(source_files, count=1, seed=42, destination=tmp_path / "frozen")
    payload = json.loads((tmp_path / "frozen" / "tasks.json").read_text())
    payload["taskcraft"] = [payload["taskcraft"][0], payload["taskcraft"][0]]
    (tmp_path / "frozen" / "tasks.json").write_text(json.dumps(payload))
    with pytest.raises(DatasetFreezeError, match="manifest"):
        load_frozen(tmp_path / "frozen")
    assert frozen["taskcraft"][0].id == "valid-task"


def test_preflight_reports_missing_source_reason(tmp_path, source_files):
    environments = {"PhysicianBench": {"sources": {
        "workflow": {"tasks": None, "missing_reason": "PhysicianBench 尚无 Workflow 生成产物"},
        "taskcraft": {"tasks": str(source_files["taskcraft"])},
    }}}
    report = preflight_datasets(environments, count=1, seed=42)
    entry = report["PhysicianBench"]
    assert entry["freezable"] is False
    assert entry["sources"]["workflow"] == {
        "status": "missing", "reason": "PhysicianBench 尚无 Workflow 生成产物"}
    assert entry["sources"]["taskcraft"]["status"] == "ok"
    assert entry["sources"]["taskcraft"]["valid"] == 1
    assert entry["blockers"] == ["workflow: PhysicianBench 尚无 Workflow 生成产物"]
    assert not list(tmp_path.iterdir())


def test_preflight_blocks_insufficient_yield_and_unverified_source(tmp_path):
    files = {
        "singleshot": _write_source(tmp_path / "singleshot", [_task("only")],
                                    [_evaluation("only", True)]),
        "medevolve": _write_source(tmp_path / "medevolve", [_task("plain", source_method="medevolve")],
                                   [_evaluation("plain", None)]),
    }
    environments = {"PhysicianBench": {"sources": {
        name: {"tasks": str(path)} for name, path in files.items()}}}
    entry = preflight_datasets(environments, count=2, seed=42)["PhysicianBench"]
    assert entry["freezable"] is False
    assert entry["sources"]["singleshot"]["status"] == "ok"
    assert "有效 1" in entry["sources"]["singleshot"]["reason"]
    assert entry["sources"]["medevolve"]["status"] == "invalid"
    assert "核验结果" in entry["sources"]["medevolve"]["reason"]
    assert len(entry["blockers"]) == 2


def test_configured_sources_are_not_yet_freezable():
    config = json.loads((ROOT / "configs/harness-evolution.json").read_text())
    report = preflight_datasets(config["environments"], count=config["tasks_per_group"],
                                seed=config["seeds"][0])
    for environment_name, environment in config["environments"].items():
        entry = report[environment_name]
        assert sorted(entry["sources"]) == sorted(environment["sources"])
        assert entry["freezable"] is False
        assert entry["blockers"]
        for method, source in entry["sources"].items():
            assert source["status"] in {"ok", "missing", "invalid"}, method
            if source["status"] != "ok":
                assert source["reason"], method


def test_experiment_config_freezes_sources_harnesses_and_budgets():
    config = json.loads((ROOT / "configs/harness-evolution.json").read_text())
    assert config["execute"] is False
    assert config["harnesses"] == list(HARNESSES)
    assert config["source_methods"] == list(SOURCE_METHODS)
    assert config["tasks_per_group"] == 1 and config["evolution_rounds"] == 1
    assert config["seeds"] == [42]
    assert config["run_dir_template"] == (
        "outputs/harness-evolution/<run_id>/<environment>/<harness>/<source_method>/<seed>")
    limits = HarnessLimits.model_validate(config["limits"])
    assert limits.execution.max_model_turns > 0 and limits.learning_timeout_seconds > 0
    assert config["experience_limit"]["max_entries"] > 0
    for environment_name, environment in config["environments"].items():
        assert environment["name"] == environment_name
        assert environment["manager_url"] and environment["seed_data"]
        assert sorted(environment["sources"]) == sorted(SOURCE_METHODS)
        for method, source in environment["sources"].items():
            if source["tasks"] is None:
                assert source["missing_reason"]
            else:
                assert source["tasks"].endswith("tasks.json")
        benchmark = environment["benchmark"]
        assert re.fullmatch(r"[0-9a-f]{40}", benchmark["commit"])
        assert benchmark["repository"].startswith("https://")
        assert benchmark["license"]
    assert config["fallback_models"] == []
    for model in config["models"].values():
        assert re.fullmatch(r"[A-Z][A-Z0-9_]*", model["api_key_env"])
        assert model["model"] and model["base_url"].startswith("http")


def test_upstream_lock_records_observed_commits_and_open_digests():
    lock = json.loads((ROOT / "docker/harnesses/upstreams.lock.json").read_text())
    assert sorted(lock["upstreams"]) == sorted(HARNESSES)
    for name, upstream in lock["upstreams"].items():
        assert re.fullmatch(r"[0-9a-f]{40}", upstream["commit"]), name
        assert upstream["repository"].startswith("https://"), name
        assert upstream["license"], name
        assert upstream["components"], name
        assert upstream["image"]["digest"] is None, name
        assert upstream["image"]["unverified"], name
        assert upstream["image"]["reference"], name
    assert lock["upstreams"]["reflexion"]["dependency_lock"] is None
    assert lock["upstreams"]["ace"]["components"] == [
        "ace/ace.py", "ace/core/reflector.py", "ace/core/curator.py", "playbook_utils.py"]
