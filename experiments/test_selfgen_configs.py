import json
from pathlib import Path
import pytest
from contracts import (
    Evidence, ExperimentConfig, GenerationDirection, PatientTrajectory, RunResult,
)
from experiments.runner import resolve_generator
from methods.selfgen import generate_tasks

KINDS = ["诊断与解释", "检查与风险分层"]
BUDGET = {"max_model_turns": 6, "max_tool_calls": 0}

@pytest.mark.parametrize("source,target", [
    ("physicianbench-comparison.json", "physicianbench-selfgen.json"),
    ("medagentbench-flash.json", "medagentbench-selfgen.json"),
])
def test_config_keeps_sampling_environment_and_model(source, target):
    root = Path(__file__).parents[1] / "configs"
    base = json.loads((root / source).read_text())
    values = json.loads((root / target).read_text())
    config = ExperimentConfig.model_validate(values)
    assert config.method == "selfgen" and config.round_patients == []
    assert values["compare_initial_strategy"] is False
    assert config.evaluation_patients
    for key in ("environment", "evolution_patients", "evaluation_patients",
                "initial_strategy", "random_seed", "evaluation"):
        assert values[key] == base[key]
    assert set(config.runtime.models) == {"generator"}
    assert set(config.runtime.budgets) == {"generator"}
    assert values["runtime"]["models"]["generator"] == base["runtime"]["models"]["generator"]
    assert config.runtime.budgets["generator"].model_dump() == BUDGET
    assert resolve_generator("selfgen") is generate_tasks

def draft(version="初稿", kinds=KINDS):
    return {"batch": {"questions": [dict(
        instruction=f"{kind}-{version}", task_type=kind, reference_answer=f"答案-{version}",
        decision_point=None,
        evidence=[dict(source_id="Observation/1", statement="原始记录事实")],
    ) for kind in kinds]}, "direction_indices": list(range(len(kinds)))}

def review(size, passed=True):
    return {"questions": [dict(candidate_index=index, passed=passed,
                               issues=[] if passed else [f"第{index}题证据不足"])
                           for index in range(size)]}

def declared_config(target):
    root = Path(__file__).parents[1] / "configs"
    return ExperimentConfig.model_validate(json.loads((root / target).read_text()))

async def run_declared_loop(monkeypatch, target, script):
    config = declared_config(target)
    assert config.runtime.budgets["generator"].model_dump() == BUDGET
    trajectory = PatientTrajectory(
        patient_id="p1", environment=config.environment.name, text="患者原始轨迹",
        sources=[Evidence(source_id="Observation/1",
                          statement='{"resourceType":"Observation","id":"1"}')])
    directions = [GenerationDirection(task_type=kind, difficulty="基础", weight=1)
                  for kind in KINDS]
    calls = []

    async def call(**kwargs):
        calls.append(kwargs)
        return RunResult(output=script[len(calls) - 1], status="completed", trace=[],
                         usage={"model_turns": 1, "tool_calls": 0})

    monkeypatch.setattr("runtime.call_llm", call)
    result = await generate_tasks(trajectory, config.environment, directions, config.runtime)
    return result, calls

@pytest.mark.asyncio
@pytest.mark.parametrize("target", ["physicianbench-selfgen.json", "medagentbench-selfgen.json"])
async def test_declared_generator_budget_completes_shortest_loop(monkeypatch, target):
    result, calls = await run_declared_loop(monkeypatch, target, [draft(), review(len(KINDS))])
    assert result.status == "completed" and result.output is not None
    assert len(calls) == 2

@pytest.mark.asyncio
@pytest.mark.parametrize("target", ["physicianbench-selfgen.json", "medagentbench-selfgen.json"])
async def test_declared_generator_budget_completes_worst_case_loop(monkeypatch, target):
    script = [draft(), review(len(KINDS), passed=False),
              draft("修订一"), review(len(KINDS), passed=False),
              draft("修订二"), review(len(KINDS))]
    result, calls = await run_declared_loop(monkeypatch, target, script)
    assert result.status == "completed" and len(result.output["questions"]) == len(KINDS)
    assert len(calls) == BUDGET["max_model_turns"] == 6
    assert [item["stage"] for item in result.trace] == [
        "candidate_generation", "self_review/1", "refine/1",
        "self_review/2", "refine/2", "self_review/3", "final_filter"]
