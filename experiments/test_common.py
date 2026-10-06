"""Public comparison boundary checks, without provider calls."""

import asyncio
import json
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import AsyncMock

import pytest

from contracts import (Budget, EnvironmentInstance, EvaluationResult, ExperimentConfig,
                       GenerationDirection, PatientTrajectory, RunResult, RuntimeConfig, TaskBatch)
from experiments import runner, summarize_run
from methods.common import GenerationBudget, normalize_usage, validate_task_batch


def directions():
    return [GenerationDirection(task_type=t, difficulty="基础", weight=1)
            for t in ("诊断与解释", "治疗规划")]


def result(questions=(), *, status="completed", usage=None):
    return RunResult(output={"questions": list(questions)} if status == "completed" else None,
                     status=status, error="stage failed" if status == "failed" else None,
                     trace=[{"raw": status}], usage=usage or {"model_turns": 1, "tool_calls": 0,
                                                              "prompt_tokens": 7, "completion_tokens": 3})


def question(task_type="诊断与解释", **kwargs):
    return dict(instruction="诊断问题", task_type=task_type, reference_answer="参考答案", evidence=[], **kwargs)


def test_constraints_keep_failed_cost_and_reject_extra_direction_or_time():
    for questions in ([question(), question()], [question(decision_point="2020-01-01")],
                      [question(task_type="药物处方")], [question()] * 3):
        rejected = validate_task_batch(result(questions), directions())
        assert rejected.status == "failed" and rejected.output is None
        assert rejected.trace[-1]["output"]["questions"] == questions
        assert rejected.usage["input_tokens"] == 7
    assert validate_task_batch(result(), directions()).output == {"questions": []}
    assert validate_task_batch(result([question()]), directions()).status == "completed"
    assert normalize_usage({})["input_tokens"] is None


def test_single_node_and_candidate_pool_limits():
    one = directions()[:1]
    assert validate_task_batch(result([question()]), one,
                               max_questions=1).status == 'completed'
    assert validate_task_batch(result([question()] * 2), one,
                               max_questions=1).status == 'failed'
    pool = [GenerationDirection(task_type=t, difficulty='基础', weight=1)
            for t in ('诊断与解释', '检查与风险分层', '治疗规划', '药物处方', '治疗重规划')]
    assert validate_task_batch(result([question(), question(task_type='药物处方')]),
                               pool, candidate_pool=True).status == 'completed'
    assert validate_task_batch(result([question()] * 2), pool,
                               candidate_pool=True).status == 'failed'
    assert validate_task_batch(result([question()] * 3), pool,
                               candidate_pool=True).status == 'failed'


def test_cumulative_budget_keeps_failed_stages_and_blocks_next_call(monkeypatch):
    config = json.loads((Path(__file__).parents[1] / "configs/experiment.json").read_text())
    runtime = RuntimeConfig.model_validate(config["runtime"])
    runtime.budgets["generator"] = Budget(max_model_turns=2, max_tool_calls=3)
    single = AsyncMock(return_value=result(status="failed"))
    agent = AsyncMock(return_value=result([question()], usage={"model_turns": 1, "tool_calls": 3,
                                                             "input_tokens": 5, "output_tokens": 2}))
    monkeypatch.setattr("runtime.call_llm", single)
    monkeypatch.setattr("runtime.run_agent", agent)

    async def run():
        budget = GenerationBudget(runtime)
        await budget.call_llm("candidate", instructions="i", input_text="p", output_type=TaskBatch)
        await budget.run_agent("filter", instructions="i", input_text="p", tools=[], output_type=TaskBatch)
        assert agent.call_args.kwargs["budget"] == Budget(max_model_turns=1, max_tool_calls=3)
        accepted = budget.finish(directions())
        assert accepted.status == "completed" and len(accepted.trace) == 2
        assert accepted.trace[0]["result"]["status"] == "failed"
        assert accepted.usage["total_tokens"] == 17
        await budget.call_llm("repair", instructions="i", input_text="p", output_type=TaskBatch)
        failed = budget.finish(directions())
        assert failed.status == "budget_exhausted" and failed.output is None
        assert failed.usage["model_turns"] == 2 and failed.usage["tool_calls"] == 3
        assert len(failed.trace) == 3

    asyncio.run(run())
    assert single.await_count == agent.await_count == 1


@pytest.mark.parametrize("method", ["taskcraft", "workflow", "singleshot", "selfgen", "envexplore"])
def test_unimplemented_method_is_explicit(method):
    with pytest.raises(ValueError, match=f"Method {method} is not implemented"):
        runner.resolve_generator(method)


def test_shared_patient_sampling_failure_accounting_and_baseline_isolation(tmp_path, monkeypatch):
    values = json.loads((Path(__file__).parents[1] / "configs/experiment.json").read_text())
    values.update(run_id="comparison", evolution_patients=["e"], evaluation_patients=["h", "h2"],
                  round_patients=[["e"]], output_dir=str(tmp_path / "main"))
    values["environment"]["name"] = "fake"
    config = ExperimentConfig.model_validate(values)
    prepared, released, sampled = [], [], {}

    async def create(environment):
        return EnvironmentInstance(instance_id="cached", endpoint="fake", environment="fake")

    async def prepare(instance, patient):
        prepared.append(patient)
        return PatientTrajectory(patient_id=patient, environment="fake", text="trajectory", sources=[])

    async def release(environment, instance):
        released.append(instance.instance_id)

    async def main_generate(trajectory, guidance, strategy, requested, runtime):
        if trajectory.patient_id != "e":
            sampled[trajectory.patient_id] = requested
        return result()

    async def baseline(trajectory, environment, requested, runtime):
        assert trajectory.patient_id != "e" and environment.name == "fake"
        assert requested == sampled[trajectory.patient_id]
        if trajectory.patient_id == "h2":
            return result(status="failed")
        return result([question(task_type=requested[0].task_type)])

    module = ModuleType("methods.singleshot")
    module.generate_tasks = baseline
    monkeypatch.setitem(sys.modules, module.__name__, module)
    for name, fn in dict(create_environment=create, prepare_trajectory=prepare,
                         release_environment=release, generate_tasks=main_generate).items():
        monkeypatch.setattr(runner, name, fn)
    monkeypatch.setattr(runner, "explore_data", AsyncMock(return_value=RunResult(output={"guidance": "data"}, trace=[], usage={}, status="completed")))
    monkeypatch.setattr(runner, "explore_environment", AsyncMock(return_value=RunResult(output={"guidance": "env"}, trace=[], usage={}, status="completed")))
    feedback = dict(quality_findings=[], difficulty_findings=[], failure_findings=[], suggestions=[])
    monkeypatch.setattr(runner, "analyze_feedback", AsyncMock(return_value=RunResult(output=feedback, trace=[], usage={}, status="completed")))
    changed = config.initial_strategy.model_copy(update={"allocation": [directions()[1]]})
    monkeypatch.setattr(runner, "update_strategy", AsyncMock(return_value=RunResult(output=changed.model_dump(), trace=[], usage={}, status="completed")))
    evaluate = AsyncMock(side_effect=lambda task, *args: EvaluationResult(task_id=task.id, status="invalid", runs=[], observed_state={}))
    monkeypatch.setattr(runner, "evaluate_task", evaluate)
    asyncio.run(runner.run_experiment(config))
    prepared.clear()
    for name in ("explore_data", "explore_environment", "analyze_feedback", "update_strategy"):
        monkeypatch.setattr(runner, name, AsyncMock(side_effect=AssertionError("MedEvolve context leaked")))
    config = config.model_copy(update={"method": "singleshot", "output_dir": str(tmp_path / "baseline")})
    asyncio.run(runner.run_experiment(config))
    assert prepared == ["h", "h2"] and released == ["cached", "cached"]
    assert not (Path(config.output_dir) / "guidance.json").exists()
    summary = summarize_run(config.output_dir)
    assert summary["evaluation"]["actual_questions"] == 1
    assert summary["evaluation"]["generation_statuses"] == {"completed": 1, "failed": 1}
    assert summary["evaluation"]["short_batches"] == 2
    assert summary["evolution"]["requested_questions"] == 0
    assert summary["costs"]["generation"]["model_turns"] == 2
    assert summary["costs"]["generation"]["total_tokens"] == 20
    assert json.loads((Path(config.output_dir) / "summary.json").read_text()) == summary
    assert evaluate.call_args.args[0].source_method == "singleshot"
