"""Workflow experiment wiring against the public environment and evaluation entry points."""

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from contracts import (EnvironmentInstance, EvaluationResult, Evidence,
                       ExperimentConfig, PatientTrajectory, RunResult)
from experiments import runner, summarize_run


@pytest.mark.parametrize('status,produced', [('completed', 1), ('completed', 0), ('failed', 0)])
def test_workflow_pool_and_public_evaluation(tmp_path, monkeypatch, status, produced):
    root = Path(__file__).resolve().parents[2]
    values = json.loads((root / 'configs/physicianbench-comparison.json').read_text())
    values.update(method='workflow', evaluation_patients=['p'], evolution_patients=[],
                  round_patients=[], output_dir=str(tmp_path), run_id='workflow-test')
    config = ExperimentConfig.model_validate(values)
    trajectory = PatientTrajectory(patient_id='p', environment=config.environment.name,
        text='[undated] Observation/o1\n{}\n\n[undated] MedicationRequest/m1\n{}',
        sources=[Evidence(source_id='Observation/o1', statement='{}'),
                 Evidence(source_id='MedicationRequest/m1', statement='{}')])
    question = dict(instruction='解释血压', task_type='诊断与解释', reference_answer='需评估',
                    evidence=[dict(source_id='Observation/o1', statement='记录存在')],
                    decision_point=None)
    async def generate(trajectory, environment, pool, runtime):
        assert len(pool) == 5 and len({d.task_type for d in pool}) == 5
        return RunResult(status=status,
            output={'questions': [question] if produced else []} if status == 'completed' else None,
            error='generate_1: failed' if status == 'failed' else None,
            trace=[{'type': 'selection', 'selected_directions': [0, 2]}],
            usage={'model_turns': 4, 'tool_calls': 0, 'input_tokens': 40,
                   'output_tokens': 20, 'total_tokens': 60})
    async def evaluate(task, trajectory, environment, evaluation_config):
        assert task.patient_id == trajectory.patient_id == 'p'
        assert task.environment == trajectory.environment == environment.name
        assert task.source_method == 'workflow'
        return EvaluationResult(task_id=task.id, observed_state={}, status='invalid', runs=[])
    evaluate_mock = AsyncMock(side_effect=evaluate)
    release_mock = AsyncMock()
    monkeypatch.setattr(runner, 'resolve_generator', lambda method: generate)
    monkeypatch.setattr(runner, 'create_environment', AsyncMock(return_value=EnvironmentInstance(
        instance_id='cache', endpoint='http://unused', environment=config.environment.name)))
    monkeypatch.setattr(runner, 'prepare_trajectory', AsyncMock(return_value=trajectory))
    monkeypatch.setattr(runner, 'release_environment', release_mock)
    monkeypatch.setattr(runner, 'evaluate_task', evaluate_mock)
    for name in ('explore_data', 'explore_environment', 'analyze_feedback', 'update_strategy'):
        monkeypatch.setattr(runner, name, AsyncMock(side_effect=AssertionError('unexpected exploration')))
    asyncio.run(runner.run_experiment(config))
    assert evaluate_mock.await_count == produced
    assert release_mock.await_count == 1
    attempts = json.loads((tmp_path / 'evaluation/attempts.json').read_text())
    assert attempts[0]['requested'] == 2 and attempts[0]['produced'] == produced
    assert len(attempts[0]['directions']) == 5 and attempts[0]['status'] == status
    summary = summarize_run(str(tmp_path))
    assert summary['costs']['generation']['model_turns'] == 4
    evaluations = json.loads((tmp_path / 'evaluation/evaluations.json').read_text())
    assert len(evaluations) == produced
    if produced:
        assert evaluations[0]['status'] == 'invalid'
