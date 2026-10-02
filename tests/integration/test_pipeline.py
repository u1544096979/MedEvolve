"""Cross-module experiment with only external HTTP services replaced."""

import json
from collections import Counter
from pathlib import Path

import httpx
import pytest
from openai import AsyncOpenAI

from contracts import ExperimentConfig
from experiments import run_experiment, summarize_run
from runtime import execution


@pytest.mark.asyncio
@pytest.mark.parametrize("instructions_role", ["system", "user"])
async def test_actual_modules_two_rounds(tmp_path, monkeypatch, instructions_role):
    config = json.loads((Path(__file__).parents[2] / 'configs/experiment.json').read_text())
    config.update(run_id='pipeline', output_dir=str(tmp_path / 'pipeline'),
                  evolution_patients=['e1', 'e2'], evaluation_patients=['h1'],
                  round_patients=[['e1', 'e2'], ['e1', 'e2']])
    config['environment'] = {'name': 'PhysicianBench', 'manager_url': 'http://manager.test',
                             'seed_data': 'local-test-image'}
    for runtime in (config['runtime'], config['evaluation']['runtime']):
        for role, model in runtime['models'].items():
            model.update(model=role, base_url='http://model.test/v1', api_key_env='PIPELINE_TEST_KEY',
                         instructions_role=instructions_role)
    monkeypatch.setenv('PIPELINE_TEST_KEY', 'local-test-key')
    model_calls, queries = Counter(), Counter()
    active, created = set(), []
    generated_strategies, completions = [], []

    def patient(identifier):
        return {'resourceType': 'Patient', 'id': identifier, 'gender': 'female'}

    def bundle(resources):
        return {'resourceType': 'Bundle', 'total': len(resources),
                'entry': [{'resource': r} for r in resources]}

    def chat(request):
        body = json.loads(request.content)
        role = body['model']
        model_calls[role] += 1
        messages = body['messages']
        user = next(m['content'] for m in reversed(messages) if m['role'] == 'user')
        tools = [m for m in messages if m['role'] == 'tool']
        call = None
        if role == 'data_explorer':
            if not tools:
                call = ('list_patients', {})
            elif len(tools) < 3:
                call = ('read_trajectory', {'patient_id': ['e1', 'e2'][len(tools)-1]})
            output = {'guidance': '按病程关联记录'}
        elif role == 'environment_explorer':
            if not tools:
                call = ('fhir_patient_search_demographics', {})
            output = {'guidance': '可查询患者信息'}
        elif role == 'verifier':
            output = {'valid': True, 'reason': '原始数据支持'}
        elif role == 'solver':
            payload = json.loads(user)
            assert set(payload) == {'instruction', 'patient_id', 'decision_point'}
            if not tools:
                call = ('fhir_patient_search_demographics', {})
            output = {'answer': 'female'}
        elif role == 'generator':
            payload = json.loads(user)
            pid = payload['trajectory']['patient_id']
            generated_strategies.append((pid, payload['strategy']['generation_guidance']))
            output = {'questions': [dict(instruction='查询患者性别', task_type=d['task_type'],
                                        reference_answer='female', decision_point=None,
                                        evidence=[dict(source_id=f'Patient/{pid}', statement='gender=female')])
                                    for d in payload['directions']]}
        elif role == 'quality':
            output = {'dimensions': {d: {'judgment': 'supported', 'reason': '原始数据支持'}
                                     for d in ['clinical_value', 'correctness', 'longitudinal_integration', 'clarity']}}
        elif role == 'completion':
            payload = json.loads(user)
            completions.append(payload)
            assert any(item['type'] == 'tool_output' for item in payload['solver_trace'])
            output = {'completed': True, 'reason': '查询结果正确'}
        elif role == 'feedback':
            payload = json.loads(user)
            assert all(t['patient_id'] != 'h1' for t in payload['tasks'])
            output = dict(quality_findings=[], difficulty_findings=[], failure_findings=[], suggestions=['整合更多记录'])
        elif role == 'updater':
            payload = json.loads(user)
            output = {**payload['strategy'], 'generation_guidance': f"strategy-{model_calls[role]}"}
        else:
            raise AssertionError(f'Unexpected model role: {role}')
        message = {'role': 'assistant', 'content': json.dumps(output, ensure_ascii=False)}
        if call:
            message = {'role': 'assistant', 'content': None, 'tool_calls': [
                {'id': f'{role}-{model_calls[role]}', 'type': 'function',
                 'function': {'name': call[0], 'arguments': json.dumps(call[1])}}]}
        return httpx.Response(200, json={'id': f'{role}-{model_calls[role]}',
            'object': 'chat.completion', 'created': 0, 'model': role,
            'choices': [{'index': 0, 'message': message, 'finish_reason': 'tool_calls' if call else 'stop'}],
            'usage': {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15}})

    def handle(request):
        host, path = request.url.host, request.url.path
        if host == 'model.test':
            return chat(request)
        if host == 'manager.test':
            if request.method == 'POST':
                iid = f'i{len(created)}'
                created.append(iid)
                active.add(iid)
                return httpx.Response(201, json={'instance_id': iid, 'environment': 'PhysicianBench',
                                                'endpoint': f'http://{iid}.test/fhir/'})
            active.remove(path.rsplit('/', 1)[1])
            return httpx.Response(204)
        iid = host.split('.')[0]
        assert iid in active
        if path.endswith('/$everything'):
            pid = path.split('/')[-2]
            queries[f"{'trajectory' if iid == 'i0' else 'state'}:{pid}"] += 1
            return httpx.Response(200, json=bundle([patient(pid)]))
        if path == '/fhir/Patient':
            ids = request.url.params['_id'].split(',')
            queries['tool_queries'] += 1
            return httpx.Response(200, json=bundle([patient(pid) for pid in ids]))
        if path.startswith('/fhir/Patient/'):
            queries['evidence_reads'] += 1
            return httpx.Response(200, json=patient(path.rsplit('/', 1)[1]))
        raise AssertionError(f'Unexpected HTTP request: {request.method} {request.url}')

    transport = httpx.MockTransport(handle)
    real_client = httpx.AsyncClient
    class TestClient(real_client):
        def __init__(self, **kwargs):
            super().__init__(transport=transport, **kwargs)

    monkeypatch.setattr(httpx, 'AsyncClient', TestClient)
    monkeypatch.setattr(execution, '_client', lambda model: AsyncOpenAI(
        api_key='local-test-key', base_url=model.base_url, max_retries=0,
        http_client=TestClient()))
    await run_experiment(ExperimentConfig.model_validate(config))
    summary = summarize_run(config['output_dir'])
    assert summary['status'] == 'completed'
    assert summary['errors'] == []
    assert summary['evolution']['actual_questions'] == 8
    assert summary['evaluation']['actual_questions'] == 2
    assert summary['evaluation']['completion_among_reviewed']['rate'] == 1
    initial = config['initial_strategy']['generation_guidance']
    assert generated_strategies == [('e1', initial), ('e2', initial), ('e1', 'strategy-1'),
                                    ('e2', 'strategy-1'), ('h1', 'strategy-2')]
    assert model_calls['generator'] == 5
    assert model_calls['feedback'] == model_calls['updater'] == 2
    assert queries['trajectory:e1'] == queries['trajectory:e2'] == queries['trajectory:h1'] == 1
    assert len(created) == 22 and not active
    assert len(completions) == 10
