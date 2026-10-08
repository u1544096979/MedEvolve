"""官方任务隔离、原生操作与评分边界。"""
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from contracts import EnvironmentConfig, EnvironmentInstance, RunResult
from harnesses.benchmarks.assets import BenchmarkUnavailable, REFSOL_URL, digest, load_cases
from harnesses.benchmarks.scoring import parse_checkpoints, score_case, write_trajectory
from harnesses.benchmarks.tools import BenchmarkNativeTools, BenchmarkSessions
from harnesses.benchmarks.worker import medagent_score
from harnesses.contracts import BenchmarkCase
from experiments.harness_benchmark import summarize_benchmark

REAL_ASYNC_CLIENT = httpx.AsyncClient


def asset_repo(tmp_path, monkeypatch):
    monkeypatch.setattr('harnesses.benchmarks.assets.subprocess.run',
        lambda command, **kwargs: SimpleNamespace(stdout='locked\n' if 'rev-parse' in command else ''))
    root = tmp_path / 'official'
    directory = root / 'src/server/tasks/medagentbench'
    directory.mkdir(parents=True)
    (directory / 'utils.py').write_text('')
    (directory / 'eval.py').write_text('from .refsol import task1\ndef eval(case_data, results, fhir_api_base):\n return task1(case_data,results,fhir_api_base)\n')
    (directory / 'refsol.py').write_text('import json\ndef task1(case_data,results,url):\n return json.loads(results.result)==case_data["sol"]\n')
    data = root / 'data/medagentbench'
    data.mkdir(parents=True)
    (data / 'funcs_v1.json').write_text('[]')
    (data / 'test_data_v2.json').write_text(json.dumps([{
        'id':'task1_1','instruction':'Find the patient named Alice','context':'Search demographics',
        'sol':['S1234567'],'eval_MRN':'S1234567'}]))
    settings = {'source_dir':str(root),'commit':'locked','repository':'official',
        'dataset_path':'data/medagentbench/test_data_v2.json',
        'refsol_source':REFSOL_URL,'refsol_sha256':digest(directory / 'refsol.py')}
    return settings, directory


def test_official_cases_keep_answers_out_of_solver_input(tmp_path, monkeypatch):
    settings, _ = asset_repo(tmp_path, monkeypatch)
    cases, manifest = load_cases('MedAgentBench', settings)
    assert cases[0].solver_input().model_dump() == {
        'instruction':'Find the patient named Alice\n\nSearch demographics',
        'patient_id':'','decision_point':None}
    assert cases[0].reference['sol'] == ['S1234567']
    assert manifest['task_ids'] == ['task1_1']


def test_official_assets_missing_or_modified_are_unavailable(tmp_path, monkeypatch):
    settings, directory = asset_repo(tmp_path, monkeypatch)
    (directory / 'refsol.py').write_text('changed')
    with pytest.raises(BenchmarkUnavailable, match='摘要不匹配'):
        load_cases('MedAgentBench', settings)
    (directory / 'refsol.py').unlink()
    with pytest.raises(BenchmarkUnavailable, match='benchmark_unavailable'):
        load_cases('MedAgentBench', settings)


@pytest.mark.asyncio
async def test_native_patient_search_does_not_inject_hidden_patient(monkeypatch):
    urls = []
    def get(request):
        urls.append(str(request.url))
        return httpx.Response(200, text='{"entry":[]}')
    monkeypatch.setattr(httpx,'AsyncClient',lambda **kwargs: REAL_ASYNC_CLIENT(transport=httpx.MockTransport(get), **kwargs))
    tools = BenchmarkNativeTools('http://instance/fhir', [])
    result = await tools.execute('GET Patient?name=Alice')
    assert result['kind'] == 'GET'
    assert urls == ['http://instance/fhir/Patient?name=Alice&_format=json']
    assert (await tools.execute('GET https://elsewhere/Patient'))['kind'] == 'error'


@pytest.mark.asyncio
async def test_native_post_is_accepted_without_network_and_finish_preserves_list(monkeypatch):
    post = AsyncMock(side_effect=AssertionError('POST must not write'))
    monkeypatch.setattr(httpx,'AsyncClient',post)
    tools = BenchmarkNativeTools('http://instance/fhir', [])
    result = await tools.execute('POST Observation\n{"valueString":"118/77 mmHg"}')
    assert result['request']['payload'] == {'valueString':'118/77 mmHg'}
    assert 'accepted' in result['reply']
    post.assert_not_called()
    assert (await tools.execute('FINISH([12,"2023-11-13"])'))['answers'] == [12,'2023-11-13']
    assert (await tools.execute('FINISH({})'))['kind'] == 'error'


def test_original_medagent_grader_uses_finish_and_accepted_posts(tmp_path, monkeypatch):
    _, directory = asset_repo(tmp_path, monkeypatch)
    payload = {'grader_directory':str(directory), 'case_data':{'sol':['S1234567']},
               'endpoint':'http://instance/fhir', 'calls':[]}
    assert medagent_score(payload)['completed'] is False
    payload['calls'] = [{'arguments':{'action':'FINISH(["S1234567"])'},
                         'output':{'kind':'FINISH','answers':['S1234567'],'reply':'FINISH'}}]
    assert medagent_score(payload)['completed'] is True
    payload['calls'][0]['output']['answers'] = ['wrong']
    assert medagent_score(payload)['completed'] is False


def test_pytest_scoring_distinguishes_clinical_failure_and_fault(tmp_path):
    report = tmp_path / 'report.xml'
    report.write_text('<testsuite><testcase name="cp1"/><testcase name="cp2"><failure message="AssertionError: missing order"/></testcase></testsuite>')
    score = parse_checkpoints(report, 1)
    assert score['completed'] is False
    assert score['checkpoint_score'] == .5
    report.write_text('<testsuite><testcase name="cp1"><error message="ImportError"/></testcase></testsuite>')
    assert parse_checkpoints(report, 1)['status'] == 'scoring_error'
    assert parse_checkpoints(report, 5)['completed'] is None
    report.write_text('<testsuite><testcase name="cp1"/></testsuite>')
    errors = tmp_path / 'judge_errors.json'; errors.write_text('["RateLimitError"]')
    assert parse_checkpoints(report, 0, errors)['status'] == 'scoring_error'


def test_trajectory_matches_official_metadata_schema(tmp_path):
    calls = [{'name':'fhir_patient_search_demographics','arguments':{'patient':'p1'},
              'output':{'entry':[{'resource':{'id':'p1'}}]}}]
    run = RunResult(status='completed', output={'answer':'done'},trace=[],usage={})
    write_trajectory(tmp_path,calls,run)
    event = json.loads((tmp_path/'logs/agent/trajectory.log').read_text().splitlines()[0])
    assert event['type'] == 'tool_call'
    assert json.loads(event['metadata']['output']) == calls[0]['output']


@pytest.mark.asyncio
async def test_file_outputs_and_instance_are_retained_until_explicit_close(tmp_path):
    instance = EnvironmentInstance(instance_id='instance',environment='PhysicianBench',endpoint='http://instance/fhir')
    environments = SimpleNamespace(create_environment=AsyncMock(return_value=instance),
        get_tools=AsyncMock(return_value=[]), release_environment=AsyncMock())
    sessions = BenchmarkSessions({}, tmp_path, environments=environments)
    task = BenchmarkCase(id='test', environment='PhysicianBench', instruction='Do task', patient_id='p1', asset_path=tmp_path)
    sid = await sessions.open(task.solver_input(), EnvironmentConfig(name='PhysicianBench',manager_url='http://manager',seed_data='seed'),client_id='runner')
    result = await sessions.invoke(sid,'write_file',{'file_path':'/workspace/output/note.md','content':'assessment','mode':'w'},client_id='runner')
    assert result['status'] == 'ok'
    assert (tmp_path/'workspace/output/note.md').read_text() == 'assessment'
    environments.release_environment.assert_not_called()
    escaped = await sessions.invoke(sid,'write_file',{'file_path':'/workspace/output/../../secret','content':'bad','mode':'w'},client_id='runner')
    assert escaped['status'] == 'error'
    await sessions.close(sid,client_id='runner')
    environments.release_environment.assert_awaited_once()


def test_summary_keeps_full_denominator_and_pairs_only_valid_cases():
    result = {'environment':'MedAgentBench','harness':'ace','source':'medevolve','seed':42,
        'planned_denominator':2,'assets':{},'runs':[],'evaluation_config':{},'snapshots_unchanged':True,
        'records':[{'state':'initial','task_id':'a','patient_id':'p1','status':'scored','completed':False},
                   {'state':'final','task_id':'a','patient_id':'p1','status':'scored','completed':True},
                   {'state':'initial','task_id':'b','patient_id':'p2','status':'scored','completed':True},
                   {'state':'final','task_id':'b','patient_id':'p2','status':'scoring_error','completed':None}]}
    summary = summarize_benchmark(result)
    assert summary['states']['final']['success_rate'] == .5
    assert summary['states']['final']['valid_denominator'] == 1
    assert summary['paired_tasks'] == 1
    assert summary['paired_lift_percentage_points'] == 100
    assert summary['comparison_valid'] is False


@pytest.mark.asyncio
@pytest.mark.parametrize('fail_execution',[False,True])
async def test_benchmark_runner_closes_each_attempt_and_freezes_experience(tmp_path,monkeypatch,fail_execution):
    from contextlib import asynccontextmanager
    from experiments import harness_benchmark as module
    initial, final = tmp_path/'initial', tmp_path/'final'
    initial.mkdir(); final.mkdir()
    (initial/'memory.md').write_text('initial'); (final/'memory.md').write_text('trained')
    case = BenchmarkCase(id='case',environment='PhysicianBench',instruction='Review MRN123',patient_id='MRN123',asset_path=tmp_path)
    config = {'run_id':'test','run_dir_template':str(tmp_path/'group'), 'models':{},
        'limits':{'execution':{'max_tool_calls':5},'timeout_seconds':5},
        'environments':{'PhysicianBench':{'name':'PhysicianBench','manager_url':'http://manager','seed_data':'seed'}}}
    monkeypatch.setattr(module,'check_group',lambda *args:{'ok':True})
    monkeypatch.setattr(module,'require_execution',lambda *args:None)
    monkeypatch.setattr(module,'benchmark_settings',lambda *args:{})
    monkeypatch.setattr(module,'load_cases',lambda *args:([case],{}))
    monkeypatch.setattr(module,'snapshots_for',lambda *args:{'initial':initial,'final':final})
    released, started, closed = [], [], []
    async def create_environment(cfg):
        return EnvironmentInstance(instance_id=str(len(started)),environment='PhysicianBench',endpoint='http://instance/fhir')
    async def release_environment(cfg,instance):
        released.append(instance.instance_id)
    envs = SimpleNamespace(create_environment=create_environment,get_tools=AsyncMock(return_value=[]),release_environment=release_environment)
    live_sessions=[]
    def sessions(settings,job):
        value=BenchmarkSessions(settings,job,environments=envs);live_sessions.append(value);return value
    monkeypatch.setattr(module,'BenchmarkSessions',sessions)
    @asynccontextmanager
    async def bridge(sessions):
        yield 'http://bridge'
    monkeypatch.setattr(module,'bridge_server',bridge)
    class Runner:
        def __init__(self,config,group_dir):
            self.path=group_dir
            assert config['frozen'] is True
        async def start(self,group,snapshot):
            started.append(snapshot.path)
        async def execute(self,task,sid):
            assert task.instruction == case.instruction
            if fail_execution:
                raise RuntimeError('execution failed')
            return RunResult(status='completed',output={'answer':'done'},trace=[],usage={})
        async def close(self):
            assert self.path.exists()
            closed.append(self.path)
    monkeypatch.setattr(module,'create_runner',lambda name,config,group_dir,bridge_url:Runner(config,group_dir))
    async def score(*args):
        assert len(released) == len(started)-1
        return {'status':'scored','completed':True,'checkpoint_score':1}
    monkeypatch.setattr(module,'score_case',score)
    result = await module.run_benchmark(config,'PhysicianBench','ace','medevolve',42)
    assert started == [initial,final]
    assert len(closed) == len(released) == 2
    assert result['snapshots_unchanged'] is True
    assert {row['status'] for row in result['records']} == ({'execution_error'} if fail_execution else {'scored'})
    assert all(session.closed for sessions in live_sessions for session in sessions._sessions.values())


def test_scorer_keeps_virtualenv_interpreter_path(tmp_path):
    from experiments.harness_benchmark import benchmark_settings
    executable = tmp_path/'venv/bin/python'
    executable.parent.mkdir(parents=True)
    executable.symlink_to('/usr/bin/python3')
    config = {'environments':{'MedAgentBench':{'benchmark':{'source_dir':str(tmp_path),'python_path':str(executable)}}}}
    assert benchmark_settings(config,'MedAgentBench')['python_path'] == str(executable)


def test_missing_official_scoring_dependencies_fail_preflight(monkeypatch):
    from harnesses.benchmarks.scoring import check_scorer
    monkeypatch.setattr('harnesses.benchmarks.scoring.subprocess.run',lambda *args,**kwargs:SimpleNamespace(returncode=1))
    check=check_scorer('PhysicianBench',{'python_path':'missing-scoring-environment'})
    assert check['ok'] is False
    assert 'benchmark' in check['error']
