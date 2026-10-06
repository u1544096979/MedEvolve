# Workflow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 根据一个患者的轨迹，从五类任务中选择两类，分别生成题目，统一内部复验，并接入现有公共环境评测。

**Architecture:** 保留四参数 generate_tasks 入口及共享数据模型。定位、两个按题型分派的生成节点、统一内部检查顺序运行，共用 GenerationBudget；公共入口负责候选方向池、任务封装和 evaluate_task。公共约束显式支持单节点输出与五类候选池，其他方法保持原有双方向调用约定。

**Tech Stack:** Python >=3.10、Pydantic >=2.10,<3、pytest、pytest-asyncio；现有 runtime、environments、evaluation 与 experiments 模块。

## Global Constraints

- 方法定义和验收依据为同目录 SPEC.md；五类任务沿用 contracts.TaskType，不修改 contracts.py。
- 定位选择两个不同题型；关注点为临床主题文本，不建立关注点分类体系。
- 各生成节点最多一题，合计最多两题；证据不足允许零题或一题，decision_point 为 null。
- 内部调用统一使用 runtime.models['generator'] 和 runtime.budgets['generator']，全部调用及失败成本进入 trace、usage。
- 不回退、不重试规划、不跨患者学习；复用公共环境、公共评测和标准模型运行层。
- 实施前使用 using-git-worktrees 技能核对并复用现有 .worktrees/workflow；保留现有未提交文件，公共代码同步仅包含本方案所需内容。
- 提交按关注点分层；Conventional Commits 的 description 和 body 使用中文。

## 文件与接口

| 文件 | 职责 |
| --- | --- |
| methods/common.py | 共享输出约束、候选池与单节点校验、累计预算和成本 |
| methods/workflow/__init__.py | 定位输出校验、顺序节点调用、证据映射、复验及结果汇总 |
| methods/workflow/prompts.py | 定位提示词、五类生成节点提示词、内部检查提示词 |
| experiments/runner.py | Workflow 候选池分派、公共校验及公共评测调用 |
| experiments/test_common.py | 公共约束和方法分派测试 |
| tests/method/test_workflow.py | 选型、节点依赖、复验、故障与累计预算测试 |
| tests/experiment/test_workflow.py | Workflow 经公共入口进入公共评测的集成测试 |

公共方法接口保持：

```python
async def generate_tasks(trajectory: PatientTrajectory,
                         environment: EnvironmentConfig,
                         directions: list[GenerationDirection],
                         runtime: RuntimeConfig) -> RunResult: ...
```

Workflow 接收五类候选方向，其他方法接收原有两个目标方向。公共评测接口保持 evaluate_task(task, trajectory, environment, config)；生成节点不调用评测，也不创建环境。

## Task 1: 公共约束与 worktree 依赖

**Files:** Modify methods/common.py、experiments/test_common.py；检查 worktree 内该公共模块是否已跟踪。

**Interfaces:** 消费 GenerationDirection、RunResult、TaskBatch；提供下列兼容接口，默认行为仍为两位置约束：

```python
def output_constraints(directions: list[GenerationDirection], *,
                       max_questions: int = 2, candidate_pool: bool = False) -> str: ...

def validate_task_batch(result: RunResult, directions: list[GenerationDirection], *,
                        max_questions: int = 2, candidate_pool: bool = False) -> RunResult: ...
```

- [ ] **Step 1:** 在 experiments/test_common.py 增加单节点和五类候选池的失败测试，复用现有 result、question、directions 辅助函数：

```python
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
```

- [ ] **Step 2:** 运行 `.venv/bin/python -m pytest experiments/test_common.py::test_single_node_and_candidate_pool_limits -q`；预期新增参数尚未实现而失败。
- [ ] **Step 3:** 增加私有方向校验函数并由两个公共函数复用。候选池要求恰好五类且各一次；普通模式要求方向数量等于 max_questions，限定 max_questions 为 1 或 2。数量约束使用 max_questions，题型数量继续使用 Counter 验证；原有字段约束、失败输出清空及成本保留逻辑不变。

```python
from typing import get_args
from contracts import TaskType

def _validate_directions(directions, *, max_questions, candidate_pool):
    if max_questions not in (1, 2):
        raise ValueError('max_questions must be one or two')
    types = [d.task_type for d in directions]
    if candidate_pool:
        if len(types) != 5 or set(types) != set(get_args(TaskType)):
            raise ValueError('candidate pool must contain each of the five task types once')
    elif len(directions) != max_questions:
        raise ValueError(f'directions must contain exactly {max_questions} generation positions')
```

提示词中的“最多两题”和方向数量改由 max_questions 参数表达；候选池校验的题型次数上限为每类一次。GenerationBudget.finish 保持原接口，方法最终传入定位后的两个方向。

- [ ] **Step 4:** 运行新增测试及 `test_constraints_keep_failed_cost_and_reject_extra_direction_or_time`、`test_cumulative_budget_keeps_failed_stages_and_blocks_next_call`；预期通过。
- [ ] **Step 5:** 提交公共模块及对应测试：`feat(common): 支持候选题型池与单节点输出约束`。worktree 的 methods/common.py 必须进入版本控制，不能依赖未跟踪副本。

## Task 2: 五类任务定位与来源校验

**Files:** Modify methods/workflow/__init__.py、methods/workflow/prompts.py、tests/method/test_workflow.py。

**Interfaces:** 临床关注点仍使用 ClinicalFocus；direction_index 范围扩大为 0..4。新增独立校验函数，旧入口在 Task 3 切换：

```python
def _selected_focuses(selection: FocusSelection,
                      directions: list[GenerationDirection],
                      trajectory: PatientTrajectory) -> list[ClinicalFocus]: ...
```

- [ ] **Step 1:** 新增 directions_pool 测试辅助函数，按 Task 1 的五类顺序构造方向。为 _selected_focuses 增加测试：选择下标 0 和 3 应保留顺序；重复下标、三条选择、缺失来源、空主题均应拒绝。

```python
def test_selected_focuses_match_pool_and_reject_duplicates():
    pool = directions_pool()
    selection = workflow.FocusSelection(focuses=[
        dict(direction_index=0, clinical_focus='血压控制', source_ids=['Observation/o1']),
        dict(direction_index=3, clinical_focus='降压用药', source_ids=['MedicationRequest/m1']),
    ])
    assert [f.direction_index for f in workflow._selected_focuses(selection, pool, patient())] == [0, 3]
    duplicate = selection.model_copy(update={'focuses': [selection.focuses[0]] * 2})
    with pytest.raises(ValueError):
        workflow._selected_focuses(duplicate, pool, patient())
```

其他拒绝用例使用 selection.model_copy 替换 focuses；缺失来源设为 Observation/missing，空主题设为空白，三条选择追加第二条。

- [ ] **Step 2:** 运行新增定位测试，确认校验函数未实现时失败。
- [ ] **Step 3:** 实现校验并修改 LOCATE 提示词，要求按轨迹适配程度从五类中选两类；weight 不参与随机采样。

```python
def _selected_focuses(selection, directions, trajectory):
    output_constraints(directions, candidate_pool=True)
    focuses = selection.focuses
    if len(focuses) != 2 or len({f.direction_index for f in focuses}) != 2:
        raise ValueError('locate must select two distinct task types')
    known = {s.source_id for s in trajectory.sources}
    for focus in focuses:
        if not 0 <= focus.direction_index < len(directions):
            raise ValueError('focus index is outside candidate pool')
        if not focus.clinical_focus.strip() or not focus.source_ids:
            raise ValueError('focus requires a topic and evidence sources')
        if set(focus.source_ids) - known:
            raise ValueError('focus sources are absent from trajectory')
    return focuses
```

- [ ] **Step 4:** 运行全部新增定位用例及现有方法测试；此任务保持旧入口可运行，新的选型函数由 Task 3 接入。
- [ ] **Step 5:** 提交：`feat(workflow): 增加五类任务定位与来源校验`。

## Task 3: 分节点生成、统一复验与公共分派

**Files:** Modify methods/workflow/__init__.py、methods/workflow/prompts.py、experiments/runner.py、tests/method/test_workflow.py、experiments/test_common.py。

**Interfaces:** generate_tasks 使用五类候选池；trace 中定位阶段保留选型，两个生成阶段分别命名 generate_0、generate_1，最终生成成本包含四次模型调用。

- [ ] **Step 1:** 将现有方法测试输入改为 directions_pool。成功替身结果改为定位、第一节点单题、第二节点单题、统一复验四份。断言调用顺序、各节点只收到自己的方向和关注点、合并候选顺序、最终筛选结果和累计 model_turns=4。新增第二节点返回两题或错题型时立即失败的用例，复验不得执行。
- [ ] **Step 2:** 运行方法测试，确认旧入口因五类输入或调用数量而失败。
- [ ] **Step 3:** prompts.py 提供 GENERATE_BY_TYPE 映射，五个键与 contracts.TaskType 一致；各节点共同遵守单题、事实引用、null 决策点要求。类型特定要求为：诊断与解释给出判断及依据；检查与风险分层给出检查建议及风险判断依据；治疗规划给出治疗选择及依据；药物处方给出轨迹可支持的药物和剂量要求；治疗重规划围绕疗效、病情变化或不良反应调整方案。不得要求轨迹未提供的事实或环境未开放的操作。

将 generate_tasks 的单次生成替换为顺序循环：

```python
focuses = _selected_focuses(FocusSelection.model_validate(located.output), directions, trajectory)
selected = [directions[f.direction_index] for f in focuses]
candidates = []
for index, (focus, direction) in enumerate(zip(focuses, selected)):
    stage = f'generate_{index}'
    generated = await budget.call_llm(
        stage,
        instructions=GENERATE_BY_TYPE[direction.task_type] + '\n' + output_constraints([direction], max_questions=1),
        input_text=json.dumps({'trajectory': _trajectory_input(trajectory),
                               'focus': focus.model_dump(mode='json'),
                               'direction': direction.model_dump(mode='json')}, ensure_ascii=False),
        output_type=TaskBatch,
    )
    validated = validate_task_batch(generated, [direction], max_questions=1)
    if validated.status != 'completed':
        return budget.finish(selected, status=validated.status, error=f'{stage}: {validated.error}')
    batch = TaskBatch.model_validate(validated.output)
    known = {source.source_id for source in trajectory.sources}
    if any(e.source_id not in known for q in batch.questions for e in q.evidence):
        return budget.finish(selected, status='failed', error=f'{stage}: unknown evidence source')
    candidates.extend(batch.questions)
```

入口首先调用 output_constraints(directions, candidate_pool=True)。定位失败用显式 RunResult 或 summarize_results 汇总并清空输出，不能把五类池传入默认双位置 finish；定位成功后的所有 finish 使用 selected。候选为空时完成返回；否则统一复验。

替换文本反解析为已有来源映射，并检查判定数量再建立字典：

```python
records = {source.source_id: source.statement for source in trajectory.sources}
if len(review.reviews) != len(questions):
    raise ValueError('internal check must judge every candidate exactly once')
by_index = {item.index: item for item in review.reviews}
if len(by_index) != len(review.reviews) or set(by_index) != set(range(len(questions))):
    raise ValueError('internal check must judge every candidate exactly once')
```

删除正则解析及 re 导入；保持内部检查只筛选不改写。定位 trace 中增加 selected_directions 及 focuses，运行原始记录仍完整保留。

公共入口在该任务一起切换，防止新旧接口错配：

```python
directions = (list(allocation) if config.method == 'workflow' else
              sampling.choices(allocation, weights=[d.weight for d in allocation], k=2))
# generator 调用之后：
result = validate_task_batch(result, directions, candidate_pool=config.method == 'workflow')
```

运行前对 Workflow allocation 调用候选池校验；输出任务数目标仍为 2，attempts 保存五类候选方向与运行成本，具体选型由生成 trace 保存。移除 test_unimplemented_method_is_explicit 中的 workflow 参数，增加 resolve_generator('workflow') 指向 workflow.generate_tasks 的正向测试。

- [ ] **Step 4:** 更新故障用例覆盖定位、两个生成节点与复验；预算为 3 时不得调用复验，预算为 4 时允许完整流程；两个节点都为空时调用数为 3。验证重复判定、缺失判定、空白字段、未知来源均不能被筛选掩盖。运行 `.venv/bin/python -m pytest tests/method/test_workflow.py experiments/test_common.py -q`，预期全部通过。
- [ ] **Step 5:** 提交：`feat(workflow): 接入双节点生成与统一复验`。该提交同时包含方法入口、公共分派及测试，避免接口切换拆成不兼容提交。

## Task 4: 当前公共环境评测接入验证

**Files:** Create tests/experiment/test_workflow.py；公共环境和评测实现仅同步当前版本，不新增 Workflow 专用实现。

**Interfaces:** 消费 runner.run_experiment(config) 和 workflow.generate_tasks；输出公共 Task、EvaluationResult、attempts、summary，复用 evaluate_task 四参数接口。

- [ ] **Step 1:** 新测试读取 configs/physicianbench-comparison.json，复制配置为 method='workflow'、evaluation_patients=['p']、evolution_patients=[]、round_patients=[]、output_dir=tmp_path。替身 create_environment 返回与配置同名的 EnvironmentInstance，prepare_trajectory 返回含两个来源的患者轨迹，release_environment 为 AsyncMock。将四个 MedEvolve 探索和更新入口设置为抛 AssertionError，证明没有调用。
- [ ] **Step 2:** 替身 Workflow 入口断言收到完整五类池，返回一题、四次调用成本和选型 trace。公共评测替身检查封装后的患者、环境、source_method 及题型，返回真实可用的 EvaluationResult 结构：

```python
async def evaluate(task, trajectory, environment, config):
    assert task.patient_id == trajectory.patient_id == 'p'
    assert task.environment == trajectory.environment == environment.name
    assert task.source_method == 'workflow'
    return EvaluationResult(task_id=task.id, observed_state={}, status='invalid', runs=[])
```

新增集成测试正文如下，覆盖一题、零题和生成故障：

```python
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
        text='[undated] Observation/o1\n{}',
        sources=[Evidence(source_id='Observation/o1', statement='{}')])
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
```

- [ ] **Step 3:** 运行 `.venv/bin/python -m pytest tests/experiment/test_workflow.py tests/experiment/test_closed_loop.py tests/evaluation/test_evaluator.py -q`。修复本方案引入的接口问题；不改变公共评测提示词、判定规则或环境能力。
- [ ] **Step 4:** 核对 worktree 的 runtime、environments、evaluation、contracts、公共 runner 与当前接口兼容。只同步必要差异，保留主项目未提交改动；环境故障、模型故障与证据拒绝按现有 EvaluationResult 表达。
- [ ] **Step 5:** 提交接入验证：`test(workflow): 验证公共实验与评测接入`。若确有依赖同步改动，按公共依赖关注点单独提交，description 使用中文。

## Task 5: 单患者验收与集成

**Files:** 运行结果保存在 outputs/workflow-validation-<run-id>/，不纳入源码提交；仅在现有配置不满足五类池时修改对应配置。

**Interfaces:** 消费公共配置和 experiments.run 命令；交付完整生成记录、公共评测结果和运行汇总。

- [ ] **Step 1:** 运行 `.venv/bin/python -m pytest tests/method/test_workflow.py experiments/test_common.py tests/experiment/test_workflow.py tests/experiment/test_closed_loop.py tests/evaluation/test_evaluator.py -q`；预期全部通过。每项新增测试对应实际行为，不测试提示词字符串或重复字段转发。
- [ ] **Step 2:** 核对 configs/physicianbench-comparison.json 的五类 allocation、单个 evaluation_patient 和生成预算至少 4；实际模型使用已授权的 gpt-4o-mini。通过环境变量提供密钥，不写入计划、配置或运行说明。
- [ ] **Step 3:** 从实施 worktree 运行公共入口，使用唯一 run-id 与新目录：

```bash
.venv/bin/python -m experiments.run --config configs/physicianbench-comparison.json --method workflow --run-id workflow-validation-20261006-01 --output-dir outputs/workflow-validation-20261006-01
```

同名运行已存在时递增末尾编号，不覆盖目录。worktree 未配置运行环境时复用或建立项目既有依赖环境，不依赖主目录的隐式导入路径。

- [ ] **Step 4:** 检查选型来源于五类候选池；两次生成各对应一个选定题型；引用复验逐题覆盖且只筛选；运行记录包含全部调用和 token。核对环境实例释放、公共评测实际执行，以及其通过、拒绝或故障结果与原始记录一致。公共拒绝不要求改成通过。
- [ ] **Step 5:** 使用 finishing-a-development-branch 技能交付集成选项与验证结果；按用户选择合并回项目当前分支。合并前保留当前分支未提交工作，按提交粒度处理冲突，不覆盖其他方法的改动。

## 验收标准

1. Workflow 收到五类候选任务，定位选择两个不同题型及可回查来源。
2. 两个节点各生成最多一题，错题型、超额题、未知来源和无效字段在复验前拒绝。
3. 复验覆盖候选恰好一次，重复或缺失判定返回阶段故障；通过题保持原内容及顺序。
4. 完整非空流程累计四次模型调用；阶段故障和预算耗尽不隐去成本或伪造题目。
5. 方法经当前公共入口进入公共环境评测；独立检出的 worktree 不依赖未跟踪公共模块。
