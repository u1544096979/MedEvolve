# 环境能力与患者并行实施计划

> 执行方式：使用 executing-plans 在独立 worktree 中按任务顺序实施；若用户选择子代理执行，则使用 subagent-driven-development。实施前读取 using-git-worktrees，保留项目当前分支所需的已完成改动。

**目标：** 同一个方法为多个患者并行生成，生成结束立即释放名额；所有最终 task 独立排队评测，生成与评测分别配置并发数，环境查询复用且真实写入隔离。

**架构：** 保留现有六方法入口、累计预算、公共评测和结果写入锁。增加生成/评测调度器，扩展现有 EvaluationEnvironments 覆盖完整运行的实例申请；统一能力来源提供执行要求与实例路由依据。

**技术栈：** Python >=3.10、asyncio、Pydantic 2、httpx、现有 Agents SDK、Docker、pytest 和 pytest-asyncio；不新增运行依赖。

## 全局约束

- 依据 [设计](design.md)，只调整同一次实验的患者生成、task 评测和环境复用，不调度多个方法同时运行，不改 Harness。
- 生成与评测分别控制；生成名额包含方法内部自检筛选，不包含公共评测；MedEvolve 本轮全部评测完成后才更新策略。
- 六个方法均接入同一能力来源并输出执行要求；不改变原有题型选择、内部阶段顺序、随机流、预算及内置筛选。
- 保持公共评测的逐条证据核验、整体核验、质量与难度各三次评分及中位数、独立解题和完成判断；不额外添加分类模型调用。
- 所有环境申请进入同一运行会话额度；共享查询禁止真实写入，PhysicianBench 试写探索和需要真实写入的 task 使用独立种子。
- MEDEVOLVE_MAX_INSTANCES 继续按逻辑实例计数；FHIR-AgentBench 一个实例含两个容器，验收区分实例数和实际容器数。
- 实施测试使用替代模型，不发起付费调用；真实环境检查只使用已准备镜像或种子。提交按关注点分层，description 与 body 使用中文。

## 基线与文件职责

main 已存在六个生成入口；`experiments/runner.py` 用一个患者名额包住生成和评测；`evaluation/session.py` 仅约束公共评测，TaskCraft 的生成和环境探索仍直接申请实例。SingleShot 的 task_types 与 Workflow 的 directions/candidate_pool 路径已在当前 `methods/common.py` 并存，沿用此接口，不重新设计分支合并。

| 文件 | 职责 |
| --- | --- |
| contracts.py、新增 environments/capabilities.py | 两个并发参数、固定能力、执行要求与核验字段 |
| methods/common.py、methods/generation.py、六个方法实现与提示 | 公共字段校验、能力注入、候选字段保留与方法特有生成边界 |
| evaluation/session.py、environments/__init__.py、environments/fhir.py、environments/errors.py | 全流程实例额度、共享借用、空闲回收与写入阻断 |
| 新增 experiments/scheduling.py、experiments/runner.py、configs/experiment.json | 生成/评测队列、独立上限、即时保存、轮次屏障与有效参数记录 |
| evaluation/evaluator.py、evaluation/prompts.py | 整体核验确定执行要求、解题路由及真实状态判定 |
| tests/method、tests/evaluation、tests/experiment、tests/environment | 六方法兼容、两类名额、共享隔离、失败清理及既有评测回归 |

源文件定位以本次代码核对为基线，执行时按对应函数重新定位，不能将已合入 main 的方法重新列为未实现。

## 六方法实施清单

| 方法 | 当前文件定位 | 必做适配及不变量 |
| --- | --- | --- |
| medevolve | methods/medevolve/__init__.py:122、:160、:186；methods/generation.py:5 | 环境探索使用运行会话；生成与反馈注入统一能力，生成执行要求；同轮指导和策略保持，轮次评测屏障保留 |
| workflow | methods/workflow/__init__.py:117；methods/workflow/prompts.py | 定位、按类型生成、内部检查接入能力；候选和最终输出保留执行要求；五类选两类、每类最多一题、累计预算保持 |
| selfgen | methods/selfgen/__init__.py:83、:94；methods/selfgen/prompts.py:8、:22 | 删除 operation_semantics 手写环境分支，由调用包装器统一注入能力；初稿/修订输出执行要求，过滤保留；通过题全字段不变，自检修订计入生成名额 |
| singleshot | methods/singleshot/__init__.py:24；methods/singleshot/prompts.py:13–14 | 删除两行环境语义副本，单次生成注入能力并输出执行要求；沿用 task_types 和公共 finish 校验，不引入探索、修订或方向抽样 |
| envexplore | methods/envexplore/__init__.py:21；methods/generation.py:5；experiments/runner.py 的固定指导绑定器 | 一次环境探索接入会话，生成注入同一能力、输出执行要求；删除 GENERATION_RULES 中提交/持久化语义副本，保留事实与证据规则；不增加数据探索或策略进化 |
| taskcraft | methods/taskcraft/__init__.py:55；ledger.py:16；stages.py 的 Pipeline、atomic/width/depth 候选构造；schemas.py；environment.py:56、:88 | 生成借用共享查询实例；Ledger 统一注入能力；各阶段输出并保留最终执行要求；每患者独立 Pipeline/Ledger/预算/随机流，只读工具视图和内部核验保持 |

TaskCraft 在研分支 `codex/taskcraft-baseline` 仍修改 `methods/taskcraft/stages.py`、`prompts.py` 与 `tests/method/test_taskcraft.py`，包括请求重试规则。若先合入在研改动，按合入后的结构适配；若先实施本方案，分支随后 rebase，逐项保留会话、能力、执行要求与原重试规则。不得以旧 stages.py 或测试文件覆盖任一侧行为，合入后重新跑该方法及公共并发回归。

## 任务 1：固定能力、执行要求与并发配置

**文件：** 修改 contracts.py；新增 environments/capabilities.py、tests/environment/test_capabilities.py；新增 tests/experiment/test_concurrency_config.py；修改 experiments/runner.py 的参数解析与配置保存。

**产出接口：**

```python
# contracts.py
ExecutionRequirement = Literal["read_only", "submit_request", "database_write"]

class EnvironmentCapabilities(BaseModel):
    name: str
    post_semantics: Literal["persistent", "accept_without_persistence", "unavailable"]
    supports_database_write: bool

# GeneratedQuestion；允许读取旧保存任务，生成校验在任务 2 要求非空。
execution_requirement: ExecutionRequirement | None = None

# Verification；单条事实核验可为空，整体核验在任务 5 要求填写。
execution_requirement: ExecutionRequirement | None = None
capability_supported: bool | None = None

# ExperimentConfig；缺省使用兼容回退，保存时填入有效值。
generation_concurrency: int | None = Field(default=None, ge=1)
evaluation_concurrency: int | None = Field(default=None, ge=1)

# environments/capabilities.py 的接口
# get_capabilities(name: str) -> EnvironmentCapabilities
# capability_instructions(name: str) -> str
# requirement_supported(name: str, requirement: ExecutionRequirement) -> bool
```

- [ ] **1. 添加配置优先级和能力测试。**

```python
# tests/experiment/test_concurrency_config.py
from contracts import ExperimentConfig
from experiments.runner import concurrency_limits

def test_config_overrides_legacy_limits(monkeypatch):
    monkeypatch.setenv("MEDEVOLVE_RUN_CONCURRENCY", "9")
    config = ExperimentConfig.model_construct(
        generation_concurrency=5, evaluation_concurrency=2)
    assert concurrency_limits(config) == (5, 2)

def test_missing_config_uses_legacy_limits(monkeypatch):
    monkeypatch.setenv("MEDEVOLVE_RUN_CONCURRENCY", "4")
    config = ExperimentConfig.model_construct(
        generation_concurrency=None, evaluation_concurrency=None)
    assert concurrency_limits(config) == (4, 4)
```

```python
# tests/environment/test_capabilities.py
from environments.capabilities import get_capabilities, requirement_supported

def test_medagentbench_submission_is_not_persistence():
    assert get_capabilities("MedAgentBench").post_semantics == "accept_without_persistence"
    assert requirement_supported("MedAgentBench", "submit_request")
    assert not requirement_supported("MedAgentBench", "database_write")

def test_fhir_agentbench_tools_are_read_only():
    assert requirement_supported("FHIR-AgentBench", "read_only")
    assert not requirement_supported("FHIR-AgentBench", "submit_request")
    assert not requirement_supported("FHIR-AgentBench", "database_write")
```

补充两个配置字段的 0/负数 Pydantic 拒绝、部分显式设置、环境变量缺失/非法沿用默认 6、旧 Task 仍能加载、未知环境明确失败。

- [ ] **2. 运行新增测试确认缺失接口时失败。**

```bash
python -m pytest tests/experiment/test_concurrency_config.py tests/environment/test_capabilities.py -q
```

- [ ] **3. 实现契约与解析。**

```python
# experiments/runner.py：沿用既有 run_concurrency 兼容解析。
def concurrency_limits(config):
    legacy = run_concurrency()
    generation = config.generation_concurrency
    evaluation = config.evaluation_concurrency
    return (generation if generation is not None else legacy,
            evaluation if evaluation is not None else legacy)
```

固定表中 PhysicianBench 是 persistent/可写，MedAgentBench 是 accept_without_persistence/不可真实写，FHIR-AgentBench 是 unavailable/不可写。get_capabilities 返回契约副本，未知环境抛 EnvironmentUnavailable。requirement_supported 对 read_only 返回真，对 submit_request 检查 POST 是否开放，对 database_write 检查真实写能力。说明文本完整列出语义和三种执行要求，不允许探索指导覆盖。

保存 config.json 前，将两个解析值填入本次 config 副本；不修改用户原配置文件。EvaluationResult 的失败分类增加 unsupported_operation，供任务 5 使用。

- [ ] **4. 运行契约和旧配置回归。**

```bash
python -m pytest tests/experiment/test_concurrency_config.py tests/environment/test_capabilities.py tests/experiment/test_closed_loop.py -q
```

- [ ] **5. 提交。** `feat(environment): 定义固定能力与独立并发参数`。

## 任务 2：六方法统一能力与生成字段

**文件：** 修改 methods/common.py、methods/generation.py、methods/{medevolve,workflow,selfgen,singleshot,envexplore}/__init__.py 与各自提示文件；修改 methods/taskcraft/{__init__,ledger,schemas,stages,prompts}.py；修改 tests/method 中六方法对应测试与 test_generation_shared.py。

**输入与产出：** 使用任务 1 的 capability_instructions 和 ExecutionRequirement；现有 generate_tasks 的位置参数保持。最终 TaskBatch 的每题包含执行要求，各方法私有候选输出保留该字段。

- [ ] **1. 添加能力输入、缺字段拒绝和筛选保留测试。**

```python
# 加入现有 tests/method/test_method.py，复用其 method/runtime/trajectory 夹具。
from unittest.mock import AsyncMock
from environments.capabilities import capability_instructions

@pytest.mark.asyncio
async def test_generation_receives_fixed_capability(method, runtime, monkeypatch):
    patient = trajectory("p1").model_copy(update={"environment": "MedAgentBench"})
    direction = GenerationDirection(task_type="诊断与解释", difficulty="查询", weight=1)
    strategy = Strategy(generation_guidance="当前指导", allocation=[direction])
    guidance = ExplorationGuidance(data_guidance="数据指导", environment_guidance="探索指导")
    call = AsyncMock(return_value=completed(TaskBatch(questions=[])))
    monkeypatch.setattr(method, "call_llm", call)
    await method.generate_tasks(patient, guidance, strategy, [direction, direction], runtime)
    assert capability_instructions("MedAgentBench") in call.call_args.kwargs["instructions"]
```

利用现有六方法的替代模型入口捕获指令：Workflow 三类节点、SelfGen 初稿/自检/修订、EnvExplore 与 SingleShot 单次生成、TaskCraft atomic/width/depth/内置核验均收到固定说明。新生成输出缺少执行要求时被公共校验拒绝；SelfGen 修订和通过题、Workflow 过滤、EnvExplore 引用过滤、TaskCraft 候选选择都保留该字段。预算和调用次数断言保持。

- [ ] **2. 运行对应方法测试，确认未注入能力的调用失败。**

```bash
python -m pytest tests/method -q
```

- [ ] **3. 逐方法接入统一来源。**

```python
# 每患者计算一次，在各自模型调用包装处加入，不另做模型调用。
fixed = capability_instructions(trajectory.environment)
instructions = original_instructions + "\n" + fixed

# methods/common.py：新生成批次的公共校验。
for question in batch.questions:
    if question.execution_requirement is None:
        raise ValueError("Generated task must specify execution_requirement")
```

按六方法清单删除重复语义，只保留临床事实、题型和方法专属规则。SelfGen 的环境白名单由 get_capabilities 接管，但不据此宣称其他方法工具适配完成；其 prompts._fact_rules 仍提取事实规则，不从统一说明中截取副本。

TaskCraft 由 Ledger 包装 llm 和 agent 指令，公开生成入口传入本环境的说明；不改变 _SearchRun 的规划、反思、动作与累计预算。InitialQuestion、CleanedQuestion、MergedQuestion、NewQuery、QueryMerge 相关已有模型输出携带执行要求，atomic/width/depth 的 GeneratedQuestion 构造传入该字段；重写题意时以新题要求为准，不能机械继承父题。通过已有调用输出字段取得要求，不增加分类请求，不按生成工具只读直接把任务标为 read_only。

保持 methods/common.py 当前 directions、task_types、max_questions 和 candidate_pool 接口；GenerationBudget.finish 仍委托公共校验。SingleShot 两题可同类型，Workflow 候选池每类最多一题，两者不互换选择规则。

- [ ] **4. 运行六方法与共享生成规则回归。**

```bash
python -m pytest tests/method tests/experiment/test_workflow.py -q
```

测试中的虚拟环境名改为对应已支持环境，真实输入仍用替代数据；生成题目夹具补齐执行要求，保留每患者独立预算和随机流断言。

- [ ] **5. 提交。** `feat(methods): 统一六方法能力说明与执行要求`。

## 任务 3：完整运行共享实例与统一额度

**文件：** 修改 evaluation/session.py、environments/{__init__,fhir,errors}.py、contracts.py、methods/medevolve/__init__.py、methods/taskcraft/__init__.py、experiments/runner.py；扩展 tests/evaluation/test_evaluation_session.py、tests/environment/test_patient_boundary.py、tests/method/test_taskcraft.py 与探索测试。

**接口：** 继续使用 EvaluationEnvironments，扩展 instance 的 phase 为 prepare/explore/generate/verify/solve，增加可选 requirement 参数；solve 未给 requirement 时保留 PhysicianBench 独立实例行为。EnvironmentInstance 增加 access_mode（shared_query/isolated，默认 isolated）。FHIRClient 增加 allow_database_write（默认 True），get_tools 根据 access_mode 绑定。

explore_environment 与 TaskCraft generate_tasks 新增可选关键字 environments。TaskCraft bind_generator 同样接受 environments 并绑定至闭包；未传会话时函数创建自己的会话且在 finally 关闭，传入时只结束本次借用。公共包装器和 invoke 转发新增关键字。

- [ ] **1. 添加跨阶段共享及额度为 1 的测试。**

```python
@pytest.mark.asyncio
async def test_generation_and_verification_share_one_instance(monkeypatch):
    from uuid import uuid4
    from contracts import EnvironmentConfig, EnvironmentInstance
    from evaluation.session import EvaluationEnvironments
    created, released = [], []
    async def create(config):
        item = EnvironmentInstance(instance_id=str(uuid4()),
            endpoint="http://fixture/fhir/", environment=config.name)
        created.append(item.instance_id)
        return item
    async def release(config, instance):
        released.append(instance.instance_id)
    monkeypatch.setattr(EvaluationEnvironments, "_create", staticmethod(create))
    monkeypatch.setattr(EvaluationEnvironments, "_release", staticmethod(release))
    config = EnvironmentConfig(name="PhysicianBench", manager_url="http://fixture", seed_data="seed")
    session = EvaluationEnvironments(instance_budget=1)
    async with session.instance(config, phase="generate") as first:
        async with session.instance(config, phase="verify") as second:
            assert first.instance_id == second.instance_id
    await session.close()
    assert len(created) == len(released) == 1
```

用 asyncio.Event 控制当前借用结束：额度 1 时独立申请等待，查询借用结束后空闲共享实例回收，独立申请能在 wait_for 超时前完成；待释放容器仍计额度。另测不同 seed/manager 不共享，同键并发只创建一次，取消创建及取消等待不泄漏额度，close 在借用结束后清理，释放失败记录实例且不将额度标为空闲。

- [ ] **2. 运行会话与工具测试，确认新 phase 或共享行为未实现时失败。**

```bash
python -m pytest tests/evaluation/test_evaluation_session.py tests/environment/test_patient_boundary.py -q
```

- [ ] **3. 扩展当前会话，接入准备、探索和生成。**

共享键为环境名、manager_url、固定 seed_data；记录当前借用数，不能仅按环境名混用。prepare/generate/verify 为查询；explore 的 PhysicianBench 为独立，其他环境共享；solve 按能力和要求选择。共享对象不得按患者修改，get_tools 每次创建独立的患者工具绑定。

会话额度覆盖创建中、存活和释放中实例。使用 Condition 协调等待、共享创建及空闲回收；锁内登记/预留，锁外执行 Docker 创建和删除。独立申请已等待时阻止新增查询借用插队，已有借用结束后删除空闲共享实例再创建独立实例；后续查询按需重建。不在持有共享借用时申请独立实例。FhirAgentBench 仍按一个逻辑实例计额度，其两容器清理由现有管理器完成。

```python
# FHIRClient.create，在真实 POST 前执行。
if not self.allow_database_write:
    raise ReadOnlyEnvironmentError("共享查询环境禁止真实写入")
```

get_tools 为 shared_query 传 allow_database_write=False；新增 ReadOnlyEnvironmentError，写工具仍记录明确拒绝。MedAgentBench 模拟 POST 不受数据库写入开关影响，保留范围检查；TaskCraft 的只读视图仍拒绝 POST，不扩大原方法工具能力。

实验入口的轨迹准备、一次探索及 TaskCraft 生成使用同一会话。TaskCraft bind_generator 仍绑定 random_seed，每次调用拥有独立 Pipeline/Ledger。取消后等全部借用清理再 close；创建失败回收预留，删除失败保留被占资源及具体错误，不静默释放额度。

- [ ] **4. 运行会话、生命周期、患者边界及 TaskCraft 回归。**

```bash
python -m pytest tests/evaluation/test_evaluation_session.py tests/environment/test_lifecycle.py tests/environment/test_patient_boundary.py tests/environment/test_medagentbench.py tests/method/test_taskcraft.py tests/method/test_method.py -q
```

- [ ] **5. 提交。** `feat(environment): 将生成与评测纳入共享实例额度`。

## 任务 4：分离生成名额与 task 评测队列

**文件：** 新增 experiments/scheduling.py、tests/experiment/test_generation_pipeline.py；修改 experiments/runner.py、configs/experiment.json、tests/experiment/test_closed_loop.py、test_workflow.py 与 tests/integration/test_pipeline.py。

**接口：** run_generation_evaluation 接收患者 plans、两种上限、generate 和 evaluate 回调；generate 返回最终 task 列表，evaluate 处理一个 task。调度器不改变方法算法，不直接保存实验文件。

- [ ] **1. 添加评测等待时后续患者仍能生成的测试。**

```python
import asyncio
import pytest
from experiments.scheduling import run_generation_evaluation

@pytest.mark.asyncio
async def test_generation_slot_is_free_while_evaluation_waits():
    evaluation_started = asyncio.Event()
    third_generated = asyncio.Event()
    finish_evaluation = asyncio.Event()
    async def generate(patient):
        if patient == 2:
            third_generated.set()
        return [patient]
    async def evaluate(task):
        evaluation_started.set()
        await finish_evaluation.wait()
    run = asyncio.create_task(run_generation_evaluation(
        [0, 1, 2], generation_limit=1, evaluation_limit=1,
        generate=generate, evaluate=evaluate))
    try:
        await asyncio.wait_for(evaluation_started.wait(), timeout=1)
        await asyncio.wait_for(third_generated.wait(), timeout=1)
        assert not run.done()
    finally:
        finish_evaluation.set()
        await run
```

用事件与活动计数分别检查两上限，单患者返回两题时两题各占 task 评测名额，生成失败/空列表不占评测，某 task 失败不丢其他 task，取消后所有回调收尾。对六方法替代入口复用同一调度矩阵，并保留其方法专属预算、方向及 seed 绑定断言。

- [ ] **2. 运行新调度测试确认模块缺失时失败。**

```bash
python -m pytest tests/experiment/test_generation_pipeline.py -q
```

- [ ] **3. 实现调度器并拆分现有 process。**

```python
# experiments/scheduling.py
import asyncio
from collections.abc import Awaitable, Callable
from typing import TypeVar

PlanT = TypeVar("PlanT")
TaskT = TypeVar("TaskT")

async def run_generation_evaluation(
    plans: list[PlanT], *, generation_limit: int, evaluation_limit: int,
    generate: Callable[[PlanT], Awaitable[list[TaskT]]],
    evaluate: Callable[[TaskT], Awaitable[None]],
) -> None:
    if generation_limit < 1 or evaluation_limit < 1:
        raise ValueError("Concurrency limits must be positive")
    queue = asyncio.Queue()
    stop = object()
    slots = asyncio.Semaphore(generation_limit)
    async def produce(plan):
        async with slots:
            tasks = await generate(plan)
        for task in tasks:
            await queue.put(task)
    async def consume():
        while True:
            task = await queue.get()
            try:
                if task is stop:
                    return
                await evaluate(task)
            finally:
                queue.task_done()
    consumers = [asyncio.create_task(consume()) for _ in range(evaluation_limit)]
    producers = [asyncio.create_task(produce(plan)) for plan in plans]
    workers = [*producers, *consumers]
    try:
        await asyncio.gather(*producers)
        for _ in consumers:
            await queue.put(stop)
        await asyncio.gather(*consumers)
    finally:
        for worker in workers:
            if not worker.done():
                worker.cancel()
        await asyncio.gather(*workers, return_exceptions=True)
```

使用有限患者批次的任务队列；生成回调结束时释放 slots，再提交评测工作，不持有生成名额等待评测。正常生成和 task 运行失败由实验回调转换为既有 RunResult/EvaluationResult，继续其他任务；取消和不可继续的异常传播至调度器并清理 worker。

runner 在并发启动前保留顺序方向采样。generate 回调执行原方法与公共输出校验，保存 attempts、generation runs 和最终 tasks，返回这些 tasks；evaluate 回调使用完整原始 trajectory 调用公共 evaluate_task，保存该 task 结果与全部用量。两个回调共用现有 artifact_lock，以患者/题下标排序保存，生成完成的任务不能等整名患者评测后才落盘。

方法的累计 GenerationBudget 仍每患者独立，策略副本同轮固定。调度器返回代表本轮全部生成与评测完成，随后才运行 MedEvolve feedback/update；初始/最终评测阶段保持顺序。准备阶段的 fit_trajectory 和生成输入裁剪保持，评测仍使用未裁剪轨迹。

configs/experiment.json 显式给出 generation_concurrency=6、evaluation_concurrency=6；用户可分别修改，例如 5 和 2。旧配置按任务 1 回退，不批量改写全量实验配置。scripts/run_full_comparison.py 的现有环境变量设置仍兼容，无需为本方案重写运行脚本。

- [ ] **4. 运行调度、闭环和集成回归。**

```bash
python -m pytest tests/experiment tests/integration -q
```

检查串行与并行的患者方向、task ID、排序、数量和成本归属一致；不比较真实模型文本相等。使用事件检查本轮最后一次评测完成之前 feedback 不运行，下一轮使用更新后的策略。保存配置含两个有效并发值。

- [ ] **5. 提交。** `feat(experiment): 分离患者生成与任务评测并发`。

## 任务 5：任务能力核验、实例路由与整体验收

**文件：** 修改 evaluation/evaluator.py、evaluation/prompts.py；新增 tests/evaluation/test_environment_routing.py、tests/environment/test_shared_generation_live.py；扩展现有 evaluator 与 MedAgentBench 集成测试。

**接口：** evaluate_task 保持当前 environments 可选参数；整体 Verification 使用任务 1 新字段，session.instance 的 solve 接收核验后的 requirement。不改变质量/难度契约或评测模型角色数量。

- [ ] **1. 添加路由和旧任务兼容测试。**

复用 tests/evaluation/test_evaluator.py 的 harness：整体 verifier 返回执行要求和 capability_supported，单条 verifier 只返回事实有效性；租约替代记录 phase/requirement，完成评价捕获实际 trace 与 observed_state。

```python
# 将已有 session 的 _shareable 扩展为此签名；测试不用模型。
from evaluation.session import EvaluationEnvironments
from contracts import EnvironmentConfig

def test_solver_routing_uses_requirement():
    session = EvaluationEnvironments(instance_budget=3)
    config = EnvironmentConfig(name="PhysicianBench", manager_url="http://fixture", seed_data="seed")
    assert session._shareable(config, "solve", requirement="read_only")
    assert not session._shareable(config, "solve", requirement="submit_request")
    assert not session._shareable(config, "solve", requirement="database_write")
```

另测 MedAgentBench 的 submit_request 共享、database_write 在 solver 前拒绝；FHIR-AgentBench 的操作要求拒绝；旧 task 无字段由整体核验补齐，新字段与题面核验冲突判 invalid，不错误进入共享写入。模型缺少整体能力字段时报明确模型失败，不能按关键词猜测。已有四维分数、三次中位数、复杂度计算和解题输入隔离测试保持。

- [ ] **2. 运行路由及评测测试确认新能力检查缺失时失败。**

```bash
python -m pytest tests/evaluation/test_environment_routing.py tests/evaluation/test_evaluator.py -q
```

- [ ] **3. 通过已有整体核验确定要求并完成路由。**

_review_material 的 task 白名单加入 execution_requirement，但标为待核验元数据。整体 verifier 在既有调用中根据 task.instruction 返回 execution_requirement 和 capability_supported；单条事实 verifier 不承担此职责。先处理环境不支持，再处理普通 valid=false，避免 unsupported_operation 被记成证据不足；生成与核验要求冲突归 ambiguous_task。环境规则不支持要求时，不能仅依赖模型返回的 capability_supported=true 放行。

```python
# 独立状态判断，不因模型声称可用而改变固定能力。
requirement = result.verification.execution_requirement
if requirement is None:
    raise ValueError("Overall verification must specify execution_requirement")
if not requirement_supported(environment.name, requirement):
    result.status = "invalid"
    result.failure_category = "unsupported_operation"
    return result
```

质量、难度、solver 和 completion 指令追加同一能力说明。solver 输入仍只含题目、患者和决策点，参考答案不进入解题请求。submit_request 看实际 URL、payload 与接受结果；database_write 看写前、写后资源及工具返回。PhysicianBench 独立 solver 开始前读取患者资源快照，结束后再次读取；observed_state 保存 before_resources、resources 和按资源类型/ID 比较的新增标识，既有资源不能算本次新建。只读和模拟提交不增加写前快照。

固定能力不支持的任务及元数据冲突不计作临床难度；feedback 保持真实失败分类。结构化路由测试证明接线与共享/隔离行为，不宣称替代模型测试证明真实语义评分准确。

- [ ] **4. 完成全量离线回归和显式启用的环境检查。**

```bash
python -m pytest tests --ignore=tests/environment/test_live_image.py --ignore=tests/environment/test_medagentbench_live.py --ignore=tests/environment/test_shared_generation_live.py --ignore=tests/environment/test_fhir_agentbench_live.py --ignore=tests/method/test_taskcraft_live.py -q
```

新 live 测试读取现有 MEDEVOLVE_TEST_IMAGE、MEDEVOLVE_MEDAGENTBENCH_TEST_IMAGE 的已加载镜像和MEDEVOLVE_FHIR_AGENTBENCH_SEED 指定的已准备种子，无资产时 skip。五患者并发读取同一实例，验证患者隔离；TaskCraft 的真实只读工具绑定拒绝 POST；MedAgentBench 提交前后资源不变；PhysicianBench 独立实例写入唯一标记，共享实例与下一独立种子均无该标记。使用 finally 清理实例，不调用模型、不拉取镜像。

同时记录逻辑实例峰值和实际 Docker 数；FHIR-AgentBench 每实例两容器，不能套用一实例一容器断言。真实样本生成需要另有已授权模型与运行请求，缺少实测时保留未验证状态。

TaskCraft 在研改动合入后再次执行 tests/method/test_taskcraft.py、tests/experiment/test_generation_pipeline.py、tests/evaluation/test_evaluation_session.py 和公共回归；验证 request_attempts 等重试、累计预算、共享绑定、执行要求、取消清理均保留。

- [ ] **5. 提交并记录实际验证范围。** `feat(evaluation): 按任务能力复用查询并隔离真实写入`。

## 实施检查点与完成标准

按五项任务依次完成，每项测试通过后提交；执行中以当前源码为准核对 TaskCraft 在研合入状态，不把旧行号或历史分支清单当作方法现状。

完成标准：六方法最终任务带执行要求；生成名额在公共评测结束前释放；两个并发上限分别生效；全部环境申请在同一运行额度内；查询和模拟提交共享、真实写入隔离；方法内部机制与公共评测规则保持；MedEvolve 轮次屏障、结果归属及失败/取消清理通过验证。
