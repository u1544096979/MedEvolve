# 环境能力与患者并行实施计划

> 执行方式：在独立 worktree 中使用 executing-plans，按任务顺序实施并设置检查点。若用户选择子代理执行，则使用 subagent-driven-development。实施前读取 using-git-worktrees，基于项目当前分支与所需未提交修改建立隔离工作区。

**目标：** 五个患者可并行生成、查询和模拟提交；真实写入任务与试写探索使用独立种子环境，受统一容器额度限制。

**架构：** 环境能力由确定的公共契约提供，生成与公共核验明确任务的执行要求。环境管理服务维护共享查询实例、独立实例和租约；实验入口按患者并发执行，同轮结束后再汇总反馈和更新策略。

**技术栈：** Python >=3.10、Pydantic 2、asyncio、FastAPI、httpx、Docker、现有 Agents SDK、pytest 与 pytest-asyncio；不新增运行依赖。

## 全局约束

- 依据 [设计文档](design.md)，保持 PhysicianBench 的真实 FHIR 写入及 MedAgentBench 的模拟 POST 语义。
- 所有方法、环境探索、解题与评测使用相同的固定能力说明；探索指导不能覆盖这些约束。
- 方法清单为 `medevolve`、`workflow`、`selfgen`、`taskcraft`、`singleshot`、`envexplore`：main 上前三项本次实施，worktree 中后三项为合并后必做项；固定能力和语义文案仅来自 `environments/capabilities.py`。
- 共享查询实例的工具入口禁止真实写入，保留患者范围和原生请求格式；模拟 POST 可以执行。
- 真实写入任务及 PhysicianBench 试写探索从独立种子开始，完成状态读取后释放容器与独立存储。
- 患者并发与容器额度分开设置；同轮策略、方向采样、任务编号、输出归属和统计口径不随完成顺序改变。
- 各任务独立验证与提交；Conventional Commits 的 description 和 body 使用中文。
- 本计划中的模型测试使用替代运行器，不调用付费模型；真实环境检查使用已加载镜像，不拉取新镜像。

## 文件与职责

| 文件 | 职责 |
| --- | --- |
| `contracts.py`、新增 `environments/capabilities.py` | 能力、任务执行要求、租约契约与固定说明 |
| `methods/common.py`、`methods/medevolve/__init__.py`、`methods/workflow/__init__.py`、`methods/selfgen/__init__.py` 及三个方法的提示文件 | main 本次实施：统一能力输入、任务要求输出与生成校验，移除方法内语义副本 |
| `environments/fhir.py`、`environments/__init__.py`、`environments/medagentbench.py` | 租约客户端、共享实例写入阻断与原生工具适配 |
| 新增 `server/pool.py`、`server/app.py`、`server/docker.py` | 同种子实例复用、全局容器额度、租约与生命周期 |
| `evaluation/evaluator.py`、`evaluation/prompts.py` | 核验任务能力、选择实例、检查实际提交和状态变化 |
| `experiments/runner.py`、新增 `experiments/concurrency.py` | 患者并发、顺序采样、结果写入、轮次屏障与会话清理 |
| `docs/comparison-experiments/common/SPEC.md:46` | 同步公共规范：查询与模拟提交共享，真实写入及 PhysicianBench 试写探索隔离 |
| `.worktrees/taskcraft/methods/taskcraft/{__init__,ledger,stages,environment}.py`、`.worktrees/taskcraft/methods/medevolve/prompts.py:44` | TaskCraft 合并后必做：阶段能力输入、候选执行要求、查询租约及重复语义清理 |
| `.worktrees/singleshot/methods/singleshot/{__init__,prompts}.py`、`.worktrees/singleshot/methods/common.py` | SingleShot 合并后必做：能力输入与执行要求、语义副本清理及公共接口统一 |
| `.worktrees/envexplore/methods/envexplore/{__init__,prompts}.py`、`.worktrees/envexplore-v2/methods/generation.py`、`.worktrees/envexplore-v2/methods/envexplore/__init__.py`、对应 `experiments/runner.py` | EnvExplore 两分支合并后必做：各版本的探索/生成能力输入、执行要求、租约与绑定入口 |

## 六个方法适配清单

行号对应当前 main 工作区及对应 worktree，合并后按函数或字段重新定位；本轮修订仅修改文档，下面的源文件变更由实施任务执行。

| 方法 | 状态与文件定位 | 固定能力说明、执行要求和环境复用适配 |
| --- | --- | --- |
| `medevolve` | main，本次实施；`methods/medevolve/__init__.py:126`、`:147`、`:173`，`methods/medevolve/prompts.py:47` | 探索、生成和反馈指令统一追加 `capability_instructions`；生成返回执行要求，反馈按实际失败分类处理；处方提示保留药物、剂量、频次和证据规则，将提交/持久化语义交给统一来源；探索按环境能力租约路由 |
| `workflow` | main，本次实施；`methods/workflow/__init__.py:111`、`:123`、`:148`、`:177`，`methods/workflow/prompts.py` | 定位、分类型生成、内部检查追加统一说明；每道候选及最终题目保留执行要求；保持 `candidate_pool=True` 的五类选两类、单题约束和公共预算 |
| `selfgen` | main，本次实施；`methods/selfgen/__init__.py:83`、`:88`、`:94`、`:98`、`:111`、`:119`、`:130`，`methods/selfgen/prompts.py:8`、`:22` | 删除 94–95 行 `operation_semantics` 手写文案，第 88–89 行的环境名检查改用统一能力来源；调用包装器对初稿、自检和修订统一追加能力说明；提示不再依赖旧输入字段，事实规则提取不能重新复制环境语义；初稿/修订候选输出执行要求，最终过滤和通过题不变校验保留字段 |
| `taskcraft` | worktree，合并后必做；分支 `codex/taskcraft-baseline`；`.worktrees/taskcraft/methods/medevolve/prompts.py:44`、`methods/taskcraft/__init__.py:55`、`:82`、`:95`，`methods/taskcraft/ledger.py:16`，`methods/taskcraft/stages.py:289`、`:389`、`:605`，`methods/taskcraft/environment.py:56`、`:88`（后五个路径均相对该 worktree） | 通过 Ledger 的模型/工具 Agent 调用统一注入能力说明；删除分支主方法提示第 44 行的提交/持久化语义副本，保留临床要求；atomic、width、depth 的生成输出与 `GeneratedQuestion` 构造传递执行要求，筛选不得丢失；查询生成改共享租约，保留原只读工具视图和随机种子绑定 |
| `singleshot` | worktree，合并后必做；分支 `codex/singleshot`；`.worktrees/singleshot/methods/singleshot/prompts.py:13`、`:14`、`methods/singleshot/__init__.py:24`、`:33`、`:37`、`:44`，`methods/common.py:45`、`:65`（后三个路径相对该 worktree） | 删除 prompts 第 13–14 行两环境语义副本，单次生成追加统一说明并输出执行要求；将 `output_constraints(task_types=...)`、`validate_task_batch` 和 `GenerationBudget.finish(task_types=...)` 与更新后的公共接口统一；保留从五种题型自主选题、两题可同类型及一次调用，无额外探索或修订 |
| `envexplore` | worktree，合并后必做；分支 `codex/envexplore`、`codex/envexplore-v2`；原分支 `.worktrees/envexplore/methods/envexplore/__init__.py:83`、`:100`、`:104`、`prompts.py:18`；v2 `.worktrees/envexplore-v2/methods/generation.py:5`、`:7`、`methods/envexplore/__init__.py:21`、`:38`，`experiments/runner.py:182`、`:194`（简写路径相对对应 worktree） | 原分支的工具生成追加统一说明、输出执行要求并使用共享查询租约，保留生成阶段只读限制；v2 删除 `GENERATION_RULES` 第 5 行字符串中的提交/持久化语义副本，保留事实规则，环境探索调用公共租约入口，生成绑定器保留固定初始策略与探索指导并追加统一能力；两个分支均按所选版本的原方法流程验收，计为一个方法 |

## 公共接口与分支合并顺序

先完成本计划五项任务，将 main 上三个方法、公共能力和执行要求校验、租约及并发调度落入 main；然后分别 rebase `codex/taskcraft-baseline`、`codex/singleshot`、`codex/envexplore`、`codex/envexplore-v2` 到更新后的 main，在各分支完成上表必做项与测试后再合并。分支之间不设未经确认的先后依赖，EnvExplore 两个版本分别核对并按选定版本合并。

每个分支都改动 `methods/common.py`，rebase 时保留本方案的公共字段、能力约束和 `execution_requirement` 校验，逐项整合方法特有选择规则；禁止以分支旧 `common.py` 整文件覆盖新契约。

SingleShot 的 `output_constraints(task_types=...)` 已与 main 的 `output_constraints(directions, *, max_questions, candidate_pool)` 分叉；其合并后必做项是同时统一 `output_constraints`、`validate_task_batch`、`GenerationBudget.finish` 和实验入口校验，公共字段与执行要求检查共用，选择规则分别处理。统一接口保留 main 参数，并增加互斥的 `task_types` 模式：

```python
def output_constraints(
    directions: list[GenerationDirection] | None = None, *,
    max_questions: int = 2, candidate_pool: bool = False,
    task_types: list[TaskType] | None = None,
) -> str: ...

def validate_task_batch(
    result: RunResult, directions: list[GenerationDirection] | None = None, *,
    max_questions: int = 2, candidate_pool: bool = False,
    task_types: list[TaskType] | None = None,
) -> RunResult: ...
```

`task_types` 与 `directions`/`candidate_pool` 互斥；SingleShot 模式允许两个同类型问题，不能用 Workflow 的候选池每类最多一题规则替代。`GenerationBudget.finish` 保留原有 `output/status/error` 参数及方向路径，新增 `task_types` 关键字路径并委托同一个输出校验；`experiments/runner.py` 保留 SingleShot 的五种允许题型输入，按该模式校验输出，不提前抽样方向。main 本次实施不提前加入 SingleShot 的方法分派或改变现有选择规则。

## 任务 1：固定环境能力与生成要求

**文件：** 修改 `contracts.py`、`methods/common.py`、`methods/medevolve/{__init__,prompts}.py`、`methods/workflow/{__init__,prompts}.py`、`methods/selfgen/{__init__,prompts}.py`；新增 `environments/capabilities.py`、`tests/environment/test_capabilities.py`；修改 `tests/method/test_method.py`、`tests/method/test_workflow.py`、`tests/method/test_selfgen.py`。本任务直接覆盖 main 三个方法；另外三个方法按上表在分支合并时适配。

**接口：**

```python
# contracts.py
ExecutionRequirement = Literal["read_only", "submit_request", "database_write"]

class EnvironmentCapabilities(BaseModel):
    name: Literal["PhysicianBench", "MedAgentBench"]
    post_semantics: Literal["persistent", "accept_without_persistence"]
    supports_database_write: bool

# GeneratedQuestion 增加字段；旧保存文件允许缺失。
execution_requirement: ExecutionRequirement | None = None

# Verification 增加字段。单条证据核验可为空；整体任务核验必须返回。
execution_requirement: ExecutionRequirement | None = None
capability_supported: bool | None = None

# EvaluationResult.failure_category 增加 unsupported_operation。

# environments/capabilities.py
def get_capabilities(name: str) -> EnvironmentCapabilities: ...
def capability_instructions(name: str) -> str: ...
def requirement_supported(
    name: str, requirement: ExecutionRequirement
) -> bool: ...
```

- [ ] **1. 添加能力与要求测试。**

```python
import pytest
from environments.capabilities import (
    capability_instructions, get_capabilities, requirement_supported,
)
from environments.errors import EnvironmentUnavailable

def test_medagentbench_acceptance_is_not_persistence():
    capability = get_capabilities("MedAgentBench")
    assert capability.post_semantics == "accept_without_persistence"
    assert not capability.supports_database_write
    assert requirement_supported("MedAgentBench", "submit_request")
    assert not requirement_supported("MedAgentBench", "database_write")
    assert "不持久化" in capability_instructions("MedAgentBench")

def test_physicianbench_supports_persistence():
    assert requirement_supported("PhysicianBench", "database_write")

def test_unknown_environment_has_no_assumed_capability():
    with pytest.raises(EnvironmentUnavailable):
        get_capabilities("unknown")
```

生成测试捕获 `call_llm` 参数，分别断言 MedEvolve 的单次生成、workflow 的定位/按题型生成/内部检查、selfgen 的初稿/自检/修订均收到同一个 `capability_instructions(trajectory.environment)`。对 main 三个方法使用相同的两环境能力测试矩阵，检查每道新生成或修订题目的 `execution_requirement` 非空，过滤后保留字段且通过题全部字段不变；selfgen 输入不再携带手写 `operation_semantics`。更新现有生成测试的题目夹具，不改变题数或预算预期。方法测试中表示实际环境的 `test` 改为已接入环境名，纯类型解析夹具仍可以使用任意字符串。

- [ ] **2. 运行新增测试，确认能力模块缺失导致失败。**

```bash
python -m pytest tests/environment/test_capabilities.py -q
```

- [ ] **3. 实现固定能力与提示接入。**

```python
CAPABILITIES = {
    "PhysicianBench": EnvironmentCapabilities(
        name="PhysicianBench", post_semantics="persistent",
        supports_database_write=True,
    ),
    "MedAgentBench": EnvironmentCapabilities(
        name="MedAgentBench", post_semantics="accept_without_persistence",
        supports_database_write=False,
    ),
}

def requirement_supported(name, requirement):
    capability = get_capabilities(name)
    return requirement != "database_write" or capability.supports_database_write
```

`get_capabilities` 返回契约副本，未知环境抛出 `EnvironmentUnavailable`。固定中文说明明确查询能力、POST 是否持久化、不可要求的操作及三种执行要求的区别。

MedEvolve 保持 `generate_tasks` 签名，从 `trajectory.environment` 取能力；workflow 与 selfgen 保持签名，校验 `environment.name == trajectory.environment` 后取同一能力。将固定说明追加到角色指令，使轨迹、策略、探索输出不能覆盖它。selfgen 在 `call` 包装器一次绑定能力说明，覆盖初稿、自检和修订，删除 `operation_semantics` 字段及其环境分支文案；`INPUT_RULES` 不再读取该字段。主方法提示中的处方规则仅保留临床要求和证据条件，selfgen 的 `_fact_rules` 保持事实规则提取，不从主方法或其他分支重新引入环境语义副本。环境探索与解题接入在任务 4 完成。

生成提示要求模型根据题面填写执行要求：纯分析及查询为 `read_only`，只要求提交为 `submit_request`，要求资源真实创建或后续读取新增结果为 `database_write`。`methods/common.py` 的生成约束同步字段列表，`validate_task_batch` 拒绝新生成输出缺少要求；已保存的旧任务仍可解析，并在整体核验阶段补齐要求。

新增失败分类用于能力不支持的题目，反馈明确该失败属于任务与环境不匹配，不作为临床难度提升。main 本次实施覆盖 `medevolve`、`workflow`、`selfgen`，不合入另外三个方法；`taskcraft`、`singleshot`、`envexplore` 已有 worktree 实现，其能力适配是合并验收的必做项。

- [ ] **4. 运行能力和方法回归。**

```bash
python -m pytest tests/environment/test_capabilities.py tests/method -q
```

预期：两环境能力确定，main 三个方法同等接收约束并输出执行要求，selfgen 的所有生成/自检/修订阶段覆盖统一说明，方法内不保留重复语义文案，原有调用数、题型及预算行为保持。

- [ ] **5. 提交本任务的契约、提示与测试。**

提交说明：`feat(environment): 定义固定能力与任务执行要求`。

## 任务 2：共享工具执行入口阻止真实写入

**文件：** 修改 `contracts.py`、`environments/fhir.py`、`environments/__init__.py`、`environments/medagentbench.py`、`environments/errors.py`；新增 `tests/environment/test_read_only.py`；保留并扩展患者边界和原生动作测试。

**接口：**

```python
# EnvironmentInstance 增加默认值，旧独立实例保持可写行为。
access_mode: Literal["shared_query", "isolated"] = "isolated"

# environments/fhir.py
def __init__(self, endpoint: str, patient_ids: list[str], *,
             allow_database_write: bool = True): ...

# environments/errors.py
class ReadOnlyEnvironmentError(PatientAccessError):
    pass
```

`get_tools` 保持原签名，通过 `instance.access_mode` 设置 `FHIRClient.allow_database_write`。MedAgentBench 的模拟 POST 仍执行解析与患者范围校验，不因共享访问模式被拒绝。

- [ ] **1. 添加阻断测试。**

```python
import pytest
from environments.errors import ReadOnlyEnvironmentError
from environments.fhir import FHIRClient

@pytest.mark.asyncio
async def test_shared_client_never_sends_a_write(monkeypatch):
    requests = []
    async def request(self, method, path, **kwargs):
        requests.append((method, path))
        raise AssertionError("共享写入不应到达 HTTP 请求")
    monkeypatch.setattr(FHIRClient, "_request", request)
    client = FHIRClient("http://fixture/fhir/", ["p1"],
                        allow_database_write=False)
    with pytest.raises(ReadOnlyEnvironmentError):
        await client.create({
            "resourceType": "Communication",
            "subject": {"reference": "Patient/p1"},
        })
    assert requests == []
```

复用 `tests/environment/test_patient_boundary.py` 的 `native_fhir` 夹具，经 `get_tools` 实际调用共享实例的创建工具，断言两个患者存储都不变；独立实例仍可写。扩展 `tests/environment/test_medagentbench.py`，将实例设为 `shared_query` 后执行原有合法 POST，断言提交仍记录且未发送 HTTP POST。现有跨患者访问拒绝测试保持。

- [ ] **2. 运行新增测试，确认缺少写入开关时失败。**

```bash
python -m pytest tests/environment/test_read_only.py -q
```

- [ ] **3. 实现工具层限制。**

```python
# FHIRClient.create 的第一项检查
if not self.allow_database_write:
    raise ReadOnlyEnvironmentError("共享查询环境不允许数据库写入")

# PhysicianBench 的 get_tools 客户端
client = FHIRClient(
    instance.endpoint, patient_ids,
    allow_database_write=instance.access_mode == "isolated",
)
```

在真实 HTTP POST 前执行阻断，不依赖工具是否隐藏或模型是否遵守提示。共享工具调用把拒绝记录为明确的工具错误；不自动新建容器重放误写动作。MedAgentBench 原生分支不改为真实写入。

- [ ] **4. 运行工具和患者边界回归。**

```bash
python -m pytest tests/environment/test_read_only.py tests/environment/test_patient_boundary.py tests/environment/test_medagentbench.py -q
```

- [ ] **5. 提交本任务。**

提交说明：`feat(environment): 阻止共享查询实例真实写入`。

## 任务 3：管理服务复用实例与统一容器额度

**文件：** 新增 `server/pool.py`、`tests/environment/test_pool.py`；修改 `contracts.py`、`server/app.py`、`server/docker.py`、`environments/__init__.py`、`tests/environment/test_lifecycle.py`。

**接口：**

```python
# contracts.py
class EnvironmentAcquireRequest(BaseModel):
    environment: EnvironmentConfig
    session_id: str
    access_mode: Literal["shared_query", "isolated"]

class EnvironmentLease(BaseModel):
    lease_id: str
    session_id: str
    instance: EnvironmentInstance

# server/docker.py
async def resolve_seed(self, config: EnvironmentConfig) -> str: ...
async def create(self, config: EnvironmentConfig, *,
                 image_id: str | None = None) -> EnvironmentInstance: ...

# server/pool.py
class EnvironmentPool:
    def __init__(self, manager: DockerManager, *, max_instances: int): ...
    async def acquire(self, request: EnvironmentAcquireRequest) -> EnvironmentLease: ...
    async def release(self, lease_id: str) -> None: ...
    async def close_session(self, session_id: str) -> None: ...
    async def close(self) -> None: ...

# environments/__init__.py
async def acquire_environment(config: EnvironmentConfig, session_id: str, *,
                              access_mode: str) -> EnvironmentLease: ...
async def release_lease(config: EnvironmentConfig, lease: EnvironmentLease) -> None: ...
async def close_environment_session(config: EnvironmentConfig, session_id: str) -> None: ...
```

新增 `POST /leases`、`DELETE /leases/{lease_id}`、`DELETE /sessions/{session_id}`。保留现有 `POST /instances` 的独立实例语义和 DELETE 接口，旧接口也经过同一额度控制；低层 `DockerManager` 仍负责单容器独立存储与启动清理。

- [ ] **1. 添加真实生命周期模型测试。**

在 `test_pool.py` 定义以下 `FakeManager`；用 `asyncio.Event` 控制创建和解题结束，不以长时间 sleep 判断并发。

```python
import asyncio
import copy
import hashlib
import uuid
from collections import Counter

import pytest
from contracts import (
    EnvironmentAcquireRequest, EnvironmentConfig, EnvironmentInstance,
)
from server.pool import EnvironmentPool

class FakeManager:
    def __init__(self):
        self.active = {}
        self.released = Counter()
        self.peak = 0
        self.gate = None
        self.fail_next = False

    async def resolve_seed(self, config):
        return "sha256:" + hashlib.sha256(config.seed_data.encode()).hexdigest()

    async def create(self, config, *, image_id=None):
        if self.gate is not None:
            await self.gate.wait()
        if self.fail_next:
            self.fail_next = False
            raise RuntimeError("创建失败")
        identifier = str(uuid.uuid4())
        instance = EnvironmentInstance(instance_id=identifier,
            endpoint=f"http://{identifier}/fhir/", environment=config.name)
        self.active[identifier] = {
            "instance": instance,
            "seed": image_id or await self.resolve_seed(config),
            "resources": copy.deepcopy({"Patient/p1": {"id": "p1"}}),
        }
        self.peak = max(self.peak, len(self.active))
        return instance

    async def release(self, identifier):
        if self.active.pop(identifier, None) is not None:
            self.released[identifier] += 1

@pytest.fixture
def fake_manager():
    return FakeManager()
```

```python
@pytest.mark.asyncio
async def test_two_sessions_reuse_the_same_query_container(fake_manager):
    pool = EnvironmentPool(fake_manager, max_instances=2)
    config = EnvironmentConfig(name="PhysicianBench",
        manager_url="http://manager", seed_data="seed-a")
    def request(session):
        return EnvironmentAcquireRequest(environment=config,
            session_id=session, access_mode="shared_query")
    first, second = await asyncio.gather(
        pool.acquire(request("run-a")), pool.acquire(request("run-b")))
    assert first.lease_id != second.lease_id
    assert first.instance.instance_id == second.instance.instance_id
    await pool.release(first.lease_id)
    assert second.instance.instance_id in fake_manager.active
    await pool.release(second.lease_id)
    await pool.close_session("run-a")
    assert second.instance.instance_id in fake_manager.active
    await pool.close_session("run-b")
    assert not fake_manager.active
```

同一夹具补充以下验收：两个真实写入租约 ID 与存储不同；不同环境或镜像 ID 不共享；一次并发创建失败唤醒等待者且不泄漏额度；取消等待者和取消创建都释放资源；双重释放不删除其他租约；`max_instances=1` 下，释放查询租约后独立任务能继续；服务退出释放全部受管实例。

- [ ] **2. 运行池测试，确认缺少模块时失败。**

```bash
python -m pytest tests/environment/test_pool.py -q
```

- [ ] **3. 实现池、接口和租约客户端。**

共享键使用 `(environment.name, immutable_image_id)`。抽取现有 Docker 镜像检查为 `resolve_seed`，保留外部数据库拒绝和本地镜像限制；创建时使用检查后的不可变 ID。共享实例的 `access_mode` 为 `shared_query`，独立实例为 `isolated`。

池使用 `asyncio.Condition` 协调创建中的预留额度、活动实例、借用计数、会话使用记录和等待请求。同键的并发查询只启动一个容器。不能持有锁等待 Docker 启动或删除；在锁内预留状态，锁外执行，完成后在锁内登记并唤醒。

共享租约结束后，容器可在会话期间保留以复用；最后使用它的会话关闭后释放。额度满且独立任务等待时，可以释放借用数为零的共享容器，即使其会话尚未关闭，下一次查询按需重建。删除完成前仍占额度。已有借用者可正常完成，等待独立任务优先于新增借用，避免 `max_instances=1` 时死锁或持续查询饿死写入任务。调用者不能持有查询租约再等待独立租约。

独立租约一旦释放就删除实例，不进入共享缓存。`close_session` 仅清理该会话租约和使用关系，不影响其他会话的借用。`close` 在 FastAPI lifespan 退出时等待创建任务结束并清理受管资源。

服务通过 `MEDEVOLVE_MAX_INSTANCES` 设置总额度，默认 2，校验为正整数；现有 `MEDEVOLVE_CONTAINER_MEMORY` 保持独立。总额度覆盖共享容器、真实写入、试写探索、旧独立接口及启动中的实例。使用单个服务进程维护额度，不启用多 worker。

租约请求可能因排队超过当前客户端的 240 秒超时；新客户端使用有限连接超时、无固定读取超时。请求断开或调用取消时，取消等待申请并回收已创建但未交付的租约；客户端收到租约后保证在 `finally` 释放。测试通过 ASGI 请求和事件覆盖断开/取消，不引入自动重放写入。

- [ ] **4. 运行池、接口及原生命周期测试。**

```bash
python -m pytest tests/environment/test_pool.py tests/environment/test_lifecycle.py -q
```

预期：跨实验会话可复用同一查询容器，活动峰值不超过配置额度，旧实例接口保持独立性。

- [ ] **5. 提交本任务。**

提交说明：`feat(server): 复用查询实例并限制容器总数`。

## 任务 4：按任务要求选择实例并核对提交与状态

**文件：** 修改 `evaluation/evaluator.py`、`evaluation/prompts.py`、`methods/medevolve/__init__.py`；新增 `tests/evaluation/test_environment_routing.py`；修改 `tests/evaluation/test_evaluator.py`、`tests/integration/test_medagentbench_integration.py`、`tests/method/test_method.py`。

**接口：**

```python
async def evaluate_task(task: Task, trajectory: PatientTrajectory,
                        environment: EnvironmentConfig,
                        config: EvaluationConfig, *,
                        session_id: str | None = None) -> EvaluationResult: ...

async def explore_environment(config: EnvironmentConfig, patient_ids: list[str],
                              runtime: RuntimeConfig, *,
                              session_id: str | None = None) -> RunResult: ...
```

直接调用没有会话 ID 时，函数创建 UUID 会话，并在结束时关闭；实验入口传会话 ID 时仅释放本次租约，最后由实验入口关闭会话。

- [ ] **1. 添加路由与语义测试。**

路由测试使用以下辅助函数替代租约客户端并记录 `access_mode`，替代运行器按 output schema 返回结果。整体 `Verification` 返回基于题面确定的要求；单条证据 Verification 不填写要求。

```python
import asyncio
import json
import uuid
from unittest.mock import patch

from contracts import (
    Budget, CompletionReview, EnvironmentConfig, EnvironmentInstance,
    EnvironmentLease, EvaluationConfig, ModelConfig, PatientTrajectory,
    QualityReview, RunResult, RuntimeConfig, SolverAnswer, Task, Verification,
)
from evaluation.evaluator import evaluate_task

def evaluate_with_requirement(name, requirement):
    modes, releases = [], []
    instruction = {
        "read_only": "读取患者记录并解释结果",
        "submit_request": "提交指定患者的处方请求",
        "database_write": "在数据库创建指定患者的处方并查询确认",
    }[requirement]
    task = Task(id="t1", patient_id="p1", environment=name,
        round_index=1, source_method="medevolve", instruction=instruction,
        task_type="药物处方", reference_answer="参考答案",
        evidence=[{"source_id": "Observation/o1", "statement": "事实"}],
        execution_requirement=requirement)
    trajectory = PatientTrajectory(patient_id="p1", environment=name,
        text="记录", sources=task.evidence)
    roles = ("verifier", "quality", "solver", "completion")
    config = EvaluationConfig(runtime=RuntimeConfig(
        models={r: ModelConfig(model=r, api_key_env="UNUSED") for r in roles},
        budgets={r: Budget(max_model_turns=4, max_tool_calls=4) for r in roles}))
    environment = EnvironmentConfig(name=name,
        manager_url="http://fixture", seed_data="seed")
    async def acquire(config, session_id, *, access_mode):
        modes.append(access_mode)
        return EnvironmentLease(lease_id=str(uuid.uuid4()), session_id=session_id,
            instance=EnvironmentInstance(instance_id=str(uuid.uuid4()),
                endpoint="http://fixture/fhir/", environment=name,
                access_mode=access_mode))
    async def release(config, lease):
        releases.append(lease.lease_id)
    async def evidence(*args):
        return {"resourceType": "Observation", "id": "o1"}
    async def tools(*args, **kwargs):
        return []
    async def state(*args):
        return {"resources": []}
    async def run(**kwargs):
        schema = kwargs["output_type"]
        if schema is Verification:
            payload = json.loads(kwargs["input_text"])
            if "task" not in payload:
                output = {"valid": True, "reason": "事实受支持"}
            else:
                supported = not (name == "MedAgentBench"
                                 and requirement == "database_write")
                output = {"valid": supported, "reason": "能力核验",
                    "execution_requirement": requirement,
                    "capability_supported": supported}
        elif schema is QualityReview:
            output = {"dimensions": {key: {"judgment": "合格", "reason": "清晰"}
                for key in ("clinical_value", "correctness",
                            "longitudinal_integration", "clarity")}}
        elif schema is SolverAnswer:
            output = {"answer": "回答"}
        elif schema is CompletionReview:
            output = {"completed": False, "reason": "回答未覆盖操作要求"}
        else:
            raise AssertionError(schema)
        return RunResult(output=output, trace=[], usage={}, status="completed")
    async def execute():
        return await evaluate_task(task, trajectory, environment, config,
                                   session_id="routing-test")
    with patch("environments.acquire_environment", acquire), \
         patch("environments.release_lease", release), \
         patch("environments.read_evidence", evidence), \
         patch("environments.get_tools", tools), \
         patch("environments.read_state", state), \
         patch("runtime.call_llm", run), patch("runtime.run_agent", run):
        result = asyncio.run(execute())
    return result, modes, len(releases)
```

```python
@pytest.mark.parametrize("environment,requirement,modes", [
    ("PhysicianBench", "read_only", ["shared_query", "shared_query"]),
    ("PhysicianBench", "submit_request", ["shared_query", "isolated"]),
    ("PhysicianBench", "database_write", ["shared_query", "isolated"]),
    ("MedAgentBench", "read_only", ["shared_query", "shared_query"]),
    ("MedAgentBench", "submit_request", ["shared_query", "shared_query"]),
    ("MedAgentBench", "database_write", ["shared_query"]),
])
def test_routing_uses_verified_requirement(environment, requirement, modes):
    result, acquired_modes, released = evaluate_with_requirement(
        environment, requirement)
    assert acquired_modes == modes
    assert released == len(modes)
    if environment == "MedAgentBench" and requirement == "database_write":
        assert result.status == "invalid"
        assert result.failure_category == "unsupported_operation"
        assert result.solver_run is None
```

另测：生成元数据标为只读，但整体核验识别数据库写入时不能进入共享解题；旧任务无字段由整体核验补齐；整体核验缺少要求时明确失败，不猜题型或关键词。探索测试断言 PhysicianBench 使用 isolated，MedAgentBench 使用 shared_query，角色指令包含固定说明。

- [ ] **2. 运行路由测试，确认旧实例调用路径导致失败。**

```bash
python -m pytest tests/evaluation/test_environment_routing.py -q
```

- [ ] **3. 实现公共核验、路由与完成证据。**

整体核验保持现有模型调用位置，不增加专门分类模型调用。固定能力说明追加到核验、质量、解题、完成及环境探索指令；`_review_material` 的 task 字段白名单加入 `execution_requirement`，将它视为待核验的生成元数据。整体核验以 `task.instruction` 为准，返回执行要求和能力是否支持；与生成字段冲突时将任务判 invalid、`ambiguous_task`，不使用不可靠元数据选择共享写入。旧任务允许用核验结果补齐。MedAgentBench 的数据库写入要求判 invalid、`unsupported_operation`，不启动 solver。先处理能力不支持的结果再处理普通 valid=false，避免错误归入证据不足。

```python
def solver_access_mode(environment_name, requirement):
    capability = get_capabilities(environment_name)
    if requirement != "read_only" and capability.supports_database_write:
        return "isolated"
    return "shared_query"
```

PhysicianBench 的 `submit_request` 仍会真实写入，因此也使用独立实例。证据核验租约在进入 solver 前释放，避免额度为 1 时嵌套等待。

真实写入 solver 执行前后分别读取同一独立实例的患者资源快照，在 `observed_state` 保留 `before_resources`、`resources` 及新增资源标识，按 `(resourceType, id)` 对比；完成评价结合快照和实际工具返回，不将既有处方误当成新建。查询与 MedAgentBench 模拟提交不增加这次写前快照。

固定完成提示使用设计中已确认的规则：提交检查 URL、payload、接受结果；持久化检查实际变化。解题输入仍只含题目、患者与决策点，不泄漏参考答案或审题证据；环境能力说明来自角色指令。

所有租约在 `finally` 释放，释放后再执行不需要环境的质量或完成模型调用。探索的两个环境采用对应租约模式，生成环境指导时显式区分接受与持久化。

- [ ] **4. 运行评测、探索和原生提交回归。**

```bash
python -m pytest tests/evaluation tests/method/test_method.py tests/integration/test_medagentbench_integration.py -q
```

替代模型测试只证明路由、材料和字段接线正确，不声称证明真实模型的语义判断准确。另为已有集成测试增加“错误剂量”与“没有 POST”材料，确认完整操作记录传给 completion，而不是只传成功提示。

- [ ] **5. 提交本任务。**

提交说明：`feat(evaluation): 按环境能力隔离写入并核对执行结果`。

## 任务 5：患者并发、结果归属与整体验证

**文件：** 修改 `contracts.py`、`experiments/runner.py`、`configs/experiment.json`、`docs/comparison-experiments/common/SPEC.md:46`；新增 `experiments/concurrency.py`、`tests/experiment/test_concurrency.py`、`tests/environment/test_shared_live.py`；修改闭环、workflow、selfgen 与 pipeline 测试的任务和环境夹具。

**接口：**

```python
# ExperimentConfig 新增。旧配置保持一个患者串行执行。
patient_concurrency: int = Field(default=1, ge=1)

# experiments/concurrency.py，T/R 使用 TypeVar，以兼容 Python 3.10。
async def bounded_map(items: list[T], limit: int,
                      worker: Callable[[T], Awaitable[R]]) -> list[R]: ...
```

- [ ] **1. 添加并发测试和可控夹具。**

```python
@pytest.mark.asyncio
async def test_bounded_map_overlaps_and_preserves_input_order():
    active = peak = 0
    two_started = asyncio.Event()
    async def worker(index):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        if active == 2:
            two_started.set()
        await asyncio.wait_for(two_started.wait(), timeout=1)
        await asyncio.sleep(0)
        active -= 1
        return index
    assert await bounded_map(list(range(5)), 2, worker) == list(range(5))
    assert peak == 2
```

扩展 `test_closed_loop.py` 的替代依赖：测试环境名改用已接入环境，数据内容仍为假数据；生成结果补齐执行要求；探索和评测替代函数接收关键字会话 ID。加入事件记录当前策略和完成轮次。并发上限 1 与 5 各跑一次，比较每患者采样方向、任务 ID、最终列表次序、任务数量和用量统计，而不是比较并发下不可保证相同的真实模型输出。

验证五个患者确实有重叠执行；反馈开始前本轮所有患者完成；下一轮始终使用更新后的策略；单患者失败不会取消其他患者；取消实验会等待已启动 worker 释放租约后关闭会话。保存中间 JSON 后重新读取，确认多个完成事件不覆盖其他患者结果。

- [ ] **2. 运行并发测试，确认模块缺失时失败。**

```bash
python -m pytest tests/experiment/test_concurrency.py -q
```

- [ ] **3. 实现患者调度与单一结果写入。**

```python
async def bounded_map(items, limit, worker):
    semaphore = asyncio.Semaphore(limit)
    async def run(item):
        async with semaphore:
            return await worker(item)
    tasks = [asyncio.create_task(run(item)) for item in items]
    try:
        return await asyncio.gather(*tasks)
    finally:
        pending = [task for task in tasks if not task.done()]
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
```

`run_experiment` 创建一个 UUID 环境会话。轨迹读取借用共享实例，整个准备阶段结束释放租约但保留会话缓存；探索和各题评测传相同会话 ID。实验结束或取消时先等患者 worker 完成清理，再调用 `close_environment_session`，无论成功失败都执行。

`generate_and_evaluate` 在启动患者 worker 前按患者原始顺序采样 directions，保持当前 `rng` 消费顺序与独立评测的重置行为。每个 worker 从不可变的同轮策略副本开始，单患者内部仍按题序评测，不再增加题级并发。

worker 将生成结果、任务和评测结果提交给单一结果聚合器，聚合器负责共享列表、manifest、runs 及 `_save`。以患者下标和题下标排序 tasks、evaluations、attempts，使用已有 stage 保存模型记录归属；每次患者生成或题目评测结束均保留 checkpoint。worker 不直接写相同文件或临时文件。

`bounded_map` 返回后聚合器处理完本轮事件，才允许 feedback/update；初始策略评测和最终独立患者评测仍按现有阶段顺序执行。预期单任务异常继续使用原有失败结果；取消不转成临床失败，清理后向调用者传播取消。

配置模板显式写 `patient_concurrency: 1`；使用者改为 5 可启用五患者并行，管理服务额度通过独立的 `MEDEVOLVE_MAX_INSTANCES` 设置，不在各方法配置里维护相互独立的容器上限。

`docs/comparison-experiments/common/SPEC.md:46` 同步为：查询与模拟提交复用共享查询实例，真实写入任务和 PhysicianBench 试写探索使用独立种子实例，保留现有患者边界与决策点限制。

- [ ] **4. 完成回归及不调用模型的真实环境检查。**

```bash
python -m pytest tests/experiment tests/integration -q
python -m pytest tests/method/test_method.py tests/method/test_workflow.py tests/method/test_selfgen.py -q
python -m pytest tests --ignore=tests/environment/test_live_image.py --ignore=tests/environment/test_medagentbench_live.py --ignore=tests/environment/test_shared_live.py -q
```

新增 `test_shared_live.py` 使用已有 live 测试的显式镜像选择方式，无配置时 skip。测试自建使用不可变本地镜像 ID 的池、设置额度 2，并在 `finally` 关闭全部会话及池：

- PhysicianBench：查询原始 Patient 列表取得五个不同患者，五个并发租约指向同一实例；每个查询结果属于对应患者。独立实例写入带唯一标记的 Communication，查询实例不出现该标记；新的独立实例也不出现它。试写探索实例用相同写入动作模拟，释放后查询实例仍干净。
- MedAgentBench：五个租约共用查询实例，执行原生 GET；用符合患者范围的 POST 提交一次请求，提交被接受但写前写后资源一致，其他任务记录不混入该提交。

```bash
MEDEVOLVE_TEST_IMAGE='<已加载的 PhysicianBench 不可变镜像 ID>' python -m pytest tests/environment/test_shared_live.py -k physicianbench -q
MEDEVOLVE_MEDAGENTBENCH_TEST_IMAGE='<已加载的 MedAgentBench 不可变镜像 ID>' python -m pytest tests/environment/test_shared_live.py -k medagentbench -q
```

实际执行时将镜像参数替换为本机 `docker image inspect` 确认的不可变 ID；不使用示例占位值运行。记录最大活动实例数、共享实例 ID、独立实例 ID 和释放结果。缺少镜像时保留测试为跳过，交付说明区分已验证与未验证项；不为完成检查调用未授权模型。

- [ ] **5. 提交本任务并交付验证结果。**

提交说明：`feat(experiment): 并行调度患者并保持轮次结果一致`。测试随所属行为提交；不单独拆分无法验证实现的测试提交。

## 实施检查点

按五项任务顺序执行。每项完成后核对其接口、测试和提交，再进入下一项；不同时改动依赖尚未确定的模块。

本次 main 完成标准是固定能力说明和执行要求覆盖 `medevolve`、`workflow`、`selfgen`，五患者查询或模拟提交可共享实例并行执行，真实写入及试写探索隔离，总容器数受管理服务额度限制，结果与清理保持正确，公共 SPEC 与实例复用规则一致。

### 合并后必做项验收

| 方法 | rebase 后及合并前必须完成 | 验证文件 |
| --- | --- | --- |
| `taskcraft` | 适配清单中的 Ledger 能力注入、三类候选执行要求、共享查询租约全部接通；分支主方法第 44 行重复语义不带回 main；`methods/common.py` 保留新公共校验，保留方法预算和只读生成约束 | `tests/method/test_taskcraft.py`：捕获 atomic/width/depth 与内置核验指令、检查候选字段和筛选输出；补充共享租约创建/释放断言 |
| `singleshot` | 删除 prompts 第 13–14 行语义副本；统一能力注入、执行要求、公共接口及实验入口校验；保留一次模型调用、自主选题、两题可同类型；与 Workflow 的 `max_questions/candidate_pool` 回归一起通过 | `tests/method/test_singleshot.py`、`tests/integration/test_singleshot_integration.py`，并回归 `tests/method/test_workflow.py` |
| `envexplore` | 两分支均按适配清单核对，合并所选版本时接通统一能力、执行要求和对应租约路径；v2 `GENERATION_RULES` 中的语义副本不带回 main；保留所选版本的探索次数、策略绑定和方法预算 | 原分支 `tests/method/test_envexplore.py`；v2 另含 `tests/method/test_envexplore_environment.py`、`tests/method/test_generation_shared.py` |

各分支验收均覆盖 PhysicianBench 与 MedAgentBench，检查所有相关生成阶段收到相同固定说明、每题执行要求经公共校验且内部筛选不丢字段；记录实际租约模式，不以生成阶段只读推定生成任务也是只读。六个方法均通过上述适配与验证后，才报告完整六方法覆盖；分支未合并时不将必做项标为已完成。
