# Harness 进化实验实施计划

> 执行方式：使用 executing-plans 按任务顺序实施；实施前使用 using-git-worktrees，从项目当前状态建立独立工作区。未经用户要求不派发子代理。

**目标：** 用 OpenClaw、Hermes Agent、ACE、Reflexion 分别从六种来源的合成任务中积累经验，在固定模型权重和独立原生 Benchmark 上比较进化收益。

**架构：** 每个“环境 × 基座 × 任务来源 × 重复种子”使用独立 Agent 容器与经验存储；每次任务尝试创建独立医疗数据实例。公共调度器向各基座发题，通过统一医疗工具服务记录执行，再将训练反馈送给各基座的学习机制。Benchmark 阶段使用只读经验快照和全新会话，评分反馈留在调度器。

**技术栈：** 现有 Python、Pydantic、FastAPI、httpx、Docker 与 pytest；OpenClaw 使用自身 Node.js 运行时，另外三个基座使用各自隔离的 Python 依赖。工具桥采用 HTTP JSON 和基座原生工具注册，不新增 MCP 服务或通用插件平台。

## 全局约束

- 依据 [实验设计](design.md)。合成任务不另切测试集；固定轮次后取最终经验状态，原生 Benchmark 不参与版本选择。
- 四个基座全部纳入实施，不增加单独的“两组任务试点”实验。组件级验收仍须证明隔离、经验更新和冻结测试正确。
- 不使用电脑上个人 OpenClaw、Hermes 的配置、技能、记忆、历史、登录缓存或服务端口。运行、暂停及清理只作用于本实验资源。
- 同一来源的冻结任务集供四个基座共用；每个基座的初始状态在不同来源之间保持一致。保留原生执行循环，不用公共固定解题器替代四者。
- 同组内任务按固定顺序进化，医疗数据每次尝试重置；最终测试每题从同一只读经验快照启动新会话。
- API 模型权重固定；执行和辅助模型显式指定，关闭备用模型。所有前台、反思与后台学习请求纳入预算和用量。
- 首版调度串行，每次仅保留一个活动 Agent 容器和一个解题医疗实例；评审所需数据实例顺序使用。二十四种组合是独立实验状态，不要求二十四个容器同时在线。
- 不依赖尚未实施的环境并发／共享租约方案，不修改其文件和接口。未来兼容变化集中在医疗工具桥中。
- 当前实际名称为 SelfRefine，配置键仍为 `selfgen`，以 `docs/comparison-experiments/baselines/selfgen/SPEC.md` 为准。
- 模型测试使用替代 API，不产生付费调用。真实联调只使用已授权的 DeepSeek V4.1 Flash 或 gpt-4o-mini，正式实验预算在执行前冻结。
- 每项改动独立验证并原子提交；提交 description 与 body 使用中文，不夹带其他未提交修改。

## 文件与职责

| 文件 | 职责 |
| --- | --- |
| `harnesses/contracts.py`、`harnesses/datasets.py`、`configs/harness-evolution.json` | 实验契约、已生成任务导入、数据冻结与配置 |
| `harnesses/isolation.py`、`harnesses/medical_bridge.py`、`harnesses/model_gateway.py` | Agent 容器、医疗工具会话、API 预算与记录 |
| `harnesses/adapters/{openclaw,hermes,ace,reflexion}.py`、`harnesses/workers/`、`docker/harnesses/` | 四个基座的启动、原生执行、学习、经验导出 |
| `evaluation/evaluator.py`、`harnesses/feedback.py` | 提取已有完成评价，评价实际 Harness 执行轨迹 |
| `harnesses/benchmarks/{common,physicianbench,medagentbench}.py` | 官方任务导入、原生判定与动作记录转换 |
| `experiments/harness_run.py`、`experiments/harness_runner.py`、`experiments/harness_summary.py` | 实验调度、产物与统计 |

全部新增 Python 目录包含 `__init__.py`。`pyproject.toml` 的包发现增加 `harnesses*`，四个基座依赖放在各自镜像内，不合并进 MedEvolve 主环境。

## 任务 1：冻结输入、上游版本和公共契约

**文件：** 新增 `harnesses/contracts.py`、`harnesses/datasets.py`、`configs/harness-evolution.json`、`docker/harnesses/upstreams.lock.json`、`tests/harness/test_datasets.py`；修改 `pyproject.toml`。

### 1.1 固定上游源码与数据来源

- [ ] 从 [OpenClaw](https://github.com/openclaw/openclaw)、[Hermes](https://github.com/NousResearch/hermes-agent)、[ACE](https://github.com/ace-agent/ace)、[Reflexion](https://github.com/noahshinn/reflexion) 的官方仓库取得源码，保留许可证。记录完整 Git SHA、镜像摘要、依赖锁文件和相关组件路径到 `upstreams.lock.json`；镜像构建禁止使用浮动 `main`、`latest`。
- [ ] 本机 OpenClaw 2026.9.2、Hermes v0.17.0 仅作为接口核对起点。读取锁定源码确认工具注册、程序入口、后台学习与冻结能力，镜像内重复验证；不自动升级个人安装。
- [ ] ACE 核对 `ace/core/reflector.py`、`ace/core/curator.py`、`playbook_utils.py`；Reflexion 核对 `hotpotqa_runs/agents.py` 和 `alfworld_runs/generate_reflections.py`。适配补丁存入 `docker/harnesses/patches/`，以差异文件保存，不把重写实现冒充官方源码。
- [ ] 任务来源固定为 `medevolve/taskcraft/workflow/singleshot/selfgen/envexplore`。读取已有 `Task` 与对应 `EvaluationResult` 文件；某个来源尚未生成或未核验时，输入预检返回缺失来源和原因，不自动出题或复制其他来源补齐。

### 1.2 定义契约

以下为新契约，引用现有 `Budget`、`ModelConfig`、`RunResult`、`SolverInput`、`Task`，不修改其既有字段。

```python
from pathlib import Path
from typing import Literal, Protocol
from pydantic import BaseModel
from contracts import Budget, ModelConfig, RunResult, SolverInput, Task

HarnessName = Literal["openclaw", "hermes", "ace", "reflexion"]

class GroupKey(BaseModel):
    run_id: str
    environment: str
    harness: HarnessName
    source_method: str
    seed: int

class HarnessLimits(BaseModel):
    execution: Budget
    learning: Budget
    max_input_tokens: int
    max_output_tokens: int
    max_total_tokens: int
    timeout_seconds: int
    learning_timeout_seconds: int

class HarnessStep(BaseModel):
    run: RunResult
    learning_runs: list[RunResult]
    learning_status: Literal[
        "updated", "unchanged", "not_triggered", "failed", "budget_exhausted"
    ]

class TrainingFeedback(BaseModel):
    task_id: str
    completed: bool | None
    failure_category: str | None
    reason: str
    reference_answer: str
    evidence: list[dict]

class Snapshot(BaseModel):
    path: Path
    digest: str

class HarnessAdapter(Protocol):
    async def start(self, group: GroupKey, initial: Snapshot) -> None: ...
    async def execute(self, task: SolverInput, tool_session: str) -> RunResult: ...
    async def learn(self, feedback: TrainingFeedback) -> HarnessStep: ...
    async def freeze(self, destination: Path) -> Snapshot: ...
    async def close(self) -> None: ...

def freeze_datasets(
    inputs: dict[str, Path], *, count: int, seed: int, destination: Path
) -> dict[str, list[Task]]: ...
```

适配器通过构造参数接收模型、预算、镜像锁和存储路径；执行调用不接收 `Task` 或 `TrainingFeedback`，避免参考答案进入解题上下文。`learn()` 不用于 Benchmark。预算字段均校验为正数；数据任务量、轮次、种子、参考答案策略及经验上限由配置显式提供。

### 1.3 冻结配置与数据

- [ ] `freeze_datasets` 只接受核验通过、环境一致、患者存在的任务；按照排序后的任务 ID 与固定种子抽样。请求数量超过最少有效产量时报错并列出各源产量，不静默改变已配置数量。
- [ ] manifest 保存输入文件 SHA256、完整任务 ID、抽样顺序、有效与无效数量、已有生成成本。四个基座只读取同一 manifest，不分别重抽。
- [ ] 冻结配置必须包含两环境各自的数据源、官方 Benchmark 源码及数据路径、六个任务来源、四个基座、模型、任务量、进化轮次、重复种子和预算；不设置会触发实际实验的隐式默认预算。配置初始为 `execute: false`，仅用于预检。
- [ ] 配置只保存 API key 的环境变量名称，运行时传入；不复制个人认证文件。运行目录格式为 `outputs/harness-evolution/<run_id>/<environment>/<harness>/<source_method>/<seed>/`。

### 1.4 验证与提交

添加包含一条有效题和一条无效题的 JSON 夹具，以及以下测试：

```python
def test_task_sampling_is_shared_across_harnesses(tmp_path, source_files):
    from harnesses.datasets import freeze_datasets
    first = freeze_datasets(source_files, count=1, seed=42,
                            destination=tmp_path / "a")
    second = freeze_datasets(source_files, count=1, seed=42,
                             destination=tmp_path / "b")
    assert first == second
    assert all(len(tasks) == 1 for tasks in first.values())
    assert all("invalid-task" not in {t.id for t in tasks}
               for tasks in first.values())
```

另测来源缺失、有效量不足、环境混用、没有核验结果时的明确报错；运行 `python -m pytest tests/harness/test_datasets.py -q`。实现前测试因模块缺失失败，实现后通过。提交：`feat(harness): 定义实验契约与冻结任务输入`。

## 任务 2：Agent 隔离、医疗工具桥和完成反馈

**文件：** 新增 `harnesses/isolation.py`、`harnesses/medical_bridge.py`、`harnesses/model_gateway.py`、`harnesses/feedback.py`、`tests/harness/test_isolation.py`、`tests/harness/test_medical_bridge.py`、`tests/harness/test_model_gateway.py`、`tests/harness/test_feedback.py`；修改 `evaluation/evaluator.py` 并扩展其原有测试。

### 2.1 固定两层隔离

- [ ] 每组从基座初始快照复制独立状态卷，创建独立容器、工作目录和会话。首次实验初始状态包含相同的基座随附资源与医疗工具说明，不包含个人资料或已经学习的医疗经验。
- [ ] OpenClaw 在容器内使用 `OPENCLAW_STATE_DIR=/state/openclaw`、`OPENCLAW_CONFIG_PATH=/state/openclaw/openclaw.json`；Hermes 使用 `HERMES_HOME=/state/hermes`。ACE、Reflexion 状态分别位于 `/state/ace`、`/state/reflexion`。这些是容器子进程配置，不修改宿主机的 HOME 或个人目录。
- [ ] Agent 容器不挂载宿主机个人目录、项目数据集、医疗数据库或 Docker socket。网络仅连公共工具桥和模型网关；医疗 FHIR 地址仅工具桥可见，不给 Agent 直接绕过患者边界访问数据库的路径。
- [ ] 每次 `execute()` 前调用既有 `create_environment()`，执行后读取状态并在 `finally` 释放；Reflexion 每次重试同样新建医疗实例。状态卷保留学习结果，环境实例不保留前题写入。
- [ ] 测试容器将经验文件以只读方式投影到基座预期位置；会话、缓存、日志使用新的可写临时目录。每题从同一快照重新开始，禁止将整个状态目录只读挂载后误伤正常日志写入。

容器操作使用 `asyncio.create_subprocess_exec` 参数数组，容器及卷带本运行 ID 标签；清理按记录的具体 ID 执行，不使用全局 prune。

### 2.2 医疗工具桥

```python
# harnesses/medical_bridge.py
async def open_tool_session(task: SolverInput, environment) -> str: ...
async def list_tools(session_id: str) -> list[dict]: ...
async def invoke_tool(session_id: str, name: str, arguments: dict) -> dict: ...
async def close_tool_session(session_id: str) -> dict: ...

# 内部路由；会话身份绑定到调用容器，不能仅凭 URL 中的 ID 换组。
# GET  /sessions/{id}/tools
# POST /sessions/{id}/calls   {"name": ..., "arguments": ...}
```

- [ ] PhysicianBench 复用 `get_tools()` 返回的名称、参数 schema 和原生执行函数，调用前继续执行现有患者限制。每次调用记录 `tool_start/tool_output`、顺序、耗时和写入状态，预算在桥端扣减。
- [ ] MedAgentBench 暴露 `medical_action(action: str)`，内部调用 `NativeTools.execute()`，保留 GET、POST、FINISH 原生文本与答案列表。POST 仍是接受但不持久化，不替换成 PhysicianBench 的写入工具；工具包装属于交互适配，在 manifest 标明。
- [ ] 会话关闭时，先保存任务需要的 `read_state()` 结果与动作记录，再释放实例。超时、取消、启动失败和重复关闭均不遗留本运行资源。

### 2.3 所有模型调用的预算与记录

- [ ] `model_gateway.py` 提供 OpenAI Chat Completions 兼容转发。四个基座的执行、辅助与后台学习客户端全部指向网关；网关转发到已授权 provider，并保留现有运行时所需会话头、用量映射和密钥脱敏行为。
- [ ] 每组、每任务、每阶段维护调用数及 token 额度。调用数与工具数在请求前硬限制，输出 token 按剩余额度限制；输入 token 计数需要匹配锁定模型的 tokenizer，有可靠计数时预留输入和输出额度。无法可靠计数时记录 token 上限为软停止条件，按实际 usage 结算并披露最后一次请求可能越界，不冒充严格 token 上限。缺失 usage 单列为未知，不能填零。
- [ ] 强制模型标识、阶段和会话归属；不允许备用模型绕过网关。锁定版本若不能把某个辅助请求指向网关，预检失败并说明具体组件，不能跳过计费后宣称预算一致。
- [ ] 确定性暂停后台学习后才能导出快照。超过学习超时或预算时停止本任务学习，保留已发生调用及失败状态，不后台继续更新最终经验。

### 2.4 提取完成评价，避免重复解题

现有 `evaluate_task()` 内部会调用固定解题器，不直接用于评价新的 Harness。提取其中的完成评价段，保留原入口行为和模型提示。

```python
# evaluation/evaluator.py
from contracts import CompletionReview, EvaluationConfig, PatientTrajectory

async def review_completion(
    task: Task, trajectory: PatientTrajectory, solver_run: RunResult,
    observed_state: dict, config: EvaluationConfig
) -> tuple[CompletionReview, RunResult]: ...

# harnesses/feedback.py
async def make_training_feedback(
    task: Task, trajectory: PatientTrajectory, run: RunResult,
    observed_state: dict, config: EvaluationConfig
) -> TrainingFeedback: ...
```

- [ ] 完成评价只使用真实 Harness 答案、工具轨迹和状态。核验与质量结果读取冻结任务集中的既有产物，不因换基座重新出题或重新试解。
- [ ] 评审进程持有参考答案和证据；完成反馈只在合成任务学习阶段送给基座。基础设施失败不触发临床失败经验更新。训练参考答案可以供学习角色读取，但不直接注入 Actor 的重试输入。
- [ ] 保持原 `evaluate_task` 的返回结构、调用顺序、失败分类和测试兼容，不改已有出题实验调度。

### 2.5 验证与提交

先写用替代容器与 FHIR 服务的失败测试，再实现以上接口。夹具验证：

```python
async def test_database_is_reset_but_group_memory_survives(rig):
    rig.group_state.write_text("查询前确认患者ID")
    first = await rig.open_session(patient="p1")
    await rig.write_message(first, "only-first-attempt")
    await rig.close_session(first)
    second = await rig.open_session(patient="p1")
    assert "only-first-attempt" not in await rig.read_messages(second)
    assert rig.group_state.read_text() == "查询前确认患者ID"
    await rig.close_session(second)
    assert rig.active_instances == 0
```

另测组 A 不能读取组 B 状态、跨患者工具调用被拒、模拟 POST 没有 HTTP 写入、答案未进入解题请求、预算耗尽后转发请求数不增加、取消后医疗实例释放。运行 `python -m pytest tests/harness/test_isolation.py tests/harness/test_medical_bridge.py tests/harness/test_model_gateway.py tests/harness/test_feedback.py tests/evaluation -q`。

按关注点提交：`feat(harness): 隔离实验状态与医疗工具会话`、`feat(harness): 记录并限制全部模型调用`、`refactor(evaluation): 提取实际执行结果完成评价`。

## 任务 3：四个基座的原生执行与经验更新

**文件：** 新增四个 `harnesses/adapters/` 模块、相应 `harnesses/workers/` 入口、`docker/harnesses/{openclaw,hermes,ace,reflexion}.Dockerfile`、`docker/harnesses/openclaw-tools/` 插件、`tests/harness/test_adapters.py`、`tests/harness/test_reflexion_memory.py`。

每个适配器实现任务 1 的 `HarnessAdapter`，统一收集 `RunResult`，不隐藏基座原生轨迹。各镜像使用任务 1 的不可变源码和依赖锁；构建过程不能读取个人配置。

### 3.1 OpenClaw

- [ ] 通过锁定版本的 `api.registerTool()` 注册医疗工具，插件清单同步声明 `contracts.tools`；工具仅转发任务 2 的组内会话，不创建第二套 FHIR 逻辑。
- [ ] 容器内启动持续运行的独立 Gateway，以 `openclaw agent --agent medical --session-id ... --message-file ... --json` 提交新题。不用可能省略后台学习的一次性本地调用替代 Gateway。
- [ ] 解题后，将公共训练评价作为该训练会话的反馈；允许原生 Skill Workshop 反思、生成与修订技能。逐条核对版本中前台结束、反馈轮次与后台学习的触发条件，等待确切的任务结束或未触发状态，不以“目录有文件”证明本次学习完成。
- [ ] 不为了触发学习强行增加模型轮次，不把反馈喂给测试题，不另写自定义 Reflector 替代原生学习。导出 Workshop 技能和允许持久化的经验文件，排除原始训练对话与答案缓存；未触发也作为有效运行记录。
- [ ] 冻结测试关闭自主学习、维护任务与写入工具，保留技能读取；每题刷新只读快照和会话目录。所有需要写状态的组件重定向到独立临时目录，不能把 writable 副本作为跨测试题经验。

### 3.2 Hermes Agent

- [ ] worker 通过 `AIAgent(..., base_url=model_gateway, save_trajectories=True)` 和 `run_conversation()` 执行；医疗工具使用锁定版本 `tools.registry.registry.register()` 注册到独立 `medical` toolset。注册代码只放在实验镜像中。
- [ ] 开启技能、记忆读取和必要的原生管理工具，关闭与医疗实验无关的消息发送、浏览器和个人外部服务；每题新的 Agent 会话，状态卷在组内保留。
- [ ] 公共反馈回到当前训练会话，供原生 `skill_manage` 和记忆机制积累经验；导出真实变更与辅助调用记录。后台审查按锁定版本的可用机制执行，不假定当前官网功能全部存在于所选版本。
- [ ] 冻结阶段禁用 `skill_manage` 写入、memory 写入、后台 review/curator，保留 `skill_view` 与记忆读取。只读挂载经验文件，日志与数据库会话记录使用临时存储；不设置 `skip_memory=True`。
- [ ] 从经验库读取之外的跨会话回查在 Benchmark 阶段关闭，避免测试题读到历史训练答案或其他测试题；导出范围包含最终技能和合规经验，不包含原始训练会话数据库。

### 3.3 ACE

- [ ] Generator 使用 ACE 的上下文与策略手册结构接入医疗工具调用；工具交互所需执行循环属于医疗适配，保存补丁。不能仅借用 ACE 名称调用现有公共 solver。
- [ ] 将真实轨迹、答案、公共完成反馈传给官方 `Reflector.reflect(question, reasoning_trace, predicted_answer, ground_truth, environment_feedback, bullets_used, ...)`，保存反思、条目标签和调用信息。
- [ ] 调用官方 `Curator.curate(current_playbook, recent_reflection, question_context, current_step, total_samples, token_budget, playbook_stats, ...)` 和官方手册合并逻辑，保存最终手册、条目 ID 和更新记录。
- [ ] 公共调度按配置轮次调用组件，不走依赖验证集并挑选 `best_playbook` 的离线总入口。固定最终手册参与 Benchmark，无验证集选择阶段。
- [ ] 冻结测试只运行 Generator；Reflector、Curator 和条目使用计数写入均禁用，空手册初始基线与最终手册使用同一执行循环和预算。

### 3.4 Reflexion

- [ ] 从官方 Actor 与反思提示适配医疗工具；每题先执行，失败后根据反馈在预算内反思和重试，每次重试重新创建医疗数据实例。
- [ ] 训练参考答案留在公共评审和反思角色，Actor 只获得文字反思和工具结果。原始反思作为本题尝试记录保存，不能直接变成跨患者答案记忆。
- [ ] 跨任务经验提炼是独立适配调用，计入学习预算。输出固定为 `task_type/tool_names/condition/procedure/pitfall/source_task_id`；持久库不包含患者 ID、个体临床值或参考答案。原始反思与可迁移经验分开保存。
- [ ] 经验选择使用非模型的固定排序：先任务类型相同，再按工具名称交集数量降序，再按经验 ID 排序；读取最多三条，在配置的经验上下文上限内依次加入，超出上限的条目不加入。Benchmark 工具类型从官方任务元数据或相同工具目录得到，禁止用参考答案推断。
- [ ] Benchmark 不生成新反思、不根据评分重试；只读取冻结经验，输出一次正常任务执行结果。运行清单标识 `reflexion_cross_task`，说明这一迁移设置不同于原版同题重试实验。

### 3.5 验证与提交

各适配器依赖注入替代模型和工具桥。`adapter_factory` 测试夹具按四个基座分别创建适配器，`clean_snapshot` 为同一基座无经验状态；公共测试覆盖：

```python
import pytest
from contracts import SolverInput

@pytest.mark.parametrize("name", ["openclaw", "hermes", "ace", "reflexion"])
async def test_frozen_adapter_reads_without_learning(
    name, adapter_factory, clean_snapshot, group_factory, bridge
):
    adapter = adapter_factory(name, frozen=True)
    await adapter.start(group_factory(name), clean_snapshot)
    before = await adapter.freeze(bridge.temp_dir / "before")
    result = await adapter.execute(
        SolverInput(instruction="查询患者已有用药", patient_id="p1"),
        bridge.session_id,
    )
    after = await adapter.freeze(bridge.temp_dir / "after")
    assert result.status == "completed"
    assert before.digest == after.digest
    assert bridge.learning_model_calls == 0
    await adapter.close()
```

新增具体测试：OpenClaw 后台未触发保持 `not_triggered`；Hermes 冻结时记忆读取仍生效；ACE 两条反馈产生官方 delta 更新且不触发验证集选择；Reflexion 同题重试重置环境、跨任务选择与排序确定、第二位患者不读取第一位患者答案。运行 `python -m pytest tests/harness/test_adapters.py tests/harness/test_reflexion_memory.py -q`。

四个基座分别提交：`feat(harness): 接入OpenClaw原生技能学习`、`feat(harness): 接入Hermes原生技能与记忆`、`feat(harness): 接入ACE策略手册进化`、`feat(harness): 接入Reflexion跨任务经验迁移`。

## 任务 4：原生 Benchmark 任务与评分器

**文件：** 新增 `harnesses/benchmarks/common.py`、`harnesses/benchmarks/physicianbench.py`、`harnesses/benchmarks/medagentbench.py`、`tests/harness/test_benchmarks.py`。

### 4.1 导入与接口

```python
from pathlib import Path
from pydantic import BaseModel
from contracts import RunResult, SolverInput

class BenchmarkCase(BaseModel):
    id: str
    solver_input: SolverInput
    native_case_path: Path

class NativeScore(BaseModel):
    case_id: str
    passed: bool | None
    status: str
    raw: dict

def load_cases(environment: str, source: Path) -> list[BenchmarkCase]: ...
async def score_case(
    case: BenchmarkCase, run: RunResult, actions: list[dict], endpoint: str
) -> NativeScore: ...
```

- [ ] 根据各环境官方任务文件导入 instruction、context、患者映射与任务 ID，必要背景拼入解题题面；原生参考解答和评分参数只留在评审进程。固定官方测试划分、任务 ID 顺序和数据哈希。
- [ ] PhysicianBench 沿当前工具来源提交 `c7efa8fd5b1e4744ada50668efe4b7e84023cbb0` 核对任务与官方评分入口；工具来源不等于完整 Benchmark 已可用，预检独立检查任务、种子、参考评分器和许可证。
- [ ] MedAgentBench 固定提交 `99260117137b09f04837a8c18d18a1107efa55ae`，复用 `src/server/tasks/medagentbench/eval.py` 的 `eval(case_data, results, fhir_api_base)` 和对应官方 `refsol.py`。源码文档要求另行获取的参考函数必须实际存在才允许正式运行。
- [ ] 缺数据、缺官方评分器或无法映射所需患者时，报 `benchmark_unavailable` 并停止该环境正式实验；不能改用模型判分冒充原生 Benchmark。

### 4.2 原生判定

- [ ] PhysicianBench 判定所需写入状态在当前执行实例上读取；MedAgentBench `results` 按官方格式从 GET、POST、FINISH 记录构造，保留模拟提交语义。评分器运行结束后释放当前实例，不能在重置后评分。
- [ ] 保存原生判定输入、原始判定输出和评分器版本。评分发生在 Agent 退出之后，结果不回传、不存入经验卷；Benchmark 不调用 `make_training_feedback()`。
- [ ] 单题运行故障按官方协议进入统计，并单列故障类别；评分器本身故障单列为 `scoring_error`，不能把异常误当普通未完成成绩发布。

### 4.3 验证与提交

先以官方公开样例及可验证的人工动作夹具验证正确／错误答案和正确／错误提交。测试片段：

```python
async def test_medagent_score_uses_native_submissions(
    benchmark_rig, native_case, good_run, bad_run
):
    good = await benchmark_rig.score(native_case, good_run)
    bad = await benchmark_rig.score(native_case, bad_run)
    assert good.passed is True
    assert bad.passed is False
    assert benchmark_rig.completion_llm_calls == 0
    assert benchmark_rig.agent_feedback_messages == []
```

另测答案隔离、评分前实例未释放、未知任务类型报错、缺 `refsol.py` 明确失败；没有官方资产时相关真实评分检查报告未验证，不伪造通过。运行 `python -m pytest tests/harness/test_benchmarks.py -q`。提交：`feat(harness): 接入原生Benchmark任务与判定`。

## 任务 5：完整调度、统计与四基座验收

**文件：** 新增 `experiments/harness_run.py`、`experiments/harness_runner.py`、`experiments/harness_summary.py`、`tests/harness/test_runner.py`、`tests/harness/test_summary.py`、`tests/harness/test_live_harnesses.py`；更新本计划的完成勾选和 `design.md` 的已确认隔离规则。

### 5.1 公共执行流程

```python
# experiments/harness_runner.py
async def run_harness_experiment(config_path: str) -> None: ...

# 每个环境、基座、重复种子：
# 1. 从初始状态进行 baseline Benchmark，关闭学习。
# 2. 每种任务来源从相同初始状态启动独立组。
# 3. 按冻结任务顺序执行 -> 完成评审 -> learn，持续到固定轮次/预算。
# 4. freeze 最终经验并关闭训练容器。
# 5. 每个 Benchmark 任务创建全新测试容器与医疗实例，加载同一只读快照。
# 6. 运行原生评分器，保存结果，关闭本题容器与医疗实例。
```

- [ ] baseline 的空经验状态与进化状态分别保存摘要；同一基座的初始资源、工具说明、模型和测试预算完全相同。每次重复从独立初始卷开始，不沿用上次经验。
- [ ] `start/execute/learn/freeze/close` 均有明确状态记录，`try/finally` 清理 Agent、医疗实例和工具会话；预算耗尽保存部分进化状态与真实训练量，不能悄悄追加任务。
- [ ] 前台消息与学习反馈分阶段保存，初始状态、任务 manifest、源代码锁和配置哈希组成运行身份。中断后不自动重放可能已执行的模型请求或医疗写入；本版不实现复杂恢复系统。
- [ ] 每题结束原子保存日志与结果；只有四个基座的接入验收全部通过、任务来源齐全、官方 Benchmark 可用、`execute=true` 且预算已冻结，才允许启动完整矩阵。

### 5.2 汇总

- [ ] 按环境、基座、任务来源、重复种子输出 baseline 成绩、最终成绩、绝对提升；成功率差值使用百分点。另保存每题通过状态和原生分母，禁止只统计成功或正常返回的任务。
- [ ] 聚合重复均值和标准差，按相同患者进行配对 bootstrap 得到提升的 95% 区间；无患者分组的任务按任务 ID 配对。重采样种子固定为 42，重复 1000 次；样本不足两组时区间留空并说明原因。
- [ ] 保留任务质量、有效产量、实际训练量、经验更新与未触发次数、失败类别。生成成本从输入产物读取并单列，不因为同一数据被四基座复用而重复声称生成四次。
- [ ] 执行、学习、完成评审、Benchmark 推理分阶段报告调用、输入／输出 token 与金额；缺价格或 usage 分别标识未知，不能用零补齐。无需新增图表、页面或独立报告系统。

### 5.3 验证

先写替代四基座、替代医疗实例和原生评分器的完整流程测试：

```python
async def test_all_groups_start_clean_and_test_without_feedback(matrix_rig):
    await matrix_rig.run(methods=6, harnesses=4, seeds=[42])
    assert len(matrix_rig.training_groups) == 24
    assert len(matrix_rig.baselines) == 4
    assert len(set(matrix_rig.state_paths)) == 24
    assert all(group.initial_digest == group.baseline_digest
               for group in matrix_rig.training_groups)
    assert matrix_rig.benchmark_learning_calls == 0
    assert matrix_rig.cross_group_reads == 0
    assert matrix_rig.active_containers == 0
```

另测每题会话重置、环境重置、快照只读、超时清理、任务数量不足阻断、baseline 与进化结果配对、故障进入分母及成本未知情况。运行：

```bash
python -m pytest tests/harness -q -m 'not live'
python -m pytest tests/evaluation tests/environment tests/experiment tests/integration -q -m 'not live'
python -m experiments.harness_run --config configs/harness-evolution.json --check
```

`--check` 只验证数据、镜像、版本、预算和评分器，不创建付费模型请求；调用 Docker inspect 可核对已有资源，不自动拉取环境镜像。

真实接入验收针对四个基座分别完成工具调用、原生学习状态、保存重载与冻结执行；用公开或人工可核对任务验证机制，不将这些验收样例作为额外对比实验或正式收益证据。直接调用命令为：

```bash
MEDEVOLVE_HARNESS_LIVE=1 python -m pytest tests/harness/test_live_harnesses.py -q
```

仅在镜像、官方数据、授权模型和明确的验收预算齐备时运行；缺项返回未验证及具体原因。验收后按冻结正式配置运行二十四组组合及四个基线，每个环境和重复种子分别执行。真实结果不通过时定位所在基座和失败机制，不扩大任务或模型范围。

### 5.4 提交与交付

提交：`feat(experiment): 调度四基座进化与冻结测试`、`feat(experiment): 汇总Benchmark提升与运行成本`。交付可运行配置、镜像和源码锁、冻结任务 manifest、各基座经验快照、真实执行记录、原生 Benchmark 结果及汇总文件。

实施完成条件是四个基座全部可在独立状态下执行、学习、冻结与测试；模型与工具预算可追踪，测试评分不会回流，容器清理和患者边界通过验收。仅完成替代运行器测试时，不声明真实实验完成。
