# Harness 进化实验实施计划

> 执行方式：使用 executing-plans 按任务顺序实施。工作区为当前 `codex/harness-evolution` worktree；每项改动独立验证、原子提交，未经用户要求不派发子代理。

**目标：** 接入 OpenClaw、Hermes Agent、ACE、Reflexion，以训练和测试两条命令完成合成任务驱动的经验进化与迁移评价。

**架构：** 四个基座共用任务读取、医疗工具桥、完成评审和结果保存，各自使用原生程序与学习机制。每组独立目录和进程、每题新会话，模型直连配置的 API；训练后保存经验，测试逐题复制临时目录并丢弃。

**技术栈：** 现有 Python、Pydantic、FastAPI、httpx、uvicorn 与 pytest；OpenClaw 使用固定版本 Node.js 程序，Hermes、ACE、Reflexion 使用各自 Python 依赖环境。

## 全局约束

1. 以 [实验设计](design.md) 为准：两环境、六任务来源、四基座，种子 42、重复 1 次，各组串行运行。
2. 每来源抽样后按任务数 70/30 划分；训练数量向下取整，两集合非空、ID 不重叠，四基座使用相同划分。
3. 每道训练题新会话，同题执行与反馈可以共用会话；合成任务只读，原生 Benchmark 接入时按其原生要求开放工具。每次尝试新医疗实例，执行、评审或评分结束后才释放，逐题实例不进入生成流程共享实例池。
4. 固定程序版本、模型和原生限制；公共评审复用 `runtime/context.py` 的输入预算与完整记录选取，原生 Agent 内部调用由自身限制。执行、学习、评审及测试用量分开记录，缺少 usage 时标记未知。后台调用限制不声明为统一硬预算。
5. 保留已有未提交修改。替代模型测试不产生付费请求；真实验收及正式实验只使用已授权模型和预算。四基座真实接入全部通过后才启动完整实验。

## 文件与职责

| 文件 | 职责 |
| --- | --- |
| `contracts.py`、`harnesses/contracts.py`、`harnesses/datasets.py`、`configs/harness-evolution.json` | 项目共享数据契约、Harness 接口、任务划分和本地程序配置 |
| `harnesses/local.py`、`harnesses/snapshots.py` | 本地子进程、初始目录、测试临时目录、经验保存 |
| `harnesses/runners/{openclaw,hermes,ace,reflexion}.py` | 四份原生运行脚本，执行、学习和经验导出 |
| `harnesses/medical_bridge.py`、`harnesses/feedback.py` | 工具会话与完成评审复用 |
| `experiments/harness_train.py`、`experiments/harness_test.py`、`experiments/harness_summary.py` | 两条命令、串行调度与结果汇总 |

新增 Python 包包含 `__init__.py`，沿用 `pyproject.toml` 的 `harnesses*` 包发现。各基座依赖安装在各自环境，不合并到项目依赖中。工具桥绑定本地地址，使用现有调用身份和患者限制。

实施前在当前 worktree 同步 main 的共享层，保留已提交方案和未提交修改，并验证合并后的共享层测试。共享 Pydantic 数据结构定义在根目录 `contracts.py`，runner 使用普通名称字典分派；新增测试的替代依赖在各文件内定义，异步测试显式标注 `@pytest.mark.asyncio`。训练与测试命令的测试放在 `tests/experiment/`。

## 任务 1：任务划分与本地运行配置

**文件：** 修改 `contracts.py`、`harnesses/contracts.py`、`harnesses/datasets.py`、`configs/harness-evolution.json`；新增 `harnesses/local.py`、`tests/harness/test_splits.py`、`tests/harness/test_local.py`。

**接口：** 保留 `freeze_datasets()` 和 `load_frozen()`；增加 `split_tasks()`。`TaskSplit` 等共享数据模型定义在根目录 `contracts.py`，复用 Harness 已有数据字段和接口约定。四个本地 runner 提供 `start/execute/learn/freeze/close`，构造参数接收配置、组目录和工具桥地址。

```python
from pathlib import Path
from contracts import Task
from pydantic import BaseModel

class TaskSplit(BaseModel):
    train: list[Task]
    test: list[Task]

def split_tasks(tasks: list[Task], *, seed: int = 42) -> TaskSplit:
    import random
    ordered = sorted(tasks, key=lambda task: task.id)
    if len({task.id for task in ordered}) != len(ordered):
        raise ValueError("任务 ID 重复")
    random.Random(seed).shuffle(ordered)
    n_train = len(ordered) * 7 // 10
    if n_train == 0 or n_train == len(ordered):
        raise ValueError("训练集和测试集均须非空")
    return TaskSplit(train=ordered[:n_train], test=ordered[n_train:])
```

- [ ] 写测试：10题得到7/3；改变输入顺序不改变划分；训练与测试ID互斥；单题报错；不同基座读取相同清单。执行 `python -m pytest tests/harness/test_splits.py -q`，确认新增功能尚未实现时失败。
- [ ] 按 main 的新版 `Task`、`EvaluationResult` 解析输入，保留四项质量评分及难度；仅接受核验有效且 `read_only_task=True` 的任务，范围不匹配保留 `unsupported_operation`，缺只读核验的历史任务报告需重新核验。不兼容的旧质量结构明确报错，不静默降格。实现划分，manifest 保存比例、ID、患者重叠和源文件摘要；跨集合完全重复的题目时报出ID并停止该组，近似变体纳入数据验收。
- [ ] 配置保存每个基座程序路径、上游提交、Python路径、初始状态模板和实验目录；沿用已授权模型配置及 `execute: false`。任务数须满足非空划分，单题夹具仍可用于组件测试。
- [ ] 实现本地子进程与目录操作。子进程单独传入环境变量，保留宿主机 HOME；原始快照保存组内经验，测试复制到 `TemporaryDirectory`，以 `finally` 清理；导出明确包含的经验路径，排除原始会话和答案缓存。
- [ ] 运行划分、本地目录及现有数据测试。提交：`feat(harness): 支持任务划分与本地实验目录`。

划分测试直接使用已有任务夹具：

```python
import json
from pathlib import Path
from contracts import Task
from harnesses.datasets import split_tasks

def test_split_is_seven_three_and_input_order_independent():
    fixture = Path(__file__).parent / "fixtures/frozen_source/tasks.json"
    template = Task.model_validate(json.loads(fixture.read_text())[0])
    tasks = [template.model_copy(update={"id": f"t-{i}"}) for i in range(10)]
    first = split_tasks(tasks, seed=42)
    second = split_tasks(list(reversed(tasks)), seed=42)
    assert first == second
    assert (len(first.train), len(first.test)) == (7, 3)
    assert not {t.id for t in first.train} & {t.id for t in first.test}
```

本地进程调用示例：

```python
process = await asyncio.create_subprocess_exec(
    *command, cwd=str(group_dir), env={**os.environ, **group_env},
    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
)
try:
    stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
except (asyncio.TimeoutError, asyncio.CancelledError):
    process.kill()
    await process.communicate()
    raise
```

该代码放入 `harnesses/local.py` 的 `run_command(command, *, group_dir, group_env, timeout)`，返回退出码、stdout、stderr；报错不打印密钥。对于持久 Gateway，另保存该子进程对象并在关闭时等待其退出，只清理实验进程。

## 任务 2：OpenClaw 与 Hermes 本地接入

**文件：** 新增 `harnesses/runners/__init__.py`、`harnesses/runners/openclaw.py`、`harnesses/runners/hermes.py`、`tests/harness/test_local_openclaw.py`、`tests/harness/test_local_hermes.py`；修改 `harnesses/medical_bridge.py`。复用 `docker/harnesses/openclaw-tools/` 中的医疗工具插件源码。

**接口：** `create_runner(name, *, config, group_dir, bridge_url)` 返回实现现有 `HarnessAdapter` 协议的本地 runner。各 runner 的 `execute(task, tool_session)` 返回真实原生答案和轨迹；`learn(feedback)` 返回实际学习状态；`freeze(destination)` 导出可重新加载的经验。

新增仅供宿主调度使用的 `ToolSessions.inspect(session_id, *, client_id)`，返回已校验身份的 `ToolSession`，提供同一实例、动作和事件供评审读取；不增加向 Agent 暴露 FHIR 地址的 HTTP 接口。评分仍在实例关闭前进行。

`ToolSessions.open(..., read_only: bool)` 显式接收工具权限并转发给 `get_tools()`；训练及合成测试传 `True`，官方 Benchmark 按原生要求传 `False`。MedAgentBench 复用 `ReadOnlyNativeTools`，只读会话的工具说明只列 GET、FINISH，POST 明确拒绝；PhysicianBench 只读会话不暴露创建工具。边界测试同时覆盖只读拒绝和原生任务所需写入可用。

- [ ] 写 OpenClaw 测试：两道训练题会话不同，同题反馈沿用本题会话；输出技能保存重载后可读；个人配置路径和服务未被使用。写 Hermes 测试：两题新会话、组内记忆共享、测试关闭写入但保留读取。运行两份新增测试确认失败。
- [ ] OpenClaw 直接启动固定版本 Gateway，显式传入本组配置、状态和工作目录，选择与个人服务无冲突的实验端口；保留子进程句柄。复用医疗插件，把本题工具会话写入本组状态目录，使用 `openclaw agent --agent medical --session-id ... --message-file ... --json` 执行。provider 的 base URL 直接取模型配置。
- [ ] OpenClaw 每题生成新会话ID，保存该ID供反馈回合使用。解析原生响应中的答案文本、模型用量、工具信息和错误，转换成 `RunResult`；CLI 标准输出的日志不冒充答案。学习按锁定版本实际触发条件等待对应任务结束，历史 Workshop 记录不作为当前任务结束证据。
- [ ] Hermes 在指定 Python 环境的独立子进程中、导入前设置 `HERMES_HOME`。按锁定源码使用 `AIAgent`、`run_conversation()` 与工具注册接口，将医疗调用转发到工具桥；同题反馈送回本题会话，下一题新建 Agent。冻结状态保留技能与记忆读取，关闭管理写入与后台整理，导出技能及允许迁移的记忆文件。
- [ ] 运行上述测试及 `tests/harness/test_medical_bridge.py`，各自提交：`feat(harness): 接入OpenClaw本地训练与冻结测试`、`feat(harness): 接入Hermes本地训练与冻结测试`。

OpenClaw、Hermes runner 提供只读属性 `current_session_id`，会话选择必须满足：

```python
first = runner.current_session_id
await runner.learn(feedback)
assert runner.current_session_id == first
await runner.execute(next_task, next_tool_session)
assert runner.current_session_id != first
```

其中 `feedback` 为已完成本题执行的 `TrainingFeedback`，`next_task` 为另一道 `SolverInput`，工具会话由 `ToolSessions.open()` 创建。runner 的 `learn()` 遇到 `completed=None` 时只记录故障，不更新临床经验。

## 任务 3：ACE 与 Reflexion 官方组件接入

**文件：** 新增 `harnesses/runners/ace.py`、`harnesses/runners/reflexion.py`、`tests/harness/test_local_ace.py`、`tests/harness/test_local_reflexion.py`；修改 `harnesses/runners/__init__.py` 的四项名称映射。

**接口：** 同任务2的本地 runner 接口；独立 Python 环境执行官方组件，模型直接使用配置中的 API。保存上游 SHA、依赖版本和医疗适配差异；执行时先检查官方组件实际存在。

- [ ] 写 ACE 测试：两条真实反馈调用官方反思、策展和合并逻辑，最终手册重载有效；冻结执行不调用学习组件。写 Reflexion 测试：临床失败后反思重试，每次新医疗实例；下一位患者仅获得可迁移经验；冻结执行不反思或按评分重试。运行新增测试确认失败。
- [ ] ACE 复用官方 Generator 的上下文与策略手册结构接入医疗工具循环；将真实轨迹、答案、参考答案和完成反馈传给官方 `Reflector.reflect()`，再调用 `Curator.curate()` 与官方手册合并逻辑。按锁定源码的参数名调用，使用固定轮次最终手册。
- [ ] Reflexion 复用官方 Actor 和文字反思提示。失败后先完成本次评审，再释放实例；在固定重试次数内反思并重新创建实例。原始参考答案不传入 Actor 重试输入；跨任务经验提炼的调用单独记录为学习成本。
- [ ] 跨任务经验固定字段为 `task_type/tool_names/condition/procedure/pitfall/source_task_id`，不存患者标识、个体临床值或逐题答案。选择顺序固定为任务类型相同优先、工具名称交集数量降序、经验ID升序，最多三条并受配置的上下文上限限制。训练元数据从任务清单读取，测试元数据从公开题目元数据和工具目录读取；无任务类型时按空类型处理，不从答案推断。runner 构造配置中的 `task_metadata` 保存按题目与患者生成的稳定键到类型和工具名的映射，`execute()` 仍只接收 `SolverInput`。
- [ ] 运行两份新增测试，提交：`feat(harness): 接入ACE官方策略更新组件`、`feat(harness): 接入Reflexion反思与跨任务经验`。

冻结执行的组件调用断言：

```python
await runner.execute(test_input, tool_session)
assert native_calls["reflect"] == 0
assert native_calls["curate"] == 0
assert directory_digest(final_snapshot.path) == final_snapshot.digest
```

`native_calls` 为测试注入官方组件的调用记录；训练测试也须断言其大于零，不能通过完全跳过官方组件取得通过结果。

## 任务 4：生成任务测试评分

**文件：** 新增 `tests/harness/test_scoring.py`；复用已提交的 `evaluation/evaluator.py:review_completion()` 与 `harnesses/feedback.py`，本轮不新增评分模块。

**接口：** 合成任务测试直接调用公共完成评审，参考答案只留在评审进程：

```python
review, _ = await review_completion(
    task, trajectory, run, observed_state, config, record=result.runs.append,
)
```

- [ ] 写评分测试：正确与错误的答案得到相应完成判定；评分完成前医疗实例未释放；评审故障单列为 `scoring_error` 且不写入经验库。运行 `python -m pytest tests/harness/test_scoring.py -q` 确认失败。
- [ ] 复用 `review_completion(task, trajectory, solver_run, observed_state, config, *, record=None)`，保留现有完成判定材料、只读能力说明与失败语义；测试只评价真实 runner 的答案和查询轨迹，不重新解题、不以数据库写入作为完成条件。评审通过 `runtime.call_llm()` 自动应用公共输入预算，记录材料缩减情况，原始材料保留。
- [ ] 运行评分测试和现有评价测试，提交：`feat(harness): 接入合成任务测试评分`。

医疗实例生命周期在测试中必须满足：

```python
session_id = await registry.open(
    solver_input, environment, client_id=client_id, read_only=True
)
try:
    context = registry.inspect(session_id, client_id=client_id)
    run = await runner.execute(solver_input, session_id)
    review, _ = await review_completion(
        task, trajectory, run, observed_state, config, record=result.runs.append,
    )
finally:
    await registry.close(session_id, client_id=client_id)
```

**后续扩展（本轮不实施）：** 原生 Benchmark 评分按以下约定接入，官方提交号与评分入口已记录在 `configs/harness-evolution.json`：

```python
class BenchmarkCase(BaseModel):
    id: str
    solver_input: SolverInput
    native_data: dict

class NativeScore(BaseModel):
    case_id: str
    passed: bool | None
    status: str
    raw: dict

def load_cases(environment: str, source: Path) -> list[BenchmarkCase]:
    """读取官方题目、患者映射及评分材料。"""

async def score_case(case: BenchmarkCase, run: RunResult, *,
                     actions: list[dict], endpoint: str,
                     output_dir: Path, config: dict) -> NativeScore:
    """在仍存活的本题医疗实例上运行官方原生评分。"""
```

PhysicianBench 使用官方 pytest 入口（`JOB_DIR`、指向当前实例的 `FHIR_BASE_URL`、`trajectory.log`），MedAgentBench 使用官方 `eval(case_data, results, fhir_api_base)` 与 `refsol.py`；官方资产缺失时报 `benchmark_unavailable`，不使用公共模型评分代替原生分数。

## 任务 5：训练、测试命令与四基座验收

**文件：** 新增 `experiments/harness_train.py`、`experiments/harness_test.py`、`experiments/harness_summary.py`、`tests/experiment/test_harness_local.py`、`tests/harness/test_live_harnesses.py`；更新 `configs/harness-evolution.json`。

**接口：** 两个命令均支持 `--config --environment --harness --source --check`。`--check` 只验证本组合的数据、版本、程序、目录和评分器；实际运行需要配置 `execute: true` 及已授权模型和预算。未指定组合时依照配置顺序串行调用相同流程，不引入并行调度。

```bash
python -m experiments.harness_train --config configs/harness-evolution.json \
  --environment PhysicianBench --harness openclaw --source medevolve
python -m experiments.harness_test --config configs/harness-evolution.json \
  --environment PhysicianBench --harness openclaw --source medevolve
```

- [ ] 写贯通测试，用四种替代原生组件运行10题7/3划分，断言仅7题反馈进入学习；测试分别读取初始状态与最终经验，3题测试反馈不进入学习。断言新题会话不同、题间临时目录不同、评分后释放实例、快照不变。
- [ ] 训练入口读取或创建共同冻结清单，准备独立组目录，按轮次执行训练题。每次尝试以 `read_only=True` 创建独立医疗实例，不借用生成流程的共享实例；通过 `prepare_trajectory(context.instance, patient_id)` 保存原始患者材料。执行后收集状态和真实工具轨迹，调用 `make_training_feedback()`；评审应用公共输入预算并记录缩减信息。学习结束后关闭工具会话，故障跳过学习并记录，所有清理置于 `finally`。
- [ ] 测试入口分别运行初始状态和最终状态；每题复制经验到独立临时目录并启动新会话，关闭学习。合成测试使用任务 4 的公共完成评审；逐题保存原始结果，关闭进程与实例后丢弃目录。经验导出与重新加载的目录布局必须一致。
- [ ] 汇总按环境、基座、来源分别保存测试成绩、基线、绝对提升、逐题分母和失败类型；按患者配对 bootstrap 1000次，种子42，小于两组时区间为空。成本按生成、执行、学习、评审及测试分开记录，缺价格或用量标记未知，生成成本按任务来源仅计一次。
- [ ] 运行替代测试及四基座真实接入验收，独立提交命令与汇总改动：`feat(experiment): 提供Harness训练和测试命令`、`feat(experiment): 汇总测试成绩与成本`。真实验收缺程序、数据、授权或预算时报告未验证，不产生付费请求。

正式运行前的验证命令：

```bash
python -m pytest tests/harness tests/experiment/test_harness_local.py tests/evaluation tests/environment -q
python -m experiments.harness_train --config configs/harness-evolution.json --check
python -m experiments.harness_test --config configs/harness-evolution.json --check
```

真实验收由 `MEDEVOLVE_HARNESS_LIVE=1` 门控，默认跳过，并逐基座检查程序版本、医疗调用、学习状态、保存重载、冻结读取和目录独立性。仅在明确授权后执行：

```bash
MEDEVOLVE_HARNESS_LIVE=1 python -m pytest tests/harness/test_live_harnesses.py -q
```

**完成条件：** 四个基座均能通过训练命令积累经验，通过测试命令评价生成任务测试集；原始记录、成本与评分可追溯，测试不更新最终经验，医疗实例在评分结束后释放。交付配置、版本记录、任务划分、经验快照和测试结果文件。
