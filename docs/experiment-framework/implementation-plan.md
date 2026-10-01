# MedEvolve Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在独立项目中运行医疗环境探索、任务生成、公共评测和出题策略进化实验。

**Architecture:** 实验入口在本机运行，远程服务管理独立 Docker 环境。标准 Agent 使用 OpenAI Agents SDK，方法模块组织各角色并更新出题指导与生成配比，评测模块保持固定。

**Tech Stack:** Python、OpenAI Agents SDK、Pydantic、httpx、FastAPI、Docker；配置和结果使用 JSON 文件。

## Global Constraints

- 项目根目录为 `/Users/guagua/code/MedEvolve`，实施使用独立 Git worktree。
- 实验与模型调用在本机，环境与数据在远程；环境工具直接访问分配的实例地址。
- 每次探索、核验和解题使用独立初始状态；容器数据存储相互隔离。
- 进化对象为出题指导及生成配比，固定模型权重、公共评测和解题器配置。
- 先完成主方法；五个对比方法在各自接入阶段依据原始来源实现。
- 不增加训练依赖、数据库、工作流引擎或自定义模型工具调用循环。
- 模型名称、连接配置、环境镜像和初始数据通过实验配置指定；真实模型调用前确认模型与成本。

## 文件与接口

```text
configs/experiment.json           环境、角色模型、预算、患者分组和轮数
runtime/schemas.py                Task、Strategy、AgentRun、TaskFeedback
runtime/agents.py                 SDK Agent 创建、运行与记录提取
environments/client.py            申请和释放远程实例
environments/physicianbench.py    PhysicianBench 原生工具包装与状态读取
environments/medagentbench.py     MedAgentBench 原生工具包装与状态读取
environments/fhir_agentbench.py   FHIR-AgentBench 原生工具包装与状态读取
server/app.py                     创建和释放接口
server/docker_env.py              镜像启动、就绪检查和资源清理
methods/medevolve/prompts.py       探索、生成和更新的初始提示词
methods/medevolve/run.py           单轮生成与策略更新
evaluation/prompts.py             固定核验、评分和解题提示词
evaluation/run.py                 公共评测与结果归因
experiments/run.py                实验入口、预算与文件保存
experiments/summarize.py          读取结果生成逐轮汇总
tests/                           生命周期、工具调用、隔离与闭环验证
pyproject.toml                    包配置与必要依赖
```

使用普通函数，不设继承式方法注册体系。环境配置选择对应适配模块；各适配模块暴露 `make_tools(endpoint)` 和 `read_state(endpoint, task)`。

### 1. 标准 Agent 运行层

**文件：** `pyproject.toml`、`runtime/schemas.py`、`runtime/agents.py`、`tests/test_agent_runtime.py`。

核心数据结构使用 Pydantic；结构检查针对输出能否被后续步骤读取，不将结构检查当作医学正确性判断。

```python
class Task(BaseModel):
    id: str
    instruction: str
    task_type: str
    environment: str
    patient_ids: list[str]
    visible_scope: dict
    completion: dict
    evidence: list[dict]
    round_index: int
    source_method: str

class Strategy(BaseModel):
    guidance: str
    allocation: dict[str, int]

class AgentRun(BaseModel):
    final_output: Any
    trajectory: list[dict]
    usage: dict
    status: str
    error: str | None = None

class TaskFeedback(BaseModel):
    task_id: str
    verification: dict
    quality: dict
    solver: dict
    failure_category: str | None = None
```

其中 `allocation` 的值为各生成方向的尝试次数；方向描述可包含题型与目标难度，不定义额外的模式匹配器。

- [ ] 建立包配置与上述结构，实验配置为各角色指定模型、最大模型轮次和工具调用预算。
- [ ] 实现 SDK 调用入口，正常返回和达到预算时都保存已有轨迹。`max_turns` 约束模型轮次；工具预算在工具包装处计数，二者分别记录。

```python
async def run_agent(
    agent: Agent, text: str, max_turns: int
) -> AgentRun:
    result = await Runner.run(agent, input=text, max_turns=max_turns)
    return AgentRun(
        final_output=result.final_output,
        trajectory=[item.to_input_item() for item in result.new_items],
        usage={
            "requests": result.context_wrapper.usage.requests,
            "input_tokens": result.context_wrapper.usage.input_tokens,
            "output_tokens": result.context_wrapper.usage.output_tokens,
        },
        status="completed",
    )
```

- [ ] 用可控模型响应验证一次工具调用后继续运行并返回最终结果，另验证预算耗尽保存中间轨迹；不访问收费模型。
- [ ] 提交：`feat(runtime): 接入标准 Agent 运行与记录`。

验收：研究代码没有自行解析 `tool_calls` 并组织多轮模型调用；有模型输出、工具输出和用量记录。

### 2. 远程环境与本机工具

**文件：** `server/app.py`、`server/docker_env.py`、`environments/client.py`、三个环境适配文件、`tests/test_environment.py`。

管理协议固定为两个操作：

```text
POST /instances
输入：environment, seed_data
输出：instance_id, endpoint

DELETE /instances/{instance_id}
输出：released
```

`endpoint` 是本机能够直接访问的原生环境地址，不能返回远程容器内部地址。初始数据引用对应服务器预先准备的数据。

- [ ] 对照各环境原始源码列出原生工具名称、参数、支持的写入操作及启动要求；分别填写适配模块，不依据旧只读封装推断能力。
- [ ] 实现 Docker 启停：为实例分配独立端口和可写存储，服务就绪后返回地址，启动失败清理本次创建资源，释放只清理该实例。
- [ ] 实现本机申请和释放函数，并由 `finally` 保证实验异常时释放实例。

```python
async def create_instance(
    manager_url: str, environment: str, seed_data: str
) -> dict:
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{manager_url}/instances",
            json={"environment": environment, "seed_data": seed_data},
        )
        response.raise_for_status()
        return response.json()

async def release_instance(manager_url: str, instance_id: str) -> None:
    async with httpx.AsyncClient() as client:
        response = await client.delete(f"{manager_url}/instances/{instance_id}")
        response.raise_for_status()
```

- [ ] 将原生操作包装为 SDK `function_tool`：参数类型和描述沿用环境定义，闭包绑定本次实例地址；验证环境允许的一次读取和一次写入。
- [ ] 创建两个相同初始数据实例，对实例 A 写入后确认实例 B 状态未变，释放 A 后确认 B 可访问；提交 `feat(environment): 接入独立远程医疗环境`。

验收：Agent 业务工具只请求分配的环境地址；原生写入能力可用，实例互不影响。三个环境分别完成原生能力验证后才声明可运行。

### 3. 探索、生成与进化

**文件：** `methods/medevolve/prompts.py`、`methods/medevolve/run.py`、`tests/test_evolution.py`。

输入输出：

```python
UPDATE_INSTRUCTIONS = """根据本轮反馈更新出题指导和生成配比。
将有效的查询经验和出题模式合并到 guidance 中。
优先提高成立且清晰的任务的难度，不能把工具故障或证据不足当作难度。
allocation 表示各生成方向的尝试次数，非负且总和与输入策略相同。
只输出符合 Strategy 的结果，不修改评测标准或解题器配置。"""

async def generate_round(
    config: dict, strategy: Strategy, round_index: int
) -> tuple[list[Task], AgentRun]:
    exploration = await explore(config, strategy)
    tasks = await generate(config, strategy, exploration, round_index)
    return tasks, exploration

async def update_strategy(
    config: dict, strategy: Strategy, feedback: list[TaskFeedback]
) -> Strategy:
    agent = Agent(
        name="strategy_updater",
        model=config["models"]["updater"],
        instructions=UPDATE_INSTRUCTIONS,
        output_type=Strategy,
    )
    run = await run_agent(
        agent,
        json.dumps({
            "strategy": strategy.model_dump(),
            "feedback": [item.model_dump() for item in feedback],
        }, ensure_ascii=False),
        config["max_turns"]["updater"],
    )
    return run.final_output
```

- [ ] 定义探索提示词：根据五类初始方向查询患者资料，记录观察、具体资源证据与可成立的任务机会。
- [ ] 实现 `explore(config, strategy) -> AgentRun`：申请独立实例，挂载原生工具，运行探索并释放实例。
- [ ] 定义生成输出 `TaskBatch(BaseModel)`，字段 `tasks: list[Task]`；实现 `generate(config, strategy, exploration, round_index) -> list[Task]`，以探索记录和配比作为输入。
- [ ] 实现指导更新提示词：保留有效查询经验与出题模式，根据质量和失败归因调整生成方向，逐轮追求更难且成立的任务；尝试次数总和保持配置预算。
- [ ] 使用固定探索和反馈样本验证生成读取证据、更新策略被下一轮读取、更新后的尝试次数总和不变；提交 `feat(medevolve): 实现探索出题与策略进化`。

验收：任务来自真实探索记录；模式存于自然语言指导；进化不修改解题器和评审提示词。

### 4. 公共核验、质量评分与固定解题

**文件：** `evaluation/prompts.py`、`evaluation/run.py`、`tests/test_evaluation.py`。

接口：`evaluate_task(config: dict, task: Task) -> TaskFeedback`。输入评审时隐藏来源方法；完成标准仅提供给核验和评分，不提供给解题 Agent。

- [ ] 为核验创建独立初始实例，检查任务证据、前提及完成标准；无法成立的任务记录原因，不进入有效任务难度统计。
- [ ] 固定质量提示词，分别返回临床价值、正确性、纵向整合需求、清晰度的评价与依据；首版保留各维结果，不擅自加入加权总分。
- [ ] 在另一个初始实例运行固定解题器，输入任务指令和允许访问范围，通过适配模块的 `read_state(endpoint, task)` 获取操作结果。
- [ ] 按任务完成标准比较答案、证据或写后状态；分别记录临床未完成、证据不足、题意含糊、工具故障和环境故障。
- [ ] 用含正确操作但错误最终文字的写入样本验证状态核验，用环境故障样本验证其不进入临床完成率分母；提交 `feat(evaluation): 实现公共任务评测`。

验收：有效任务完成率与运行故障分开统计；写入任务有状态证据；各轮评测条件相同。

### 5. 实验入口与结果汇总

**文件：** `configs/experiment.json`、`experiments/run.py`、`experiments/summarize.py`、`tests/test_experiment.py`。

配置包括：环境、管理服务地址、初始数据引用、患者分组、各角色模型、轮数、每轮尝试次数、模型轮次预算和工具预算。进化患者与最终评测患者使用不相交列表。

- [ ] 实现顺序主流程；调用 `generate_round`、`evaluate_task` 和 `update_strategy`，无需额外调度框架。

```python
async def run_experiment(config: dict, strategy: Strategy) -> None:
    for round_index in range(config["rounds"]):
        tasks, exploration = await generate_round(config, strategy, round_index)
        feedback = [await evaluate_task(config, task) for task in tasks]
        next_strategy = await update_strategy(config, strategy, feedback)
        save_round(config, round_index, strategy, next_strategy,
                   tasks, exploration, feedback)
        strategy = next_strategy
```

- [ ] 实现 `save_round(config, round_index, strategy, next_strategy, tasks, exploration, feedback) -> None`，写入 `outputs/<run_id>/round_<index>/`，保存使用策略、更新策略、任务、探索和评测记录；全程模型轨迹附在对应角色记录中。
- [ ] 实现结果汇总：逐轮列出尝试数、有效任务数、质量维度、有效任务解题完成率、故障数和模型用量；价格有明确配置时计算成本，否则保留 token 用量。
- [ ] 用替代 Agent 返回值和环境响应完成两轮集成验证，检查策略传递、患者分组、产物完整及全部实例释放。
- [ ] 在明确模型配置后运行一轮真实环境实验；固定最终指导在保留患者上生成并评测，确认汇总读取文件时不会调用模型；提交 `feat(experiment): 串联实验与结果汇总`。

验收：配置可以驱动完整主方法，逐轮产物足以检查质量与难度变化；最终评测使用保留患者与固定评测条件。

## SDK 参考

- [Agent 与结构化输出](https://developers.openai.com/api/docs/guides/agents/define-agents)
- [Runner 多轮运行](https://developers.openai.com/api/docs/guides/agents/running-agents)
- [function_tool 快速开始](https://developers.openai.com/api/docs/guides/agents/quickstart?lang=python)
