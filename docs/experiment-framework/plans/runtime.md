# 模型运行模块 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 提供统一结构、单次模型请求及标准 SDK Agent 运行。

**Architecture:** 公共结构先提交，其他模块据此并行开发。单次调用和 Agent 调用分开实现，使用同一种 RunResult。

**Tech Stack:** Pydantic、OpenAI Python SDK、OpenAI Agents SDK、pytest、httpx。

## Global Constraints

以 [主 specification](../design.md) 的第 5 节为接口约定，引用根目录 `contracts.py`。实施使用独立 worktree，不修改其他模块文件。使用替代响应进行本地验证；真实模型调用与服务器部署在配置和授权明确后执行。失败保留记录，不伪造成功输出。

## 文件归属与依赖

创建 `contracts.py`、`pyproject.toml`、`runtime/__init__.py`、`runtime/models.py`、`runtime/llm.py`、`runtime/agent.py`、`tests/runtime/`、`tests/fixtures/`。依赖环境和其他方法代码之前先交付公共结构与样例。

包配置统一声明 pydantic、openai、openai-agents、httpx、fastapi、uvicorn、docker，以及测试依赖 pytest、pytest-asyncio。实际采用版本在安装时记录，不导入 AgentEvolver 的依赖清单。顶层包包含 contracts 与五个业务目录；根目录保留 contracts.py，不要求每个模块重定义结构。

## Task 1：公共结构与样例

- [ ] 按 specification 第 5 节建立全部模型。DataExploration、EnvironmentExploration 只有 guidance 字段；定义 JsonObject 为 JSON 字典的类型别名。提供 SolverAnswer，只有 answer: str，作为固定解题输出。
- [ ] 在 `tests/fixtures/` 写入完整 JSON 样例：两位进化患者、一位评测患者、TaskBatch、Strategy、成功与故障 EvaluationResult；样例使用虚构标识，不使用真实患者资料。
- [ ] 在 `tests/runtime/test_contracts.py` 验证所有样例可以 model_validate，往返 JSON 后保持相等；失败评测允许 verification 为 None。执行 `pytest tests/runtime/test_contracts.py -q`，通过后提交公共基础。

```python
def test_solver_input_excludes_answers():
    from contracts import SolverInput
    data = SolverInput(instruction="读取记录", patient_id="p1", decision_point=None)
    assert set(data.model_dump()) == {"instruction", "patient_id", "decision_point"}
```

## Task 2：单次调用

- [ ] 在 models.py 根据 ModelConfig 构造 AsyncOpenAI 客户端，密钥从指定环境变量读取，不写入轨迹；关闭 SDK 自动请求重试，使单次入口保持一次请求。
- [ ] 在 llm.py 实现 `call_llm`，传递系统指导、输入和 output_type 的 JSON 结构。支持结构化输出的模型使用官方结构化接口；其他已配置模型可在一次响应内输出 JSON，再本地校验，不进行自动修复调用。连接能力配置属于 models.py，不根据失败响应重试猜测协议。

```python
client = AsyncOpenAI(api_key=os.environ[model.api_key_env],
                     base_url=model.base_url, max_retries=0)
raw = await client.chat.completions.create(
    model=model.model,
    messages=[{"role": "system", "content": instructions},
              {"role": "user", "content": input_text}],
)
output = output_type.model_validate_json(raw.choices[0].message.content)
```

以上为本地 JSON 校验路径。返回 `output.model_dump(mode="json")`，保存请求、原始响应与 usage；拒答、空输出、解析失败或传输失败均为 failed，output 为 None。

- [ ] 替代客户端记录请求次数；验证合法 JSON、非法 JSON、空响应、传输异常都只请求一次。执行 `pytest tests/runtime/test_llm.py -q`。
- [ ] 提交 `feat(runtime): 实现单次模型调用与记录`。

## Task 3：标准 Agent

- [ ] 在 agent.py 实现 `run_agent`；使用 models.py 的客户端配置创建 SDK 模型，将 instructions、tools 和 output_type 交给 Agent，max_model_turns 交给 Runner 的 max_turns。

```python
agent = Agent(name="experiment_role", instructions=instructions,
              model=sdk_model, tools=tools, output_type=output_type)
result = await Runner.run(agent, input=input_text,
                          max_turns=budget.max_model_turns)
```

`sdk_model` 在 models.py 中由 ModelConfig 创建；原生提供商配置和兼容 Chat Completions 的连接配置分别使用 SDK 对应模型，不将 base_url 直接当作模型名称。

- [ ] 为函数工具增加本次运行共享计数器，调用到 max_tool_calls 后停止运行，禁止超额操作；并行工具调用也使用同一计数器。工具调用预算不等于 Runner 模型轮次。
- [ ] 将 SDK 新增运行条目、最终输出及用量转成 RunResult；达到预算或异常时保存已发生的模型与工具条目。工具错误标明来源，不能只返回一段“工具失败”的成功输出。SDK 云端 tracing 在本地实验配置中关闭，文件轨迹由调用方保存。
- [ ] 使用 SDK 可控模型响应验证“工具调用→工具结果→最终结构化输出”，再验证轮次耗尽和工具预算耗尽。执行 `pytest tests/runtime -q`，提交 `feat(runtime): 实现标准 Agent 执行`。

## 交付验收

其他模块只依赖两种公共入口；单次调用无工具、无隐藏修复循环；Agent 使用 SDK；失败保留轨迹与用量。交付公共基础提交与完整模型运行提交。

参考：[结构化输出](https://developers.openai.com/api/docs/guides/structured-outputs)、[SDK Runner](https://developers.openai.com/api/docs/guides/agents/running-agents)。
