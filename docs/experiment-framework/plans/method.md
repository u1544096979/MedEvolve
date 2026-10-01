# 主方法模块 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现两种初始探索、单次出题、独立反馈分析与策略更新。

**Architecture:** 初始探索沉淀两份固定指导；单患者轨迹驱动一次出题调用；反馈分析与策略更新各一次调用。轮次和缓存归实验入口管理。

**Tech Stack:** Pydantic、公共模型入口、SDK function_tool。

## Global Constraints

以 [主 specification](../design.md) 的第 5 节为接口约定，引用根目录 `contracts.py`。实施使用独立 worktree，不修改其他模块文件。使用替代响应进行本地验证；真实模型调用与服务器部署在配置和授权明确后执行。失败保留记录，不伪造成功输出。

## 文件归属与接口

创建 `methods/medevolve/exploration.py`、`generation.py`、`feedback.py`、`strategy.py`、`prompts.py`、`api.py`、`tests/method/` 及 __init__.py。api.py 导出 specification 的五个主方法接口。依赖 runtime 的 call_llm/run_agent 和 environments 的创建、工具、释放接口，不导入 evaluation 或 experiments。

prompts.py 的初始角色指令如下，输出结构由公共模型入口传递：

```python
DATA_EXPLORATION = """自由读取给定患者的长期轨迹，归纳数据结构、
病程关联、信息变化和潜在出题方向。总结可以跨患者使用的数据探索指导，
不要在最终指导中列出具体患者、患者 ID 或某位患者的答案。
不能将缺失记录补成事实。"""

ENVIRONMENT_EXPLORATION = """通过工具交互了解参数、返回结构、
可用读写操作及操作约束。仅访问允许范围内的患者。
输出可用于后续出题的环境指导，不包含具体患者记录。
写入尝试必须属于环境原生支持的操作。"""

GENERATION = """根据单个患者的长期轨迹、数据探索指导、环境探索指导、
当前出题指导及两个生成方向，尝试生成两道不同的医疗任务。
每题给出题目、题型、参考答案和原始数据证据；引用输入中的 source_id。
证据不足时允许少出题，不能凭空补充事实或重复改写同一问题。
决策点未启用时输出空决策点；启用时题目须仅依赖此前可查询信息。
需要写入的任务必须在环境能力范围内。"""

ANALYZE_FEEDBACK = """根据本轮任务、患者轨迹和固定评测记录，分别分析
质量问题、难度表现、失败原因与改进建议。保留执行结果中的事实，
不要将环境故障、工具故障、模型故障或无效任务解释为难度提升。
评价难度时结合任务成立性和解题轨迹，不只观察完成率。"""

UPDATE_STRATEGY = """根据当前策略与反馈，更新自然语言出题指导及题型、
难度方向权重，追求更高质量和更高难度。保留有效模式，调整低质量模式。
题型从五类初始方向选择，权重非负且至少一个正权重。
不修改患者分组、初始探索指导、每患者两题的预算或公共评测条件。"""
```

## Task 1：两种初始探索

- [ ] 定义 DATA_EXPLORATION 提示词：自由读取可用轨迹，归纳跨患者数据特点、关联方式和出题机会，不在最终指导中包含患者 ID 或特定患者答案。
- [ ] explore_data 将传入的轨迹建立本地索引，提供 list_patients 和 read_trajectory 两个工具；仅索引传入轨迹，不读取磁盘的评测目录。

```python
index = {item.patient_id: item for item in trajectories}
@function_tool
async def read_trajectory(patient_id: str) -> str:
    return index[patient_id].model_dump_json()
```

通过 run_agent 输出 DataExploration，返回 RunResult。list_patients 返回 index 的键；不把所有完整轨迹重复塞进每轮提示词。

- [ ] 定义 ENVIRONMENT_EXPLORATION 提示词：归纳工具参数、返回形式、能力和约束；允许在独立实例中验证原生支持的操作，最终指导不包含具体患者记录。
- [ ] explore_environment 申请实例，用 get_tools(instance, patient_ids) 绑定进化患者范围；输出 EnvironmentExploration，finally 释放。工具错误和模型失败保留为运行失败。
- [ ] 替代 run_agent 验证数据探索只访问传入患者，环境探索发生写入也会释放，失败路径也释放。执行 `pytest tests/method/test_exploration.py -q`，提交 `feat(medevolve): 实现初始数据与环境探索`。

## Task 2：一次调用出题

- [ ] 定义 GENERATION 提示词：输入单患者轨迹、两份初始指导、出题指导、两个方向，尝试生成两题；每题包含题目、合法题型、参考答案与来源证据，证据不足时允许不足两题，不重复改写同一问题。
- [ ] generate_tasks 仅调用 call_llm，不调用 run_agent 或环境接口。directions 长度必须为 2；输出类型为 TaskBatch。任务运行信息由 experiments 添加。

```python
payload = {
    "trajectory": trajectory.model_dump(mode="json"),
    "guidance": guidance.model_dump(mode="json"),
    "strategy": strategy.model_dump(mode="json"),
    "directions": [d.model_dump(mode="json") for d in directions],
}
return await call_llm(instructions=GENERATION,
    input_text=json.dumps(payload, ensure_ascii=False), output_type=TaskBatch,
    model=runtime.models["generator"])
```

- [ ] 替代 call_llm 捕获输入，验证只含当前患者、指导与两个方向，调用数为 1；参考证据必须来自当前轨迹。超出两题或未知 source_id 时返回 failed 并保留原始记录，不隐式追加修复请求。执行 `pytest tests/method/test_generation.py -q`，提交 `feat(medevolve): 实现轨迹驱动的单次出题`。

## Task 3：反馈与更新分开

- [ ] analyze_feedback 按 task_id 关联任务、对应患者轨迹和评测，单次 call_llm 输出 Feedback。提示词分别分析质量、难度、失败和建议，运行故障不解释为题目难度提升。
- [ ] update_strategy 单次 call_llm 读取 Strategy 与 Feedback，输出 Strategy；初始两份探索指导、评审和解题配置不在可更新输出内。

```python
payload = {"strategy": strategy.model_dump(mode="json"),
           "feedback": feedback.model_dump(mode="json")}
return await call_llm(instructions=UPDATE_STRATEGY,
    input_text=json.dumps(payload, ensure_ascii=False), output_type=Strategy,
    model=runtime.models["updater"])
```

- [ ] 检查 allocation 题型为五类初始方向、权重非负且总和正；无效策略返回 failed，实验入口保留原策略。反馈分析失败时不调用更新。
- [ ] 替代两次调用验证 analyze_feedback 与 update_strategy 各调用一次，更新只接收反馈结果与旧策略，工具故障不变成临床未完成；执行 `pytest tests/method -q`，提交 `feat(medevolve): 实现反馈分析与策略更新`。

## 交付验收

两个初始探索分别保存结果；任务生成不交互环境；反馈和更新不是一个请求；全部输出使用公共结构。交付五个接口与固定角色提示词。
