# 公共评测模块 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用固定条件核验任务、评价质量并测试解题表现。

**Architecture:** 评审可读取完整轨迹与参考答案；解题只读题目和受限工具。核验与解题使用不同独立实例，完成评价检查答案与操作状态。

**Tech Stack:** 公共模型接口、公共环境接口、Pydantic。

## Global Constraints

以 [主 specification](../design.md) 的第 5 节为接口约定，引用根目录 `contracts.py`。实施使用独立 worktree，不修改其他模块文件。使用替代响应进行本地验证；真实模型调用与服务器部署在配置和授权明确后执行。失败保留记录，不伪造成功输出。

## 文件归属与接口

创建 `evaluation/api.py`、`evaluation/prompts.py`、`evaluation/solver.py`、`tests/evaluation/` 及 __init__.py。公共入口 evaluate_task 按 specification 签名；依赖 runtime、environments 和 contracts，不导入主方法代码。

角色配置固定为 verifier、quality、solver、completion。固定提示词包含在 prompts.py，不由反馈或更新改写。SolverAnswer 由 contracts 提供。

```python
VERIFY = """核对任务和参考答案是否得到原始数据及工具能力支持。
检查证据与题目是否一致；不把运行失败当作任务不成立。
后续真实治疗只能支持事实回顾，不能自动作为临床决策的唯一答案。"""
QUALITY = """分别评价临床价值、正确性、纵向整合需求和清晰度，
每个维度返回 judgment 与 reason。整合需求看作答是否需要关联信息，
不按引用数量计分，不计算总分。"""
SOLVER = """根据任务指令和患者范围，通过可用工具完成任务。
只依据工具可见的信息；需要操作时执行实际操作。
最终 answer 陈述结果和依据，不能声称未执行的操作已成功。"""
COMPLETION = """根据题目、参考答案、证据、实际解题轨迹与环境状态，
判断任务是否完成。合理等价答案可接受，写入操作以实际状态为依据。
只陈述最终文字不能证明写入成功；运行故障不自动归类为临床错误。"""
```

## Task 1：核验与质量

- [ ] 核验先检查证据 source_id 是否属于该轨迹；未知引用标记 invalid 并附原因，不视为运行故障。
- [ ] 为核验申请独立实例，结合题目、参考答案、证据、完整轨迹和原生工具输出 Verification；支持写入的任务允许验证可行性。finally 释放。运行失败返回 failed，verification 为空。
- [ ] 核验 valid 为 False 时返回 invalid，不试解；valid 为 True 时单次 call_llm 输出 QualityReview，四维判断与理由分别保存，不计算加权总分。
- [ ] 替代环境和模型验证无效证据不会调用解题，核验传输失败不会被记录为任务无效，来源方法不进入评审 payload。执行 `pytest tests/evaluation/test_verification.py -q`，提交 `feat(evaluation): 实现证据核验与质量评价`。

## Task 2：固定解题与状态核验

- [ ] 构造 SolverInput，仅传任务指令、患者 ID 和可选决策点。给 run_agent 的 instructions 使用固定解题提示词，不拼接参考答案、证据、初始指导或缓存轨迹。

```python
solver_input = SolverInput(instruction=task.instruction,
    patient_id=task.patient_id, decision_point=task.decision_point)
instance = await create_environment(environment)
try:
    tools = await get_tools(instance, [task.patient_id], task.decision_point)
    solver_run = await run_agent(instructions=SOLVER,
        input_text=solver_input.model_dump_json(), tools=tools,
        output_type=SolverAnswer, model=config.runtime.models["solver"],
        budget=config.runtime.budgets["solver"])
    observed_state = await read_state(instance, task)
finally:
    await release_environment(environment, instance)
```

- [ ] 决策点能力未实现时返回明确的失败结果，禁止换成完整环境继续运行。解题与核验实例不得相同。
- [ ] 完成评审一次 call_llm，输入题目、参考答案、数据证据、SolverAnswer、工具轨迹与 observed_state，输出 CompletionReview。写入任务不能只看最终文字；查询任务依据答案和证据评价。
- [ ] 组装 EvaluationResult：保存所有评测模型 RunResult，预算耗尽记录预算状态；环境、工具、模型故障与临床未完成区分，缺少可靠完成评价时 completion 为空。
- [ ] 替代解题函数验证 payload 没有参考答案、证据或完整轨迹；构造“回答声称写入成功但状态未改变”的样例，验证 completed 为 False；注入环境异常验证 finally 释放。执行 `pytest tests/evaluation -q`，提交 `feat(evaluation): 实现固定解题与状态评测`。

```python
def test_solver_payload_has_no_reference(task):
    data = SolverInput(instruction=task.instruction,
        patient_id=task.patient_id, decision_point=task.decision_point).model_dump()
    assert "reference_answer" not in data
    assert "evidence" not in data
    assert "trajectory" not in data
```

## 交付验收

固定解题器看不到答案；评审与方法无关；写入有状态依据；不成立与运行失败区分；全部实例可清理。完成率只对有效且有可靠完成评价的任务统计，预算耗尽与外部故障单列。
