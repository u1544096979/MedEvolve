# MedEvolve Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 通过五个模块构成可运行的医疗任务自进化实验。

**Architecture:** 共用主 specification 中的数据结构与函数签名。公共结构先落地，各模块在独立 worktree 实现，最终由实验入口完成集成。

**Tech Stack:** Python、Pydantic、OpenAI Agents SDK、OpenAI Python 客户端、httpx、FastAPI、Docker、pytest。

## Global Constraints

[主 specification](design.md) 是患者分组、方法流程、模块接口和失败状态的唯一约定。出题、反馈分析、策略更新各一次调用。代码实施使用独立 worktree；真实模型调用使用已确认的模型与成本配置。

## 模块计划与文件归属

| 计划 | 可编辑范围 | 交付接口 |
| --- | --- | --- |
| [环境](plans/environment.md) | environments/、server/、tests/environment/ | 创建、工具、状态、释放、轨迹 |
| [模型运行](plans/runtime.md) | contracts.py、pyproject.toml、runtime/、tests/runtime/、tests/fixtures/ | 公共结构、单次调用、Agent 运行 |
| [主方法](plans/method.md) | methods/medevolve/、tests/method/ | 两种探索、出题、反馈、更新 |
| [评测](plans/evaluation.md) | evaluation/、tests/evaluation/ | evaluate_task |
| [实验](plans/experiment.md) | experiments/、configs/、tests/experiment/ | 实验运行与汇总 |

其他模块所需依赖在模型运行模块的公共基础任务统一加入 pyproject.toml。各模块不自行修改共享结构或其他模块文件，不接入五个对比算法。

## 实施顺序

- [ ] 模型运行模块首先定义 contracts.py，按 specification 构造覆盖成功与失败结果的 JSON 样例，并建立 pyproject.toml；提交 `chore(core): 建立公共接口与项目依赖`。
- [ ] 五个模块使用同一份公共结构提交开始各自 worktree。环境与模型运行提供接口实现；主方法、评测与实验使用替代函数开发，不等待真实服务或收费模型。
- [ ] 各模块完成其计划中的验证，交付变更提交、测试结果和原生服务依赖；公共契约有变动时同步后再集成。
- [ ] 集成各模块，完成两轮无收费模型的闭环：两个进化患者、一个评测患者，每患者尝试 2 题；检查患者边界、策略传递、产物和实例释放。
- [ ] 在环境镜像、数据和模型配置就绪后运行一轮真实实验，随后固定策略运行评测组；汇总必须只读取文件。

## 集成验收

初始探索只能访问进化患者；生成输入为单患者缓存轨迹；出题无工具调用；反馈分析与更新分开；评测不向解题器泄露答案与完整轨迹；写入实例隔离；决策点不支持时明确拒绝；各轮记录有效产量、质量、有效任务完成率、故障和模型用量。

最终交付为能按配置运行的主方法与公共实验层。对比方法按公共任务和评测接口单独接入。
