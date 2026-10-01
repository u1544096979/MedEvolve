# 实验入口模块 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 串联初始准备、逐轮进化、最终评测与文件汇总。

**Architecture:** 实验入口管理患者分组、轨迹缓存、轮次和保存。通过公共接口调用各模块，使用固定样例开发，不把其他模块实现复制进实验入口。

**Tech Stack:** Python、JSON、公共 contracts、pytest。

## Global Constraints

以 [主 specification](../design.md) 的第 5 节为接口约定，引用根目录 `contracts.py`。实施使用独立 worktree，不修改其他模块文件。使用替代响应进行本地验证；真实模型调用与服务器部署在配置和授权明确后执行。失败保留记录，不伪造成功输出。

## 文件归属与接口

创建 `configs/experiment.json`、`experiments/run.py`、`experiments/storage.py`、`experiments/summarize.py`、`tests/experiment/` 及 __init__.py。公共入口 run_experiment、summarize_run 按 specification 签名。

实验配置使用 contracts.ExperimentConfig。runtime 角色为 data_explorer、environment_explorer、generator、feedback、updater；evaluation.runtime 角色为 verifier、quality、solver、completion。模型名称和地址来自实际实验配置，密钥只写环境变量名称。配置中的 method 首版仅支持 medevolve；选择其他方法明确报错。

## Task 1：分组、轨迹缓存与初始探索

- [ ] 检查两组患者不相交，round_patients 的每个 ID 属于进化组；轮数为 round_patients 长度，禁止另设矛盾轮数字段。
- [ ] 逐个患者通过 create_environment、prepare_trajectory、finally release_environment 整理轨迹，写入 `outputs/<run_id>/trajectories/evolution/<patient_id>.json` 和 `evaluation/<patient_id>.json`；实际路径对 patient_id 做安全编码。
- [ ] explore_data 仅传进化组轨迹；explore_environment 仅传进化患者列表。恢复 DataExploration 和 EnvironmentExploration，组成 ExplorationGuidance；失败停止进入迭代，保存已有记录。
- [ ] 替代轨迹接口记录患者访问，验证每位只整理一次、后续使用缓存，探索未接收评测患者。执行 `pytest tests/experiment/test_preparation.py -q`，提交 `feat(experiment): 实现患者分组与初始准备`。

## Task 2：逐轮进化

- [ ] 用固定随机种子和当前 allocation 权重，为本轮每患者抽取两个 GenerationDirection；同一固定策略和种子能够重现抽样。

```python
rng = random.Random(config.random_seed)
directions = rng.choices(strategy.allocation,
    weights=[item.weight for item in strategy.allocation], k=2)
```

- [ ] 对单患者调用 generate_tasks，校验 TaskBatch 后附加 Task 元数据；任务 ID 使用 run_id、round_index、patient_id 的安全编码和题目位置构成。不能让模型提供来源方法或轮次。
- [ ] 逐题 evaluate_task，汇总本轮任务与评测后调用 analyze_feedback，再调用 update_strategy。只在成功且合法时替换策略；反馈失败跳过更新，更新失败保留旧策略，两种失败均保存状态。

```python
feedback_run = await analyze_feedback(tasks, round_trajectories,
                                      evaluations, config.runtime)
if feedback_run.status == "completed":
    feedback = Feedback.model_validate(feedback_run.output)
    update_run = await update_strategy(strategy, feedback, config.runtime)
    if update_run.status == "completed":
        strategy = Strategy.model_validate(update_run.output)
```

- [ ] generation 失败记录尝试但不生成虚构任务；一个患者少于两题保存实际产量。每个角色 RunResult 与其调用输入对应，所有评测运行均保存。
- [ ] 写入 `round_<index>/strategy_used.json`、`strategy_next.json`、`tasks.json`、`evaluations.json`、`feedback.json`、`runs.json`；用替代函数跑两轮，验证下一轮读取上轮更新策略，每患者生成只调用一次。执行 `pytest tests/experiment/test_rounds.py -q`，提交 `feat(experiment): 串联逐轮出题与进化`。

## Task 3：最终评测与汇总

- [ ] 使用最终固定 Strategy 和初始指导，对评测组逐患者单次生成并公共评测，产物写入 `final_evaluation/`；不调用反馈或更新。评测完成后策略保持不变。
- [ ] summarize_run 只读产物，报告尝试数、实际产量、有效任务数、质量各维判断分布、有效且有可靠完成评价的任务完成率、运行故障、预算耗尽和 token 用量；报告分母和未完成评测数量。
- [ ] usage 缺失字段记为空，不填零；仅配置明确模型价格时计算费用。结果按环境分别统计，不直接比较不同环境的原始分数。
- [ ] 集成测试使用两位进化患者、一位评测患者、两轮替代模型与环境；验证初始探索一次、评测无更新、轨迹复用、2 题预算、产物关联和实例全部释放。汇总时替代所有模型入口为“调用即失败”，确认仍可完成。
- [ ] 执行 `pytest tests/experiment -q`；合并五模块后执行 `pytest tests -q`。在明确镜像、数据和模型配置后执行 `python -m experiments.run --config configs/experiment.json`，用 `python -m experiments.summarize --run-dir <实际运行目录>` 读取结果；提交 `feat(experiment): 实现最终评测与实验汇总`。

## 交付验收

缓存不跨初始数据设置误用；评测患者不参与策略学习；实际出题调用数可追溯；汇总不调用模型或启动环境。真实实验结束后保存完整输出目录与配置。
