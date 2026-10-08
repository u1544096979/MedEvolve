# 随机四分之一进化集实施计划

> **For agentic workers:** 使用 subagent-driven-development 实施并审查。

**Goal:** 按固定随机种子划分25%真实患者进化集，分两轮更新策略，独立运行新协议。

**Architecture:** cohort记录显式round_patients；配置生成器传递现有ExperimentConfig字段。反馈按患者适配预算后汇总，复用现有模型调用与用量结构。部署在独立输出目录，历史已完成基线按患者筛选复用。

**Tech Stack:** Python、Pydantic、pytest、SSH。

## 约束

仅两个环境、六方法、GPT-5.4 Nano和GPT-4o Mini。两裁判、解题模型等于生成模型。生成32、评测64。无凭证落盘与输出。

## 任务1：配置与反馈

- [ ] 修改scripts/emit_comparison_configs.py，优先使用cohort显式round_patients，验证两轮覆盖进化集且无重复；历史未提供字段保持兼容。
- [ ] 修改methods/medevolve反馈路径，按患者生成预算内反馈，汇总Feedback字段，保留每个调用trace、usage及失败，不将所有轨迹拼成单次请求。
- [ ] 增加配置划分与长反馈边界测试，运行相关测试，原子中文提交。

## 任务2：部署与结果复用

- [ ] 与官方患者名单核验100个MedAgentBench丰富轨迹。
- [ ] 构建随机25%cohort及两轮分组，确保历史进化患者不进入评测名单。
- [ ] 为六方法构建新批次；按patient_id筛选历史已完成基线任务、评测、attempt及runs，保留来源与协议身份；仅完整覆盖的批次标记完成。
- [ ] 停止旧入口并保留中断记录，同步提交、切换配置到独立产物目录，启动新入口。
- [ ] 核对实际配置患者数、两轮不交叉、双裁判与解题身份、PID和新进化调用，更新定时检查路径。
