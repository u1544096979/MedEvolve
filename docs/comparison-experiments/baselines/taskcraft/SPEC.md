# TaskCraft 医疗适配规范

## 方法来源

论文：[TaskCraft: Automated Generation of Agentic Tasks](https://arxiv.org/abs/2506.10055)。作者源码：[OPPO-PersonalAI/TaskCraft](https://github.com/OPPO-PersonalAI/TaskCraft)。

本机参考仓库 /Users/guagua/code/TaskCraft，已检查提交 1e405e0b02c4738654e2d885b16677b46c9fe969。该提交作为适配依据，不能将旧 AgentEvolver 的同名实现当成原始源码。

## 核心机制

按源码保留原子生成、深度扩展、宽度扩展及各阶段的内置核验。原子生成从内容抽取候选结论并构造问题，使用 VerifyAgent 过滤；深度扩展通过反向关系搜索构造更长的问题链并验证；宽度扩展组合多个问题，检查可分解性与答案一致性。

原始答案及答案组织方式由对应源码阶段决定，再映射为候选 reference_answer。不得将这些阶段统一替换为一次普通出题，也不得引入旧实现的答案操作库来代替原始机制。

## 源码入口

| 功能 | 源码 |
|---|---|
| 原子生成与过滤 | taskcraft/gen_atomic_task.py |
| 深度扩展 | taskcraft/gen_depth_based_task.py |
| 宽度扩展与检查 | taskcraft/gen_width_based_task.py |
| 搜索和内置核验 | taskcraft/agent_tools.py |
| 方法提示词 | taskcraft/prompts/ |

## 医疗适配边界

输入内容映射为指定患者的轨迹；任务搜索和验证所用工具映射为该患者的医疗环境工具。保留原算法阶段、依赖关系、判断分支和提示词语义，替换领域内容及工具描述。每个上游阶段对环境能力的依赖须逐项检查，无法映射时明确报告，不用任意提示词近似代替。

优先复用原始函数；若原始同步调用或 Agent 接口无法接入公共运行层，移植对应阶段并列出源码位置与适配差异。输出格式映射、证据补齐和医疗环境替换属于适配，不宣称完整原环境复现。

## 公共接口和评测

遵循 ../../common/SPEC.md 的 generate_tasks。返回 TaskBatch 和完整生成成本，环境工具绑定当前患者。原算法筛选与公共评测分别执行、分别计入生成及评测成本。只开放当前指定患者的事实，不进行 MedEvolve 的数据探索或策略进化。

源码循环和内置核验的预算应显式配置；模型统一采用实验配置，不能保留源码默认的高价模型自动发起调用。
