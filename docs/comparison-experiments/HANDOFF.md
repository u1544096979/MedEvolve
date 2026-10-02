# MedEvolve 会话交接

## 项目入口

仓库 /Users/guagua/code/MedEvolve，工作分支 codex/experiment-framework。实现使用独立 worktree，集成检查后合回该分支。原 AgentEvolver 仓库只用于参考；其 archive/fork-medevolve 分支保留旧五个对比方向文档，旧实现不是复现依据。

## 已确认方案

本机运行实验流程，远程 Docker 提供隔离医疗环境。使用标准 OpenAI Agents SDK 的工具 Agent；主方法单次调用生成任务，初始数据探索与环境探索分别执行，策略自进化不训练模型参数。对比方法优先作者源码，其次论文或明确方向定义。

公共边界见 common/SPEC.md，公共实验接入任务见 common/PRD.md。主方法见 main/SPEC.md 和 main/PRD.md。TaskCraft 见 baselines/taskcraft/SPEC.md 和 PRD.md。其余四个方法的论文对应关系尚需确认，不能自行匹配名称相近论文。

## 已实现状态

截至提交 39de17e，环境、模型运行、主方法、公共评测和实验入口已集成。核验按引用直接读取原始记录，逐条比较证据声明，全部通过后核验题目与候选答案。52 项本地测试通过，1 项跳过；正确与错误引用已在正式环境分别复测。

正式运行输出：/Users/guagua/code/MedEvolve/outputs/physicianbench-evidence-review-20261003-040214。

两轮进化计划 8 题，生成 6 题；5 题证据无效，1 题进入完整评测但解题未通过。两轮反馈与策略更新完成。第二轮一组出题和独立患者出题因题型超出指定范围失败，因此尚无有效的独立评测结果，不能宣称进化效果成立。环境探索该次返回指导但没有调用工具。

## 正式环境和模型

PhysicianBench 管理服务 http://10.7.0.15:18084，SSH 别名 szu-hosp-nuc；种子 fhir-full:v1。容器内数据库自动整理已关闭，以处理既有启动故障。

实验使用 OpenLux 中转 gpt-4o-mini，system 指令，允许并行工具调用。认证只从环境变量读取；配置模板有环境和患者占位符，不能直接当成正式配置运行。上述真实运行目录的 config.json 是实际患者和环境配置依据。

gpt-4o-mini 和 DeepSeek-v4.1-Flash 已授权用于实验，其他旗舰或高价模型须先确认。DeepSeek 在既有中转链路出现 system 指令不遵守和无效工具参数，目前正式实验使用 gpt-4o-mini。

## section 启动指令

读取本文件、common/SPEC.md，以及负责 section 的 SPEC.md 和 PRD.md。先确认源码或论文来源与当前契约；只修改所属文件，跨模块接口由公共实验 section 统一处理。文档的新骨架须经用户确认后再写正文；不自动启用 superpowers。只验证关键机制和共享边界，不增加独立框架、额外标注体系或无授权模型调用。

## 当前优先任务

主方法：对齐候选答案与核验口径，修正生成时的证据引用与题型遵守。公共层：冻结方法入口并接入对比运行。TaskCraft：按作者源码制定医疗适配。Workflow、SingleShot、SelfGen、EnvExplore：先确认论文对应关系或明确采用方向定义。
