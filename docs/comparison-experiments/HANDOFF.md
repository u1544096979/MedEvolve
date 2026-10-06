# MedEvolve 会话交接

## 项目入口

仓库 /Users/guagua/code/MedEvolve，工作分支 codex/experiment-framework。实现使用独立 worktree，集成检查后合回该分支。原 AgentEvolver 仓库只用于参考；其 archive/fork-medevolve 分支保留旧五个对比方向文档，旧实现不是复现依据。

## 已确认方案

本机运行实验流程，远程 Docker 提供隔离医疗环境。使用标准 OpenAI Agents SDK 的工具 Agent；主方法单次调用生成任务，初始数据探索与环境探索分别执行，策略自进化不训练模型参数。对比方法优先作者源码，其次论文或明确方向定义。

公共边界见 common/SPEC.md，公共实验接入任务见 common/PRD.md。主方法见 main/SPEC.md 和 main/PRD.md。TaskCraft 见 baselines/taskcraft/SPEC.md 和 PRD.md。Workflow 见 [SPEC.md](baselines/workflow/SPEC.md) 与 [PLAN.md](baselines/workflow/PLAN.md)。SingleShot 见 [SPEC.md](baselines/singleshot/SPEC.md) 与 [PLAN.md](baselines/singleshot/PLAN.md)。SelfRefine 见 [SPEC.md](baselines/selfgen/SPEC.md) 与 [PLAN.md](baselines/selfgen/PLAN.md)。EnvExplore 见 [SPEC.md](baselines/envexplore/SPEC.md) 与 [PLAN.md](baselines/envexplore/PLAN.md)，是保留环境探索、使用固定初始策略的主方法消融，支持 PhysicianBench 与 MedAgentBench。

## 已实现状态

环境、模型运行、主方法、公共评测和实验入口已集成。主方法按原始来源记录组织生成输入，单次生成时使用本批题型及数量约束；引用逐条核验后，再判断题目与候选答案。64 项本地检查通过，1 项跳过；正确与错误引用、合理等价答案已通过真实模型复核。

正式运行输出：/Users/guagua/code/MedEvolve/outputs/physicianbench-flash-closed-loop-20261003-145454。详细结果见 main/GENERATION.md。

两轮进化生成 8/8 题，核验通过 8/8，解题完成 7/8；1 题因质量评价缺少 dimensions 字段记为 model_error。两轮反馈与策略更新均完成，运行故障未当作难度提升。独立患者生成 2/2 题，核验通过且解题完成 1/2；另一题答案包含未列入 evidence 的真实轨迹记录，被公共核验拒绝。不能据单名独立评测患者宣称进化收益成立。

SelfRefine（配置键 selfgen）已实现并完成两个环境的公共验收：PhysicianBench 两名评测患者生成 4/4 题、核验通过 4/4、解题完成 4/4；MedAgentBench 生成 2/2 题、核验通过 2/2、解题完成 1/2。运行目录 outputs/physicianbench-selfgen-acceptance-2p-flash 与 outputs/medagentbench-selfgen-acceptance，详细结果见 [baselines/selfgen/RESULTS.md](baselines/selfgen/RESULTS.md)。

## 正式环境和模型

PhysicianBench 本机管理服务 http://127.0.0.1:18084；种子 fhir-full:v1。镜像归档 /Users/guagua/data/physical_benchmark_data/physicianbench-fhir-v1.tar.gz，来源为 szu-hosp-4090 的 /gpfs/flash/home/gwh/data/physical_benchmark_data。Docker 服务已启动，单容器内存上限 4 GB，实验逐次创建、使用和释放环境，同时最多一个实例。容器内数据库自动整理关闭。

实验使用 PI 中 OpenCode Go provider 的 https://opencode.ai/zen/go/v1 和 deepseek-v4.1-flash，全部模型角色一致，认证从 OPENCODE_API_KEY 读取。运行时携带会话请求头，工具 Agent 使用 JSON object 与明确的最终 schema，兼容字符串中的原始换行并保留响应 trace。配置为 configs/physicianbench-flash.json。

gpt-4o-mini 和 DeepSeek V4.1 Flash 已授权用于实验，其他旗舰或高价模型须先确认。当前模型仍可能产生 schema 字段缺失，须计为模型故障；不得自动改写候选答案或放宽公共核验。

## section 启动指令

读取本文件、common/SPEC.md，以及负责 section 的 SPEC.md 和实施文档（EnvExplore 为 PLAN.md，其他 section 为 PRD.md）。按对应 SPEC 的源码依据或方向定义确认当前契约；只修改所属文件，跨模块接口由公共实验 section 统一处理。文档的新骨架须经用户确认后再写正文；不自动启用 superpowers。只验证关键机制和共享边界，不增加独立框架、额外标注体系或无授权模型调用。

## 当前优先任务

主方法：依据已对齐口径检查答案全部事实的引用覆盖，保留公共核验拒绝与模型格式故障的真实记录。公共层：冻结方法入口并接入对比运行。TaskCraft：按作者源码制定医疗适配。Workflow、SingleShot：按各自方向 SPEC 和 PLAN 实现。EnvExplore：按 SPEC 和 PLAN 实现一次环境探索、固定策略单次生成及两个环境的公共评测接入。公共生成入口、预算与结果汇总约定冻结后，五个方法可分配独立 session 并行实现。
