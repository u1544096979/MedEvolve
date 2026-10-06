# SelfRefine 实现与验收记录

本方法是单患者自反馈迭代修订 baseline：生成初稿，由同一模型自评当前批次，再按反馈修订，通过即停。配置键 method 与 source_method 保留 selfgen，展示名为 SelfRefine，实现目录为 `methods/selfgen/`。算法依据见 [SPEC.md](SPEC.md)，实施步骤见 [PLAN.md](PLAN.md)。

## 实现状态

- 入口：`async def generate_tasks(trajectory, environment, directions, runtime) -> RunResult`，沿用五类题型与公共采样给出的两个方向。
- 循环：1 次初稿 → 最多 3 次自评 → 最多 2 次修订，共 6 次模型调用、0 工具调用；当前批次全部通过立即结束。
- 版本控制：候选与原始方向位置一一对应，修订删除题目后索引保持原始值；通过题目在修订中保持原样；第三次自评后仅保留通过题目、筛除其余。
- 终止语义：模型故障、结构故障或预算耗尽停止本批且 output 为 null；正常筛除全部候选返回 completed 与空批次。
- 不使用数据探索指导、环境探索指导、进化策略、跨患者经验或公共评测反馈。

## 配置

| 项目 | PhysicianBench | MedAgentBench |
| --- | --- | --- |
| 配置 | `configs/physicianbench-selfgen.json` | `configs/medagentbench-selfgen.json` |
| 评测患者 | MRN4460967357、MRN2299051392 | S2874099 |
| 生成模型 | deepseek-v4.1-flash | deepseek-v4.1-flash |
| 评测模型 | verifier、quality、solver、completion 均为 deepseek-v4.1-flash | 同左 |
| 生成预算 | 6 轮 / 0 工具 | 6 轮 / 0 工具 |
| 评测预算 | 30 轮 / 200 工具（每角色） | 同左 |
| 环境 | 管理服务 127.0.0.1:18084，种子 fhir-full:v1 | 管理服务 127.0.0.1:18084，种子 jyxsu6/medagentbench@sha256:3fb83d7e |

生成预算按方法单批次调用上限声明为 6 轮 / 0 工具。公共评测规则、评测侧预算与患者材料均未按方法调整。

## 验收运行

| 环境 | 运行目录 | 患者 | 生成调用 | 出题 | 核验通过 | 解题完成 | 错误 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PhysicianBench | `outputs/physicianbench-selfgen-acceptance-2p-flash` | MRN4460967357、MRN2299051392 | 4（每名 2 次） | 4/4 | 4/4 | 4/4 | 0 |
| MedAgentBench | `outputs/medagentbench-selfgen-acceptance` | S2874099 | 2 | 2/2 | 2/2 | 1/2 | 0 |

两个环境均按通过即停结束：初稿后第一次自评全部通过即进入筛选，未产生修订轮次。PhysicianBench 覆盖药物处方、诊断与解释、检查与风险分层；四题四维质量齐全（临床价值 4 高，正确性 3 高 1 正确，纵向整合 4 高，清晰度 3 高 1 清晰），解题共 88 次原生 FHIR 工具调用，含一次处方写入任务。MedAgentBench 覆盖药物处方与诊断与解释，解题走原生 GET / POST / FINISH，完成 1/2，未完成题目按临床类别记录。

## 成本

| 环境 | 生成 | 公共评测 | 每题有效题生成成本 |
| --- | --- | --- | --- |
| PhysicianBench | 4 轮调用，277,957 tokens | 70 轮、88 工具调用，1,196,022 tokens | 69,489 tokens |
| MedAgentBench | 2 轮调用，659,763 tokens | 30 轮、11 工具调用，971,544 tokens | 329,882 tokens |

最坏路径为单患者 6 次生成调用：PhysicianBench 每次约 7 万 tokens、MedAgentBench 每次约 33 万 tokens，对应单患者生成上限约 42 万与 200 万 tokens。

## 公共口径核验

- trace 保存 `candidate_generation`、`self_review/1..3`、`refine/1..2` 与最终筛选记录，含原始输入输出、逐题反馈、保留索引与版本。
- 独立解题只接收题目、患者与决策点；候选答案、证据与自评反馈逐条核对未进入解题输入。
- PhysicianBench 写入任务按实际返回与持久化状态判定；MedAgentBench 提交请求不按数据库写入判定。
- 两个环境 decision_point 均为 null。自评通过不等于事实正确：PhysicianBench 出现自评通过、但被公共核验以证据声明与原始记录不符为由拒绝的题目。

## 遗留与口径

1. 环境模型统一由公共口径决定：PhysicianBench 侧各方法配置当前模型不一致（本方法为 deepseek-v4.1-flash，EnvExplore 为 gpt-4o-mini）。实测该方法在 gpt-4o-mini 上的多轮修订会因结构故障终止批次，相关小样本目录保留在 `outputs/` 中，不作为验收依据。
2. PhysicianBench 评测患者组：本配置声明 2 名，其他方法配置声明 1 名；患者组统一由公共口径决定。
3. 未启用数据探索、环境探索、进化策略与公共评测回流。
