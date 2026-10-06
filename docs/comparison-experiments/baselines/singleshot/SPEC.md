# SingleShot 设计规范

## 方法定义与目标

SingleShot 是本项目的单次生成对比方法，按本规范报告，不对应某篇已确认的论文。每名评测患者通过一次模型调用，同时生成最多两道任务、候选参考答案和逐条证据。

目标是在 PhysicianBench 与 MedAgentBench 中，与 main 的公共实验框架使用一致的患者数据、模型配置、公共核验、解题和统计口径。SingleShot 从全部五种题型中自主选择适合患者资料的任务，不使用公共方向采样。方法不执行数据探索、环境探索、自答复核、内部模型筛选、反馈分析或策略更新。

## main 基准与文件边界

项目根目录为 `/Users/guagua/code/MedEvolve`。方法规范位于 `docs/comparison-experiments/baselines/singleshot/SPEC.md`，实施计划位于同目录 PLAN.md。本规范定义在已有 SingleShot 实现上完成 main 与双环境适配的要求。

实施基准为开始实施时本地 main 最新已提交版本。实施 agent 记录其完整提交号，在已有 `.worktrees/singleshot`、`codex/singleshot` 分支同步该版本并保留现有方法实现。同步使用已有 worktree，不另建平行实现。main 工作目录中的未提交修改不自动纳入基准；需要依赖的公共改动必须先有明确的提交和归属。

### 已有实现与适配范围

SingleShot 实现提交为 `f18013ce9d5d1420906dcae7a81855c01a64975f`，分支为 `codex/singleshot`，worktree 为 `/Users/guagua/code/MedEvolve/.worktrees/singleshot`。main 已登记 singleshot 方法分派，但尚未包含 methods/singleshot/ 代码；实施须先同步公共更新，再以该 worktree 中的已有实现完成适配。

| 项目 | 已有实现 | 本次适配要求 |
| --- | --- | --- |
| 生成函数与提示词 | 独立 generate_tasks、GENERATION，使用 GenerationBudget，一次调用 | 保留方法边界，根据公共契约及双环境限制调整 |
| 轨迹输入 | 完整 text；sources 只传 source_id | 与 main 的轨迹材料口径一致，sources 同时传原始 statement |
| 题型选择与输出 | 旧实现接收两个 directions，最多两题，失败不追加调用 | 接收全部五种题型并自主选择，校验题型集合与最多两题，不设置题型配额 |
| 环境与评测 | worktree 的环境层只接受 PhysicianBench | 从 main 同步 MedAgentBench 适配、原生执行器及公共评测更新 |
| 配置与运行记录 | 无 SingleShot 专属配置；已有 PhysicianBench 运行与三项方法测试 | 提供双环境配置，补齐 MedAgentBench 和更新后公共链路验收 |

已有真实记录位于 worktree 的 `outputs/singleshot-local-20261004-201849/`：PhysicianBench 单次生成两题，其中一题被核验拒绝，另一题通过核验并完成解题。这些记录仅证明该旧版本的 PhysicianBench 链路，不替代同步 main 后的双环境验收。

main 对齐指公共契约、环境适配、运行协议、实验入口和评测口径兼容，不要求复制主方法的生成算法、探索指导或进化策略。已有 SingleShot 实现作为适配起点，不重写环境客户端、LLM 客户端或 Agent 执行器。

SingleShot 的 generate_tasks 和生成提示词在 methods/singleshot/ 内独立维护，不调用 methods.medevolve.generate_tasks，也不导入 MedEvolve 的生成提示词。可按本方法的输入与单次生成要求调整提示词；复用范围为公共数据契约、模型调用、预算统计、输出约束及评测接口。

| 组件 | 职责 | 文件边界 |
| --- | --- | --- |
| SingleShot | 完整输入、单次生成、状态与成本 | methods/singleshot/__init__.py、methods/singleshot/prompts.py |
| 公共生成层 | 字段与题型集合约束、预算与结果汇总 | methods/common.py、contracts.py，增加必要的自由选题校验，保留其他方法的方向校验 |
| 公共实验入口 | 轨迹准备、传入五种题型、元数据、评测 | experiments/runner.py，singleshot 分派不采样两个方向 |
| 环境与运行层 | 双环境轨迹、证据回读、独立实例与原生协议 | environments/、runtime/，复用已有适配 |
| 配置与验收 | 双环境配置及方法边界检查 | configs/、tests/method/、tests/integration/ |

仅在公共入口不能接入方法时做必要兼容修改。其他方法算法、公共评测提示词、患者划分和统计规则不属于本方法的修改范围。

## 流程与输入隔离

公共轨迹准备 → 提供全部五种题型 → 模型自主选题并一次生成 → 公共证据与整题核验 → 质量评价 → 独立解题 → 完成评价与汇总。

SingleShot 只在 evaluation_patients 上生成任务，不使用 evolution_patients 或 round_patients 学习、探索或更新策略。公共入口直接提供 contracts.TaskType 定义的全部五种题型；模型依据患者资料选择适合的题型与具体问题。

task_types 是允许生成的题型集合，不是五个待完成的题目位置。一次最多生成两题，允许选择相同或不同题型，不要求每种题型都生成，也不按权重分配题数。SingleShot 不接收 directions，不依赖 initial_strategy.allocation 中的 difficulty 或 weight。

生成模型接收当前患者的 trajectory、全部五种 task_types 和字段、引用及数量约束。trajectory 包含患者标识、环境名称、完整 text，以及 sources 中的 source_id 和原始记录 statement；不只传来源 ID，不为节省输入而截断、抽样或概括患者事实。

环境的固定能力限制可由已有适配定义或方法提示词按 environment.name 提供，只描述已知原生协议和操作语义，不进行模型探索或环境试写。输入不包含主方法的数据探索指导、环境探索产物、generation_guidance、历史反馈、更新策略或公共评测答案。公共 ExperimentConfig 可保留 initial_strategy 字段以兼容现有配置，但它不参与 SingleShot 生成。

轨迹文本是患者数据，不能覆盖生成职责与输出接口。生成模型不获得工具；公共轨迹准备和公共评测的环境调用不计为 SingleShot 内部工具调用。

## 共享接口与输出

沿用公共 PatientTrajectory、EnvironmentConfig、TaskType、RuntimeConfig、RunResult 和 TaskBatch 数据契约。SingleShot 使用自主选题入口：

```python
async def generate_tasks(
    trajectory: PatientTrajectory,
    environment: EnvironmentConfig,
    task_types: list[TaskType],
    runtime: RuntimeConfig,
) -> RunResult: ...
```

调用模型前校验 trajectory.environment 与 environment.name 一致，task_types 完整包含公共五种题型且不重复。环境由公共层验证并准备，不在方法中重新创建实例。

生成使用 runtime.models["generator"]、runtime.budgets["generator"] 和 GenerationBudget.call_llm；以 TaskBatch 作为结构化输出，复用公共字段校验、预算与结果汇总。公共约束及结果校验增加按允许题型集合校验的路径，检查最多两题、合法题型、非空字段与 null 决策点，不套用要求恰好两个 directions 的 output_constraints 或题型位置配额。GenerationBudget.finish 的接入须支持该校验路径，不能先按两个方向拒绝自由选题输出。

experiments/runner.py 在 singleshot 分支传入全部五种 task_types，并使用同一题型集合校验返回结果，不执行该方法的方向采样。其他方法保留现有 directions 参数、采样和校验行为；公共核验、解题与统计判定保持一致。

TaskBatch.questions 最多两题，证据不足允许少出题或返回空列表。每题包含非空 instruction、task_type、非空 reference_answer、evidence 和 decision_point。task_type 必须属于允许的五种中文题型，两题可同类型。两个环境的 decision_point 均为 null。

题型为诊断与解释、检查与风险分层、治疗规划、药物处方、治疗重规划。患者、环境、轮次、方法和任务编号由公共入口附加；source_method 为 singleshot。

## 事实、答案与证据

生成提示词由 SingleShot 独立组织，模型依据实际资料选择有临床价值且有证据支持的题目，不额外指定难度分类，也不要求必须跨时间或整合多条记录。题干明确时间范围、作答或操作要求；事实真实性、引用有效性、候选答案合理性和环境可执行性遵循本节及公共评测要求。缺失信息、歧义或环境故障不能作为任务难度的依据。

每条患者事实对应当前患者轨迹中的真实 ResourceType/id。evidence.statement 只陈述该原始记录支持的事实，核对药物、剂量、数值、单位与时间；跨记录推断写在候选答案中并分别引用依据。活动处方不证明实际服用、疗效或耐受性，记录未出现某事实不证明该事实不存在。

每道非空任务须有非空证据。候选答案不是金标准；允许有依据的临床等价方案，真实后续治疗不是临床决策题的唯一答案。SingleShot 不在生成后调用模型纠正引用、重写答案或补题。

## 双环境约束

| 项目 | PhysicianBench | MedAgentBench |
| --- | --- | --- |
| 配置名称 | PhysicianBench | MedAgentBench |
| 患者轨迹 | 公共 prepare_trajectory 和 FHIR 来源记录 | 公共 prepare_trajectory 与已有 snapshot 回退 |
| 生成 | 缓存轨迹，单次 call_llm，无工具 | 缓存轨迹，单次 call_llm，无工具 |
| 解题 | 原生 FHIR FunctionTool，经公共 run_agent | NativeTools 的 GET / POST / FINISH，经公共原生执行器 |
| 证据与状态 | 公共 read_evidence、read_state | 公共 read_evidence、read_state |
| 操作语义 | 按真实返回及状态核对持久化操作 | POST 接受请求，不证明数据库写入 |
| 决策点 | null | null |

题目要求必须符合对应环境的现有能力。MedAgentBench 的操作题可要求提交具体请求，不得把 POST 接受语义描述为数据库创建或更新成功。PhysicianBench 的持久化操作依据已有工具真实能力表达，不推定任意资源均可写入。

解题和完成评价复用公共流程，不能假设两环境使用相同工具对象、工具名称或日志结构。环境故障不得成为放宽题目要求或核验标准的理由。

## 公共核验与评测

统一调用 evaluate_task，不按方法名设置特殊判定。公共层检查引用是否属于患者轨迹，按 source_id 回读原始记录，逐条核验事实，再核验题目与候选答案的依据充分性。缺失证据或不存在的引用由公共核验拒绝，不引入方法内部筛选。

通过核验后按共同的 clinical_value、correctness、longitudinal_integration、clarity 四个维度评价。固定 solver 只接收题干、患者、决策点和对应环境工具，不接收候选答案、证据或生成上下文。

完成评价根据题干、实际答案、操作轨迹和环境状态判定。分析、解释、评估或规划任务按文字作答要求判断；要求提交请求时核对请求和接受结果；要求持久化操作时核对实际资源变化。临床合理等价答案使用公共判定口径。

核验 invalid 与运行 failed 分别保存，核验拒绝的任务不进入解题。公共核验或解题的结果不触发再次生成。

## 模型、预算与故障处理

只要求生成侧 generator 角色，不要求 data_explorer、environment_explorer、feedback 或 updater。公共评测角色和预算沿用 main。模型名称、服务地址和密钥环境变量来自配置，不在方法里固定模型或读取其他方法的私有配置。

每名患者最多一次生成模型调用、零次生成工具调用。预算允许时仅调用一次；max_model_turns 为零时不调用模型，返回 budget_exhausted。预算上限大于一也不得追加调用。公共 runtime 的客户端重试保持关闭，不能绕过单次调用约束。

| 条件 | 处理 | 保存内容 |
| --- | --- | --- |
| 环境不匹配、题型集合不完整/重复或缺必要角色 | 调用模型前拒绝 | 可定位的输入或配置错误 |
| 生成预算为零 | budget_exhausted，无有效输出 | 零次模型调用及预算原因 |
| 请求失败、拒答、输出无法解析 | failed，无有效输出，不重试 | 已有 trace、实际用量和错误 |
| 数量、题型、决策点或公共字段约束不合法 | 公共校验返回 failed，不修复 | 原始被拒绝输出、trace 和成本 |
| 空批次 | completed，生成题数为零 | 已发生的调用、用量及原始结果 |

生成 completed 仅表示批次满足结构、题型集合和数量约束，不表示事实已核验正确。实例创建、回读、解题或释放失败按公共环境与评测状态记录；公共层已创建的实例按既有 finally 释放流程处理。

## 配置与运行产物

双环境分别提供 method="singleshot" 的可运行配置，复用对应 main 配置的环境镜像、患者、种子、模型角色和公共评测预算；initial_strategy 仅作为公共配置兼容字段保留，不用于题型或难度选择。生成配置可显式设 max_model_turns=1、max_tool_calls=0；不改变同环境对比使用的模型、数据或公共评测要求。运行编号与输出目录独立，避免覆盖其他方法产物。

PhysicianBench 配置以 main 的 configs/physicianbench-flash.json 为模板，MedAgentBench 配置以 configs/medagentbench-flash.json 为模板，分别提供 configs/physicianbench-singleshot.json 和 configs/medagentbench-singleshot.json。现有模板的模型为 deepseek-v4.1-flash；实施时以基准提交中的对应配置为准，不沿用旧运行记录或旧 PRD 中固定的 gpt-4o-mini 要求。旧 PRD 的从零实现步骤和公共配置交接安排由本规范及 PLAN.md 的适配任务替代。

复用公共 config、attempts、tasks、evaluations、runs、summary 和错误产物。usage 记录 model_turns、tool_calls、input_tokens、output_tokens、total_tokens；未知 token 保留未知状态，不作为零消耗。

生成与公共评测成本分别统计，失败、空批次与被拒绝输出的生成成本也计入总生成成本。不得只报告通过核验的任务成本，不能把 solver 工具调用算作生成工具调用。

## 验收标准

| 编号 | 要求 | 验证证据 |
| --- | --- | --- |
| A1 | 与已记录的 main 提交兼容，公共接口与分派正常 | worktree 的基准提交、公共入口运行结果、相关回归检查 |
| A2 | 一次生成、零工具，无探索、筛选、修复或进化 | 成功、失败、拒绝和零预算路径的调用记录；两个环境的参数化检查 |
| A3 | 完整真实输入与自主选题约束 | 正文及原始 sources、全部五种题型进入输入；最多两题、同题型两题、少题/空批次与非法题型检查；入口及结果校验不使用两个方向配额 |
| A4 | 两环境复用原生公共评测，答案隔离及操作语义正确 | 公共核验、solver 输入和轨迹；请求接受与数据库写入区分 |
| A5 | 双环境配置和成本产物可直接使用 | 每个环境至少一名真实评测患者的生成、核验、解题产物和分阶段成本 |

本地检查聚焦调用边界、原始输入、预算、输出拒绝以及两种原生解题协议的公共入口连接。已有公共证据与完成判定测试直接复用，不为提示词字符串或简单字段转发建立重复测试。

真实验收每个环境至少运行一名评测患者，保留实际失败、空批次或核验拒绝，不为了通过验收更换事实、降低标准或自动重试。若该患者没有通过核验的任务，须如实报告未覆盖真实解题链路；双环境完整链路验收仍未完成。模型和服务使用项目已授权配置，新增付费或高价模型调用另行确认。

实施计划文件为同目录 PLAN.md，按上述接口、文件边界和验收编号组织任务。
