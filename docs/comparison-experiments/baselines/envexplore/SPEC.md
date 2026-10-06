# EnvExplore 设计规范

## 方法定义

EnvExplore 是 MedEvolve 的消融对比方法：保留患者轨迹整理、环境探索、单患者任务生成和公共评测，不执行数据探索、反馈分析与策略更新。生成指导及题型难度配比固定使用实验的 initial_strategy。

该方法按本项目定义报告，不对应已确认的论文。与主方法比较时，环境初始数据、患者划分、模型角色配置、初始策略、生成方向安排及公共评测规则保持一致。方法差异仅为数据探索和自进化是否启用。

## 目标与非目标

目标是比较同一医疗环境下，主方法的数据探索与自进化对任务生成的影响。EnvExplore 使用相同初始策略、生成规则、环境探索和公共评测，只固定策略并省略数据探索。PhysicianBench 与 MedAgentBench 都是必需支持的环境，不能只通过一个环境验收。

本方法不增加逐患者自由工具出题、内部模型筛选、自动答案修复、单独标注体系或新的 Agent 执行器。实验结果只报告实际任务与评测表现，不把模型核验当成人工金标准。

## 组件职责与依赖

| 组件 | 职责与输出 | 依赖与文件边界 |
| --- | --- | --- |
| 公共实验入口 | 验证配置、准备轨迹、执行一次探索、绑定固定上下文、采样方向并保存产物 | experiments/runner.py，现有 contracts 和环境层 |
| 共享生成规则 | 提供与主方法一致的轨迹组织、临床与引用规则；各方法描述自己的输入 | methods/common.py、主方法及 EnvExplore 的生成提示词 |
| EnvExplore 生成入口 | 用轨迹、环境指导、固定策略及方向单次生成 TaskBatch，返回带成本的 RunResult | methods/envexplore/，共享生成约束与 runtime.call_llm |
| 环境与运行层 | 提供患者隔离、独立实例、证据回读及对应原生工具协议 | environments/、runtime/；复用主方法环境探索入口 |
| 公共评价与统计 | 逐条证据和整题核验、四维评价、独立解题、完成判定及成本汇总 | evaluation/、experiments/summary.py；各方法相同判定 |

共享生成规则与模型输入保持单一来源。抽取时保持主方法的输入内容及临床规则不变，不能因接入 EnvExplore 改写主方法的算法或候选答案口径。

## 流程与患者边界

患者轨迹整理 → 一次环境探索 → 固定策略单患者出题 → 公共核验 → 质量评价 → 独立解题 → 完成评价。

环境探索在每次实验开始时执行一次，仅访问主方法同一划分中的探索患者。当前配置使用 evolution_patients 指定这组患者；EnvExplore 使用该列表探索环境能力，不执行进化轮次。evaluation_patients 不参与探索。探索患者列表为空时配置校验失败，不自动改用评测患者。

环境探索复用主方法 explore_environment，输出不绑定具体患者的 EnvironmentExploration，描述工具能力、参数、返回结构、操作限制和故障。工具 Schema 声明与实际验证结果必须区分；未验证的能力不能写成已验证。指导中不包含患者 ID、来源 ID、具体诊疗事实或试写内容。

环境探索可按主方法规则在独立实例中验证原生读写能力。试写记录仅属于该探索实例，不进入出题轨迹、参考答案或公共核验材料。探索完成、失败或预算耗尽均释放实例，探索实例不能传给生成或解题阶段。

生成使用公共环境层准备的评测患者轨迹，包括完整正文及 sources 中的原始记录。每名患者一次 call_llm 同时生成题目、候选参考答案及证据，不向生成模型开放工具。环境指导与初始策略在整次实验中固定，公共评测结果不改变后续策略。

## 共享接口与上下文

沿用 [公共规范](../../common/SPEC.md) 的契约及患者批次调用方式：

```python
async def generate_tasks(
    trajectory: PatientTrajectory,
    environment: EnvironmentConfig,
    directions: list[GenerationDirection],
    runtime: RuntimeConfig,
) -> RunResult: ...
```

公共实验入口在完成环境探索后，通过本次运行绑定的调用包装器向方法传入 environment_guidance 和固定 strategy。方法实现采用以下签名，公共批次调用仍使用上述四个参数：

```python
async def generate_tasks(
    trajectory: PatientTrajectory,
    environment: EnvironmentConfig,
    directions: list[GenerationDirection],
    runtime: RuntimeConfig,
    *,
    environment_guidance: EnvironmentExploration,
    strategy: Strategy,
) -> RunResult: ...
```

environment_guidance 来自本次 explore_environment 的已完成结果，strategy 为 config.initial_strategy。公共入口的本次运行闭包负责传递这两个参数，方法解析保持现有显式映射。不使用模块全局变量存放指导、策略或实例，不为每名患者重复环境探索。

生成材料由 trajectory、environment_guidance、strategy.generation_guidance 与两个 directions 组成。trajectory.environment 必须与 environment.name 一致，directions 必须包含两个生成位置。输入不包含数据探索指导、反馈或更新后的策略。

输出为 TaskBatch，questions 最多两题，证据不足时允许少出题或返回空列表。每题包含 instruction、task_type、reference_answer、evidence、decision_point。题型数量不得超过指定方向中对应题型的位置数，当前两个环境的 decision_point 均为 null。患者、环境、轮次、方法与任务编号由公共入口附加。

## 事实、答案与证据

复用 [主方法规范](../../main/SPEC.md) 的事实选取、引用与候选答案规则，以及主方法生成提示词中的临床约束。共享规则来自同一处实现；EnvExplore 的输入描述只包含患者轨迹、环境指导、固定策略和两个方向。

先从 trajectory.sources 选择原始事实，再构造题目、答案和证据。source_id 使用真实的 ResourceType/id，必须属于当前患者的输入来源集合；每条 statement 只陈述该原始记录支持的事实。跨记录推断放在候选答案中，并分别引用其依据。

非空题目必须具有非空证据、题干和候选答案。核对药物、剂量、数值、单位及事件时间；活动处方不证明实际服用、疗效或耐受性。不能将轨迹未出现某事实解释为该事实不存在，也不能用空查询、故障或探索试写补充患者事实。

题干明确时间范围、必要条件和作答或操作要求；相对时间给出明确时间与时区。候选答案区分记录事实、临床推断和材料限制，允许有依据的临床等价方案。生成流程完成不等于候选题目核验正确。

## 公共核验与解题

统一调用 evaluate_task，不按 source_method 改变判定。公共层在独立环境按 source_id 回读原始记录，逐条检查证据声明，再判断题目明确性、候选答案回应性及依据充分性。核验拒绝的题目不进入解题。

质量评价采用主方法相同的四个维度。固定解题 Agent 仅收到题干、患者、决策点及该环境的原生工具，不收到候选答案、证据、环境探索指导或策略。完成评价根据实际操作轨迹和环境状态判断，接受临床合理等价答案。

缺失证据、事实不支持、运行故障与预算耗尽分别记录，不能解释为临床难度提升。候选题目、被拒绝的输出及核验理由保留在运行产物中。

## 运行状态与故障处理

| 条件 | 处理与输出 | 记录要求 |
| --- | --- | --- |
| 配置缺角色、探索组为空、患者组交叉或环境不匹配 | 调用模型前拒绝；不改用评测患者探索 | 保存可定位的配置或输入错误 |
| 探索失败、预算耗尽或指导无法解析/为空 | 终止实验，不进入生成 | 保留探索 RunResult、实际用量与失败原因 |
| 实例创建或释放失败 | 记录环境故障，不将该次探索判为成功 | 已获得的实例在 finally 中执行释放 |
| 生成预算耗尽或结构/方向不合法 | 无有效输出；分别返回 budget_exhausted 或 failed | 保留原始输出、trace 和调用成本 |
| 候选没有证据或引用不在轨迹中 | 来源检查逐题剔除；其余题继续公共核验 | 保存被剔除题目与原因，允许最终空批次 |

成功解析的候选只表示生成完成。公共核验 invalid 与运行 failed 分别保存；没有核验结果时不能计算成核验通过。公共核验失败后不在方法内补写答案或重新出题。

## 两个环境的支持要求

| 项目 | PhysicianBench | MedAgentBench |
| --- | --- | --- |
| 环境名称 | PhysicianBench | MedAgentBench |
| 患者轨迹 | 公共 prepare_trajectory 与 FHIR 来源记录 | 公共 prepare_trajectory 与现有 snapshot 回退 |
| 探索与解题工具 | 原生 FHIR FunctionTool | NativeTools 的 GET / POST / FINISH 协议 |
| 模型执行 | 公共 run_agent | 公共 run_agent 分派到现有原生执行器 |
| 证据回读 | 公共 read_evidence | 公共 read_evidence |
| 写入判定 | 结合真实返回与持久化状态 | POST 接受请求，不证明数据库写入 |
| 决策点 | null | null |

根据 environment.name 复用现有适配器，不建立 EnvExplore 专属环境客户端或 Agent 循环。PhysicianBench 的工具对象检查、工具名称和日志结构不能成为 MedAgentBench 的前置假设。

MedAgentBench 的操作任务明确要求提交请求，完成评价按原生协议和实际执行记录判定，不以新增数据库记录作为提交成功的必要条件。PhysicianBench 的持久化操作按其真实能力核对。环境故障不能作为放宽答案或操作要求的理由。

## 模型、预算与运行产物

环境探索使用 runtime.models["environment_explorer"] 及 runtime.budgets["environment_explorer"]；患者生成使用 runtime.models["generator"] 及 runtime.budgets["generator"]。对应角色与主方法使用相同配置，不在方法中固定模型名称。

不要求 data_explorer、feedback 或 updater 角色。每批生成最多一次模型调用、零次工具调用；预算不足时不调用模型、不产出有效任务。探索预算独立累计，不从每名患者的 generator 预算重复扣除。

usage 统一记录 model_turns、tool_calls、input_tokens、output_tokens、total_tokens。保存环境探索的全部调用与故障记录；患者生成与公共评测分别保存。汇总同时报告一次性探索成本、患者生成成本和公共评测成本，方法总成本包含探索与全部生成尝试，不能仅统计最终通过的题目。

## 实现与验收边界

方法代码位于 methods/envexplore/；实验入口负责一次环境探索、固定策略上下文、方向采样、任务元数据和产物汇总。复用现有 contracts、环境层、运行层及公共评测，不修改其他方法的算法或公共判定口径。

两个环境均须通过本地接口检查及至少一名真实评测患者的完整生成、核验和解题检查。真实生成不足两题、核验拒绝或环境故障按实际结果报告，不将本地测试通过当作真实效果证明。

### 可验证的验收要求

| 编号 | 要求 | 验证证据 |
| --- | --- | --- |
| A1 | 只省略数据探索与自进化，保留一次环境探索及固定策略 | 禁用函数未调用，多个评测患者共享同一指导与初始策略 |
| A2 | 患者边界及方向采样与主方法对齐 | 探索/评测组无交叉；同种子和初始配比得到同一方向安排 |
| A3 | 使用真实轨迹，单次生成最多两题 | 输入包含正文和原始 sources；模型调用最多一次、工具调用为零 |
| A4 | 生成与核验分离，解题答案隔离 | 错误事实被公共核验拒绝；solver 请求不含候选答案或证据 |
| A5 | 两个环境按原生协议运行 | FunctionTool 与 NativeTools 各自完成探索和解题；POST 提交不冒充持久化 |
| A6 | 故障、预算与实例释放正确 | 失败或耗尽不产出有效任务，已创建实例执行释放，失败成本保留 |
| A7 | 成本与产物完整 | 一次探索只计一次，全部生成及评测调用可从 runs 回溯 |

本地检查采用模拟环境和模型验证调用边界、状态与上下文，公共事实判定复用既有测试。真实验收分别选择至少一名探索患者与一名不重叠的评测患者，核对任务、来源记录、公共评价及操作状态；拒绝和故障保留原始结果。

实施计划位于 [PLAN.md](PLAN.md)，按组件接口、依赖顺序和上述验收编号组织任务。
