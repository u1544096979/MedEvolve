# 只读任务与患者并行实施计划

## 目标与约束

同一个方法同时为多个患者生成查询任务；生成完成即释放生成名额，所有最终任务进入统一公共评测。生成与 task 评测分别配置并发数，同一环境及固定种子在一次运行中复用一个逻辑实例。

实施依据 [设计方案](design.md)。保留六方法内部阶段、题型选择、累计预算、随机流、筛选机制及公共评测指标；不改变 Harness，不调度多个方法同时运行。环境探索、生成、内置核验和公共解题全部只读，MedAgentBench 模拟 POST 也禁止。环境种子初始化和实验输出文件保存沿用现有机制。

实施使用独立 worktree，按下列五项任务分层提交。测试使用替代模型，不调用付费模型；真实环境检查使用已准备镜像和种子，不拉取新资产。提交 description 与 body 使用中文。

## 基线与文件清单

main 已存在 `medevolve`、`workflow`、`selfgen`、`singleshot`、`envexplore`、`taskcraft` 六个入口。当前 runner 的患者名额同时包住生成与评测；EvaluationEnvironments 仅覆盖公共评测，PhysicianBench 解题仍逐 task 创建实例；探索与 TaskCraft 生成仍需接入同一运行会话。

| 文件 | 实施职责 |
| --- | --- |
| contracts.py；新增 environments/capabilities.py | 两个并发参数、统一只读任务要求、固定能力和核验字段 |
| methods/common.py、methods/generation.py、六方法实现与提示 | 统一说明、只读任务输出、候选字段保留及方法机制兼容 |
| evaluation/session.py、environments/{__init__,fhir,medagentbench,errors}.py | 完整运行共享会话、患者工具绑定、修改请求阻断及清理 |
| 新增 experiments/scheduling.py；experiments/runner.py、configs/experiment.json | 独立生成与评测队列、配置回退、即时保存和轮次屏障 |
| evaluation/{evaluator,prompts}.py；tests/{method,evaluation,experiment,environment,integration}；docs/comparison-experiments/common/SPEC.md | 只读题面核验、回答完成判定、并发回归和公共实验规范对齐 |

`common/SPEC.md` 当前环境复用说明位于第 59 行，实施时改为：所有阶段仅执行查询任务，同一环境及固定种子在一次运行中复用一个共享逻辑实例，患者会话与输出独立；具体遵循环境能力与患者并行设计。源文件行号为定位参考，实施时以函数及当前内容为准。

## 六方法适配清单

| 方法 | 当前定位 | 必做适配与不变量 |
| --- | --- | --- |
| medevolve | methods/medevolve/__init__.py:122、160、186；methods/generation.py:5 | 探索接入共享会话；探索、生成、反馈注入只读说明；同轮策略固定，全部评测结束才更新 |
| workflow | methods/workflow/__init__.py:117；methods/workflow/prompts.py | 定位、按类型生成与内部检查限定查询任务；保留五类选两类、每类最多一题及预算 |
| selfgen | methods/selfgen/__init__.py:83、94；methods/selfgen/prompts.py:8、22 | 删除 operation_semantics 环境文案副本；初稿、自检、修订统一注入；保留通过题字段与修订预算 |
| singleshot | methods/singleshot/__init__.py:24；methods/singleshot/prompts.py:13–14 | 删除重复环境语义，统一注入并输出只读要求；保留 task_types、一次调用和两题可同类型 |
| envexplore | methods/envexplore/__init__.py:21；methods/generation.py:5；runner 固定指导绑定器 | 一次只读探索使用共享实例；GENERATION_RULES 删除提交与持久化语义副本；固定策略，无逐患者探索 |
| taskcraft | methods/taskcraft/__init__.py:55；ledger.py:16；stages.py、schemas.py；environment.py:56、88 | 生成共享实例，独立工具绑定；Ledger 和各阶段只产生查询任务；保留 Pipeline、候选、内置核验、预算和随机流隔离 |

TaskCraft 在 `codex/taskcraft-baseline` 继续迭代 stages.py、prompts.py 和机制测试，包含 request_attempts 重试规则。若先合入在研改动，按合入结构适配；若先实施本方案，再 rebase 分支并保留双方会话、只读约束、候选字段及重试行为。合入后重新执行 TaskCraft 机制与公共并发回归。

SelfGen 环境白名单和 TaskCraft 工具协议需逐项核对；能力说明接入不代表所有三环境均适配完成。保持 methods/common.py 已并存的 directions、task_types、max_questions、candidate_pool 接口。

## 任务 1：统一只读说明与并发配置

**文件：** contracts.py、environments/capabilities.py、experiments/runner.py、configs/experiment.json；新增能力与配置测试。

1. 建立固定能力来源，记录三环境真实语义和本轮只读限制。提供 `get_capabilities(name)` 和 `capability_instructions(name)`；说明禁止持久化写入及模拟提交，允许基于既有记录形成临床建议。
2. GeneratedQuestion 增加 `execution_requirement: Literal["read_only"] | None = None`，兼容旧保存任务；新生成公共校验要求明确填写 read_only。Verification 增加可选 `read_only_task: bool | None = None`，仅整体核验必须返回，单条事实核验不承担范围判断。
3. ExperimentConfig 增加可选正整数 generation_concurrency 与 evaluation_concurrency；分别按显式值、有效 MEDEVOLVE_RUN_CONCURRENCY、默认 6 解析。运行 config.json 保存有效值，示例配置显式给出 6 和 6，可分别设为 5 和 2。
4. 能力测试覆盖三环境统一限制；配置测试覆盖独立覆盖、旧配置回退和非正整数拒绝。旧任务字段缺失可读取，新输出缺失或其他要求不能通过公共校验。
5. 运行相关测试并提交：`feat(config): 增加只读任务约束与独立并发参数`。


## 任务 2：六方法统一生成查询任务

**文件：** methods/common.py、methods/generation.py、六方法实现与提示、TaskCraft schemas.py/stages.py/ledger.py/environment.py；tests/method。

1. 按六方法清单将统一说明注入所有已有模型调用包装器，涵盖探索、生成、内部检查、修订及反馈。删除重复环境语义，保留方法专属事实、证据与题型规则。
2. 所有新最终任务填写 read_only；候选扩展、修订和过滤保留字段。TaskCraft 在已有阶段输出与候选构造中传递要求，不增加模型调用；不得因生成工具只读就跳过题面只读约束。
3. 通过替代模型捕获各节点指令与输出，验证六方法均使用同一说明；用提交医嘱、模拟 POST、依赖新增资源的候选检查内部提示和既有筛选约束。实际题面最终由公共整体核验判断，公共字段校验不能替代语义核验。
4. 保留各方法现有调用次数、预算、题型选择、引用过滤和每患者随机流断言；支持组合分别记录，不补造尚未适配的协议。
5. 运行 methods 与既有方法实验回归并提交：`feat(methods): 统一六方法只读任务生成`。

## 任务 3：所有阶段复用只读共享实例

**文件：** evaluation/session.py、environments/{__init__,fhir,medagentbench,errors}.py、methods/medevolve/__init__.py、methods/taskcraft/__init__.py、experiments/runner.py；相关环境、会话与方法测试。

1. 扩展 EvaluationEnvironments 支持 prepare/explore/generate/verify/solve，所有阶段共享同一配置实例，PhysicianBench solve 也共享。键包含环境名、manager_url 和固定 seed_data，同键并发首次申请只创建一次。沿用逻辑实例额度，创建中、存活与释放中均计数。
2. 轨迹准备、环境探索、TaskCraft 生成及公共评测全部接入运行会话。explore_environment、TaskCraft generate_tasks 及 bind_generator 接受可选 environments 参数；入口单独调用时自建会话并清理，运行内调用只结束借用，不关闭共享实例。TaskCraft binder 继续绑定 random_seed。
3. 公共工具入口提供只读视图，覆盖所有阶段。函数工具按环境声明的查询能力筛选，FHIR 客户端默认保持既有兼容行为、只读绑定显式禁用修改请求；MedAgentBench 视图仅允许 GET/FINISH，在 source.execute 前拒绝 POST。TaskCraft 沿用生成只读视图，统一底层限制，不能修改来源工具对象。
4. 每患者独立允许 ID、工具对象、上下文、轨迹和输出路径；不修改共享实例患者字段。记录借用数，close 在全部借用收尾后释放一次；失败、取消创建和取消等待均清理，释放失败保留具体错误。测试同键复用、不同配置不混用、五患者并发、跨患者访问拒绝、各阶段修改请求拒绝和取消清理。
5. 运行会话、生命周期、患者边界、环境协议及 TaskCraft 回归并提交：`feat(environment): 全流程复用只读共享实例`。

单环境运行峰值为一个逻辑实例。PhysicianBench 与 MedAgentBench 各一个物理容器；FHIR-AgentBench 为两个物理容器。测试分别统计逻辑实例和物理容器，不将两者等同。共享实例在运行结束释放，单患者生成完成只释放名额与借用。

## 任务 4：分离患者生成与 task 评测调度

**文件：** 新增 experiments/scheduling.py、tests/experiment/test_generation_pipeline.py；experiments/runner.py、tests/experiment、tests/integration。

1. 调度器接收患者 plans、generation_limit、evaluation_limit，以及 generate/evaluate 回调。生成回调执行单患者完整方法流程并返回最终 tasks；评测回调执行一个 task 的完整公共评测。
2. 患者生成名额在 generate 返回或失败时释放，再将最终 tasks 放入评测队列；不能持有生成名额等待评测。评测 worker 数遵守独立上限，单患者两题分别作为队列项目。有限患者批次可沿用内存队列，不增加额外排队参数。
3. runner 保留并发前的顺序方向采样、生成输入裁剪及公共评测原始轨迹；generate 立即保存 attempts、生成记录和最终 tasks，evaluate 保存单 task 结果。共用现有输出锁，按稳定患者/题下标组织文件、ID 与成本归属。
4. 事件测试验证评测阻塞期间下一患者仍能生成、两上限独立生效、空输出不评测、单任务失败不丢其他结果、取消收尾，以及六方法内部预算与顺序保持。调度器返回时本轮所有生成和评测结束；MedEvolve 此后才 feedback/update，初始与最终评测阶段顺序保持。
5. 运行调度、闭环、方法实验与集成回归并提交：`feat(experiment): 分离患者生成与任务评测并发`。

旧 scripts/run_full_comparison.py 环境变量配置继续兼容，无需重写运行脚本；不批量改写全量实验配置。

## 任务 5：只读题面核验与整体验收

**文件：** evaluation/{evaluator,prompts}.py、docs/comparison-experiments/common/SPEC.md；新增只读任务评测与共享环境 live 测试，扩展现有评测测试。

1. 整体 verifier 在既有调用中依据实际题面和参考答案返回 read_only_task，判断能否仅查询既有资源完成。单条证据 verifier 保持事实核验职责。整体字段缺失视为模型输出错误；false 在质量、难度和 solver 前判 invalid，failure_category 使用明确任务范围分类，如 unsupported_operation。
2. 新任务的 read_only 标签不能覆盖题面核验；旧任务缺少标签仍通过同一判断。只要求给出诊断、解释、建议可以通过；提交、持久化、模拟 POST 和依赖新建记录必须拒绝。不用题型、关键词或生成工具替代整体判断，不增加分类模型调用。
3. 质量、难度、solver 和 completion 使用统一只读说明。保留质量和难度各三次评分与中位数、既有复杂度规则和参考答案输入隔离。completion 对照回答与查询轨迹判定，不要求新增资源或提交成功。工具越界尝试记录拒绝与失败原因，不伪装为完成。
4. 将 common/SPEC.md 的环境与任务范围说明对齐为只读共享实例；完成离线回归和显式启用的环境检查。TaskCraft 在研分支合入后再跑机制、预算、重试、共享绑定、患者边界及公共并发测试。
5. 测试通过后提交并记录实际验证范围：`feat(evaluation): 核验只读任务并对齐公共评测`。

离线回归排除需要真实环境或模型的测试：

```bash
python -m pytest tests --ignore=tests/environment/test_live_image.py --ignore=tests/environment/test_medagentbench_live.py --ignore=tests/environment/test_fhir_agentbench_live.py --ignore=tests/method/test_taskcraft_live.py --ignore=tests/environment/test_shared_generation_live.py -q
```

新增 live 检查使用 MEDEVOLVE_TEST_IMAGE、MEDEVOLVE_MEDAGENTBENCH_TEST_IMAGE 的已加载镜像或 MEDEVOLVE_FHIR_AGENTBENCH_SEED 指定的已准备种子，无资产时 skip。五患者只读并发覆盖准备、探索、生成和解题的共享绑定，验证同一实例、患者范围、只读工具，以及结束后的实例释放；修改请求拒绝使用拦截计数确认没有向服务发送，不执行真实写入。检查期间不调用模型，使用 finally 清理。

真实模型对只读题面的遵守率需在另外明确授权的生成运行中验证；替代模型测试只证明接线、字段和调度规则。

## 完成标准

六方法的新任务明确标记 read_only，公共核验拒绝提交及写入要求；所有 Agent 阶段仅开放查询；五患者生成与 task 评测分别遵守上限，生成名额在评测结束前释放；同一环境和固定种子仅创建一个共享逻辑实例；患者上下文、文件和成本记录独立；方法机制、公共指标、MedEvolve 轮次屏障及失败/取消清理通过验证。
