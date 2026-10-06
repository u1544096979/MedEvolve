# MedEvolve Specification

## 1. 研究目标与实验边界

系统通过更新出题指导和生成配比，逐轮生成质量更好、难度更高的医疗任务。模型权重固定。质量评价、固定解题器的提示词、模型和预算保持一致。

项目在独立的 MedEvolve 仓库实现。参考 AgentEvolver 的环境交互、探索、任务总结与核验分工，适用代码和提示词按需复用。运行代码不依赖其训练数据集、PPO、训练 tokenizer 或经验训练模块。

五类初始任务方向为诊断与解释、检查与风险分层、治疗规划、药物处方、治疗重规划。出题模式由探索与反馈沉淀为自然语言指导。

### 患者与轮次

患者固定分为进化组与评测组，两组不相交。进化患者参与初始数据探索、迭代出题与反馈更新。评测患者只在固定策略后的评测阶段使用，不参与初始探索或策略更新。

初始阶段整理并缓存患者轨迹；缓存评测患者轨迹不意味着向探索角色开放它们。各轮按配置中的进化患者列表逐个出题，每次只传一个患者，每患者每轮尝试生成 2 题。每轮可以配置不同患者列表，分组本身不变。

题型与难度方向配比控制这 2 个生成位置的安排，不改变生成数量。不足 2 题时记录实际产量，不补造任务。最终评测默认使用最终策略；各轮策略快照允许在相同评测条件下比较，评测结果不回流。

## 2. 主流程

```text
整理并缓存患者轨迹
        │
        ├─ 数据探索：自由读取多位进化患者的轨迹 ─► 数据探索指导
        └─ 环境探索：交互了解原生工具与约束   ─► 环境探索指导
                                                        │
进化患者单人轨迹 + 两份初始指导 + 当前策略 ────────────────┘
        │
        ▼
单次 LLM 出题（尝试 2 题）
        │
        ▼
公共核验、质量评价、固定 Agent 试解
        │
        ▼
单次 LLM 反馈分析 ─► 单次 LLM 策略更新 ─► 下一轮出题
        │
        ▼
固定策略，在评测患者上生成并评测
```

### 两种初始探索

数据探索是主方法的独立组成部分。探索 Agent 自由读取多位进化患者的缓存轨迹，归纳数据组织方式、病程关联、可利用的信息变化和出题机会。输出为不绑定具体患者的抽象指导；原始读取轨迹单独保存。

环境探索 Agent 在独立实例中了解工具名称、参数、返回结构、原生读写能力与操作约束。输出为抽象的环境指导。它不替代数据探索，也不将具体患者观察当成公共指导。

两种探索只在初始阶段运行，其指导在实验内保持固定。历轮经验和新模式进入可更新的出题指导。

### 单次任务生成

一次调用接收一个患者的缓存轨迹、两份初始指导、当前出题指导，以及本次两个题型／难度方向。生成角色没有环境查询工具。

每题生成题目、题型、参考答案和数据证据。患者 ID、环境名称、任务 ID、轮次和来源方法由调用端附加，不能依赖模型自行命名。证据引用轨迹中的来源 ID。参考答案经公共核验后使用，不建设单独标注体系，也不称为金标准。

### 反馈与更新

反馈分析接收本轮任务、其患者轨迹、核验、质量评价、解题轨迹和操作状态，输出质量问题、难度表现、失败原因与改进建议。模型分析与执行记录分开保存，完成情况和故障事实不能由反馈模型改写。

策略更新只接收当前策略与反馈分析结果，更新出题指导和配比。生成配比为带题型与难度描述的权重列表；实验入口按权重和固定随机种子抽取每患者的两个生成方向。只检查合法题型、非负权重和至少一个正权重，不限制自然语言模式如何变化。

## 3. 公共环境与模型运行

### 环境

模型调用与实验入口在本机；Docker 服务和数据存储在远程。远程管理服务创建实例、返回可由本机访问的原生服务地址、释放实例。本机业务工具直接访问对应实例。

轨迹整理、环境探索、任务核验和固定解题分别使用独立实例，均从指定初始数据启动。实例之间不共享可写存储。写入仅限环境原生开放的能力；任务核验和解题不能继承探索产生的写入。

患者轨迹的具体内容由环境适配决定。统一提供按时间组织的轨迹正文与原始记录引用，保留用于出题和核验的关键事实。证据引用须可回查原始数据，压缩不能凭空补充事实。初始数据不可变时，同一实验内复用缓存。

各环境适配独立实现；只有通过实际原生工具和隔离验证的环境才声明可运行。PhysicianBench、MedAgentBench、FHIR-AgentBench 的具体启动、工具与数据要求按其原始实现核对。

### 模型运行

提供两个入口：单次 LLM 调用、SDK Agent 运行。单次入口完成一次模型请求，不隐式增加反思或修复调用；解析失败保存原始响应并返回失败状态。Agent 入口使用 OpenAI Agents SDK 的 Agent、Runner 与工具封装，由 SDK 组织多轮调用。

共享运行结果包含最终结构化输出、调用轨迹、模型用量、完成状态与错误。模型轮次预算与工具调用预算分开记录。模型名称、连接地址和 API key 环境变量由配置指定；不硬编码模型或价格。

## 4. 公共评测与决策点

公共评测先核验题目和参考答案是否有数据证据支持，评价质量，再运行固定解题 Agent。首版分别记录临床价值、正确性、纵向整合需求和清晰度的评价与依据，不设加权总分。

解题 Agent 仅接收题目、患者 ID、环境工具和可选决策点，不接收参考答案、证据列表或完整缓存轨迹。评审角色可读取这些材料。查询任务评价答案；写入任务同时检查操作后状态。运行故障与临床未完成分开记录，不将故障当成难度提升。

方法名称不进入评审输入。主方法与对比方法在同一环境内共用初始数据、患者范围、评测配置和解题预算；方法的探索及生成算法保持各自特点。

### 决策点

决策点是可选任务字段，默认为空。启用时，解题查询工具只返回决策点之前的信息；完整轨迹中的之后信息可供出题与核验生成参考答案。后续真实治疗不是临床决策题的唯一正确方案。

时间可见范围由环境适配保证，覆盖列表检索、按 ID 获取与关联读取。预留字段和能力入口；首版未实现时间过滤的环境必须明确拒绝带决策点的任务，不能静默暴露完整记录。具体适配完成后，再开放此类任务。

## 5. 共享数据与接口

`contracts.py` 是共享结构的唯一代码定义，归属模型运行模块的基础任务。五个模块引用它，不建立各自同名结构。所有函数为异步，只有文件保存与结果汇总使用同步函数。

以下类型映射为 Pydantic 模型；`JsonObject` 表示可 JSON 序列化的字典。时间点采用含时区的 ISO 8601 字符串，由环境解释其可见语义。

| 类型 | 字段 |
| --- | --- |
| EnvironmentConfig | name: str, manager_url: str, seed_data: str |
| EnvironmentInstance | instance_id: str, endpoint: str, environment: str |
| Evidence | source_id: str, statement: str |
| PatientTrajectory | patient_id: str, environment: str, text: str, sources: list[Evidence] |
| ExplorationGuidance | data_guidance: str, environment_guidance: str |
| GenerationDirection | task_type: str, difficulty: str, weight: float |
| Strategy | generation_guidance: str, allocation: list[GenerationDirection] |
| GeneratedQuestion | instruction: str, task_type: str, reference_answer: str, evidence: list[Evidence], decision_point: str 或空 |
| TaskBatch | questions: list[GeneratedQuestion] |
| Task | GeneratedQuestion 的字段，加 id: str, patient_id: str, environment: str, round_index: int, source_method: str |
| ModelConfig | model: str, base_url: str 或空, api_key_env: str |
| Budget | max_model_turns: int, max_tool_calls: int |
| RunResult | output: JsonObject 或空, trace: list[JsonObject], usage: JsonObject, status: str, error: str 或空 |
| Verification | valid: bool, reason: str |
| QualityJudgment | 单裁判输出，四个维度（`factual_accuracy`、`clinical_soundness`、`evidence_sufficiency`、`task_clarity`）各含 `score: int` 与 `reason: str` |
| QualityReview | 四个固定维度 `factual_accuracy`、`clinical_soundness`、`evidence_sufficiency`、`task_clarity`，每维含 `score: int`、`scores: list[int]`、`reason: str` |
| ComplexityMetrics | `evidence_count`、`resource_type_count`、`time_span_days` |
| DifficultyJudgment | 单裁判输出，含 `reasoning_depth`、`reason` |
| DifficultyScore | `reasoning_depth`、`reasoning_depths`、`reason`、`complexity_metrics` |
| CompletionReview | completed: bool, reason: str |
| EvaluationResult | task_id: str, verification: Verification 或空, quality: QualityReview 或空, difficulty: DifficultyScore 或空, completion: CompletionReview 或空, solver_run: RunResult 或空, observed_state: JsonObject, status: str, failure_category: str 或空, runs: list[RunResult] |
| Feedback | quality_findings: list[str], difficulty_findings: list[str], failure_findings: list[str], suggestions: list[str] |
| SolverInput | instruction: str, patient_id: str, decision_point: str 或空 |
| SolverAnswer | answer: str |
| DataExploration | guidance: str |
| EnvironmentExploration | guidance: str |
| RuntimeConfig | models: dict[str, ModelConfig], budgets: dict[str, Budget] |
| EvaluationConfig | runtime: RuntimeConfig，角色为 verifier、quality、difficulty、solver、completion |
| ExperimentConfig | run_id: str, method: str, environment: EnvironmentConfig, runtime: RuntimeConfig, evaluation: EvaluationConfig, evolution_patients: list[str], evaluation_patients: list[str], round_patients: list[list[str]], initial_strategy: Strategy, random_seed: int, output_dir: str |

RunResult.status 为 completed、budget_exhausted 或 failed；EvaluationResult.status 为 completed、invalid 或 failed。failure_category 为 clinical、insufficient_evidence、ambiguous_task、tool_error、environment_error、model_error 或空。无法完成核验时 verification 为空，不能伪造为任务不成立。

### 环境接口

```python
async def create_environment(config: EnvironmentConfig) -> EnvironmentInstance: ...
async def get_tools(instance: EnvironmentInstance, patient_ids: list[str],
                    decision_point: str | None = None) -> list[Tool]: ...
async def read_state(instance: EnvironmentInstance, task: Task) -> JsonObject: ...
async def release_environment(config: EnvironmentConfig,
                              instance: EnvironmentInstance) -> None: ...
async def prepare_trajectory(instance: EnvironmentInstance,
                             patient_id: str) -> PatientTrajectory: ...
```

Tool 为 OpenAI Agents SDK 的工具类型。工具绑定本次实例；解题时绑定单个患者与可见范围。环境探索传入进化患者列表；固定解题传入仅含当前患者的列表。工具不能访问列表之外的患者。

### 模型运行接口

```python
async def call_llm(*, instructions: str, input_text: str,
                   output_type: type[BaseModel], model: ModelConfig) -> RunResult: ...
async def run_agent(*, instructions: str, input_text: str, tools: list[Tool],
                    output_type: type[BaseModel], model: ModelConfig,
                    budget: Budget) -> RunResult: ...
```

RunResult.output 为 output_type 校验后的字典，成功结果可通过 `output_type.model_validate(result.output)` 恢复模型。失败不返回伪造的成功对象。

### 主方法接口

```python
async def explore_data(trajectories: list[PatientTrajectory],
                       runtime: RuntimeConfig) -> RunResult: ...
async def explore_environment(config: EnvironmentConfig, patient_ids: list[str],
                              runtime: RuntimeConfig) -> RunResult: ...
async def generate_tasks(trajectory: PatientTrajectory, guidance: ExplorationGuidance,
                         strategy: Strategy, directions: list[GenerationDirection],
                         runtime: RuntimeConfig) -> RunResult: ...
async def analyze_feedback(tasks: list[Task], trajectories: list[PatientTrajectory],
                           evaluations: list[EvaluationResult],
                           runtime: RuntimeConfig) -> RunResult: ...
async def update_strategy(strategy: Strategy, feedback: Feedback,
                          runtime: RuntimeConfig) -> RunResult: ...
```

初始探索的输出模型分别为只有 guidance: str 的 DataExploration、EnvironmentExploration。任务生成输出为 TaskBatch；反馈输出为 Feedback；更新输出为 Strategy。directions 必须有两个元素，生成角色按照各位置方向尝试出题。数据探索的工具仅读取传入的缓存轨迹。

### 公共评测与实验接口

```python
async def evaluate_task(task: Task, trajectory: PatientTrajectory,
                        environment: EnvironmentConfig,
                        config: EvaluationConfig) -> EvaluationResult: ...
async def run_experiment(config: ExperimentConfig) -> None: ...
def summarize_run(output_dir: str) -> JsonObject: ...
```

只有实验入口负责患者分组、缓存路径、轮次和任务运行信息。主方法不调度公共评测，评测不更新策略。未接入的对比方法在配置入口返回明确错误，不替代为主方法运行。

## 6. 模块分工与产物

```text
MedEvolve/
  contracts.py
  configs/experiment.json
  environments/       管理客户端、环境适配、轨迹整理
  server/             Docker 生命周期与管理接口
  runtime/            单次调用、SDK Agent、模型配置和记录
  methods/medevolve/  初始探索、出题、反馈分析与更新
  evaluation/         公共核验、质量、固定解题与完成评价
  experiments/        缓存、患者分配、轮次、保存和汇总
  tests/
  outputs/
```

每次运行保存配置、两份初始指导、患者轨迹、各轮使用策略与更新策略、任务、评测、反馈和所有模型运行记录。轨迹按进化组和评测组分目录。调用成本有配置价格时计算，否则报告 token 用量。

五份模块计划分别限定文件归属。先提交公共结构和包配置，再并行实现各模块；通过固定样例和替代调用验证接口后，执行完整闭环集成。实现阶段使用独立 worktree；共享结构变更须同步所有引用方。
