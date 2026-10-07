# 公共对比实验规范

## 实验目标

比较 MedEvolve、TaskCraft、Workflow、SingleShot、SelfRefine、EnvExplore 在相同医疗环境中的任务生成质量、任务难度表现及生成成本。TaskCraft 以作者源码作为算法依据；Workflow、SingleShot、SelfRefine、EnvExplore 按明确的方向定义实现。具有论文来源的方法优先作者源码，其次论文。

## 公共边界

统一环境初始数据、患者划分、模型配置、公共评测提示词及固定解题 Agent。各方法保留自己的出题算法、方法提示词和内置筛选，不共享 MedEvolve 特有的数据探索指导与进化经验。方法内部筛选不能代替公共评测，其调用计入生成成本。

同一对比环境中所有方法使用同一个环境适配；算法专属的探索或执行器由该方法实现。新增医疗环境适配属于公共环境任务，不能混入某个对比方法。首轮使用已联调的 PhysicianBench。

## 共享输入输出

沿用 contracts.py 的 PatientTrajectory、GenerationDirection、RuntimeConfig、TaskBatch、RunResult、Task 和 EvaluationResult。公共字段不另建同名模型。

对比方法生成入口：

```python
async def generate_tasks(
    trajectory: PatientTrajectory,
    environment: EnvironmentConfig,
    directions: list[GenerationDirection],
    runtime: RuntimeConfig,
) -> RunResult: ...
```

除 Workflow 外，directions 对应两个目标任务。Workflow 接收五类候选方向，由定位节点根据患者轨迹选择两类，再分别生成；公共入口不为其提前抽样两个题型，输出校验遵循其候选池与选型约束，详见 [Workflow 规范](../baselines/workflow/SPEC.md)。成功输出 TaskBatch，最多两题；允许证据不足时少出题。算法内部候选、扩展与失败尝试记入运行记录和用量，不能通过只统计最终成功题隐去成本。Task 的患者、环境、轮次、方法和编号由实验入口附加。

SingleShot 使用自主选题入口，第三参数为 contracts.TaskType 定义的全部五种题型：

```python
async def generate_tasks(
    trajectory: PatientTrajectory,
    environment: EnvironmentConfig,
    task_types: list[TaskType],
    runtime: RuntimeConfig,
) -> RunResult: ...
```

task_types 是允许生成的题型集合，不是题目位置：集合须完整包含五种题型且不重复，一次最多两题，两题可同类型，不要求覆盖全部题型，也不按权重分配题数。公共输出约束与结果校验据此提供按题型集合校验的路径，检查最多两题、题型属于该集合、字段非空与 decision_point 为 null；其他方法保留 directions 的两个方向配额校验。生成输入包含完整轨迹与原始 sources，不含主方法的数据探索指导、环境探索产物、生成指导或评测答案。

六种方法内部生成及筛选统一使用 runtime.models["generator"]，每患者生成预算上限统一为 max_model_turns=800、max_tool_calls=800。预算取 runtime.budgets["generator"]，model_turns 对多阶段方法指本批累计模型调用数；工具调用单独累计。800/800 是上限，各方法保留自身生成流程与停止规则，不为消耗额度追加调用。公共实验 section 提供公共输出约束和轻量运行结果汇总函数，各方法调用而不另写一份。usage 统一保存 model_turns、tool_calls、input_tokens、output_tokens、total_tokens；单次接口的 prompt_tokens、completion_tokens 映射为输入、输出 token。trace 按阶段附加原始运行记录，保留失败信息。

环境复用 create_environment、prepare_trajectory、get_tools、read_evidence、read_state、release_environment。模型复用 call_llm 与 run_agent。评测复用 evaluate_task，禁止对方法名称设置不同判定规则。

EnvExplore 保留与主方法相同的一次环境探索，使用 environment_explorer 角色及其独立预算；患者生成仍使用 generator。实验入口将环境指导和 initial_strategy 绑定到本次运行的生成包装器，患者批次保持四参数调用。探索成本单列并计入方法总成本，不重复计入每名患者预算。EnvExplore 支持现有 PhysicianBench 和 MedAgentBench 公共适配，具体约定见 [方法规范](../baselines/envexplore/SPEC.md)。

## 对比口径

同环境内固定患者组、目标任务数及外部预算上限，记录实际生成题数、证据核验通过率、四维质量判断、解题完成情况、失败类别、模型调用和 token 用量。各方法调用次数可以不同，差异必须进入成本结果。

进化患者用于 MedEvolve 探索和策略更新，评测患者不参与策略学习。所有方法在相同评测患者上生成任务；冻结 MedEvolve 的最终策略后评测。方向对比的题型采样安排由公共入口管理，源码方法有自己的采样机制时须注明适配差异；SingleShot 不采样方向，由模型在允许题型集合内自主选题。

核验包括逐条引用与事实一致性、题目和参考答案支持性；之后才进入质量评价、解题和完成评价。原始记录是事实依据，生成模型和核验模型的结论都不是金标准。临床合理等价答案不能仅因措辞或真实后续治疗不同而被否定。

## 环境和可见范围

查询与模拟提交复用共享查询 Docker 实例，真实写入任务及 PhysicianBench 试写探索使用独立种子实例，具体遵循 [环境能力与患者并行设计](../../environment-capabilities/design.md)。首轮 decision_point 为 null；仅在环境实现完整时间过滤后开放决策点任务。凭环境真实开放能力进行读写。

## 方法来源状态

TaskCraft 为作者源码的医疗适配；Workflow、SingleShot、SelfRefine 分别对应固定阶段生成、单次生成、单患者自反馈迭代修订。EnvExplore 为主方法消融，保留环境探索、使用固定初始策略，不执行数据探索与自进化。这四项按本项目 SPEC 定义报告。
