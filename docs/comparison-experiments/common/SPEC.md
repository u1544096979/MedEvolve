# 公共对比实验规范

## 实验目标

比较 MedEvolve、TaskCraft、Workflow、SingleShot、SelfGen、EnvExplore 在相同医疗环境中的任务生成质量、任务难度表现及生成成本。方法来源按作者源码、论文、明确的方向定义依次确定。尚未确认论文对应关系，不等同于没有公开源码。

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

directions 对应两个目标任务。成功输出 TaskBatch，最多两题；允许证据不足时少出题。算法内部候选、扩展与失败尝试记入运行记录和用量，不能通过只统计最终成功题隐去成本。Task 的患者、环境、轮次、方法和编号由实验入口附加。

环境复用 create_environment、prepare_trajectory、get_tools、read_evidence、read_state、release_environment。模型复用 call_llm 与 run_agent。评测复用 evaluate_task，禁止对方法名称设置不同判定规则。

## 对比口径

同环境内固定患者组、目标任务数及外部预算上限，记录实际生成题数、证据核验通过率、四维质量判断、解题完成情况、失败类别、模型调用和 token 用量。各方法调用次数可以不同，差异必须进入成本结果。

进化患者用于 MedEvolve 探索和策略更新，评测患者不参与策略学习。所有方法在相同评测患者上生成任务；冻结 MedEvolve 的最终策略后评测。方向对比的题型采样安排由公共入口管理，源码方法有自己的采样机制时须注明适配差异。

核验包括逐条引用与事实一致性、题目和参考答案支持性；之后才进入质量评价、解题和完成评价。原始记录是事实依据，生成模型和核验模型的结论都不是金标准。临床合理等价答案不能仅因措辞或真实后续治疗不同而被否定。

## 环境和可见范围

各次探索和解题采用独立的远程 Docker 实例。首轮 decision_point 为 null；仅在环境实现完整时间过滤后开放决策点任务。凭环境真实开放能力进行读写。

## 方法来源状态

TaskCraft 已确认作者仓库与论文。Workflow、SingleShot、SelfGen、EnvExplore 的旧文档只有方向定义，论文对应关系待确认；不得自行替换成名称相似的论文方法。
