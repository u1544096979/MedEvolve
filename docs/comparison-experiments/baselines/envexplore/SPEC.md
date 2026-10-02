# EnvExplore 方法对比规范

## 定义

EnvExplore 为本实验的“环境探索生成”方向对比，按本文件定义实现，不对应某篇已确认的论文，也不以论文复现名义报告。没有 MedEvolve 特有的数据探索和跨轮进化。

## 流程

每个患者创建独立远程环境实例，使用绑定该患者的原生工具，让标准工具 Agent 查询并选材，终局返回 TaskBatch。输入患者 ID、两个方向、工具 Schema 和公共输出要求；出题 Agent 不接收完整缓存轨迹、主方法探索指导或进化策略。该方法的选材来源是本次实际工具查询。

## 答案与证据

工具 Agent 根据实际查询结果同时生成题目、候选参考答案和证据。source_id 对应真实返回的资源记录；不将查询为空或工具故障解释为临床事实。公共层按同一规则核验候选答案。

## 输入输出

遵循 [公共规范](../../common/SPEC.md) 的 generate_tasks。directions 指定两个目标方向，输出 TaskBatch 最多两题；每题包含 instruction、task_type、reference_answer、evidence 和 decision_point，首轮 decision_point 为 null。患者、环境、方法和任务编号由公共入口附加。

公共输出约束、题型定义与评测提示词一致；方法内部提示词体现上述方向区别。运行状态及成本必须包含全部内部调用，不能只保留最后一份模型结果。

## 文件边界

实现位于 methods/envexplore/。不修改其他方法、公共评测或患者划分。复用标准运行层，不创建自己的 LLM 客户端或 Agent 循环。
