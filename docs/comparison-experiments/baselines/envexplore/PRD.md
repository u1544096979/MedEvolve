# EnvExplore 实施任务

## 目标与依赖

实现 SPEC.md 定义的环境探索生成方向。依赖公共实验 section 提供共享生成入口、输出约束、运行结果汇总和预算约定。公共配置与分派由公共实验 section 集成，本 section 负责 methods/envexplore/ 及必要的方法测试。

## 实施步骤

1. 在 methods/envexplore/ 实现公共入口，复用 create_environment、get_tools、release_environment。
2. 工具限定 trajectory.patient_id；首轮 decision_point 为 null，只传患者与方向，不把轨迹正文带入 Agent。
3. 通过 run_agent 输出 TaskBatch，模型使用 runtime.models['generator']，预算使用 runtime.budgets['generator']。
4. 确认运行轨迹有实际工具查询；未查询却直接出题的结果按该方向机制未成立记录失败，不以 Schema 推测患者事实。
5. 使用 finally 释放实例，正常、故障及预算耗尽都检查释放；本方法实例不能传给公共解题。

## 最小验收

至少实际查询当前患者，证据来自本次返回记录；没有缓存轨迹直接供出题。预算耗尽不当成有效任务，实例结束后释放。 验证关键机制和一个真实患者即可，不为字段转发或提示词字符串建立重复测试。

## 交付

实现提交、所需配置项、小样本任务和公共评测结果。真实调用使用已经授权的 gpt-4o-mini；对方法内部判定、公共核验和生成成本分别记录。接口冲突交给公共实验 section 处理。
