# MedEvolve 实施计划

## 模块职责

公共环境提供远程实例、工具与患者轨迹；模型运行提供 LLM 与标准 Agent；主方法完成探索和自进化；公共评测判断任务质量与完成情况；实验入口管理患者、轮次及产物。

## 接口

共享结构及函数签名见[主 specification](design.md)第 5 节。公共结构先建立，各模块引用同一份 contracts.py。

| 模块 | 文件归属 | 计划 |
| --- | --- | --- |
| 公共环境 | environments/、server/ | [环境](plans/environment.md) |
| 模型运行 | contracts.py、pyproject.toml、runtime/ | [模型运行](plans/runtime.md) |
| 主方法 | methods/medevolve/ | [主方法](plans/method.md) |
| 公共评测 | evaluation/ | [评测](plans/evaluation.md) |
| 实验入口 | experiments/、configs/ | [实验](plans/experiment.md) |

## 实施任务

1. 模型运行 session 先提交 contracts.py 与 pyproject.toml，形成所有模块共用的基础提交。
2. 五个 session 从同一基础提交建立各自 worktree，分别实施环境、模型运行、主方法、公共评测和实验入口计划。
3. 各 session 只修改所属模块；尚未完成的依赖通过替代调用开发。共享接口调整由集成 session 统一同步。
4. 集成 session 检查接口与关键验收，合并模块，使用可控模型和环境跑两轮完整闭环。
5. 在镜像、数据与模型配置明确后执行真实实验。

每个模块交付变更提交、验证结果及实际运行所需配置。模块内可使用 subagent 协助实施。实施使用独立 worktree。真实模型调用使用已确认的配置。对比方法通过公共环境与公共评测接口单独接入。

## 关键验收

- 环境隔离：实例写入互不影响，结束后释放。
- 数据边界：评测患者不参与进化，解题 Agent 看不到参考答案。
- 闭环运行：初始探索、出题、评测、反馈、更新串联，下一轮使用更新策略。
- 结果可信：写入读取实际状态，运行故障单独记录，结果可保存与汇总。

测试集中在上述关键路径，不要求每个小函数、每个字段或重复模拟场景单独建立测试。
