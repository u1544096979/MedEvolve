# Task 1 实施报告

## 实现内容

- 新增 `environments/capabilities.py`，集中描述 PhysicianBench、MedAgentBench、FHIR-AgentBench 的读取与写入语义，并提供统一的只读实验说明。能力读取返回副本，未知环境名称会报错。
- `GeneratedQuestion.execution_requirement` 与 `Verification.read_only_task` 均为兼容旧数据的可选字段。
- `ExperimentConfig` 增加正整数生成与评测并发配置。解析优先级为显式配置、有效的 `MEDEVOLVE_RUN_CONCURRENCY`、默认值 6；运行目录中的 `config.json` 保存解析后的有效并发数。
- 示例配置显式设置两类并发均为 6。新增能力、契约兼容及并发配置测试。

## 验证

- `/Users/guagua/code/MedEvolve/.venv/bin/python -m pytest tests/experiment/test_capabilities_and_concurrency.py tests/experiment/test_generator_budget_configs.py tests/experiment/test_workflow.py -q`：16 passed。
- `git diff --check`：通过。

## 自审

- 环境的真实写入语义与本轮统一只读限制分开表达，避免把 PhysicianBench 或 MedAgentBench 描述成底层只读环境。
- 旧任务和旧实验配置仍可载入；本任务未修改任务公共输出校验，也未改变患者调度。
- 新增字段缺失时不会推断生成结果已声明只读；公共输出强制校验留待后续任务。

## 未验证项

- 未运行完整离线测试集。
- 当前执行仍沿用既有患者并发调度；独立生成与评测并发将在后续任务接入。
