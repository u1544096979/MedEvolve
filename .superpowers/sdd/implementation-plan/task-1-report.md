# Task 1 实施报告

## 状态

已完成并提交，功能提交：`a937f7a feat(harness): 支持任务划分与本地实验目录`。

## 接口

- `contracts.TaskSplit`：共享 `train: list[Task]`、`test: list[Task]` 数据模型，并由 `harnesses.contracts` 导出。
- `harnesses.datasets.split_tasks(tasks, *, seed=42)`：按任务 ID 排序后固定种子随机打散，按 70/30 向下取整；要求两边非空、ID 不重复，发现题面与参考答案完全相同的任务跨集合时抛出 `DatasetFreezeError` 并列出 ID。
- `freeze_datasets()` 与 `load_frozen()` 原有签名保留。冻结清单新增训练/测试 ID、比例、患者重叠和逐题评测元数据；单题组件夹具的 `split` 为 `null`。
- 输入通过根目录 `Task`、`EvaluationResult` 解析。只接纳 `verification.valid` 与 `verification.read_only_task` 均为真且患者有效的任务；历史核验缺少只读字段时明确报告需要重新核验。质量四维、难度和失败类别写入 manifest，质量结构不兼容时由 Pydantic 报出解析错误。`unsupported_operation` 按共享枚举保留。
- `harnesses.local.run_command(command, *, group_dir, group_env, timeout)` 返回 `(returncode, stdout, stderr)`，继承宿主环境并合并组环境；超时或取消时终止并回收子进程。
- `prepare_group_directory(template, destination)` 从初始模板新建组目录；`temporary_experience(snapshot)` 提供自动清理的测试副本；`export_experience(source, destination, *, include)` 仅导出显式路径并沿用快照过滤规则。
- `configs/harness-evolution.json` 保持 `execute: false`，将任务数设为 10，并为四个基座声明源码目录、程序路径、锁定 SHA、Python 路径、初始状态模板及实验目录。

## 验证

命令：

```text
/Users/guagua/code/MedEvolve/.venv/bin/python -m pytest tests/harness/test_datasets.py tests/harness/test_local.py -q
```

结果：`32 passed in 0.54s`。

```text
/Users/guagua/code/MedEvolve/.venv/bin/python -m pytest tests/harness tests/evaluation -q
```

结果：`163 passed in 3.75s`。

另外执行 `git diff --check`，无空白错误；用项目 `.venv` 解析 `configs/harness-evolution.json` 成功。测试均为本地假数据和临时目录，没有模型调用、FHIR 服务或 Docker 调用。

## 边界与待后续处理

- 四个原生基座 runner 仍由后续任务实现；此提交提供可复用的进程和目录 helper，并未新增 runner 工厂或抽象框架。
- 配置中的 `local_sources/...` 与 `configs/harness-evolution/initial/...` 是后续本地接入所用的相对路径约定，当前工作树没有对应上游源码与模板。
- 完全重复题目会阻断划分；近似变体没有自动相似度阈值或分类器，仍需在数据验收环节人工核对。
- 本地遗留的 `harnesses/adapters/__init__.py`、`tests/harness/test_adapters.py` 改动属于先前工作，本次没有暂存或提交。
