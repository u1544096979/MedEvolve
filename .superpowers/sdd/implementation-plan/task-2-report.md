# Task 2 实施报告：预算耗尽候选的公共评测接续

## 结果

TaskCraft 达到累计预算上限后保留已通过内置核验的候选，并以 `budget_exhausted` 状态返回。公共汇总与批次校验保留并完整验证有效输出；实验入口仅在生成批次解析时接受该状态。超预算、失败和非法批次不进入评测，非生成解析仍只接受 `completed`。工具预算耗尽后，Ledger 不会为核验重试继续发起模型调用。

TaskCraft 阶段算法、阶段参数、固定种子选择和原子候选接纳条件均未改动；现有 `agent_score` 为 0 或 `None` 时仍依源码逻辑接纳原子题。

## RED / GREEN

- RED：新增公共批次测试首先失败 2 项：耗尽批次输出被清空，非法耗尽批次未转为失败。
- 扩充定向测试后，另复现候选未保留、工具耗尽后继续模型重试的问题。假运行时的 `agent_calls` 是 `llm_calls` 子集，故测试用非重复的 `llm_calls` 计数模型轮数；公共校验测试的题目辅助函数显式提供默认 `decision_point=None`，以和规范化后的 TaskBatch 序列化形状一致。
- GREEN：公共与 TaskCraft 定向测试 6 项通过；公共实验接续及超预算测试 2 项通过。
- 最终完整测试：`PYTHONPATH=$PWD /Users/guagua/code/MedEvolve/.venv/bin/python -m pytest experiments/test_common.py tests/method/test_taskcraft.py tests/experiment/test_closed_loop.py -q`，结果 **106 passed**。
- `git diff --check` 通过。

## 文件

- `methods/common.py`：验证并保留预算耗尽状态下的有效 TaskBatch；汇总允许耗尽状态携带批次。
- `methods/taskcraft/__init__.py`：耗尽时按现有 Pipeline 选择规则，从候选池选择输出。
- `methods/taskcraft/ledger.py`：标记工具耗尽后，阻断同次重试流程的后续模型调用。
- `experiments/runner.py`：仅生成批次解析允许 `budget_exhausted`，其余解析路径保持现状。
- `experiments/test_common.py`、`tests/method/test_taskcraft.py`、`tests/experiment/test_closed_loop.py`：覆盖公共校验、已核验候选保留、工具预算停止、评测接续与实际超预算拒绝。
- `docs/comparison-experiments/common/SPEC.md`、`docs/comparison-experiments/baselines/taskcraft/SPEC.md`：记录耗尽输出契约。

## 自检与边界

- 实际超预算检查仍由实验入口在批次校验后执行，并清空输出；测试覆盖超过模型预算 1 次时不评测。
- 失败状态会清空输出；耗尽但缺少输出时保留空输出；耗尽状态携带非法字段、题型或超过题数限制会被拒绝。
- 非生成解析仍要求 `completed`；没有改动 `runtime/execution.py`，没有调用真实模型或外部环境。
- 工作区仅包含本任务改动，无额外范围项。

## 提交

- `981b282974b778162157ace852f91ca9998d48bc` — `fix(taskcraft): 保留预算耗尽后的合格候选`
