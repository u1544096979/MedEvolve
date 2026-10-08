# 实验二官方 Benchmark 评测

## 评测对象

PhysicianBench 与 MedAgentBench 官方任务分别用于同环境的外部评价。四个基座 OpenClaw、Hermes、ACE、Reflexion，以及六种任务来源沿用实验二配置。70% 合成任务用于经验训练，30% 合成任务作为辅助验证。官方任务、参考答案和评分反馈不参与训练。

## 公平条件

每组初始经验与最终经验使用相同官方任务、模型、工具、轮次和超时。每题使用临时经验副本、新会话和独立医疗实例，关闭学习及评分后重试。测试前后核对冻结经验摘要。基座初始经验必须与训练记录一致；缺失最终经验不能替代为初始经验。

## 官方资产与评分

锁定官方仓库提交和任务文件摘要。PhysicianBench 使用 tasks/v1 原始题目和 pytest 检查点，提供 workspace/output 文件、trajectory.log 和本题 FHIR 服务。支持原生医疗写入与 write_file 工具，评分在独立进程内执行。MedAgentBench 默认使用官方配置对应的 test_data_v2.json 共 300 题；test_data_v1.json 共 100 题可显式选择并单独报告；使用官方 eval.py 与单独发布的 refsol.py。任务输入只包含 instruction 与 context，不传入 sol 或 eval_MRN。原生 GET 查询完整实例，POST 接受但不持久化，FINISH 保留 JSON 答案列表。参考评分文件记录官方下载地址与 SHA-256。

## 结果

按环境、基座和任务来源保存逐题初始与最终成绩、官方检查点、运行状态、执行用量、资产摘要和经验摘要。成功率使用完整计划任务数作为分母，运行与评分故障单列；同时报告有效评分数量与同题配对提升。PhysicianBench 另报平均检查点得分。成功率提升使用百分点。

## 验证与运行

默认测试使用替代模型和伪造外部接口，验证官方资产缺失、任务与答案隔离、评分解析、实例保留与释放、冻结经验不变及配对汇总。NUC 验证实际官方资产、评分进程和医疗实例。真实基座测试须具备锁定源码、依赖和训练快照；正式全量运行使用明确的执行开关。

### 运行命令

官方源码位于 `.benchmark_sources/physicianbench` 与 `.benchmark_sources/medagentbench`，提交号由 `configs/harness-evolution.json` 锁定。MedAgentBench 的 `refsol.py` 从官方 README 指向的 [独立下载地址](https://stanfordmedicine.box.com/s/fizv0unyjgkb1r3a83rfn5p3dc673uho) 获取，摘要须匹配配置。评分解释器通过每个环境的 `benchmark.python_path` 设置；依赖可在独立虚拟环境安装。

1. 安装官方评分依赖：`.venv/bin/python -m pip install -e '.[benchmark]'`。
2. 检查两个官方任务集和评分依赖：`.venv/bin/python -m experiments.harness_benchmark --check-assets`。
3. 检查具体实验组：`.venv/bin/python -m experiments.harness_benchmark --environment PhysicianBench --harness ace --source medevolve --check`。
4. 配置 `execute: true`、模型密钥环境变量与 `MEDEVOLVE_HARNESS_LIVE=1` 后，运行同一命令并省略 `--check`；`--task-id` 可重复指定，用于小样本接入验证。省略环境、基座或来源参数会遍历配置中的相应组合。

训练产物目录须包含 `train.json`、`initial-experience/`、`final-experience/`；评测不依赖原始合成任务文件重新可用。结果保存在该组的 `benchmark/` 下，包含 `config.json`、`assets.json`、`records.json`、`benchmark.json`、`summary.json` 与逐题 `jobs/<state>/<task_id>/`。逐题文件、原生评分日志、工具记录和结果独立保存。

汇总同时提供完整任务分母的成功率和有效同题配对成绩。`comparison_valid` 仅在全部题目两次评分有效且经验摘要保持不变时为真。PhysicianBench 的平均检查点分数以有效评分任务计算，并列出有效任务数；裁判调用异常属于评分故障。官方助手未提供裁判 usage 时，裁判用量与费用标记为未知。
