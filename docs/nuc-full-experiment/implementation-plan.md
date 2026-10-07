# NUC 实验一全量运行 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 NUC 完成三个环境、六种方法、四个生成模型与三个公共裁判的实验一，优先完成两个生成模型。

**Architecture:** 通过 EvaluationConfig 的可选裁判列表扩展现有公共评测；矩阵调度复用配置生成器、run_experiment 和原始产物，不复制生成或评测算法。配置隔离模型，批次隔离失败，读取原始调用记录汇总成本。

**Tech Stack:** Python 3.12、Pydantic 2、asyncio、pytest、Docker、SSH。

## Global Constraints

- 工作目录为 /Users/guagua/.codex/worktrees/nuc-full-matrix/MedEvolve；不得修改其他 worktree。
- 三个质量与难度裁判依次为 deepseek-v4.1-flash、gpt-5-nano、gpt-5.4-nano；解题模型与任务生成模型相同。
- 四生成模型独立实验；优先 deepseek-v4.1-flash 与 gpt-5-nano；不得静默替换模型。
- 保留原始轨迹评测、核验失败语义、三裁判中位数、所有原始分数及理由。
- 原始调用用量是成本来源，缺失用量保持未知；失败及重试成本不能丢失。
- 复用现有接口；不实现实验二，不引入数据库或新依赖；commit description 与 body 使用中文。

### Task 1: 公共评测支持三个不同模型裁判

**Files:** Modify contracts.py, evaluation/evaluator.py; Test tests/evaluation/test_evaluator.py。

**Interfaces:** EvaluationConfig 新增 judges: list[ModelConfig]，默认空列表，长度仅允许 0 或 3。为空时沿用旧配置的 quality/difficulty 模型重复三次；不为空时质量和难度均按 judges 顺序各调用一次。RuntimeConfig 与 evaluate_task 的参数接口不变。

- [ ] 新增配置边界测试：空列表和三个裁判通过，1、2、4 个拒绝；端到端 fake runtime 测试检查质量与难度实际分别调用三个不同模型、中位数及 solver 模型保留。
- [ ] 运行 `/Users/guagua/code/MedEvolve/.venv/bin/python -m pytest tests/evaluation -q`，确认新测试在实现前失败。
- [ ] 实现 ModelConfig 列表校验，model_run 接收可选 model override；为每次模型运行的 trace 添加 `model_identity`，包含 role、model、judge_index（若适用），不得包含 key。三个裁判通过 asyncio.gather 并发调用，返回顺序按配置固定。
- [ ] 新测试通过，原有 evaluation 与 contracts 相关测试通过；提交 `feat(evaluation): 支持独立三模型公共裁判`。

### Task 2: 全量矩阵调度与成本导出

**Files:** Create scripts/run_experiment_matrix.py, experiments/matrix.py, tests/experiment/test_matrix.py。

**Interfaces:** CLI 接受 `--config`，读取矩阵 JSON，字段为 models（ModelConfig 列表）、judges（三个 ModelConfig）、slugs（三环境）、methods（六方法）、cohort_root、output_root、manager_url、seed_data_overrides、model_concurrency、generation_concurrency、evaluation_concurrency、batch_size。默认使用现有 templates 与 cohort-full.json。通过调用 scripts/emit_comparison_configs.py 生成组内配置，再覆盖模型、裁判及 Linux 环境路径，验证 ExperimentConfig，调用 experiments.runner.run_experiment。CLI 不做 API 凭证加载。

- [ ] 新增测试：两个模型配置的 output_dir/run_id 不冲突；每个 solver 对应 generator，judges 正确；MedEvolve 仅加载本组策略；已完成批次跳过，未完成批次采用新 attempt 目录；某一组失败后另一组仍运行；未知 token 不计为已知零。
- [ ] 运行 `/Users/guagua/code/MedEvolve/.venv/bin/python -m pytest tests/experiment/test_matrix.py -q`，确认新增行为测试先失败。
- [ ] 调度每个模型一条工作线，模型内依次运行 slugs 与 methods；两条工作线并行，批内 concurrency 均为 12，最大实例通过运行环境变量设为 2。MedEvolve 先执行本组进化，再发出冻结策略批次配置。根据 manifest 选择跳过或 attempt-N，保存进度到 output_root/matrix-status.json。调度出错记录组与异常，继续独立组。
- [ ] 从各运行 runs.json、config.json、evaluation/tasks.json、evaluation/evaluations.json 汇总 token 明细 JSON/CSV，按 model_identity 的实际模型与阶段分类；未标注的生成记录用该运行 generator 配置识别，探索和反馈按 config 的角色识别。遍历所有 attempt 计入成本，但最终任务/有效题数量只计算成功选用的批次。保留缺失用量计数、生成题数及有效题数、每题生成成本，输出 cost-summary.json/csv 和矩阵阶段结果。
- [ ] 运行覆盖矩阵边界的测试，完整测试通过后提交 `feat(experiment): 添加全量模型矩阵调度与成本汇总`。

### Task 3: NUC 同步、验证与持久运行

**Files:** NUC /home/gwh/code/MedEvolve；运行配置与产物放在 outputs/nuc-full-experiment-20261007，凭证位于 /home/gwh/.config/medevolve/api.env（权限 600）。

- [ ] 验证 outputs 归档 SHA256，解包；完成外部患者数据同步，验证 seed dump SHA256 并匹配 NUC amd64 Docker 镜像。
- [ ] 通过用户已有 API 查询模型及结构化输出能力，验证工具调用，不输出凭证；四个模型必须分别标明可用或具体故障。修复请求参数兼容问题保持运行层接口不变。
- [ ] 三个环境真实 smoke test，两个优先模型的端到端小样本，检查解题模型、三裁判身份、token 记录及错误分类。测量患者数与耗时，估算全量耗时。
- [ ] 合并本分支到 main 并 push，NUC 使用同一 commit；持久启动两个模型工作线，进度与日志落盘。设置定期检查，仅在完成、故障或需要用户输入时通知；启动后读取实际任务及评测产物确认正在推进。
- [ ] 按运行结果修复必要问题，重复失败三次时停止该故障路径并记录原因，其他独立组继续。优先两模型完成后推进余下两模型。全部结束时汇总结果与成本，关闭本次任务创建且已不用的环境实例。
