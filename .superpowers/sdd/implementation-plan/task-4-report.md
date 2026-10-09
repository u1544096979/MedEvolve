# Task 4 实施报告：三裁判联合评分

## 实现

- 新增 `JointJudgment` 与 `EvaluationConfig.scoring_mode`。联合模式用质量模型或配置中的三名裁判并发请求三次，每个响应同时给出四维质量与推理深度；separate 模式继续分别请求质量和难度。
- 新增可复用接口：`review_scores(task, trajectory, verification, config, *, request_semaphore=None, record=None) -> tuple[QualityReview, DifficultyScore, list[RunResult]]`。Task 5 可传入 `asyncio.Semaphore(24)` 限制评分请求并发；`record` 在聚合前逐条接收实际运行记录。聚合失败时抛出 `ScoringError`，其 `runs` 保留本轮真实请求，`quality` / `difficulty` 保存已完成的部分聚合结果。
- 联合提示沿用既有四维质量和难度评分标准。评分响应只作为单条联合 `RunResult` 记录，成功路径不重复追加，因此 usage 只计一次。
- 启用所有受版本控制的 PhysicianBench、MedAgentBench 实验配置及配置生成器的 joint 模式。FHIR-AgentBench 配置和 `configs/experiment.json` 默认值保持 separate。

## 涉及文件

`contracts.py`、`evaluation/evaluator.py`、`evaluation/prompts.py`、`evaluation/__init__.py`、`scripts/emit_comparison_configs.py`、`tests/evaluation/test_evaluator.py`、`tests/experiment/test_emit_comparison_configs.py`，以及受版本控制的 PhysicianBench / MedAgentBench 配置文件。

## 验证

- `PYTHONPATH="$PWD" /Users/guagua/code/MedEvolve/.venv/bin/python -m pytest tests/evaluation/test_evaluator.py tests/experiment/test_emit_comparison_configs.py -q`：58 passed。
- `PYTHONPATH="$PWD" /Users/guagua/code/MedEvolve/.venv/bin/python -m pytest tests/evaluation tests/experiment/test_closed_loop.py tests/experiment/test_emit_comparison_configs.py -q`：85 passed。

## 未执行

未调用任何真实模型或外部服务；Openlux gpt-4o-mini pilot 按委派要求留待后续执行。
