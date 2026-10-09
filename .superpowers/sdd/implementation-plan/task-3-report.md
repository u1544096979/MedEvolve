# Task 3 实施报告：EnvExplore 冻结指导

## 实现内容

- `ExperimentConfig` 增加 `exploration_only`，并限制该选项只能用于 EnvExplore。
- 增加 `FrozenEnvironmentGuidance`，保存环境名、独立探索患者、指导正文、完整探索 `RunResult` 和运行配置。
- 独立探索运行只执行环境探索并写出冻结指导；普通 EnvExplore 单次运行仍可探索后继续生成。
- EnvExplore 批次加载冻结指导后校验环境、完成状态、指导正文和患者集合，不要求或调用探索器。
- 对比配置生成器为 PhysicianBench 和 MedAgentBench 生成单一探索配置，并让所有测试批次引用同一冻结文件。
- 对比驱动器先校验完整测试患者集与探索患者集，再运行探索并验证冻结文件，最后才运行测试批次。FHIR-AgentBench 沿用原有逐批探索分支。
- 批次 summary 将探索成本标为 `frozen_exploration_reference`。它不进入批次 `method` 成本或 `runs`，独立探索运行仍在其自身 summary 中计入一次。

## 涉及文件

- `contracts.py`
- `experiments/runner.py`
- `experiments/summary.py`
- `scripts/emit_comparison_configs.py`
- `scripts/run_full_comparison.py`
- `experiments/test_envexplore.py`
- `experiments/test_envexplore_summary.py`
- `tests/experiment/test_emit_comparison_configs.py`

## 验证

通过：

```text
PYTHONPATH="$PWD" /Users/guagua/code/MedEvolve/.venv/bin/python -m pytest tests/method/test_envexplore.py tests/method/test_envexplore_environment.py experiments/test_envexplore.py experiments/test_envexplore_configs.py experiments/test_envexplore_summary.py tests/experiment/test_emit_comparison_configs.py -q
46 passed
```

更广的 `tests/experiment experiments` 回归结果为 145 passed、1 failed。唯一失败是 `tests/experiment/test_patient_input_budget.py::test_oversized_patient_does_not_block_other_patients` 仍断言存在生成失败时 manifest 为 `completed`；当前阶段账本会将该 run 标记为 `failed`。该测试不属于本任务范围，且修改会违背已集成的 manifest 状态语义，因此保留原状。

## 约束与风险

- 未执行真实模型、FHIR 服务或 Docker 调用；探索器、轨迹准备、生成和评测均通过 mock 验证。
- 冻结文件包含运行模型标识和 API key 环境变量名，不包含 API key 值。
- 批次成本中的冻结探索项是可追溯引用，不应与独立探索运行的成本再次求和。
