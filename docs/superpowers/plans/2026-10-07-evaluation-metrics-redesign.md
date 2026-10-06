# 评测指标体系重构实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 evaluation 模块的质量评价改成四维 1-5 分数（三裁判取中位数），新增难度评价（推理深度 1-4 + 自动复杂度指标），并让汇总输出质量与难度分布。

**Architecture:** 契约层新增裁判输出结构与聚合结果结构；评测流程在 verification 之后串行调用三次质量裁判、三次难度裁判，聚合函数取中位数并保留原始分数；难度另含由 evidence 与轨迹自动计算的 complexity_metrics；汇总模块把评测记录转成均值、中位数、分布与高质量占比。

**Tech Stack:** Python 3.10+、pydantic v2、pytest + pytest-asyncio。设计文档：`docs/superpowers/specs/2026-10-07-evaluation-metrics-redesign.md`。

## Global Constraints

- 评测流程固定为：`verification → quality(3 裁判) → difficulty(3 裁判) → solver → completion`。
- 裁判数量固定为 3（`JUDGE_COUNT = 3`），最终分数取三个分数的中位数，同时保留三个原始分数与每位裁判的 reason。
- 质量四维键名固定：`factual_accuracy`、`clinical_soundness`、`evidence_sufficiency`、`task_clarity`；每维 score 为 1-5 整数。
- 难度层级固定为 1-4：1 事实查询、2 单步推理、3 整合推理、4 临床决策；`reasoning_depth` 为三裁判中位数。
- `ComplexityMetrics` 三个字段：`evidence_count` = `task.evidence` 中 source_id 数量；`resource_type_count` = source_id 前缀去重数量（`"Observation/123"` → `"Observation"`）；`time_span_days` = 任务引用的 evidence 在 `trajectory.sources` 中对应记录的最早到最晚时间跨度（天），无可用时间时为 `null`。
- 裁判单次输出只含分数与 reason（`QualityJudgment` / `DifficultyJudgment`）；`QualityReview` / `DifficultyScore` 是聚合后的最终结果，`scores`、`reasoning_depths`、`complexity_metrics` 由聚合函数填充。
- 不改生成模块（`methods/*` 的生成相关代码、`methods/common.py`）、不改 `feedback` / `strategy_update` 的业务逻辑。唯一例外：Task 6 按用户确认把 FEEDBACK 提示词里的旧维度名换成新维度名。
- 汇总保持既有键名不动：把 `quality_judgments` 换成 `quality_summary`，新增 `difficulty_summary`，其余键（`actual_questions`、`validity_among_verified`、`completion_among_*`、`quality_reviews` 等）不改名、不删除。
- `time_span_days` 只保存在每条评测记录里，不进入汇总输出（设计文档 4.2 的 `complexity_metrics` 只列 `avg_evidence_count` 与 `avg_resource_types`）。
- 测试命令（必须在 worktree 根目录执行，`PYTHONPATH` 用于压过主仓库的 editable 安装）：
  ```bash
  cd /Users/guagua/code/MedEvolve/.worktrees/evaluation-metrics
  PYTHONPATH="$PWD" ../../.venv/bin/python -m pytest -q
  ```
  基线：`171 passed, 3 skipped`。
- 代码库既有约定：不为提示词字符串和字段转发建立重复测试（见 `docs/comparison-experiments/common/SPEC.md` 的方法实现要求），因此 Task 2、Task 3、Task 6 不新增测试，靠全量测试与内容核对验收。
- commit message 规范：`<type>(<scope>): 中文描述`；body 用中文写「为什么」，description 只写「做了什么」。

---

## File Structure

| 文件 | 职责 | 任务 |
| --- | --- | --- |
| `contracts.py` | 裁判输出结构与聚合结果结构 | Task 1 |
| `evaluation/prompts.py` | 质量与难度裁判提示词 | Task 2 |
| `configs/*.json`（6 个） | 评测运行配置的 difficulty 角色 | Task 3 |
| `evaluation/evaluator.py` | 三裁判调用、中位数聚合、复杂度自动计算 | Task 4 |
| `tests/evaluation/test_evaluator.py` | 裁判机制、聚合、复杂度的单元测试 | Task 4 |
| `tests/integration/test_pipeline.py`、`tests/integration/test_medagentbench_integration.py` | 真实模块的假模型响应 | Task 4 |
| `experiments/summary.py` | 质量与难度汇总函数 | Task 5 |
| `tests/experiment/test_closed_loop.py` | 汇总断言 | Task 1、Task 5 |
| `methods/medevolve/prompts.py` | FEEDBACK 维度名 | Task 6 |

---

### Task 1: 质量与难度数据契约

**Files:**
- Modify: `contracts.py`（`QualityReview` 附近与 `EvaluationResult`）
- Test: `tests/experiment/test_closed_loop.py:85` 与同文件 `:133` 附近

**Interfaces:**
- Consumes: 无（本任务是底层）。
- Produces:
  - `DimensionJudgment(score: int, reason: str)`
  - `QualityJudgment(factual_accuracy, clinical_soundness, evidence_sufficiency, task_clarity: DimensionJudgment)`
  - `QualityDimension(score: int, scores: list[int], reason: str)`
  - `QualityReview(factual_accuracy, clinical_soundness, evidence_sufficiency, task_clarity: QualityDimension)`
  - `ComplexityMetrics(evidence_count: int, resource_type_count: int, time_span_days: float | None)`
  - `DifficultyJudgment(reasoning_depth: int, reason: str)`
  - `DifficultyScore(reasoning_depth: int, reasoning_depths: list[int], reason: str, complexity_metrics: ComplexityMetrics)`
  - `EvaluationResult.difficulty: DifficultyScore | None`

- [ ] **Step 1: 把闭环测试改成新契约（先红）**

在 `tests/experiment/test_closed_loop.py` 中，把 `evaluate` 假实现里的 `quality=QualityReview(dimensions={...})` 换成：

```python
quality=QualityReview(**{name: QualityDimension(score=4, scores=[4, 4, 4], reason="clear")
                         for name in ("factual_accuracy", "clinical_soundness",
                                      "evidence_sufficiency", "task_clarity")}),
```

导入行改为：

```python
from contracts import (
    CompletionReview, DifficultyScore, ComplexityMetrics, EvaluationConfig, Feedback,
    ...
)
```
（只加 `ComplexityMetrics`、`DifficultyScore`、`QualityDimension`，其余保持原样。）

同时删掉这一行断言（旧汇总键在 Task 5 之前没有对应新键，Task 5 会补新断言）：

```python
    assert summary["evolution"]["quality_judgments"]["clarity"]["good"] == (4 if failure else 8)
```

- [ ] **Step 2: 运行测试确认失败**

```bash
PYTHONPATH="$PWD" ../../.venv/bin/python -m pytest tests/experiment/test_closed_loop.py -q
```
Expected: FAIL，`QualityReview` 不接受 `dimensions`（或 `QualityDimension` 未定义），报 `TypeError` / `ValidationError` / `ImportError`。

- [ ] **Step 3: 写契约**

在 `contracts.py` 中把现有的 `class QualityReview(BaseModel): dimensions: JsonObject` 整块替换为：

```python
class DimensionJudgment(BaseModel):
    score: int
    reason: str


class QualityJudgment(BaseModel):
    factual_accuracy: DimensionJudgment
    clinical_soundness: DimensionJudgment
    evidence_sufficiency: DimensionJudgment
    task_clarity: DimensionJudgment


class QualityDimension(BaseModel):
    score: int
    scores: list[int]
    reason: str


class QualityReview(BaseModel):
    factual_accuracy: QualityDimension
    clinical_soundness: QualityDimension
    evidence_sufficiency: QualityDimension
    task_clarity: QualityDimension


class ComplexityMetrics(BaseModel):
    evidence_count: int
    resource_type_count: int
    time_span_days: float | None


class DifficultyJudgment(BaseModel):
    reasoning_depth: int
    reason: str


class DifficultyScore(BaseModel):
    reasoning_depth: int
    reasoning_depths: list[int]
    reason: str
    complexity_metrics: ComplexityMetrics
```

在 `EvaluationResult` 中 `quality: QualityReview | None = None` 之后加一行：

```python
    difficulty: DifficultyScore | None = None
```

- [ ] **Step 4: 运行测试确认通过**

```bash
PYTHONPATH="$PWD" ../../.venv/bin/python -m pytest -q
```
Expected: `171 passed, 3 skipped`。

- [ ] **Step 5: 提交**

```bash
git add contracts.py tests/experiment/test_closed_loop.py
git commit -m "feat(contracts): 新增质量与难度评分数据结构"
```
body 说明：质量维度改为 1-5 数值分数并保留三裁判原始分数，难度增加推理深度与自动复杂度指标；裁判单次输出用 Judgment，聚合结果用 Review/DifficultyScore。

---

### Task 2: 质量与难度评测提示词

**Files:**
- Modify: `evaluation/prompts.py`（`QUALITY` 整段替换，新增 `DIFFICULTY`）
- Test: 无新增测试（Global Constraints 的提示词测试约定）

**Interfaces:**
- Consumes: Task 1 的 `QualityJudgment`、`DifficultyJudgment` 字段名。
- Produces: 模块级常量 `QUALITY`、`DIFFICULTY`，供 `evaluation/evaluator.py` 以 `prompts.QUALITY`、`prompts.DIFFICULTY` 使用。

- [ ] **Step 1: 替换 QUALITY**

把 `evaluation/prompts.py` 里的 `QUALITY` 整段替换为：

```python
QUALITY = """评价医疗任务的质量。分别评价四个维度，每个维度给出1-5分的score和评分理由reason。

factual_accuracy（事实准确性）：
5分=所有事实完全准确；4分=核心准确但有次要瑕疵；3分=多数准确但有1-2处非关键错误；
2分=关键事实错误；1分=多处严重错误或编造数据。

clinical_soundness（临床合理性）：
5分=推理严密，方案符合标准实践；4分=基本合理但不够严谨；3分=方向正确但推理简单；
2分=存在明显漏洞；1分=推理错误或方案有风险。

evidence_sufficiency（证据充分性）：
5分=每个关键事实都有明确evidence；4分=主要结论有evidence但细节不完整；3分=核心结论有evidence但推理步骤缺引用；
2分=关键结论缺evidence；1分=主要依赖未支持的推断。

task_clarity（题目清晰度）：
5分=目标明确，背景完整，无歧义；4分=基本清晰但个别细节需推断；3分=方向明确但缺背景或要求不具体；
2分=目标模糊或矛盾；1分=无法理解或严重歧义。

输出QualityJudgment，四个维度各包含score（整数1-5）和reason（字符串）。
依据task、reference_answer、evidence、trajectory和verification结果评价。"""
```

- [ ] **Step 2: 新增 DIFFICULTY**

在 `QUALITY` 之后追加：

```python
DIFFICULTY = """评估医疗任务的难度。

complexity_metrics已在输入中提供（evidence_count、resource_type_count、time_span_days）。

评估reasoning_depth（推理深度），1-4分：

1分=事实查询：直接从记录读取事实，无需推理。例如"患者血压是多少"。
2分=单步推理：基于记录+临床知识做简单判断。例如"该血压值是否异常"。
3分=整合推理：跨多条记录比较、识别趋势、整合多维信息。例如"血压控制效果如何"。
4分=临床决策：综合多维度证据、权衡风险收益、制定方案。例如"调整降压药方案"。

评分依据task.instruction的要求和evidence的复杂度。reason说明为何归为该层级。
输出DifficultyJudgment，包含reasoning_depth（整数1-4）和reason（字符串）。"""
```

- [ ] **Step 3: 运行全量测试**

```bash
PYTHONPATH="$PWD" ../../.venv/bin/python -m pytest -q
```
Expected: `171 passed, 3 skipped`。

- [ ] **Step 4: 核对提示词与设计文档一致**

逐条核对（设计文档 1.2、1.3、2.2、2.4）：四个维度键名与五级描述、四个难度层级与示例、输出结构名 `QualityJudgment` / `DifficultyJudgment`。提示词里不得出现旧维度名 `clinical_value`、`correctness`、`longitudinal_integration`、`clarity`。

- [ ] **Step 5: 提交**

```bash
git add evaluation/prompts.py
git commit -m "feat(evaluation): 改造质量评分提示词并新增难度评估提示词"
```
body 说明：旧提示词只要求文本 judgment，无法统计对比；新提示词要求 1-5 数值评分，难度提示词定义 1-4 推理深度层级。

---

### Task 3: 评测配置增加 difficulty 角色

**Files:**
- Modify: `configs/experiment.json`、`configs/medagentbench-flash.json`、`configs/medagentbench-selfgen.json`、`configs/physicianbench-comparison.json`、`configs/physicianbench-flash.json`、`configs/physicianbench-selfgen.json`（各文件 `evaluation.runtime.models` 与 `evaluation.runtime.budgets`）
- Test: 无新增测试（配置为纯数据，`tests/integration/test_pipeline.py` 会真实读取 `configs/experiment.json`）

**Interfaces:**
- Consumes: `runtime.models` / `runtime.budgets` 的 `dict[str, ModelConfig]` / `dict[str, Budget]` 结构。
- Produces: 每个评测配置里存在 `difficulty` 角色，供 Task 4 的 `config.runtime.models["difficulty"]` 使用。

- [ ] **Step 1: 在 6 个配置的 models 中加入 difficulty**

在每个文件的 `evaluation.runtime.models.quality` 块之后插入与同文件 `quality` 完全相同的块，键名改为 `difficulty`。例如 `configs/experiment.json`：

```json
        "difficulty": {
          "model": "gpt-4o-mini",
          "base_url": "https://api.openlux.ai/v1",
          "api_key_env": "OPENLUX_API_KEY",
          "instructions_role": "system",
          "parallel_tool_calls": true
        },
```
`configs/medagentbench-flash.json`、`configs/medagentbench-selfgen.json`、`configs/physicianbench-flash.json`、`configs/physicianbench-selfgen.json` 用各自的 `quality` 块内容（`deepseek-v4.1-flash` / `https://opencode.ai/zen/go/v1` / `OPENCODE_API_KEY`），`configs/physicianbench-comparison.json` 用 `gpt-4o-mini` / `https://api.openlux.ai/v1` / `OPENLUX_API_KEY`。以同文件 `quality` 块为准逐字复制，不要凭记忆填写。

- [ ] **Step 2: 在 6 个配置的 budgets 中加入 difficulty**

同样复制同文件 `quality` 的预算块，键名改为 `difficulty`：

```json
        "difficulty": {
          "max_model_turns": 30,
          "max_tool_calls": 200
        }
```

- [ ] **Step 3: 校验 JSON 与角色齐全**

```bash
PYTHONPATH="$PWD" ../../.venv/bin/python -c "
import json, glob
for path in sorted(glob.glob('configs/*.json')):
    runtime = json.load(open(path))['evaluation']['runtime']
    assert 'difficulty' in runtime['models'], path
    assert 'difficulty' in runtime['budgets'], path
    assert runtime['models']['difficulty'] == runtime['models']['quality'], path
print('ok')
"
```
Expected: 输出 `ok`。

- [ ] **Step 4: 运行全量测试**

```bash
PYTHONPATH="$PWD" ../../.venv/bin/python -m pytest -q
```
Expected: `171 passed, 3 skipped`。

- [ ] **Step 5: 提交**

```bash
git add configs
git commit -m "chore(configs): 评测运行配置增加 difficulty 角色"
```
body 说明：难度评分需要独立的模型与预算条目，先让配置就位，评测流程随后使用。

---

### Task 4: 三裁判评测流程与复杂度自动计算

**Files:**
- Modify: `evaluation/evaluator.py`
- Test: `tests/evaluation/test_evaluator.py`（改造 harness 与故障表，新增 4 个测试）
- Test: `tests/integration/test_pipeline.py:76` 附近（假模型质量/难度响应）
- Test: `tests/integration/test_medagentbench_integration.py:50` 附近（假模型 schema 分支）

**Interfaces:**
- Consumes: Task 1 的 `QualityJudgment`、`QualityReview`、`QualityDimension`、`DifficultyJudgment`、`DifficultyScore`、`ComplexityMetrics`；Task 2 的 `prompts.QUALITY`、`prompts.DIFFICULTY`；Task 3 的 `config.runtime.models["difficulty"]`。
- Produces:
  - `_compute_complexity(task: Task, trajectory: PatientTrajectory) -> ComplexityMetrics`
  - `_aggregate_quality_scores(runs: list[RunResult]) -> QualityReview`
  - `_aggregate_difficulty_scores(runs: list[RunResult], complexity: ComplexityMetrics) -> DifficultyScore`
  - `evaluate_task` 在 `result.quality` 与 `result.difficulty` 上写入聚合结果。

- [ ] **Step 1: 改造测试 harness（先红）**

`tests/evaluation/test_evaluator.py` 的导入加上新契约名：

```python
from contracts import (
    Budget, CompletionReview, ComplexityMetrics, DifficultyJudgment, EnvironmentConfig,
    EnvironmentInstance, Evidence, EvaluationConfig, ModelConfig, PatientTrajectory,
    QualityJudgment, QualityReview, RunResult, RuntimeConfig, SolverAnswer, Task,
    Verification,
)
```

`harness` 里角色增加 difficulty：

```python
    roles = ["verifier", "quality", "difficulty", "solver", "completion"]
```

`SimpleNamespace` 增加两个可调字段：

```python
        quality_scores=[4, 4, 4], depths=[2, 2, 2],
```

假 `run` 的质量分支替换为按裁判序号取分，并新增难度分支：

```python
        if role == "quality":
            index = sum(1 for call in h.model_calls[:-1]
                        if call["output_type"] is QualityJudgment)
            output = {key: {"score": h.quality_scores[index], "reason": "supported"}
                      for key in ("factual_accuracy", "clinical_soundness",
                                  "evidence_sufficiency", "task_clarity")}
        elif role == "difficulty":
            index = sum(1 for call in h.model_calls[:-1]
                        if call["output_type"] is DifficultyJudgment)
            output = {"reasoning_depth": h.depths[index], "reason": "跨记录整合"}
```

`test_solver_receives_only_patient_question_and_decision_point` 的 `len(result.runs) == 5` 改为 `== 10`。

故障表 `test_faults_release_instances_without_clinical_conclusions` 的参数改为：

```python
    ("evidence", "environment_error", 1, 0),
    ("verifier", "model_error", 1, 1),
    ("verifier_budget", "model_error", 1, 1),
    ("quality", "model_error", 1, 5),
    ("difficulty", "model_error", 1, 8),
    ("solver", "model_error", 2, 9),
    ("solver_tool_error", "tool_error", 2, 9),
    ("solver_environment_error", "environment_error", 2, 9),
    ("solver_model_error", "model_error", 2, 9),
    ("solver_unknown_error", "model_error", 2, 9),
    ("solver_budget", "model_error", 2, 9),
    ("state", "environment_error", 2, 9),
    ("completion", "model_error", 2, 10),
```
并在该测试函数体末尾追加：

```python
    if failure == "difficulty":
        assert result.quality is not None
        assert result.difficulty is None
```

新增四个测试（追加到文件末尾）：

```python
def test_three_judges_take_median_and_keep_raw_scores(harness):
    h = harness
    h.quality_scores = [5, 3, 4]
    h.depths = [4, 1, 3]
    result = evaluate(h)
    factual = result.quality.factual_accuracy
    assert factual.score == 4
    assert factual.scores == [5, 3, 4]
    assert factual.reason == "J1: supported | J2: supported | J3: supported"
    assert result.quality.task_clarity.score == 4
    assert result.difficulty.reasoning_depth == 3
    assert result.difficulty.reasoning_depths == [4, 1, 3]
    assert result.difficulty.reason == "J1: 跨记录整合 | J2: 跨记录整合 | J3: 跨记录整合"
    assert len([c for c in h.model_calls
                if c["output_type"] is QualityJudgment]) == 3
    assert len([c for c in h.model_calls
                if c["output_type"] is DifficultyJudgment]) == 3


def test_difficulty_receives_only_task_and_computed_complexity(harness):
    h = harness
    result = evaluate(h)
    quality_call = next(c for c in h.model_calls if c["output_type"] is QualityJudgment)
    assert set(json.loads(quality_call["input_text"])) == {"task", "trajectory", "verification"}
    difficulty_call = next(c for c in h.model_calls if c["output_type"] is DifficultyJudgment)
    payload = json.loads(difficulty_call["input_text"])
    assert set(payload) == {"task", "complexity_metrics"}
    assert payload["task"]["instruction"] == h.task.instruction
    assert payload["complexity_metrics"] == {"evidence_count": 1, "resource_type_count": 0,
                                            "time_span_days": None}
    assert result.difficulty.complexity_metrics == ComplexityMetrics(
        evidence_count=1, resource_type_count=0, time_span_days=None)


def test_complexity_metrics_count_sources_types_and_time_span(harness):
    h = harness
    h.task = h.task.model_copy(update={"evidence": [
        Evidence(source_id="Observation/1", statement="首次检验"),
        Evidence(source_id="Condition/2", statement="首次诊断"),
    ]})
    h.trajectory = h.trajectory.model_copy(update={"sources": [
        Evidence(source_id="Observation/1", statement=json.dumps(
            {"resourceType": "Observation", "effectiveDateTime": "2026-01-01T00:00:00+08:00"})),
        Evidence(source_id="Condition/2", statement=json.dumps(
            {"resourceType": "Condition", "onsetDateTime": "2026-01-11T00:00:00+08:00"})),
        Evidence(source_id="Observation/9", statement=json.dumps(
            {"resourceType": "Observation", "effectiveDateTime": "2027-05-01T00:00:00+08:00"})),
        Evidence(source_id="Patient/3", statement="不是 JSON"),
    ]})
    result = evaluate(h)
    assert result.difficulty.complexity_metrics == ComplexityMetrics(
        evidence_count=2, resource_type_count=2, time_span_days=10.0)


@pytest.mark.parametrize("field, scores", [("quality_scores", [6, 4, 4]), ("depths", [5, 2, 2])])
def test_out_of_range_judgments_fail_evaluation(harness, field, scores):
    h = harness
    setattr(h, field, scores)
    result = evaluate(h)
    assert result.status == "failed"
    assert result.failure_category == "model_error"
    assert result.completion is None
```

同时改 `tests/integration/test_pipeline.py` 假模型：把 `elif role == 'quality':` 分支换成新结构，并在其后加入难度分支：

```python
        elif role == 'quality':
            output = {d: {'score': 4, 'reason': '原始数据支持'}
                      for d in ['factual_accuracy', 'clinical_soundness',
                                'evidence_sufficiency', 'task_clarity']}
        elif role == 'difficulty':
            output = {'reasoning_depth': 2, 'reason': '跨记录整合'}
```

`tests/integration/test_medagentbench_integration.py`：`roles` 增加 `"difficulty"`，导入增加 `DifficultyJudgment`、`QualityJudgment`，假 `run` 的分支改为：

```python
        elif schema is QualityJudgment:
            output = {key: {"score": 4, "reason": "满足要求"}
                      for key in ("factual_accuracy", "clinical_soundness",
                                  "evidence_sufficiency", "task_clarity")}
        elif schema is DifficultyJudgment:
            output = {"reasoning_depth": 2, "reason": "跨记录整合"}
```

- [ ] **Step 2: 运行测试确认失败**

```bash
PYTHONPATH="$PWD" ../../.venv/bin/python -m pytest tests/evaluation/test_evaluator.py -q
```
Expected: FAIL —— 假模型返回的 `QualityJudgment` 结构与旧评测代码期望的 `dimensions` 不匹配，且难度角色未被调用（`next()` 找不到 `DifficultyJudgment` 调用）或 `result.difficulty` 不存在。

- [ ] **Step 3: 改造 evaluator 常量与导入**

`evaluation/evaluator.py` 顶部导入增加：

```python
import statistics
from datetime import datetime, timezone
```

`from contracts import (...)` 增加 `ComplexityMetrics, DifficultyJudgment, DifficultyScore, QualityDimension, QualityJudgment`（`QualityReview` 保留）。

把

```python
QUALITY_DIMENSIONS = {
    "clinical_value", "correctness", "longitudinal_integration", "clarity"
}
```

替换为

```python
QUALITY_DIMENSIONS = (
    "factual_accuracy", "clinical_soundness", "evidence_sufficiency", "task_clarity",
)
JUDGE_COUNT = 3
TIME_FIELDS = (
    "effectiveDateTime", "onsetDateTime", "performedDateTime", "occurrenceDateTime",
    "authoredOn", "sent", "start", "date", "recordedDate", "issued",
)
```

- [ ] **Step 4: 新增聚合与复杂度函数**

在 `_output` 之后加入：

```python
def _judge_outputs(runs: list[RunResult], schema: type[BaseModel]) -> list[BaseModel]:
    if len(runs) != JUDGE_COUNT:
        raise ValueError(f"judging requires exactly {JUDGE_COUNT} runs")
    return [_output(run, schema) for run in runs]


def _aggregate_quality_scores(runs: list[RunResult]) -> QualityReview:
    judges = _judge_outputs(runs, QualityJudgment)

    def aggregate(dimension: str) -> QualityDimension:
        judgments = [getattr(judge, dimension) for judge in judges]
        scores = [judgment.score for judgment in judgments]
        for score in scores:
            if not 1 <= score <= 5:
                raise ValueError(f"{dimension}.score must be 1-5")
        return QualityDimension(
            score=int(statistics.median(scores)), scores=scores,
            reason=" | ".join(f"J{index}: {judgment.reason}"
                              for index, judgment in enumerate(judgments, start=1)),
        )

    return QualityReview(
        factual_accuracy=aggregate("factual_accuracy"),
        clinical_soundness=aggregate("clinical_soundness"),
        evidence_sufficiency=aggregate("evidence_sufficiency"),
        task_clarity=aggregate("task_clarity"),
    )


def _aggregate_difficulty_scores(runs: list[RunResult],
                                 complexity: ComplexityMetrics) -> DifficultyScore:
    judges = _judge_outputs(runs, DifficultyJudgment)
    depths = [judge.reasoning_depth for judge in judges]
    for depth in depths:
        if not 1 <= depth <= 4:
            raise ValueError("reasoning_depth must be 1-4")
    return DifficultyScore(
        reasoning_depth=int(statistics.median(depths)), reasoning_depths=depths,
        reason=" | ".join(f"J{index}: {judge.reason}"
                          for index, judge in enumerate(judges, start=1)),
        complexity_metrics=complexity,
    )


def _statement_time(statement: str) -> datetime | None:
    try:
        resource = json.loads(statement)
    except (TypeError, ValueError):
        return None
    if not isinstance(resource, dict):
        return None
    for field in TIME_FIELDS:
        value = resource.get(field)
        if not isinstance(value, str):
            continue
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            continue
        return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)
    return None


def _compute_complexity(task: Task, trajectory: PatientTrajectory) -> ComplexityMetrics:
    source_ids = {evidence.source_id for evidence in task.evidence}
    resource_types = {source_id.split("/", 1)[0]
                      for source_id in source_ids if "/" in source_id}
    moments = []
    for source in trajectory.sources:
        if source.source_id not in source_ids:
            continue
        moment = _statement_time(source.statement)
        if moment is not None:
            moments.append(moment)
    span = None
    if moments:
        span = (max(moments) - min(moments)).total_seconds() / 86400
    return ComplexityMetrics(
        evidence_count=len(source_ids), resource_type_count=len(resource_types),
        time_span_days=span,
    )
```

- [ ] **Step 5: 改动评测流程**

角色校验行改为：

```python
        for role in ("verifier", "quality", "difficulty", "solver", "completion"):
            if role not in config.runtime.models:
                raise ValueError(f"missing evaluation model: {role}")
```

把质量评测整块（从 `phase = "model_error"` 到 `result.quality = quality`）替换为：

```python
        phase = "model_error"
        material["verification"] = result.verification.model_dump()
        result.quality = _aggregate_quality_scores([
            await model_run(runtime.call_llm, role="quality", schema=QualityJudgment,
                            instructions=prompts.QUALITY, payload=material)
            for _ in range(JUDGE_COUNT)
        ])
        complexity = _compute_complexity(task, trajectory)
        result.difficulty = _aggregate_difficulty_scores([
            await model_run(runtime.call_llm, role="difficulty", schema=DifficultyJudgment,
                            instructions=prompts.DIFFICULTY,
                            payload={"task": material["task"],
                                     "complexity_metrics": complexity.model_dump()})
            for _ in range(JUDGE_COUNT)
        ], complexity)
```

- [ ] **Step 6: 运行测试确认通过**

```bash
PYTHONPATH="$PWD" ../../.venv/bin/python -m pytest -q
```
Expected: `175 passed, 3 skipped`（171 基线 - 1 个被参数化替换的旧测试 + 5 个新增用例：3 个新测试 + 2 个参数化用例）。

- [ ] **Step 7: 提交**

```bash
git add evaluation/evaluator.py tests/evaluation/test_evaluator.py \
        tests/integration/test_pipeline.py tests/integration/test_medagentbench_integration.py
git commit -m "feat(evaluation): 三裁判打分并自动计算任务复杂度"
```
body 说明：单次文本 judgment 无法统计对比，改成三裁判各打 1-5 分取中位数并保留原始分数；难度用三裁判 1-4 层级加 evidence 与轨迹自动计算的复杂度指标。

---

### Task 5: 质量与难度汇总

**Files:**
- Modify: `experiments/summary.py`
- Test: `tests/experiment/test_closed_loop.py`（假评测结果与汇总断言）

**Interfaces:**
- Consumes: Task 4 写入 `evaluations.json` 的 `quality`（四维 `score`/`scores`/`reason`）与 `difficulty`（`reasoning_depth`、`complexity_metrics`）。
- Produces:
  - `summarize_quality(evaluations: list[dict]) -> JsonObject`，键为四个维度名，值含 `mean`、`median`、`distribution`、`high_quality_rate`
  - `summarize_difficulty(evaluations: list[dict]) -> JsonObject`，含 `reasoning_depth`（`mean`、`distribution`）与 `complexity_metrics`（`avg_evidence_count`、`avg_resource_types`）
  - 汇总文件夹结果新增 `quality_summary`、`difficulty_summary` 两个键，替换原 `quality_judgments`

- [ ] **Step 1: 改造闭环测试（先红）**

`tests/experiment/test_closed_loop.py` 的假 `evaluate` 增加难度字段：

```python
        quality = QualityReview(**{
            name: QualityDimension(score=4, scores=[4, 4, 4], reason="clear")
            for name in ("factual_accuracy", "clinical_soundness",
                         "evidence_sufficiency", "task_clarity")})
        difficulty = DifficultyScore(
            reasoning_depth=2, reasoning_depths=[2, 2, 2], reason="整合",
            complexity_metrics=ComplexityMetrics(evidence_count=3, resource_type_count=2,
                                                time_span_days=5.0))
        return EvaluationResult(task_id=task.id, verification=Verification(valid=True, reason="supported"),
                                quality=quality, difficulty=difficulty,
                                completion=CompletionReview(completed=True, reason="done"),
                                solver_run=solver, runs=[solver], observed_state={}, status="completed")
```

在 `test_two_rounds_and_read_only_summary` 中，把原来的 `quality_judgments` 断言位置（Task 1 已删除该行）替换为：

```python
    reviews = 4 if failure else 8
    assert summary["evolution"]["quality_summary"]["task_clarity"] == {
        "mean": 4, "median": 4.0, "distribution": {"4": reviews}, "high_quality_rate": 1.0}
    assert summary["evolution"]["difficulty_summary"]["reasoning_depth"] == {
        "mean": 2, "distribution": {"2": reviews}}
    assert summary["evolution"]["difficulty_summary"]["complexity_metrics"] == {
        "avg_evidence_count": 3, "avg_resource_types": 2}
    assert summary["evaluation"]["quality_summary"]["factual_accuracy"]["high_quality_rate"] == 1.0
```

- [ ] **Step 2: 运行测试确认失败**

```bash
PYTHONPATH="$PWD" ../../.venv/bin/python -m pytest tests/experiment/test_closed_loop.py -q
```
Expected: FAIL，`KeyError: 'quality_summary'`。

- [ ] **Step 3: 写汇总函数**

`experiments/summary.py` 顶部的 `from collections import Counter, defaultdict` 改为 `from collections import Counter`，并新增 `import statistics`。在 `summarize_run` 之前加入：

```python
QUALITY_DIMENSIONS = (
    "factual_accuracy", "clinical_soundness", "evidence_sufficiency", "task_clarity",
)


def summarize_quality(evaluations: list[dict]) -> JsonObject:
    reviews = [evaluation["quality"] for evaluation in evaluations
               if evaluation.get("quality")]
    if not reviews:
        return {}
    summary = {}
    for dimension in QUALITY_DIMENSIONS:
        scores = [review[dimension]["score"] for review in reviews]
        summary[dimension] = {
            "mean": statistics.mean(scores),
            "median": statistics.median(scores),
            "distribution": {str(score): count for score, count in sorted(Counter(scores).items())},
            "high_quality_rate": sum(score >= 4 for score in scores) / len(scores),
        }
    return summary


def summarize_difficulty(evaluations: list[dict]) -> JsonObject:
    scores = [evaluation["difficulty"] for evaluation in evaluations
              if evaluation.get("difficulty")]
    if not scores:
        return {}
    depths = [score["reasoning_depth"] for score in scores]
    evidence_counts = [score["complexity_metrics"]["evidence_count"] for score in scores]
    resource_counts = [score["complexity_metrics"]["resource_type_count"] for score in scores]
    return {
        "reasoning_depth": {
            "mean": statistics.mean(depths),
            "distribution": {str(depth): count for depth, count in sorted(Counter(depths).items())},
        },
        "complexity_metrics": {
            "avg_evidence_count": statistics.mean(evidence_counts),
            "avg_resource_types": statistics.mean(resource_counts),
        },
    }
```

- [ ] **Step 4: 接入 summarize_folders**

把 `summarize_folders` 里的

```python
        quality = defaultdict(Counter)
        for evaluation in evaluations:
            for dimension, assessment in (evaluation.get("quality") or {}).get("dimensions", {}).items():
                if isinstance(assessment, dict) and "judgment" in assessment:
                    quality[dimension][str(assessment["judgment"])] += 1
```

替换为

```python
        quality = summarize_quality(evaluations)
        difficulty = summarize_difficulty(evaluations)
```

并把返回字典里的

```python
            "quality_judgments": dict(quality),
```

替换为

```python
            "quality_summary": quality,
            "difficulty_summary": difficulty,
```

其余键（`quality_reviews` 等）保持原样。

- [ ] **Step 5: 运行测试确认通过**

```bash
PYTHONPATH="$PWD" ../../.venv/bin/python -m pytest -q
```
Expected: `175 passed, 3 skipped`。

- [ ] **Step 6: 提交**

```bash
git add experiments/summary.py tests/experiment/test_closed_loop.py
git commit -m "feat(experiments): 汇总质量与难度分布"
```
body 说明：数值评分才能做统计对比，汇总输出每个维度的均值、中位数、分布与高质量占比，以及推理深度分布与平均复杂度。

---

### Task 6: 反馈提示词对齐新的质量维度名

**Files:**
- Modify: `methods/medevolve/prompts.py`（`FEEDBACK` 的建议一行）
- Test: 无新增测试（Global Constraints 的提示词测试约定）

**Interfaces:**
- Consumes: Task 1 的四个维度名。
- Produces: `FEEDBACK` 文本中提到新的四个维度名。

- [ ] **Step 1: 改维度名**

把 `FEEDBACK` 里的

```
建议应针对证据支持、临床价值、纵向整合、清晰度和生成方向，避免补造数据。
```

替换为

```
建议应针对事实准确性、临床合理性、证据充分性和题目清晰度，以及生成方向，避免补造数据。
```

文件其余部分（含 `区分质量问题、临床难度、证据不足、题意歧义及工具、环境、模型运行故障。`）保持不变。

- [ ] **Step 2: 运行全量测试**

```bash
PYTHONPATH="$PWD" ../../.venv/bin/python -m pytest -q
```
Expected: `175 passed, 3 skipped`。

- [ ] **Step 3: 确认没有残留旧维度名**

```bash
PYTHONPATH="$PWD" ../../.venv/bin/python -c "
from methods.medevolve import prompts
text = prompts.FEEDBACK
for stale in ('clinical_value', 'longitudinal_integration', '临床价值', '纵向整合'):
    assert stale not in text, stale
assert '事实准确性' in text and '证据充分性' in text
print('ok')
"
```
Expected: 输出 `ok`。

- [ ] **Step 4: 提交**

```bash
git add methods/medevolve/prompts.py
git commit -m "feat(medevolve): 反馈提示词改用新的质量维度名"
```
body 说明：质量维度改名后旧提示词引用的维度已不存在，反馈建议无法对应到实际评分项。

---

## Self-Review

**Spec coverage：** 数据结构（1.1、2.1）→ Task 1；质量 prompt（1.3）与难度 prompt（2.4）→ Task 2；评测流程与聚合（3.1、3.2、3.3）→ Task 4；自动复杂度（2.3，含 `time_span_days` 的真实解析，取代设计文档示例里的 `None` 占位）→ Task 4；汇总（4.1、4.2）→ Task 5；配置（实施步骤 5）→ Task 3；本地测试（实施步骤 6）→ 每个任务的测试步骤。设计文档的实施步骤 7（六方法全量重跑）按用户确认不在本计划内。

**类型一致性：** `QualityJudgment`（裁判输出）/ `QualityReview`（聚合）与 `DifficultyJudgment` / `DifficultyScore` 的区分在 Task 1、2、4 中一致；`_compute_complexity`、`_aggregate_quality_scores`、`_aggregate_difficulty_scores` 的签名在 Task 4 定义并在该任务内使用；汇总读取的键名（`quality`、`difficulty`、`reasoning_depth`、`complexity_metrics`）与 Task 1、Task 4 写出的字段一致。

**已知取舍：** 汇总函数接收 `evaluations.json` 记录（dict），不是设计文档 4.1 注解里的 `EvaluationResult` 对象 —— 沿用该模块既有的按文件读取方式；`time_span_days` 不进汇总输出，按设计文档 4.2 的字段清单执行。
