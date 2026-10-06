# 医疗任务生成评测指标体系重构设计

## 设计目标

重构evaluation模块的质量和难度评估指标，使其能够：
1. 用数值化评分替代文本judgment，支持统计对比
2. 引入客观的难度度量（自动计算 + 模型打分推理深度）
3. 支持6个方法（MedEvolve、TaskCraft、Workflow、SingleShot、SelfGen、EnvExplore）的质量、难度、成本对比

## 改造范围

**只修改evaluation模块**：
- `evaluation/evaluator.py` - 评测流程
- `evaluation/prompts.py` - 评测prompt
- `contracts.py` - 新增质量/难度相关的数据结构
- `experiments/summary.py` - 汇总逻辑适配新指标

**不修改**：
- 生成模块（methods/medevolve/、methods/*/的generation相关代码）
- 反馈模块（feedback和strategy_update保持原样；用户确认的唯一例外是 FEEDBACK 提示词补入难度评价输入并改用新的质量维度名）

## 整体架构

### 评测流程

保持现有流程，改造quality评测，新增difficulty评测：

```
verification → quality_scorer → difficulty_scorer → solver → completion
```

- verification：保持不变，逐条核验evidence
- quality_scorer：**改造** - 用新的四维数值评分替代旧的四维文本judgment
- difficulty_scorer：**新增** - 自动计算复杂度 + 模型打分推理深度
- solver：保持不变
- completion：保持不变

### 多裁判打分机制

quality和difficulty都使用**三裁判模型投票**：
- 每个裁判独立打分
- 取三个分数的**中位数**作为最终分数
- 保留所有裁判的原始分数和reason用于归因分析

## 详细设计

### 一、质量评价维度重构

#### 1.1 数据结构

```python
# contracts.py
class QualityDimension(BaseModel):
    score: int  # 最终分数（三裁判中位数），1-5分
    scores: list[int]  # 三个裁判的原始分数
    reason: str  # 合并的评分依据

class QualityReview(BaseModel):
    factual_accuracy: QualityDimension      # 事实准确性
    clinical_soundness: QualityDimension    # 临床合理性
    evidence_sufficiency: QualityDimension  # 证据充分性
    task_clarity: QualityDimension          # 题目清晰度
```

单个裁判只输出 score 与 reason（`DimensionJudgment`，四维组合为 `QualityJudgment`）；
`scores` 与合并后的 `reason` 由评测侧聚合函数写入 `QualityReview`，裁判不产出这两个字段。

#### 1.2 四个质量维度定义

**1. factual_accuracy（事实准确性）**

评估题目和答案中的医疗事实是否与evidence一致：

- 5分：所有事实（数值、单位、药物名、剂量、时间）完全准确，与evidence严格对应
- 4分：核心事实准确，但存在次要细节瑕疵（如单位换算、时间格式表述）
- 3分：多数事实准确，但有1-2处非关键事实错误或不够精确
- 2分：存在关键事实错误（如药物剂量错误、检验值错误）
- 1分：多处严重事实错误，或编造evidence未支持的患者数据

**2. clinical_soundness（临床合理性）**

评估临床判断、推理、方案是否符合医学逻辑：

- 5分：临床推理严密，诊疗方案符合指南和标准实践，风险评估充分
- 4分：临床判断基本合理，但个别推理步骤不够严谨或方案略显保守/激进
- 3分：临床方向正确，但推理简单或遗漏重要考虑因素
- 2分：临床判断存在明显漏洞或违背常规实践
- 1分：临床推理错误，或方案存在明显风险

**3. evidence_sufficiency（证据充分性）**

评估evidence是否足以支撑答案的全部关键事实和结论：

- 5分：每个关键事实和结论都有明确的evidence支撑，引用完整
- 4分：主要结论有evidence，但次要细节的引用不够完整
- 3分：核心结论有evidence，但部分推理步骤缺少直接引用
- 2分：关键结论缺少evidence支持，或引用不足以支撑声明
- 1分：答案主要依赖未被evidence支持的推断或编造

**4. task_clarity（题目清晰度）**

评估instruction是否明确、可操作、无歧义：

- 5分：任务目标明确，必要背景完整，操作要求清晰，无歧义
- 4分：任务基本清晰，但个别细节需要合理推断
- 3分：任务方向明确，但缺少关键背景信息或操作要求不够具体
- 2分：任务目标模糊，或包含矛盾信息，或缺少必要的时间/范围限定
- 1分：任务无法理解或存在严重歧义，无法执行

#### 1.3 Prompt设计

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

### 二、难度评估设计

#### 2.1 数据结构

```python
# contracts.py
class ComplexityMetrics(BaseModel):
    evidence_count: int           # evidence中source_id的数量
    resource_type_count: int      # 涉及的FHIR资源类型数
    time_span_days: float | None  # 最早到最晚记录的天数跨度

class DifficultyScore(BaseModel):
    reasoning_depth: int  # 最终层级（三裁判中位数），1-4
    reasoning_depths: list[int]  # 三个裁判的原始层级
    reason: str  # 合并的评分依据
    complexity_metrics: ComplexityMetrics  # 自动计算指标
```

单个裁判只输出 `reasoning_depth` 与 `reason`（`DifficultyJudgment`）；`reasoning_depths` 与 `complexity_metrics` 由评测侧聚合函数写入。

#### 2.2 推理深度层级定义

**Level 1（事实查询）**  
直接从单条或少数记录读取事实，无需推理。  
示例："患者的血压是多少？"、"开具了哪些药物？"

**Level 2（单步推理）**  
基于单条或少数记录+临床知识做简单判断。  
示例："该血压值是否异常？"、"该药物剂量是否合理？"

**Level 3（整合推理）**  
需要跨多条记录比较、识别趋势、整合多维信息。  
示例："血压控制效果如何？"、"患者的疾病进展趋势？"

**Level 4（临床决策）**  
需要综合多维度证据、权衡风险收益、制定方案。  
示例："调整降压药方案"、"制定进一步检查计划"

#### 2.3 自动计算指标

```python
def _compute_complexity(task: Task, trajectory: PatientTrajectory) -> ComplexityMetrics:
    """自动计算复杂度指标"""
    source_ids = {e.source_id for e in task.evidence}
    
    # 统计资源类型（从source_id前缀提取，如"Observation/123" -> "Observation"）
    resource_types = {sid.split('/')[0] for sid in source_ids if '/' in sid}
    
    # 时间跨度：解析任务引用记录的时间字段，口径与 environments 的 _record_time 一致
    # 平铺字段 → effectivePeriod/onsetPeriod/performedPeriod/period 的 start → context.period.start
    # 不带时区的值按 UTC 处理；无可用时间时为 None
    moments = [_statement_time(source.statement) for source in trajectory.sources
               if source.source_id in source_ids]
    moments = [moment for moment in moments if moment is not None]
    time_span_days = ((max(moments) - min(moments)).total_seconds() / 86400
                      if moments else None)
    
    return ComplexityMetrics(
        evidence_count=len(source_ids),
        resource_type_count=len(resource_types),
        time_span_days=time_span_days,
    )
```

#### 2.4 Prompt设计

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

complexity_metrics 已作为输入提供，最终 DifficultyScore 由评测侧写入该指标与三位裁判的原始层级。

### 三、评测流程改造

#### 3.1 evaluator.py改动

```python
async def evaluate_task(task: Task, trajectory: PatientTrajectory,
                        environment: EnvironmentConfig,
                        config: EvaluationConfig) -> EvaluationResult:
    # ... 现有verification逻辑 ...
    
    if not result.verification.valid:
        result.status = "invalid"
        return result
    
    # === 改造：Quality评测 - 三裁判 ===
    quality_runs = []
    for judge_idx in range(3):
        run = await model_run(
            runtime.call_llm, role="quality", schema=QualityJudgment,
            instructions=prompts.QUALITY, payload=material,
        )
        quality_runs.append(run)
    
    # 聚合三个裁判的质量评分
    result.quality = _aggregate_quality_scores(quality_runs)
    
    # === 新增：Difficulty评测 - 三裁判 ===
    complexity = _compute_complexity(task, trajectory)
    difficulty_runs = []
    for judge_idx in range(3):
        run = await model_run(
            runtime.call_llm, role="difficulty", schema=DifficultyJudgment,
            instructions=prompts.DIFFICULTY,
            payload={
                "task": material["task"],
                "complexity_metrics": complexity.model_dump()
            },
        )
        difficulty_runs.append(run)
    
    # 聚合三个裁判的难度评分
    result.difficulty = _aggregate_difficulty_scores(difficulty_runs, complexity)
    
    # ... 现有solver和completion逻辑 ...
```

#### 3.2 聚合函数

```python
import statistics

def _aggregate_quality_scores(runs: list[RunResult]) -> QualityReview:
    """聚合三个裁判的质量评分"""
    judges = [_output(run, QualityJudgment) for run in runs]
    
    def aggregate_dimension(dim_name: str) -> QualityDimension:
        dims = [getattr(judge, dim_name) for judge in judges]
        scores = [d.score for d in dims]
        reasons = [d.reason for d in dims]
        
        # 中位数作为最终分数
        final_score = int(statistics.median(scores))
        # 合并reason
        combined_reason = " | ".join(f"J{i+1}: {r}" for i, r in enumerate(reasons))
        
        return QualityDimension(
            score=final_score,
            scores=scores,
            reason=combined_reason
        )
    
    return QualityReview(
        factual_accuracy=aggregate_dimension("factual_accuracy"),
        clinical_soundness=aggregate_dimension("clinical_soundness"),
        evidence_sufficiency=aggregate_dimension("evidence_sufficiency"),
        task_clarity=aggregate_dimension("task_clarity"),
    )

def _aggregate_difficulty_scores(
    runs: list[RunResult],
    complexity: ComplexityMetrics
) -> DifficultyScore:
    """聚合三个裁判的难度评分"""
    judges = [_output(run, DifficultyJudgment) for run in runs]
    
    depths = [j.reasoning_depth for j in judges]
    reasons = [j.reason for j in judges]
    
    final_depth = int(statistics.median(depths))
    combined_reason = " | ".join(f"J{i+1}: {r}" for i, r in enumerate(reasons))
    
    return DifficultyScore(
        reasoning_depth=final_depth,
        reasoning_depths=depths,
        reason=combined_reason,
        complexity_metrics=complexity,
    )
```

#### 3.3 校验逻辑

```python
# 裁判数量固定为 3；每个裁判的 reason 必须非空，分数必须在区间内，再取中位数
if len(quality_runs) != 3:
    raise ValueError("quality scoring requires exactly three judges")
for judge in quality_judges:  # 每个裁判一次 QualityJudgment
    for dim_name in ["factual_accuracy", "clinical_soundness",
                     "evidence_sufficiency", "task_clarity"]:
        dim = getattr(judge, dim_name)
        if not dim.reason.strip():
            raise ValueError(f"{dim_name}.reason must not be empty")
        if not (1 <= dim.score <= 5):
            raise ValueError(f"{dim_name}.score must be 1-5")

if len(difficulty_runs) != 3:
    raise ValueError("difficulty scoring requires exactly three judges")
for judge in difficulty_judges:  # 每个裁判一次 DifficultyJudgment
    if not judge.reason.strip():
        raise ValueError("reason must not be empty")
    if not (1 <= judge.reasoning_depth <= 4):
        raise ValueError("reasoning_depth must be 1-4")
```

### 四、汇总逻辑改造

#### 4.1 summary.py新增函数

输入为 evaluations.json 记录（dict）。实现中会跳过结构早于数值评分的旧记录，以及缺少四个维度数值分数或难度字段的记录。

```python
QUALITY_DIMENSIONS = (
    "factual_accuracy", "clinical_soundness", "evidence_sufficiency", "task_clarity",
)


def _quality_scores(evaluation: dict) -> list[int] | None:
    review = evaluation.get("quality")
    if not isinstance(review, dict):
        return None
    scores = []
    for dimension in QUALITY_DIMENSIONS:
        assessment = review.get(dimension)
        if not isinstance(assessment, dict) or not isinstance(assessment.get("score"), int):
            return None
        scores.append(assessment["score"])
    return scores


def _difficulty_score(evaluation: dict) -> dict | None:
    score = evaluation.get("difficulty")
    if not isinstance(score, dict) or not isinstance(score.get("reasoning_depth"), int):
        return None
    metrics = score.get("complexity_metrics")
    if not isinstance(metrics, dict):
        return None
    if not all(isinstance(metrics.get(field), int)
               for field in ("evidence_count", "resource_type_count")):
        return None
    return {"reasoning_depth": score["reasoning_depth"],
            "evidence_count": metrics["evidence_count"],
            "resource_type_count": metrics["resource_type_count"]}


def summarize_quality(evaluations: list[dict]) -> JsonObject:
    # Legacy records keep the retired `dimensions` shape and cannot be aggregated.
    score_lists = [scores for evaluation in evaluations
                   if (scores := _quality_scores(evaluation)) is not None]
    if not score_lists:
        return {}
    summary = {}
    for index, dimension in enumerate(QUALITY_DIMENSIONS):
        scores = [score_list[index] for score_list in score_lists]
        summary[dimension] = {
            "mean": float(statistics.mean(scores)),
            "median": float(statistics.median(scores)),
            "distribution": {str(score): count for score, count in sorted(Counter(scores).items())},
            "high_quality_rate": sum(score >= 4 for score in scores) / len(scores),
        }
    return summary


def summarize_difficulty(evaluations: list[dict]) -> JsonObject:
    entries = [entry for evaluation in evaluations
               if (entry := _difficulty_score(evaluation)) is not None]
    if not entries:
        return {}
    depths = [entry["reasoning_depth"] for entry in entries]
    evidence_counts = [entry["evidence_count"] for entry in entries]
    resource_counts = [entry["resource_type_count"] for entry in entries]
    return {
        "reasoning_depth": {
            "mean": float(statistics.mean(depths)),
            "distribution": {str(depth): count for depth, count in sorted(Counter(depths).items())},
        },
        "complexity_metrics": {
            "avg_evidence_count": float(statistics.mean(evidence_counts)),
            "avg_resource_types": float(statistics.mean(resource_counts)),
        },
    }
```

#### 4.2 输出格式

每个方法的汇总结果：

```json
{
  "method": "medevolve",
  "evaluation": {
    "actual_questions": 10,
    "validity_among_verified": {"numerator": 9, "denominator": 10, "rate": 0.9},
    "completion_among_reviewed": {"numerator": 8, "denominator": 10, "rate": 0.8},
    "quality_summary": {
      "factual_accuracy": {
        "mean": 4.2,
        "median": 4.0,
        "distribution": {"3": 1, "4": 5, "5": 4},
        "high_quality_rate": 0.9
      },
      "clinical_soundness": {...},
      "evidence_sufficiency": {...},
      "task_clarity": {...}
    },
    "difficulty_summary": {
      "reasoning_depth": {
        "mean": 2.8,
        "distribution": {"1": 1, "2": 3, "3": 5, "4": 1}
      },
      "complexity_metrics": {
        "avg_evidence_count": 4.5,
        "avg_resource_types": 2.3
      }
    }
  },
  "costs": {...}
}
```

## 成本影响

### 评测成本增加

- 原方案每个任务：1次quality + 1次solver + 1次completion = 3次模型调用
- 新方案每个任务：3次quality + 3次difficulty + 1次solver + 1次completion = 8次模型调用
- 评测阶段成本增加约**2.5倍**

### 优势

- 质量和难度评分更稳定（三裁判投票）
- 数值化评分支持统计对比
- 难度评估客观可验证

## 实施步骤

1. 修改`contracts.py`：新增QualityDimension、DifficultyScore、ComplexityMetrics
2. 修改`evaluation/prompts.py`：改造QUALITY prompt，新增DIFFICULTY prompt
3. 修改`evaluation/evaluator.py`：
   - 改造quality评测为三裁判
   - 新增difficulty评测（三裁判）
   - 新增聚合函数和校验逻辑
4. 修改`experiments/summary.py`：新增summarize_quality和summarize_difficulty函数
5. 更新配置文件：在evaluation.runtime.models和budgets中添加difficulty角色
6. 本地测试：用少量任务验证评分逻辑和汇总函数
7. 全量重跑：对6个方法的评测患者重新评测

## 配置示例

```json
{
  "evaluation": {
    "runtime": {
      "models": {
        "verifier": {...},
        "quality": {...},
        "difficulty": {...},  // 新增
        "solver": {...},
        "completion": {...}
      },
      "budgets": {
        "verifier": {...},
        "quality": {"max_model_turns": 10, "max_tool_calls": 0},
        "difficulty": {"max_model_turns": 10, "max_tool_calls": 0},  // 新增
        "solver": {...},
        "completion": {...}
      }
    }
  }
}
```

## 验收标准

1. 质量评分：每个维度返回1-5整数，reason非空
2. 难度评分：reasoning_depth返回1-4整数，complexity_metrics自动计算正确
3. 三裁判机制：每个评分保留3个原始分数，最终分数为中位数
4. 汇总逻辑：正确计算均值、中位数、分布和高质量占比
5. 6个方法可对比：相同评测患者上的质量、难度、成本可直接对比
