# 实验正确性与评价成本 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. 执行方式由用户选择。

**Goal:** 修复实验结果与统计的正确性，使 EnvExplore 复用冻结指导，并通过三裁判联合评分降低公共评价成本。

**Architecture:** 沿用现有 TaskBatch、RunResult、EvaluationResult 与异步编排接口。TaskCraft 的逐来源证据在方法边界转换；实验完成状态在编排层汇总；联合评分和精简材料在评价层处理，真实请求只记账一次。

**Tech Stack:** Python 3.12、Pydantic v2、pytest、pytest-asyncio、httpx、现有模型 SDK；不增加依赖。

## Global Constraints

- 设计依据：`docs/experiment-corrections/design.md`。
- 实验对象为 PhysicianBench 和 MedAgentBench；FHIR-AgentBench 不纳入本次配置调整与真实实验验收。
- 公共 Verifier、各方法题型选择机制和主要算法保持现有行为。
- 任务级状态和 failure_category 沿用现有契约；不通过 trace 中的局部失败直接判定方法整体失败。
- 三个独立评分请求分别返回质量和难度，最终各维度取中位数；同一模型允许独立调用三次。
- 小样本为两个环境各三道已有题，共六道；复用完整轨迹和核验结果，只重做评分，不重跑生成和 Solver。
- 真实模型只使用实验已有模型配置，不自动改用高价模型；密钥来自现有环境，不进入文件。
- 使用 using-git-worktrees 从实施开始时当前分支创建独立工作树，保留主工作区论文改动。代码、测试与文档提交按关注点分层，description 与 body 使用中文。

## 工作树与验证

先检查当前任务附件，复用适合本次工作的工作树；没有时创建。所有代码命令在实施工作树根目录执行，使用主项目已有虚拟环境，并明确工作树导入路径：

```bash
PYTHONPATH="$PWD" /Users/guagua/code/MedEvolve/.venv/bin/python -m pytest tests/method/test_taskcraft.py -q
```

每项先运行新增边界测试确认失败，再修改实现并运行该项现有测试。五项完成后运行一次完整默认套件，默认套件不能调用真实模型、FHIR 或 Docker：

```bash
PYTHONPATH="$PWD" /Users/guagua/code/MedEvolve/.venv/bin/python -m pytest -q
```

## Task 1: TaskCraft 逐来源证据

**Files:**

- Modify: `methods/taskcraft/schemas.py`、`methods/taskcraft/prompts.py`。
- Modify: `methods/taskcraft/stages.py`。
- Test: `tests/method/test_taskcraft.py`。

**Interfaces:** `Conclusion` 与 `BackwardContext` 增加 `evidence: list[Evidence]`；公共 Evidence 不变。`Pipeline.evidence(records: list[Evidence]) -> tuple[list[Evidence], list[str]]` 保留来源与事实配对，并返回去重来源供 locator 使用。

- [ ] **Step 1: 添加错配与错误引用保留测试。** 在现有 patched fixture 中，让结论阶段返回两条不同来源事实，其中一条使用不存在的来源；驱动现有原子阶段入口后断言公共候选完整保留配对，不复制结论、不过滤错误来源。测试数据：

```python
conclusion_output = {
    "conclusions": [{
        "conclusion": "检验异常并有相关诊断", "R": "异常与诊断关联",
        "evidence": [
            {"source_id": BLOCK_A, "statement": "血钾升高"},
            {"source_id": "Condition/missing", "statement": "模型声称已有诊断"},
        ],
    }],
}
expected = [
    (BLOCK_A, "血钾升高"),
    ("Condition/missing", "模型声称已有诊断"),
]
assert [(e.source_id, e.statement) for e in candidate.question.evidence] == expected
```

对深度关系使用同样的两来源不同事实输入；合并题保留同来源不同事实，只消除完全重复的配对。已有纯 source_ids 测试响应同步为逐来源证据响应。

- [ ] **Step 2: 运行失败测试。**

```bash
PYTHONPATH="$PWD" /Users/guagua/code/MedEvolve/.venv/bin/python -m pytest tests/method/test_taskcraft.py -q -k 'evidence or atomic or depth'
```

- [ ] **Step 3: 更新适配结构及转换。** 两种模型输出结构引入 Evidence；提示词要求每项 statement 只表述该来源支持的事实。sources 标识从 evidence 派生，不作为另一份可独立漂移的模型输出。转换核心：

```python
def evidence(self, records: list[Evidence]) -> tuple[list[Evidence], list[str]]:
    accepted, seen = [], set()
    for record in records:
        key = (record.source_id, record.statement)
        if key not in seen:
            seen.add(key)
            accepted.append(record)
    return accepted, list(dict.fromkeys(record.source_id for record in accepted))
```

原子结论经 initial-question detail 传递 evidence；深度 hop 保存 evidence；`_relation_evidence` 合并每个 hop 的配对。结论与关系本身仍用于 TaskCraft 的阶段推理，不能替换为原始记录正文。空证据由公共核验判定；不得因为错误来源或缺失证据在适配层提前删除题目。

- [ ] **Step 4: 运行 TaskCraft 与公共核验边界。**

```bash
PYTHONPATH="$PWD" /Users/guagua/code/MedEvolve/.venv/bin/python -m pytest tests/method/test_taskcraft.py tests/evaluation/test_evaluator.py -q
```

- [ ] **Step 5: 单独提交。** `fix(taskcraft): 保留逐来源证据配对`。

## Task 2: 批次状态与统计

**Files:**

- Modify: `experiments/runner.py`、`experiments/run.py`、`experiments/summary.py`。
- Modify: `methods/common.py`、`scripts/run_full_comparison.py`。
- Test: `tests/experiment/test_closed_loop.py`、`tests/method/test_generation_shared.py`、`tests/method/test_taskcraft.py`。

**Interfaces:** manifest 增加 `generation` 和 `evaluation` 对象，各包含 `status`、`failed`、`incomplete`；总 status 沿用 running/completed/failed。summary 分开返回 `generation_yield`、`generation_failure_rate`、`evaluation_failure_rate`。attempts 的 produced/requested 和任务评价结果是计数来源。

- [ ] **Step 1: 添加端到端状态测试。** 复用 `config_for` 和 `install_dependencies`，将生成输出分别设为运行失败、合法一题、合法空批次、预算耗尽且有合法输出、预算耗尽且无输出；评价分别返回 invalid、completed/clinical、failed，并覆盖返回 failed 而非抛异常的情况。核心断言：

```python
manifest = json.loads((run_dir / "manifest.json").read_text())
assert manifest["status"] == "failed"
assert manifest["generation"]["failed"] == 1
assert manifest["evaluation"]["failed"] == 1

summary = summarize_run(str(run_dir))
assert summary["evaluation"]["generation_yield"]["denominator"] == 2
assert summary["evaluation"]["evaluation_failure_rate"]["numerator"] == 1
```

合法少产出及 invalid/clinical 的独立运行必须得到 completed；用例不能依赖固定提示词或参数转发。

Token 测试将一个已知用量模型请求、一个未知用量模型请求与纯工具记录混合，断言已知量累计一次，未知模型请求计数一，工具请求不增加缺失数；再驱动 summary 验证嵌套账本没有重复累计。

- [ ] **Step 2: 运行失败测试。**

```bash
PYTHONPATH="$PWD" /Users/guagua/code/MedEvolve/.venv/bin/python -m pytest tests/experiment/test_closed_loop.py tests/method/test_generation_shared.py tests/method/test_taskcraft.py -q -k 'status or failure or summary or token or exhaust'
```

- [ ] **Step 3: 实现阶段计数与最终状态。** 所有轮次和测试批次都累计阶段结果。生成 failed 和非法输出计 failed，预算耗尽无有效输出计 incomplete；存在合法输出则依据方法最终状态处理。EvaluationResult.failed 和任务评价缺失分别计 failed/incomplete。探索、策略更新及环境不可恢复故障计总失败；环境释放失败同样不能留下 completed。

```python
manifest["generation"] = generation_counts
manifest["evaluation"] = evaluation_counts
stage_failed = any(
    stage["failed"] or stage["incomplete"]
    for stage in (generation_counts, evaluation_counts)
)
manifest["status"] = "failed" if stage_failed or experiment_failed else "completed"
```

失败先保存 manifest 和可用结果，再使 CLI 返回非零。比较脚本只跳过最终完成的运行；旧 manifest 没有阶段信息时，依据已有 attempts/evaluations 检查，不把历史错误 completed 当作可靠完成。遇到非空输出目录仍沿用现有保护，不增加自动删除或恢复执行。

规范化用量保留 `runs_without_known_tokens`；工具账本已有显式零缺失标记，汇总必须消费它。只排除明确的工具记录，不能仅因 model_turns 默认零就把未知模型请求排除。生成失败率不使用 error 文本或 short_batches 作为分子。

- [ ] **Step 4: 运行相关回归。**

```bash
PYTHONPATH="$PWD" /Users/guagua/code/MedEvolve/.venv/bin/python -m pytest tests/experiment/test_closed_loop.py tests/method/test_generation_shared.py tests/method/test_taskcraft.py experiments/test_envexplore_summary.py -q
```

- [ ] **Step 5: 分层提交。** 状态修复为 `fix(experiments): 汇总生成与评估完成状态`；统计修复为 `fix(summary): 分离产出率失败率与模型用量`。每次只暂存该关注点的改动。

## Task 3: EnvExplore 冻结指导

**Files:**

- Modify: `contracts.py`、`experiments/runner.py`、`experiments/summary.py`。
- Modify: `scripts/emit_comparison_configs.py`、`scripts/run_full_comparison.py`。
- Test: `tests/experiment/test_emit_comparison_configs.py`。
- Test: `experiments/test_envexplore.py`、`experiments/test_envexplore_configs.py`、`experiments/test_envexplore_summary.py`。

**Interfaces:** ExperimentConfig 增加 `exploration_only: bool = False`，仅 EnvExplore 可设为 true。沿用 guidance_path，按 method 解析：medevolve 使用现有 ExplorationGuidance；envexplore 使用以下冻结文件结构：

```python
class FrozenEnvironmentGuidance(BaseModel):
    environment: str
    exploration_patients: list[str]
    guidance: EnvironmentExploration
    exploration_run: RunResult
    runtime: RuntimeConfig
```

指导文件为探索运行的 `environment_guidance.json`，其中 exploration_run 必须 completed、guidance 非空。冻结文件中的配置只含现有环境变量名，不含密钥。使用方能追溯成本和探索名单。

- [ ] **Step 1: 添加跨批次复用测试。** 在现有 config/wired fixture 中，先以 exploration_only 执行独立探索；再令两个测试批次读取该文件。断言探索仅一次，两个批次正常生成，探索与所有测试患者互斥，测试运行 runs 不含探索请求。以下结果为必需断言：

```python
exploration.assert_awaited_once()
assert all(row["stage"] != "explore_environment" for row in batch_runs)
assert frozen["environment"] == config.environment.name
assert frozen["exploration_run"]["status"] == "completed"
assert not set(frozen["exploration_patients"]) & all_test_patients
```

补充失败指导、环境不匹配、患者交叉和缺失文件测试；这些情况不得触发测试生成。配置脚本测试验证只生成一个探索配置，所有测试配置引用同一指导路径。

- [ ] **Step 2: 运行失败测试。**

```bash
PYTHONPATH="$PWD" /Users/guagua/code/MedEvolve/.venv/bin/python -m pytest experiments/test_envexplore.py experiments/test_envexplore_configs.py tests/experiment/test_emit_comparison_configs.py -q
```

- [ ] **Step 3: 实现探索和消费分支。** 新配置先由 contracts 校验；有冻结指导时不要求测试配置具备探索器预算，也不调用探索器；exploration_only 时不准备测试轨迹、不生成或评价任务。未提供指导的单次独立实验仍能执行探索再生成，保存同样的冻结结构。

脚本为两个目标环境创建探索配置，成功后生成引用冻结指导的测试配置；沿用既有 evolution_patients 作为独立探索名单，不复用 medevolve 的探索结果。探索指导属于各自方法运行。总成本汇总读取一次探索 RunResult，测试批次引用它而不复制到 runs。

```python
if config.method == "envexplore" and config.guidance_path:
    frozen = FrozenEnvironmentGuidance.model_validate_json(
        Path(config.guidance_path).read_text(encoding="utf-8")
    )
    environment_guidance = frozen.guidance
```

真实执行前验证 frozen.environment、指导正文、探索完成状态和全测试集互斥。FHIR-AgentBench 配置生成行为保持现有分支。

- [ ] **Step 4: 运行 EnvExplore 回归。**

```bash
PYTHONPATH="$PWD" /Users/guagua/code/MedEvolve/.venv/bin/python -m pytest tests/method/test_envexplore.py tests/method/test_envexplore_environment.py experiments/test_envexplore.py experiments/test_envexplore_configs.py experiments/test_envexplore_summary.py tests/experiment/test_emit_comparison_configs.py -q
```

- [ ] **Step 5: 单独提交。** `feat(envexplore): 复用独立探索的冻结指导`。

## Task 4: 三裁判联合评分

**Files:**

- Modify: `contracts.py`、`evaluation/prompts.py`、`evaluation/evaluator.py`。
- Modify: `scripts/emit_comparison_configs.py` 中目标环境的评价配置。
- Test: `tests/evaluation/test_evaluator.py`、`tests/experiment/test_emit_comparison_configs.py`。

**Interfaces:** 新增联合响应模型；EvaluationResult 最终结构保持现状。增加 `review_scores(task, trajectory, verification, config) -> tuple[QualityReview, DifficultyScore, list[RunResult]]`，同时供 evaluator 与 Task 5 评分对照消费，调用计账由上层追加真实 runs。

```python
class JointJudgment(BaseModel):
    quality: QualityJudgment
    difficulty: DifficultyJudgment
```

EvaluationConfig 增加 `scoring_mode: Literal["separate", "joint"] = "separate"`；两个目标环境实验配置使用 joint。历史与 FHIR-AgentBench 配置维持 separate，确保对照基线可运行。joint 模式使用 config.judges 或 quality 角色模型独立调用三次；显式裁判名单在 joint 模式必须恰好三项，历史 separate 模式保留现有兼容行为。

- [ ] **Step 1: 添加调用与聚合边界测试。** 修改现有 harness 识别 JointJudgment，分别提供质量分数 `[2, 5, 4]` 和深度 `[4, 1, 3]`；通过完整 evaluate_task 验证联合请求三次、分数各取中位数、三份真实用量仅累计一次：

```python
assert result.quality.factual_accuracy.score == 4
assert result.quality.factual_accuracy.scores == [2, 5, 4]
assert result.difficulty.reasoning_depth == 3
assert result.difficulty.reasoning_depths == [4, 1, 3]
joint_runs = [r for r in result.runs if r.output and "quality" in r.output]
assert len(joint_runs) == 3
```

补充一裁判失败、缺字段、分数越界和空理由用例；必须返回 failed 且保留请求 trace。历史 separate 用例仍验证六次评分请求，配置测试验证目标环境启用 joint。

- [ ] **Step 2: 运行失败测试。**

```bash
PYTHONPATH="$PWD" /Users/guagua/code/MedEvolve/.venv/bin/python -m pytest tests/evaluation/test_evaluator.py tests/experiment/test_emit_comparison_configs.py -q
```

- [ ] **Step 3: 合并评分提示与请求。** 联合提示完整保留 QUALITY 四维 1–5 及 DIFFICULTY 深度 1–4 的评分标准。输入使用完整 review_material，附加 verification 与代码计算的 complexity_metrics。三次并发请求各含一个 judge_index，公共 runs 只存联合响应。

```python
judgments = [JointJudgment.model_validate(run.output) for run in joint_runs]
quality_scores = [judge.quality for judge in judgments]
difficulty_scores = [judge.difficulty for judge in judgments]
```

聚合函数改为消费已验证 judgment 列表，分别复用现有中位数与理由汇总；不能构造额外带相同 usage 的伪 RunResult。review_scores 的 separate 分支沿用现有六次请求。Solver 的隔离输入、Verifier 和 Completion 不变。

- [ ] **Step 4: 运行评价与闭环回归。**

```bash
PYTHONPATH="$PWD" /Users/guagua/code/MedEvolve/.venv/bin/python -m pytest tests/evaluation tests/experiment/test_closed_loop.py tests/experiment/test_emit_comparison_configs.py -q
```

- [ ] **Step 5: 单独提交。** `feat(evaluation): 合并三裁判质量与难度评分`。

## Task 5: 评分输入与快速对照

**Files:**

- Create: `evaluation/material.py`、`scripts/compare_review_inputs.py`。
- Modify: `contracts.py`、`evaluation/evaluator.py`。
- Create: `tests/evaluation/test_review_material.py`、`tests/experiment/test_compare_review_inputs.py`。

**Interfaces:** EvaluationConfig 增加 `review_input_mode: Literal["full", "compact"] = "full"`。`evaluation.material.review_material(task, trajectory, verification, complexity, *, mode) -> dict` 只供质量/难度评分；Verifier、Solver 和 Completion 不消费精简材料。

对照 CLI 为 `scripts/compare_review_inputs.py --run-dir <path>`，可重复传入已有 run 目录，另指定 `--output-dir <path>`。从目标环境各选择三道有合法完整核验结果的既有任务，按 task.id 排序固定选择；读取任务所属轨迹、config 与 evaluation。来源或数量不足时明确报错，不补跑任务。

- [ ] **Step 1: 添加矛盾上下文与降级测试。** 构造引用“开始用药”的记录及未引用“因过敏停药”的记录，精简材料必须包含两者；补充阴性检查、不同时间检验值、否定诊断、自由文本和未知资源字段用例。不得仅根据题目关键词过滤记录。

```python
material = review_material(task, trajectory, verification, complexity, mode="compact")
assert material["cited_records"][0]["source_id"] == "MedicationRequest/start"
serialized = json.dumps(material, ensure_ascii=False)
assert "因过敏停药" in serialized
assert "未见感染" in serialized
assert "2024-03-05" in serialized and "2024-03-06" in serialized
```

对照脚本通过 AsyncMock 替代 review_scores，验证三种模式、同题同模型、原始结果持久化、一次请求一次统计，以及不会调用 generate_tasks、evaluate_task 或 Solver。

- [ ] **Step 2: 运行失败测试。**

```bash
PYTHONPATH="$PWD" /Users/guagua/code/MedEvolve/.venv/bin/python -m pytest tests/evaluation/test_review_material.py tests/experiment/test_compare_review_inputs.py -q
```

- [ ] **Step 3: 实现确定性材料精简与对照入口。** 引用记录读取 trajectory.sources 中的原始 JSON；其他记录全部进入上下文，按时间保留。结构精简优先消除重复来源标识、患者引用和格式开销；保留状态、日期、概念、数值与单位、临床引用、剂量、原因、关联就诊、叙述文本和 extension。不删除未知临床字段，不增加摘要 LLM，不截断自由文本；无法可靠解析的记录原样保留。

```python
return {
    "task": task.model_dump(include={
        "instruction", "task_type", "reference_answer", "evidence",
        "patient_id", "environment", "decision_point",
    }),
    "cited_records": cited_records,
    "clinical_context": clinical_context,
    "verification": verification.model_dump(mode="json"),
    "complexity_metrics": complexity.model_dump(mode="json"),
}
```

通过同一 review_scores 分别执行 separate/full、joint/full、joint/compact。评分请求使用已有模型与裁判配置；历史基线只有提示标准、模型、输入和裁判数兼容时才能复用。保存 `cases.json`、`runs.json`、`comparison.json`：包括任务及源文件路径、三个模式原始评分、逐维分差、运行失败、真实 Token 和未知用量。所有请求数以实际 runs 为准，不用估算替代。

- [ ] **Step 4: 离线回归及快速真实对照。** 先执行完整默认套件；通过后使用已有环境中的模型密钥和两个环境的既有 run 目录运行六题评分对照。实现期间查验既有产物再确定具体 run 路径，不写入虚构路径。

```bash
PYTHONPATH="$PWD" /Users/guagua/code/MedEvolve/.venv/bin/python -m pytest -q
```

实际对照最多 72 次评分请求；有可靠旧基线时减少至 36 次。报告每项评分差异、实际 Token 和失败情况。对中位数变化至少一分的项目检查理由与原始记录；遗漏矛盾证据、解释不清的偏移或没有 Token 节省时，正式实验维持 full。样本不足以证明统计等价，不扩大真实实验规模。

- [ ] **Step 5: 单独提交并检查交付。** `feat(evaluation): 增加评分材料精简与小样本对照`。成果保留纯净的代码、测试、现行文档与运行产物；不提交密钥或历史 outputs。检查独立工作树的最终 diff，合回当前分支时保留既有论文文件改动。
