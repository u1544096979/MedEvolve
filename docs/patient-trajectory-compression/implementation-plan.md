# 患者轨迹输入预算 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 六种任务生成方法按模型输入预算选取完整原始记录，并在多阶段请求与工具历史增长时保持请求可容纳。

**Architecture:** 公共记录选取位于 `environments/fitting.py`，公共计数与阶段材料适配位于 `runtime/context.py`。单次请求、SDK 和原生循环统一执行预算检查；方法层只处理 TaskCraft 分块、搜索记忆及部分轨迹标记。实验入口保留完整缓存用于核验，患者级材料错误单独记录。

**Tech Stack:** Python ≥ 3.10、Pydantic v2、openai==2.54.0、openai-agents==0.20.0、tiktoken、pytest、httpx.MockTransport。

## Global Constraints

- 六方法：medevolve、taskcraft、workflow、singleshot、selfgen、envexplore。
- 上下文上限减去输出额度、提示词、输出 schema、工具声明与消息余量；显式 `context_window` 和 `max_output_tokens` 优先，兼容旧配置使用 128,000 / 16,384。
- GPT-4o-mini 使用匹配分词器；不支持的模型使用 UTF-8 字节数作为保守估计。
- 保留完整原始记录，正文与来源同步；候选或焦点引用的来源保持可见；选取规则不依赖方法和出题方向。
- 原始患者缓存保持完整，证据核验读取完整轨迹与原始来源；无法容纳的必需材料在模型调用前明确失败，单个患者错误不阻塞其他患者。
- SDK 与原生历史按完整交互缩减，工具调用与结果配对；TaskCraft 保留任务、当前规划和近期完整步骤。
- 输出额度与本地预留一致；没有发起的请求不计模型轮次；默认测试不调用付费模型、FHIR 服务或 Docker。
- 不修改 main 上既有未提交改动，不修改 `scripts/run_full_comparison.py`，不加入用户未确认的摘要、检索、额外生成或重构。
- commit 的 type/scope 保留英文，description 和 body 使用中文；每个提交只涉及一个关注点，依赖先于消费者。
- 过程记录、子代理报告、review package 仅保存在该计划的 git-ignored `.superpowers/sdd/` 工作目录。

---

### Task 1: 公共预算与六方法实现

**Files:**
- Create: `runtime/context.py`, `tests/runtime/test_context.py`
- Modify: `contracts.py`, `environments/fitting.py`, `runtime/execution.py`, `runtime/native.py`, `pyproject.toml`
- Modify: `methods/common.py`, `methods/medevolve/__init__.py`, `methods/medevolve/compression.py`, `methods/taskcraft/agent.py`, `methods/taskcraft/stages.py`, `methods/workflow/__init__.py`, `experiments/runner.py`, `scripts/emit_comparison_configs.py`
- Test: `tests/method/test_compression.py`, `tests/runtime/test_native.py`, `tests/environment/test_trajectory_fitting.py`

**Interfaces:**
- Consumes: `PatientTrajectory`, `ModelConfig`, `GenerationBudget`、现有方法入口及 SDK `RunConfig.call_model_input_filter`。
- Produces: `PatientTrajectory.input_truncated: bool = False`；`count_tokens(text: str, model: ModelConfig) -> int`；`input_limit(model: ModelConfig) -> int`；`output_limit(model: ModelConfig) -> int`。
- Produces: `fit_patient(trajectory: PatientTrajectory, model: ModelConfig, reserved_tokens: int = 8192) -> PatientTrajectory`；`prepare_input(input_text: str, model: ModelConfig, *, instructions: str, schema: dict, tools: list | None = None) -> tuple[str, dict]`；`fit_history(items: list[dict], model: ModelConfig, *, fixed_tokens: int, prefix_count: int, group_start) -> tuple[list[dict], int]`。
- Produces: `fit_trajectory(trajectory, budget_chars, *, measure=payload_chars, protected_source_ids=None)`，保留旧调用兼容性。

- [ ] **Step 1: 记录新增边界测试的失败证据。** 计划工作目录中的 `seed/` 提供完整实现与测试起点；这些输入仅辅助实施，不进入提交。先放入 `tests/runtime/test_context.py`，运行测试，记录缺少公共预算模块时的失败结果。

```bash
cp .superpowers/sdd/implementation-plan/seed/tests/runtime/test_context.py tests/runtime/test_context.py
.venv/bin/python -m pytest tests/runtime/test_context.py -q
```

- [ ] **Step 2: 接入公共预算与记录选取。** 以工作目录中的完整代码起点为基础，完成列出的接口；审查并修正起点中影响 Global Constraints 的缺陷。关键契约如下，测试必须验证行为而非仅验证传参。

```python
from runtime.context import fit_patient, prepare_input

fitted = fit_patient(original, model)
assert original.model_dump() == original_snapshot
assert all(source in original.sources for source in fitted.sources)
prepared, entry = prepare_input(stage_payload, model, instructions=instructions, schema=schema)
assert entry["input_tokens_estimate"] <= entry["input_limit"]
```

按依赖接入：配置与契约 → 记录选取与计数 → 单次、SDK、原生请求 → 方法及实验入口。不得通过截断单条必需临床记录、删除候选题事实或切断工具调用配对来满足预算。

- [ ] **Step 3: 运行针对性验证和完整测试。** 单条超大记录、不可定位正文、已引用来源、JSON 转义、中英文、多患者反馈、工具循环及 TaskCraft 记忆缩减均须覆盖；针对新发现的实际缺陷补充失败测试后修复。

```bash
.venv/bin/python -m pytest tests/runtime/test_context.py tests/environment/test_trajectory_fitting.py tests/runtime/test_execution.py tests/runtime/test_native.py tests/method/test_compression.py tests/method/test_taskcraft.py -q
.venv/bin/python -m pytest -q
git diff --check
```

- [ ] **Step 4: 分层提交并报告。** 提交依赖、公共记录与预算、请求执行、方法接入、测试五类关注点；每次提交保持代码可导入，测试与其依赖一起落位。报告写入本计划工作目录的 `task-1-report.md`，包含 RED/GREEN 命令与实际结果。

### Task 2: 六方法集成与患者隔离验证

**Files:**
- Create: `tests/method/test_long_trajectory.py`, `tests/experiment/test_patient_input_budget.py`
- Modify: `docs/patient-trajectory-compression/README.md`, `docs/patient-trajectory-compression/input-budget.md`
- 必须修复的集成缺陷只涉及 Task 1 的现有文件，不新增未要求的机制。

**Interfaces:**
- Consumes: Task 1 的公共预算接口、六方法 `generate_tasks`、`runner.run_experiment`、`RunResult` 和运行产物。
- Produces: 六方法实际请求在配置上下文内的集成验证、单患者错误隔离与完整缓存验证，以及与最终实现一致的说明。

- [ ] **Step 1: 放入完整集成测试并运行。** 起点测试通过真实运行层，HTTP 和患者工具使用本地替身；计数使用模型实际读取的消息正文，JSON 传输转义不计为正文。核对测试断言是否能捕获真实超限及临床记录丢失，发现缺陷先记录失败证据。

```bash
cp .superpowers/sdd/implementation-plan/seed/tests/method/test_long_trajectory.py tests/method/test_long_trajectory.py
cp .superpowers/sdd/implementation-plan/seed/tests/experiment/test_patient_input_budget.py tests/experiment/test_patient_input_budget.py
.venv/bin/python -m pytest tests/method/test_long_trajectory.py tests/experiment/test_patient_input_budget.py -q
```

- [ ] **Step 2: 完成集成校验。** 每个方法都须在长于模型上下文的患者输入上完成合法任务输出；SelfGen、Workflow 和 TaskCraft 必须验证后续阶段。生成后的原始缓存与输入快照一致；必需单条记录超限只产生该患者的零模型调用失败，其他患者完成生成。新增测试须用可观察请求、产物与调用计数证明这些关系。

```python
assert result.status == "completed"
assert result.output["questions"]
assert original.model_dump() == original_snapshot
assert attempts[0]["status"] == "failed"
assert attempts[0]["usage"]["model_turns"] == 0
assert attempts[1]["status"] == "completed"
```

- [ ] **Step 3: 对齐说明并完成验证。** README 链接到 `input-budget.md`；正文只描述最终预算、记录选择、六方法接入与无法容纳的必需材料处理，不包含审查或调试过程。

```bash
.venv/bin/python -m pytest -q
git diff --check
```

- [ ] **Step 4: 原子提交测试与文档。** 测试提交使用 `test(generation): 覆盖六方法长轨迹与患者隔离`；文档提交使用 `docs(generation): 说明患者轨迹输入预算`。报告写入本计划工作目录的 `task-2-report.md`，包含命令、结果、覆盖边界及关注点。
