# EnvExplore Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现保留环境探索、固定初始策略的 MedEvolve 消融方法，在 PhysicianBench 与 MedAgentBench 上使用相同公共核验和解题流程。

**Architecture:** 公共入口每次实验复用主方法执行一次环境探索，将结果与 initial_strategy 绑定到四参数生成包装器。生成模型读取患者完整轨迹及原始来源，一次返回 TaskBatch，复用共享提示词规则、预算和公共输出校验。环境原生执行路径、证据核验、独立解题与完成评价沿用现有公共层。

**Tech Stack:** Python >=3.10、asyncio、Pydantic >=2.10,<3、OpenAI 2.54.0、OpenAI Agents SDK 0.20.0、httpx >=0.27,<1、pytest >=8,<10、pytest-asyncio >=0.24,<2；依赖版本沿用 pyproject.toml，不新增依赖。

## Global Constraints

EnvExplore 是 MedEvolve 的消融对比方法：保留患者轨迹整理、环境探索、单患者任务生成和公共评测，不执行数据探索、反馈分析与策略更新。
生成指导及题型难度配比固定使用实验的 initial_strategy。
每批生成最多一次模型调用、零次工具调用；预算不足时不调用模型、不产出有效任务。
当前两个环境的 decision_point 均为 null。
公共评测结果不改变后续策略。
不使用模块全局变量存放指导、策略或实例，不为每名患者重复环境探索。
MedAgentBench 的操作任务明确要求提交请求，完成评价按原生协议和实际执行记录判定，不以新增数据库记录作为提交成功的必要条件。
环境指导中不包含患者 ID、来源 ID、具体诊疗事实或试写内容。
共享生成规则与模型输入保持单一来源。

---

## 执行准备与文件边界

依据：[SPEC.md](SPEC.md)，验收编号 A1–A7。代码相对路径均以本次实施工作区为根；文档保留在 docs/comparison-experiments/baselines/envexplore/。

实施前使用 superpowers:using-git-worktrees，在当前项目状态上建立独立工作区；当前分支是否为主分支不影响该要求。检查当前未提交改动及已存在的 envexplore 工作区，保留所需公共层版本，不把其他人的未提交改动覆盖或当成本任务提交。文档编写阶段不执行以下代码、测试或真实实验。

进入实施工作区后，先执行 `python -m pytest -q tests/method tests/runtime tests/environment tests/evaluation experiments/test_common.py`，记录已有失败和依赖跳过。使用项目已有 .venv 的解释器；工作区内没有 .venv 时，从主仓库已有环境运行并确保当前工作目录为实施工作区。不安装新依赖，不将已有失败误报为本任务引入。

| 文件 | 责任 | 任务 |
| --- | --- | --- |
| methods/generation.py | 共享生成规则与轨迹序列化 | 1 |
| methods/medevolve/prompts.py、__init__.py | 使用共享规则，保持主方法行为 | 1 |
| tests/method/test_generation_shared.py | 主方法单次生成与输入回归 | 1 |
| methods/envexplore/__init__.py | 固定上下文单次生成及引用来源检查 | 2 |
| methods/envexplore/prompts.py | EnvExplore 输入说明与共享规则 | 2 |
| tests/method/test_envexplore.py | 生成、预算、结构与来源边界 | 2 |
| experiments/runner.py | 一次探索、上下文绑定、失败终止 | 3 |
| experiments/test_envexplore.py、test_common.py | 实验边界及方法解析 | 3 |
| experiments/summary.py、test_envexplore_summary.py | 一次性探索成本及失败成本 | 4 |
| tests/method/test_envexplore_environment.py | 主方法探索复用、双环境及释放 | 4 |
| configs/physicianbench-envexplore.json、medagentbench-envexplore.json | 同环境主方法配置的消融版本 | 5 |
| experiments/test_envexplore_configs.py | 配置一致性 | 5 |

依赖顺序：任务 1 → 任务 2 → 任务 3 → 任务 4 → 任务 5。每个任务完成后独立检查及提交。每个勾选步骤只执行一个动作，通常 2–5 分钟；写入较长代码块时按函数拆成多个该时长的操作，真实环境调用以实际时间为准。

## Task 1：共享生成规则并保持主方法行为

**Files:** Create `methods/generation.py`、`tests/method/test_generation_shared.py`；Modify `methods/medevolve/prompts.py` 的 GENERATION 定义和 `methods/medevolve/__init__.py` 的 `_trajectory_input`。

**Interfaces:**

Consumes：`PatientTrajectory`、现有 `methods.medevolve.prompts.GENERATION`，主方法 `generate_tasks(trajectory, guidance, strategy, directions, runtime) -> RunResult`。

Produces：`methods.generation.GENERATION_RULES: str`、`MAIN_GENERATION: str`、`ENVEXPLORE_GENERATION: str`、`trajectory_payload(trajectory: PatientTrajectory) -> dict`。主方法已有函数签名不变。

- [ ] **Step 1：添加行为回归测试**

在 `tests/method/test_generation_shared.py` 写入：

```python
import json
from unittest.mock import AsyncMock
import pytest
from contracts import (
    Budget, Evidence, ExplorationGuidance, GenerationDirection,
    ModelConfig, PatientTrajectory, RunResult, RuntimeConfig, Strategy,
)
from methods import medevolve
from methods.generation import trajectory_payload

@pytest.mark.asyncio
async def test_main_generation_keeps_material_and_one_call(monkeypatch):
    trajectory = PatientTrajectory(
        patient_id="p1", environment="PhysicianBench", text="完整轨迹正文",
        sources=[Evidence(source_id="Condition/c1", statement='{"id":"c1"}')],
    )
    directions = [GenerationDirection(task_type=t, difficulty="基础", weight=1)
                  for t in ("诊断与解释", "治疗规划")]
    guidance = ExplorationGuidance(data_guidance="数据指导", environment_guidance="环境指导")
    strategy = Strategy(generation_guidance="初始策略", allocation=directions)
    runtime = RuntimeConfig(
        models={"generator": ModelConfig(model="mock", api_key_env="UNUSED")},
        budgets={"generator": Budget(max_model_turns=1, max_tool_calls=0)},
    )
    call = AsyncMock(return_value=RunResult(
        output={"questions": []}, trace=[], usage={"model_turns": 1}, status="completed",
    ))
    monkeypatch.setattr(medevolve, "call_llm", call)
    result = await medevolve.generate_tasks(trajectory, guidance, strategy, directions, runtime)
    assert result.status == "completed"
    call.assert_awaited_once()
    body = json.loads(call.await_args.kwargs["input_text"])
    assert body["trajectory"] == trajectory.model_dump(mode="json")
    assert body["guidance"] == guidance.model_dump(mode="json")
    assert body["strategy"] == {"generation_guidance": "初始策略"}
    assert body["directions"] == [d.model_dump(mode="json") for d in directions]
    assert trajectory_payload(trajectory) == body["trajectory"]
```

- [ ] **Step 2：验证测试失败**

Run：`python -m pytest -q tests/method/test_generation_shared.py`。
Expected：新文件导入 `methods.generation` 失败；如果共享模块已存在，先核对接口并用行为断言识别缺失，不能覆盖已有实现。

- [ ] **Step 3：抽取规则，保持文本与主方法输入不变**

在实施工作区执行下述一次性抽取，完整复制当前生成规则，保留其他提示词和未涉及代码：

```python
import ast
from pathlib import Path

path = Path("methods/medevolve/prompts.py")
source = path.read_text()
node = next(n for n in ast.parse(source).body
            if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "GENERATION" for t in n.targets))
original = ast.literal_eval(node.value)
marker = "一次输出 TaskBatch"
header, marker, tail = original.partition(marker)
assert marker, "生成提示词缺少共享规则起点"
rules = marker + tail
new_header = (
    "你是医疗任务生成角色。\n"
    "输入包含一个患者的缓存轨迹、固定的环境探索指导、固定初始策略，\n"
    "以及按位置排列的两个生成方向。按照这两个位置的题型和难度分别尝试生成一道任务。\n"
)
shared = (
    '"""Shared task generation rules and patient material."""\n\n'
    'from contracts import PatientTrajectory\n\n'
    f'GENERATION_RULES = {rules!r}\n'
    f'MAIN_GENERATION = {header!r} + GENERATION_RULES\n'
    f'ENVEXPLORE_GENERATION = {new_header!r} + GENERATION_RULES\n\n'
    'def trajectory_payload(trajectory: PatientTrajectory) -> dict:\n'
    '    return trajectory.model_dump(mode="json")\n'
)
Path("methods/generation.py").write_text(shared)
lines = source.splitlines(keepends=True)
lines[node.lineno - 1:node.end_lineno] = [
    "from methods.generation import MAIN_GENERATION as GENERATION\n"
]
path.write_text("".join(lines))
from methods.generation import MAIN_GENERATION
assert MAIN_GENERATION == original
```

随后在 `methods/medevolve/__init__.py` 添加 `from methods.generation import trajectory_payload`，将 `_trajectory_input` 函数体替换为 `return trajectory_payload(trajectory)`。保留该函数名，兼容其现有内部调用。主方法的预算、指导、策略和调用次数不改。

- [ ] **Step 4：验证新测试及主方法回归**

Run：`python -m pytest -q tests/method/test_generation_shared.py tests/method/test_method.py tests/method/test_compression.py`。
Expected：新增测试 PASS；已记录基线之外无新增失败，主方法指导与患者输入保持一致。

- [ ] **Step 5：原子提交**

```bash
git add methods/generation.py methods/medevolve/prompts.py methods/medevolve/__init__.py tests/method/test_generation_shared.py
git commit -m "refactor(generation): 抽取共享出题规则与轨迹输入"
```

## Task 2：固定上下文单次生成与来源检查

**Files:** Create/Modify `methods/envexplore/__init__.py`、`methods/envexplore/prompts.py`、`tests/method/test_envexplore.py`。已有旧实现时按本任务替换入口与对应测试，不合入旧工具出题行为。

**Interfaces:**

Consumes：任务 1 的 `trajectory_payload`、`ENVEXPLORE_GENERATION`；公共 `GenerationBudget` 和 `validate_task_batch`；已完成的 `EnvironmentExploration` 及 `Strategy`。

Produces：SPEC 定义的六参数方法函数（四个位置参数加两个必填关键字参数），返回 `RunResult`。来源拒绝记录类型为 `evidence_references_rejected`。

- [ ] **Step 1：添加真实机制的失败测试**

将下列独立测试文件写入 `tests/method/test_envexplore.py`：

```python
import json
from unittest.mock import AsyncMock
import pytest
from contracts import (
    Budget, EnvironmentConfig, EnvironmentExploration, Evidence,
    GenerationDirection, ModelConfig, PatientTrajectory, RunResult,
    RuntimeConfig, Strategy,
)
from methods import envexplore

@pytest.fixture
def ctx():
    ds = [GenerationDirection(task_type=t, difficulty="基础", weight=1)
          for t in ("诊断与解释", "治疗规划")]
    return dict(
        trajectory=PatientTrajectory(patient_id="p1", environment="PhysicianBench",
            text="完整原始轨迹", sources=[Evidence(source_id="Condition/c1",
                statement='{"resourceType":"Condition","id":"c1"}')]),
        environment=EnvironmentConfig(name="PhysicianBench", manager_url="http://unused", seed_data="seed"),
        directions=ds,
        runtime=RuntimeConfig(models={"generator": ModelConfig(model="mock", api_key_env="UNUSED")},
            budgets={"generator": Budget(max_model_turns=1, max_tool_calls=0)}),
        environment_guidance=EnvironmentExploration(guidance="已验证可读取诊断"),
        strategy=Strategy(generation_guidance="固定初始策略", allocation=ds),
    )

def candidate(source="Condition/c1", evidence=True):
    return dict(instruction="解释现有诊断", task_type="诊断与解释",
        reference_answer="依据记录解释", decision_point=None,
        evidence=[dict(source_id=source, statement="诊断记录")] if evidence else [])

def completed(questions):
    return RunResult(output={"questions": questions}, trace=[{"raw": "response"}],
        usage={"model_turns": 1, "tool_calls": 0, "input_tokens": 7, "output_tokens": 3},
        status="completed")

@pytest.mark.asyncio
async def test_one_call_full_material_and_fixed_context(ctx, monkeypatch):
    call = AsyncMock(return_value=completed([candidate()]))
    agent = AsyncMock(side_effect=AssertionError("生成阶段调用工具 Agent"))
    monkeypatch.setattr("runtime.call_llm", call)
    monkeypatch.setattr("runtime.run_agent", agent)
    result = await envexplore.generate_tasks(**ctx)
    call.assert_awaited_once()
    agent.assert_not_awaited()
    body = json.loads(call.await_args.kwargs["input_text"])
    assert body["trajectory"] == ctx["trajectory"].model_dump(mode="json")
    assert body["guidance"] == {"environment_guidance": ctx["environment_guidance"].guidance}
    assert body["strategy"] == {"generation_guidance": ctx["strategy"].generation_guidance}
    assert result.status == "completed" and len(result.output["questions"]) == 1
    assert result.usage["model_turns"] == 1 and result.usage["tool_calls"] == 0
    assert result.usage["total_tokens"] == 10

@pytest.mark.asyncio
@pytest.mark.parametrize("bad", [candidate(evidence=False), candidate("Condition/missing")])
async def test_bad_sources_drop_only_bad_candidate(ctx, monkeypatch, bad):
    monkeypatch.setattr("runtime.call_llm", AsyncMock(return_value=completed([bad, candidate()])))
    ctx["directions"] = [ctx["directions"][0], ctx["directions"][0]]
    result = await envexplore.generate_tasks(**ctx)
    assert result.status == "completed" and result.output["questions"] == [candidate()]
    rejection = next(t for t in result.trace if t.get("type") == "evidence_references_rejected")
    assert rejection["questions"][0]["question"] == bad
    assert result.trace[0]["result"]["output"]["questions"][0] == bad

@pytest.mark.asyncio
async def test_zero_budget_and_failed_call_keep_status(ctx, monkeypatch):
    call = AsyncMock(return_value=RunResult(trace=[{"raw": "error"}],
        usage={"model_turns": 1, "input_tokens": 7, "output_tokens": 3},
        status="failed", error="provider offline"))
    monkeypatch.setattr("runtime.call_llm", call)
    ctx["runtime"].budgets["generator"].max_model_turns = 0
    stopped = await envexplore.generate_tasks(**ctx)
    assert stopped.status == "budget_exhausted" and stopped.output is None
    call.assert_not_awaited()
    ctx["runtime"].budgets["generator"].max_model_turns = 1
    failed = await envexplore.generate_tasks(**ctx)
    assert failed.status == "failed" and failed.output is None
    assert failed.usage["total_tokens"] == 10
    call.assert_awaited_once()

@pytest.mark.asyncio
@pytest.mark.parametrize("questions", [[], [candidate(evidence=False)]])
async def test_empty_or_all_rejected_has_no_tasks(ctx, monkeypatch, questions):
    monkeypatch.setattr("runtime.call_llm", AsyncMock(return_value=completed(questions)))
    result = await envexplore.generate_tasks(**ctx)
    assert result.status == "completed" and result.output == {"questions": []}

@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["time", "type", "structure"])
async def test_invalid_output_is_not_repaired(ctx, monkeypatch, change):
    q = candidate()
    if change == "time": q["decision_point"] = "2024-01-01"
    if change == "type": q["task_type"] = "药物处方"
    response = completed([q])
    if change == "structure": response.output = {"questions": "invalid"}
    call = AsyncMock(return_value=response)
    monkeypatch.setattr("runtime.call_llm", call)
    result = await envexplore.generate_tasks(**ctx)
    assert result.status == "failed" and result.output is None
    assert result.usage["total_tokens"] == 10
    call.assert_awaited_once()

@pytest.mark.asyncio
async def test_bad_input_stops_before_model(ctx, monkeypatch):
    call = AsyncMock()
    monkeypatch.setattr("runtime.call_llm", call)
    ctx["environment"] = ctx["environment"].model_copy(update={"name": "MedAgentBench"})
    with pytest.raises(ValueError, match="environment"):
        await envexplore.generate_tasks(**ctx)
    call.assert_not_awaited()
```

- [ ] **Step 2：运行并确认失败**

Run：`python -m pytest -q tests/method/test_envexplore.py`。
Expected：入口缺失或不接受新的关键字上下文，行为测试失败。

- [ ] **Step 3：写入最小完整方法实现**

`methods/envexplore/prompts.py`：

```python
from methods.generation import ENVEXPLORE_GENERATION as GENERATION
```

`methods/envexplore/__init__.py`：

```python
import json
from contracts import (
    EnvironmentConfig, EnvironmentExploration, GenerationDirection,
    PatientTrajectory, RunResult, RuntimeConfig, Strategy, TaskBatch,
)
from methods.common import GenerationBudget, output_constraints, validate_task_batch
from methods.generation import trajectory_payload
from .prompts import GENERATION

__all__ = ["generate_tasks"]

async def generate_tasks(
    trajectory: PatientTrajectory, environment: EnvironmentConfig,
    directions: list[GenerationDirection], runtime: RuntimeConfig, *,
    environment_guidance: EnvironmentExploration, strategy: Strategy,
) -> RunResult:
    output_constraints(directions)
    if trajectory.environment != environment.name:
        raise ValueError("Patient trajectory and environment do not match")
    if not environment_guidance.guidance.strip():
        raise ValueError("environment guidance must not be empty")
    budget = GenerationBudget(runtime)
    run = await budget.call_llm(
        "generate", instructions=GENERATION + "\n" + output_constraints(directions),
        input_text=json.dumps({
            "trajectory": trajectory_payload(trajectory),
            "guidance": {"environment_guidance": environment_guidance.guidance},
            "strategy": {"generation_guidance": strategy.generation_guidance},
            "directions": [d.model_dump(mode="json") for d in directions],
        }, ensure_ascii=False), output_type=TaskBatch,
    )
    checked = validate_task_batch(run, directions)
    if checked.status != "completed":
        return budget.finish(directions, status=checked.status, error=checked.error).model_copy(
            update={"trace": [{"stage": "generate", "result": checked.model_dump(mode="json")}]})
    known = {e.source_id for e in trajectory.sources}
    kept, rejected = [], []
    for question in TaskBatch.model_validate(checked.output).questions:
        problems = []
        if not question.evidence:
            problems.append("Task has no evidence references")
        missing = sorted({e.source_id for e in question.evidence} - known)
        if missing:
            problems.append("Evidence sources are absent from trajectory: " + ", ".join(missing))
        if problems:
            rejected.append({"question": question.model_dump(mode="json"), "problems": problems})
        else:
            kept.append(question)
    result = budget.finish(directions, output=TaskBatch(questions=kept))
    if rejected:
        result = result.model_copy(update={"trace": [*result.trace,
            {"type": "evidence_references_rejected", "questions": rejected}]})
    return result
```

调用只使用一次 `GenerationBudget.call_llm`。对已验证结构的候选逐题筛查来源，不新增事实判定调用；保留原始运行成本与被拒绝输出。

- [ ] **Step 4：验证测试通过及主方法兼容性**

Run：`python -m pytest -q tests/method/test_envexplore.py tests/method/test_generation_shared.py tests/method/test_method.py`。
Expected：新测试 PASS，既有基线之外无新增失败；空批次、引用拒绝、模型失败和预算耗尽分别保存。

- [ ] **Step 5：原子提交**

```bash
git add methods/envexplore/__init__.py methods/envexplore/prompts.py tests/method/test_envexplore.py
git commit -m "feat(envexplore): 添加固定策略单次生成与来源检查"
```

## Task 3：一次环境探索与固定上下文的公共接入

**Files:** Modify `experiments/runner.py` 的配置校验及探索分支、`experiments/test_common.py` 的未实现方法参数；Create `experiments/test_envexplore.py`。

**Interfaces:**

Consumes：任务 2 的方法入口；`runner.explore_environment(environment, patient_ids, runtime) -> RunResult`、公共 `invoke`/`parse`/`_save`、`config.initial_strategy`。

Produces：运行内绑定的 `generator(trajectory, environment, directions, runtime) -> RunResult`；一次 `explore_environment` 运行记录及 `environment_guidance.json`；固定 initial_strategy 的公共 evaluation 产物。

- [ ] **Step 1：写入完整公共接入测试**

在 `experiments/test_envexplore.py` 写入：

```python
import asyncio
import json
import random
from pathlib import Path
from unittest.mock import AsyncMock
import pytest
from contracts import (
    EnvironmentInstance, EvaluationResult, Evidence, ExperimentConfig,
    PatientTrajectory, RunResult,
)
from experiments import runner

@pytest.fixture
def config(tmp_path):
    values = json.loads((Path(__file__).parents[1] / "configs/experiment.json").read_text())
    values.update(method="envexplore", run_id="envexplore-test", evolution_patients=["train"],
        evaluation_patients=["p1", "p2"], round_patients=[], output_dir=str(tmp_path))
    values["runtime"]["models"] = {r: values["runtime"]["models"][r]
                                  for r in ("environment_explorer", "generator")}
    values["runtime"]["budgets"] = {r: values["runtime"]["budgets"][r]
                                   for r in ("environment_explorer", "generator")}
    return ExperimentConfig.model_validate(values)

@pytest.fixture
def wired(config, monkeypatch):
    prepared = []
    async def prepare(instance, patient):
        prepared.append(patient)
        return PatientTrajectory(patient_id=patient, environment=config.environment.name,
            text="患者原始记录", sources=[Evidence(source_id="Condition/c1", statement='{"id":"c1"}')])
    instance = EnvironmentInstance(instance_id="trajectory", endpoint="http://unused",
                                   environment=config.environment.name)
    monkeypatch.setattr(runner, "create_environment", AsyncMock(return_value=instance))
    release = AsyncMock()
    monkeypatch.setattr(runner, "release_environment", release)
    monkeypatch.setattr(runner, "prepare_trajectory", prepare)
    exploration = AsyncMock(return_value=RunResult(output={"guidance": "已验证读取能力"},
        trace=[{"type": "tool_output", "output": {"id": "trial-record"}}],
        usage={"model_turns": 2, "tool_calls": 1, "input_tokens": 8, "output_tokens": 4},
        status="completed"))
    monkeypatch.setattr(runner, "explore_environment", exploration)
    forbidden = []
    for name in ("explore_data", "analyze_feedback", "update_strategy"):
        mock = AsyncMock(side_effect=AssertionError(name + " 不应调用"))
        monkeypatch.setattr(runner, name, mock)
        forbidden.append(mock)
    async def generate(**kwargs):
        body = json.loads(kwargs["input_text"])
        return RunResult(output={"questions": [{"instruction": "解释原始诊断",
            "task_type": body["directions"][0]["task_type"], "reference_answer": "诊断解释",
            "evidence": [{"source_id": "Condition/c1", "statement": "记录事实"}],
            "decision_point": None}]}, trace=[],
            usage={"model_turns": 1, "tool_calls": 0, "input_tokens": 7, "output_tokens": 3},
            status="completed")
    call = AsyncMock(side_effect=generate)
    monkeypatch.setattr("runtime.call_llm", call)
    evaluate = AsyncMock(side_effect=lambda task, *args: EvaluationResult(
        task_id=task.id, observed_state={}, status="invalid", runs=[]))
    monkeypatch.setattr(runner, "evaluate_task", evaluate)
    return prepared, release, exploration, forbidden, call, evaluate

def test_once_exploration_fixed_strategy_and_public_sampling(config, wired):
    prepared, release, exploration, forbidden, call, evaluate = wired
    before = config.initial_strategy.model_dump(mode="json")
    asyncio.run(runner.run_experiment(config))
    exploration.assert_awaited_once_with(config.environment, ["train"], config.runtime)
    assert prepared == ["p1", "p2"] and call.await_count == evaluate.await_count == 2
    release.assert_awaited_once()
    for mock in forbidden: mock.assert_not_awaited()
    assert config.initial_strategy.model_dump(mode="json") == before
    rng = random.Random(config.random_seed)
    expected = [rng.choices(config.initial_strategy.allocation,
        weights=[d.weight for d in config.initial_strategy.allocation], k=2) for _ in range(2)]
    for index, entry in enumerate(call.await_args_list):
        body = json.loads(entry.kwargs["input_text"])
        assert body["guidance"] == {"environment_guidance": "已验证读取能力"}
        assert body["strategy"] == {"generation_guidance": before["generation_guidance"]}
        assert body["directions"] == [d.model_dump(mode="json") for d in expected[index]]
        assert "trial-record" not in entry.kwargs["input_text"]
    root = Path(config.output_dir)
    assert json.loads((root / "environment_guidance.json").read_text()) == {"guidance": "已验证读取能力"}
    assert not (root / "rounds").exists() and not (root / "data_exploration_patients.json").exists()
    runs = json.loads((root / "runs.json").read_text())
    assert sum(r["stage"] == "explore_environment" for r in runs) == 1
    assert all(e.args[0].source_method == "envexplore" for e in evaluate.await_args_list)

@pytest.mark.parametrize("status,output", [
    ("failed", None), ("budget_exhausted", None),
    ("completed", {"guidance": " "}), ("completed", {"invalid": "output"}),
])
def test_invalid_exploration_stops_generation(config, wired, status, output):
    _, _, exploration, _, call, evaluate = wired
    exploration.return_value = RunResult(output=output, trace=[],
        usage={"model_turns": 2, "input_tokens": 8, "output_tokens": 4}, status=status)
    with pytest.raises(RuntimeError):
        asyncio.run(runner.run_experiment(config))
    call.assert_not_awaited()
    evaluate.assert_not_awaited()
    root = Path(config.output_dir)
    assert json.loads((root / "manifest.json").read_text())["status"] == "failed"
    recorded = json.loads((root / "runs.json").read_text())[0]["result"]
    assert recorded["status"] == status and recorded["usage"]["total_tokens"] == 12

@pytest.mark.parametrize("fault", ["empty_exploration", "missing_role", "overlap", "comparison"])
def test_bad_config_rejected_before_environment(config, wired, fault):
    if fault == "empty_exploration": config.evolution_patients = []
    if fault == "missing_role": del config.runtime.models["environment_explorer"]
    if fault == "overlap": config.evolution_patients = ["p1"]
    if fault == "comparison": config.compare_initial_strategy = True
    with pytest.raises(ValueError):
        asyncio.run(runner.run_experiment(config))
    assert wired[0] == []
    wired[2].assert_not_awaited()
```

- [ ] **Step 2：确认接入测试失败**

Run：`python -m pytest -q experiments/test_envexplore.py`。
Expected：尚未绑定环境指导/固定策略或没有执行探索，测试失败；无真实远程调用。

- [ ] **Step 3：按两个插入位置添加公共接入代码**

在 `run_experiment` 中现有 generator 角色校验之后、创建输出目录之前插入：

```python
if config.method == "envexplore":
    if not config.evolution_patients:
        raise ValueError("EnvExplore requires nonempty exploration patients")
    if ("environment_explorer" not in config.runtime.models
            or "environment_explorer" not in config.runtime.budgets):
        raise ValueError("EnvExplore requires environment_explorer model and budget")
```

现有公共患者交叉检查和 `compare_initial_strategy` 拒绝保留。轨迹准备的 evolution 分支仍只属于 medevolve；EnvExplore 只准备 evaluation 轨迹。

在主方法探索分支之后、定义 `generate_and_evaluate` 之前插入：

```python
if config.method == "envexplore":
    environment_run = await invoke(
        "explore_environment", explore_environment,
        config.environment, config.evolution_patients, config.runtime,
    )
    environment_guidance = parse("explore_environment", environment_run, EnvironmentExploration)
    if environment_guidance is None or not environment_guidance.guidance.strip():
        raise RuntimeError("EnvExplore environment exploration did not produce valid guidance")
    _save(root / "environment_guidance.json", environment_guidance.model_dump(mode="json"))
    method_generate = generator
    fixed_strategy = config.initial_strategy.model_copy(deep=True)

    async def bound_generator(trajectory, environment, directions, runtime):
        return await method_generate(
            trajectory, environment, directions, runtime,
            environment_guidance=environment_guidance,
            strategy=fixed_strategy,
        )

    generator = bound_generator
```

保持现有 `config.method == "medevolve"` 的轮次、反馈、策略更新分支；公共 evaluation 调用、固定种子采样、Task 元数据和 solver 输入不改。`explore_environment` 阶段名保持现有统计识别方式。

在 `experiments/test_common.py` 的 `test_unimplemented_method_is_explicit` 参数列表移除 envexplore，并添加以下已实现入口检查。其他方法的已有基线问题单列处理，不以放宽断言掩盖。

```python
def test_envexplore_resolves_implemented_method():
    from methods.envexplore import generate_tasks
    assert runner.resolve_generator("envexplore") is generate_tasks
```

- [ ] **Step 4：验证接入及已有患者边界测试**

Run：`python -m pytest -q experiments/test_envexplore.py experiments/test_common.py`。
Expected：EnvExplore 新测试 PASS；两名患者只触发一次探索，两次生成，公共评价照常调用；其他测试相对基线无新增失败。

- [ ] **Step 5：原子提交**

```bash
git add experiments/runner.py experiments/test_envexplore.py experiments/test_common.py
git commit -m "feat(experiments): 接入一次环境探索与固定策略生成"
```

## Task 4：双环境探索、释放故障与完整成本

**Files:** Modify `methods/medevolve/__init__.py` 的共享 `explore_environment`、`experiments/summary.py`；Create `tests/method/test_envexplore_environment.py`、`experiments/test_envexplore_summary.py`。`runtime/`、`environments/` 的现有原生分派、证据回读和写入语义保持不变。

**Interfaces:**

Consumes：`create_environment(config) -> EnvironmentInstance`、`get_tools(instance, patient_ids)`、公共 `run_agent(..., output_type=EnvironmentExploration, model, budget) -> RunResult`、`release_environment(config, instance)`；`summarize_run(output_dir) -> JsonObject`。

Produces：共享探索入口原签名不变，释放失败时保留已发生的模型成本和原始运行记录；EnvExplore 的 `summary.costs.method` 为 exploration 与 generation 之和，不包含公共 evaluation。

- [ ] **Step 1：写入双环境及释放故障测试**

`tests/method/test_envexplore_environment.py`：

```python
from unittest.mock import AsyncMock
import pytest
from contracts import (
    Budget, EnvironmentConfig, EnvironmentInstance, ModelConfig, RunResult, RuntimeConfig,
)
from environments.medagentbench import NativeTools
from methods import medevolve

@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["PhysicianBench", "MedAgentBench"])
@pytest.mark.parametrize("failure", ["none", "budget", "release", "tools"])
async def test_shared_exploration_paths_release_and_cost(monkeypatch, name, failure):
    config = EnvironmentConfig(name=name, manager_url="http://unused", seed_data="seed")
    instance = EnvironmentInstance(instance_id="isolated", endpoint="http://unused", environment=name)
    runtime = RuntimeConfig(
        models={"environment_explorer": ModelConfig(model="mock", api_key_env="UNUSED")},
        budgets={"environment_explorer": Budget(max_model_turns=2, max_tool_calls=2)},
    )
    if name == "MedAgentBench":
        tools = NativeTools(instance.endpoint, ["train"], [])
    else:
        from agents import function_tool
        @function_tool
        async def lookup() -> str:
            """Read current patient."""
            return "record"
        tools = [lookup]
    get = AsyncMock(return_value=tools)
    if failure == "tools": get.side_effect = RuntimeError("tools unavailable")
    release = AsyncMock()
    if failure == "release": release.side_effect = RuntimeError("release unavailable")
    original = RunResult(output={"guidance": "读取能力已验证"} if failure != "budget" else None,
        trace=[{"type": "tool_output", "output": "record"}],
        usage={"model_turns": 2, "tool_calls": 1, "input_tokens": 8, "output_tokens": 4},
        status="budget_exhausted" if failure == "budget" else "completed")
    agent = AsyncMock(return_value=original)
    monkeypatch.setattr(medevolve, "create_environment", AsyncMock(return_value=instance))
    monkeypatch.setattr(medevolve, "get_tools", get)
    monkeypatch.setattr(medevolve, "release_environment", release)
    monkeypatch.setattr(medevolve, "run_agent", agent)
    result = await medevolve.explore_environment(config, ["train"], runtime)
    get.assert_awaited_once_with(instance, ["train"])
    release.assert_awaited_once_with(config, instance)
    if failure == "tools":
        agent.assert_not_awaited()
        assert result.status == "failed" and result.output is None
        return
    assert agent.await_args.kwargs["tools"] is tools
    assert agent.await_args.kwargs["model"] is runtime.models["environment_explorer"]
    assert result.usage == original.usage
    assert result.trace[0] == original.trace[0]
    if failure == "release":
        assert result.status == "failed" and result.output is None
        assert "release unavailable" in result.error
    else:
        assert result.status == original.status
```

在同一 `tests/method/test_envexplore_environment.py` 追加实例创建失败测试：

```python
@pytest.mark.asyncio
async def test_create_failure_has_no_instance_to_release(monkeypatch):
    config = EnvironmentConfig(name="PhysicianBench", manager_url="http://unused", seed_data="seed")
    runtime = RuntimeConfig(
        models={"environment_explorer": ModelConfig(model="mock", api_key_env="UNUSED")},
        budgets={"environment_explorer": Budget(max_model_turns=2, max_tool_calls=2)},
    )
    release, agent = AsyncMock(), AsyncMock()
    monkeypatch.setattr(medevolve, "create_environment",
                        AsyncMock(side_effect=RuntimeError("manager unavailable")))
    monkeypatch.setattr(medevolve, "release_environment", release)
    monkeypatch.setattr(medevolve, "run_agent", agent)
    result = await medevolve.explore_environment(config, ["train"], runtime)
    assert result.status == "failed" and result.output is None
    assert "manager unavailable" in result.error
    assert result.usage["model_turns"] == 0
    release.assert_not_awaited()
    agent.assert_not_awaited()
```

`experiments/test_envexplore_summary.py`：

```python
import json
from experiments.summary import summarize_run

def test_method_cost_includes_failed_exploration_and_generation_only(tmp_path):
    config = dict(run_id="cost", method="envexplore", evolution_patients=["train"],
        evaluation_patients=[], round_patients=[], runtime={"budgets": {}},
        evaluation={"runtime": {"budgets": {}}})
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "manifest.json").write_text(json.dumps({"status": "failed", "errors": []}))
    records = []
    for stage, turns, tools, tokens in [
        ("explore_environment", 2, 1, 12),
        ("evaluation/generate/0", 1, 0, 10),
        ("evaluation/evaluate/task1", 3, 1, 30),
    ]:
        records.append(dict(stage=stage, result=dict(status="failed", trace=[],
            usage=dict(model_turns=turns, tool_calls=tools,
                input_tokens=tokens, output_tokens=0, total_tokens=tokens))))
    (tmp_path / "runs.json").write_text(json.dumps(records))
    summary = summarize_run(str(tmp_path))
    costs = summary["costs"]
    assert costs["exploration"]["total_tokens"] == 12
    assert costs["generation"]["total_tokens"] == 10
    assert costs["evaluation"]["total_tokens"] == 30
    assert costs["method"]["total_tokens"] == 22
    assert costs["method"]["model_turns"] == 3
    assert costs["method"]["tool_calls"] == 1 and costs["method"]["runs"] == 2
```

- [ ] **Step 2：验证新的故障与成本测试失败**

Run：`python -m pytest -q tests/method/test_envexplore_environment.py experiments/test_envexplore_summary.py`。
Expected：现有探索释放异常会丢失 RunResult，方法成本汇总字段不存在；正常双环境路径可已通过，不强行制造失败。

- [ ] **Step 3：保留共享探索的实际运行结果并加入方法成本汇总**

替换 `methods/medevolve/__init__.py` 的共享 `explore_environment` 函数。其探索模型、预算、提示词、患者范围、工具及成功时原始输出不变：

```python
async def explore_environment(
    config: EnvironmentConfig, patient_ids: list[str], runtime: RuntimeConfig,
) -> RunResult:
    instance = None
    result = None
    release_error = None
    try:
        instance = await create_environment(config)
        tools = await get_tools(instance, patient_ids)
        result = await run_agent(
            instructions=ENVIRONMENT_EXPLORATION,
            input_text=json.dumps({"environment": instance.environment, "patient_ids": patient_ids},
                                  ensure_ascii=False),
            tools=tools, output_type=EnvironmentExploration,
            model=runtime.models["environment_explorer"],
            budget=runtime.budgets["environment_explorer"],
        )
    except Exception as exc:
        result = RunResult(trace=[{"type": "run_error", "failure_category": "environment_error",
                                  "error": str(exc)}], usage={"model_turns": 0, "tool_calls": 0},
                           status="failed", error=f"{type(exc).__name__}: {exc}")
    finally:
        if instance is not None:
            try:
                await release_environment(config, instance)
            except Exception as exc:
                release_error = f"{type(exc).__name__}: {exc}"
    if release_error is not None:
        result = result.model_copy(update={
            "output": None, "status": "failed", "error": "release_environment failed: " + release_error,
            "trace": [*result.trace, {"type": "environment_release", "released": False,
                "instance_id": instance.instance_id, "error": release_error,
                "rejected_output": result.output}],
        })
    return result
```

公共 `run_agent` 已将正常模型故障转为带成本的 RunResult；此处不自行构建 Agent 循环。创建失败时无实例可释放，`get_tools` 故障时释放已创建实例，运行完成或预算耗尽时均释放；释放失败不抹掉已返回的模型成本。

在 `experiments/summary.py` 的记录汇总循环之后、返回字典之前加入：

```python
if config["method"] == "envexplore":
    fields = (*USAGE_FIELDS, "runs", "runs_without_known_tokens")
    costs["method"] = {
        field: sum(costs.get(category, {}).get(field, 0)
                   for category in ("exploration", "generation"))
        for field in fields
    }
```

`explore_environment` 保持归属 exploration，未知 token 的既有计数保留。`costs.method` 为展示总量，不进入原始运行记录，不重复累计到 `model_runs` 的 token 总量。其他方法不新增此字段。

- [ ] **Step 4：运行双环境真实执行器模拟及公共核验回归**

Run：`python -m pytest -q tests/method/test_envexplore_environment.py experiments/test_envexplore_summary.py experiments/test_envexplore.py tests/runtime/test_execution.py tests/runtime/test_native.py tests/environment/test_medagentbench.py tests/environment/test_lifecycle.py tests/evaluation/test_evaluator.py`。
Expected：新增故障及成本测试 PASS；已有 SDK FunctionTool 与原生 GET/POST/FINISH 的模拟执行器测试通过，公共评价的证据核验及 solver 输入隔离不变。模拟方法测试只能证明传入正确工具对象，真实执行器路径由已有 runtime 测试共同验证。

- [ ] **Step 5：原子提交**

```bash
git add methods/medevolve/__init__.py tests/method/test_envexplore_environment.py experiments/summary.py experiments/test_envexplore_summary.py
git commit -m "fix(envexplore): 保留探索释放故障成本并汇总方法用量"
```

## Task 5：双环境配置与真实患者验收

**Files:** Create `configs/physicianbench-envexplore.json`、`configs/medagentbench-envexplore.json`、`experiments/test_envexplore_configs.py`；运行产物存入独立的 `outputs/<run_id>/`。不创建专属实验执行脚本。

**Interfaces:**

Consumes：`configs/physicianbench-comparison.json`、`configs/medagentbench-flash.json` 的同环境主方法配置；公共命令 `python -m experiments.run --config PATH --run-id ID --output-dir PATH`。

Produces：仅保留所需生成角色、采用同一初始策略及患者划分的 EnvExplore 配置；两环境各一份完整探索、生成与公共评价产物。真实样本的接受或拒绝按公共结果记录。

- [ ] **Step 1：添加配置一致性测试**

在 `experiments/test_envexplore_configs.py` 写入：

```python
import json
from pathlib import Path
import pytest
from contracts import ExperimentConfig

@pytest.mark.parametrize("source,target", [
    ("physicianbench-comparison.json", "physicianbench-envexplore.json"),
    ("medagentbench-flash.json", "medagentbench-envexplore.json"),
])
def test_configs_keep_main_comparison_material(source, target):
    root = Path(__file__).parents[1] / "configs"
    base = json.loads((root / source).read_text())
    values = json.loads((root / target).read_text())
    config = ExperimentConfig.model_validate(values)
    assert config.method == "envexplore" and config.round_patients == []
    assert config.compare_initial_strategy is False
    for key in ("environment", "evolution_patients", "evaluation_patients",
                "initial_strategy", "random_seed", "evaluation"):
        assert values[key] == base[key]
    assert set(config.runtime.models) == {"environment_explorer", "generator"}
    assert set(config.runtime.budgets) == {"environment_explorer", "generator"}
    for role in ("environment_explorer", "generator"):
        assert values["runtime"]["models"][role] == base["runtime"]["models"][role]
        assert values["runtime"]["budgets"][role] == base["runtime"]["budgets"][role]
    assert config.evolution_patients and config.evaluation_patients
    assert not set(config.evolution_patients) & set(config.evaluation_patients)
    assert values["run_id"] != base["run_id"] and values["output_dir"] != base["output_dir"]
```

- [ ] **Step 2：确认测试缺少目标配置而失败**

Run：`python -m pytest -q experiments/test_envexplore_configs.py`。
Expected：两个目标文件不存在；若已存在则检查字段差异，不重新创建其他环境配置。

- [ ] **Step 3：由同环境主方法配置产生两个目标文件**

在实施工作区执行：

```python
import json
from pathlib import Path

root = Path("configs")
for source, target, run_id in (
    ("physicianbench-comparison.json", "physicianbench-envexplore.json", "physicianbench-envexplore"),
    ("medagentbench-flash.json", "medagentbench-envexplore.json", "medagentbench-envexplore"),
):
    values = json.loads((root / source).read_text())
    values.update(method="envexplore", run_id=run_id, output_dir="outputs/" + run_id,
                  round_patients=[], compare_initial_strategy=False)
    for field in ("models", "budgets"):
        values["runtime"][field] = {role: values["runtime"][field][role]
                                    for role in ("environment_explorer", "generator")}
    (root / target).write_text(json.dumps(values, ensure_ascii=False, indent=2) + "\n")
```

不更改预算上限或模型角色配置来改善结果；方法本身仅使用一次生成调用。服务地址和镜像沿用同环境主方法，服务迁移时统一更新主方法及对照配置，保留同一数据种子。

- [ ] **Step 4：验证配置与全套必要回归**

Run：`python -m pytest -q experiments/test_envexplore_configs.py tests/method tests/runtime tests/environment tests/evaluation experiments/test_common.py experiments/test_envexplore.py experiments/test_envexplore_summary.py`。
Expected：新测试全部 PASS，相对实施前基线无新增失败。需要真实镜像的环境测试按既有开关跳过；此处跳过不能代替下一步骤的真实验收。发现基线中已实现的方法仍被列作未实现时明确报告其测试位置，不降低 EnvExplore 的验收标准。

- [ ] **Step 5：提交双环境配置及对应测试**

```bash
git add configs/physicianbench-envexplore.json configs/medagentbench-envexplore.json experiments/test_envexplore_configs.py
git commit -m "feat(envexplore): 添加两个医疗环境运行配置"
```

### 真实验收步骤

以下步骤逐环境串行执行，避免同时占用多个实例。至少完成一名不重叠评测患者的完整生成与公共评价；默认模板已经包含小样本患者组，探索范围与该组主方法保持一致。不新增人工筛选患者或改变方向配比。

- [ ] **Step 6：核对服务、镜像和模型凭据配置**

```python
import json
import os
from pathlib import Path
import httpx

for filename in ("physicianbench-envexplore.json", "medagentbench-envexplore.json"):
    config = json.loads((Path("configs") / filename).read_text())
    print(config["environment"]["name"], config["environment"]["seed_data"])
    models = [*config["runtime"]["models"].values(),
              *config["evaluation"]["runtime"]["models"].values()]
    missing = sorted({m["api_key_env"] for m in models if not os.environ.get(m["api_key_env"])})
    assert not missing, "Missing environment variables: " + ", ".join(missing)
    with httpx.Client(timeout=10, trust_env=False) as client:
        response = client.get(config["environment"]["manager_url"].rstrip("/") + "/openapi.json")
        response.raise_for_status()
        assert "/instances" in response.json()["paths"]
```

Expected：两个管理服务响应、环境名称与固定种子可核对、凭据变量存在；仅显示变量名，不输出凭据。镜像需已在对应管理服务所在主机加载；实例创建失败按环境故障记录，不自动换种子。未授权旗舰或高价模型不能执行。

- [ ] **Step 7：运行 PhysicianBench 小样本**

```bash
python -m experiments.run --config configs/physicianbench-envexplore.json --run-id physicianbench-envexplore-acceptance --output-dir outputs/physicianbench-envexplore-acceptance
```

Expected：一次环境探索、每名患者至多两道候选、公共核验和解题记录保存。目录已存在时选择新的 run_id 与目录，不删除或覆盖旧实验。核验拒绝与少出题属于真实产出，不要求所有题目都通过。

- [ ] **Step 8：运行 MedAgentBench 小样本**

```bash
python -m experiments.run --config configs/medagentbench-envexplore.json --run-id medagentbench-envexplore-acceptance --output-dir outputs/medagentbench-envexplore-acceptance
```

Expected：环境探索与解题使用原生 GET/POST/FINISH，生成采用单次结构化调用；公共评价保存完整记录。环境不可用、模型故障、预算耗尽单独记录。

- [ ] **Step 9：核对两份运行产物、状态与用量**

```python
import json
from pathlib import Path

for run_id in ("physicianbench-envexplore-acceptance", "medagentbench-envexplore-acceptance"):
    root = Path("outputs") / run_id
    manifest = json.loads((root / "manifest.json").read_text())
    summary = json.loads((root / "summary.json").read_text())
    runs = json.loads((root / "runs.json").read_text())
    print(run_id, "status:", manifest["status"], "errors:", manifest["errors"])
    print("candidate/evaluation:", summary["evaluation"])
    print("costs:", summary["costs"])
    exploration = [r for r in runs if r["stage"] == "explore_environment"]
    assert len(exploration) == 1
    assert all(r["stage"] not in ("explore_data",) for r in runs)
    assert not (root / "rounds").exists()
    for record in runs:
        if "/generate/" in record["stage"]:
            assert record["result"]["usage"]["model_turns"] <= 1
            assert record["result"]["usage"]["tool_calls"] == 0
    if manifest["status"] == "completed":
        assert (root / "environment_guidance.json").exists()
        evaluations = json.loads((root / "evaluation/evaluations.json").read_text())
        for evaluation in evaluations:
            state = evaluation.get("observed_state", {})
            if run_id.startswith("medagentbench") and state.get("operation_semantics"):
                assert state["operation_semantics"]["acceptance_proves_database_write"] is False
```

打开 `environment_guidance.json` 和 `runs.json` 中生成调用的 model_request，逐项核对指导抽象性、完整轨迹、原始来源及固定策略；对照 `tasks.json` 和 `evaluations.json` 检查引用与判定理由。生成候选 reference_answer 的存在不作为正确性的证据。

对含操作要求的任务，查看 `solver_run.trace` 与 `observed_state`：MedAgentBench 的 POST 请求只能证明请求接受，PhysicianBench 的持久化要求需与实际状态一致。没有产生操作题时，以已有 `tests/environment/test_medagentbench_live.py::test_official_interaction_post_semantics_and_serial_cleanup` 在可用镜像下核对原生提交语义，不补造候选或改变方向配比。

在管理服务主机上执行 `docker ps -a --filter label=medevolve.managed=true --format '{{.Names}} {{.Status}}'`，结合本次管理服务的创建/释放记录核对没有本次遗留实例；不要删除其他运行的容器。完成状态不能单靠模型返回的 FINISH 推断。

- [ ] **Step 10：按实际结果交付**

交付两个环境的运行目录、对应实现提交和检查结果。分别报告候选数量、核验通过、解题完成、探索/生成/公共评价成本及实际失败类别。若真实环境不可用或没有完成真实验收，明确标为未完成，不能使用模拟测试结果替代。真实产物不自动提交到 Git。

## SPEC 验收覆盖

| 验收 | 对应任务与证据 |
| --- | --- |
| A1 消融边界 | 任务 2 单次生成测试；任务 3 禁用函数、一次探索及固定上下文测试 |
| A2 患者与方向 | 任务 3 配置拒绝及同种子采样；任务 5 配置一致性 |
| A3 单次真实轨迹生成 | 任务 1 材料回归；任务 2 完整输入、调用次数、零预算及空批次测试 |
| A4 公共核验与答案隔离 | 任务 2 来源检查；任务 4 既有 evaluator 的错误事实、核验拒绝和 solver 输入测试；任务 5 真实评价 |
| A5 两环境原生语义 | 任务 4 双环境工具传递和 runtime/native 回归；任务 5 两环境真实运行 |
| A6 故障与释放 | 任务 2 生成故障；任务 3 探索失败终止；任务 4 工具故障、耗尽、释放故障及生命周期回归 |
| A7 完整成本与产物 | 任务 3 runs 和候选记录；任务 4 方法成本；任务 5 真实产物逐项检查 |

## 执行方式

本计划只定义实施与验收。实施采用 superpowers:subagent-driven-development 或 superpowers:executing-plans；两种方式均在 using-git-worktrees 建立的独立工作区中逐任务执行，完成后按用户要求合回项目当前分支。
