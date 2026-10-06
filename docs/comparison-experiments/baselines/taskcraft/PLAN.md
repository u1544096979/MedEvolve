# TaskCraft 医疗适配 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将固定版本 TaskCraft 的原子、深度、宽度算法接入 MedEvolve，在 PhysicianBench 与 MedAgentBench 上完成可追溯、公平的机制和实验验收。

**Architecture:** 复用既有 TaskCraft worktree 的方法代码，以实施开始时主项目已提交公共接口为基准。三个算法阶段只依赖方法内的材料、工具、执行及预算映射；深度与宽度读取同一原子池。实验入口绑定随机种子，公共生成入口维持四参数。

**Tech Stack:** Python >=3.10、pydantic >=2.10,<3、openai 2.54.0、openai-agents 0.20.0、httpx >=0.27,<1、pytest >=8,<10；使用项目已有依赖和 Python 环境。

## Global Constraints

- 规范依据：[SPEC.md](SPEC.md)；作者源码固定为 `/Users/guagua/code/TaskCraft` 的 `1e405e0b02c4738654e2d885b16677b46c9fe969`。项目为 `/Users/guagua/code/MedEvolve`；不引用 AgentEvolver 的同名实现。
- 原子参数 16384/20/10；深度 extended_attempts=4、max_hops=2、合并外层 5 次；SearchAgent 默认 max_step=20，深度反向及核验各 10；评分采样 3、VerifyAgent 外层重试 3。模型工具函数和宽度的内外层重试按各源码调用点保留。
- 内置判据：无工具 llm_score >=1 拒绝；原子无 error 时允许 agent_score=0；深度 agent_score <=0 拒绝；宽度 state=1 且 score<1。三个已确认源码修复限定为 superset 字段、宽度本组答案、稳定索引。
- 所有调用共用 generator 总预算；不设置阶段占比、预留轮数或生成来源配额。最多两题，按 directions 临床题型配额选择；decision_point=null。答案分别为 refined_answer、最终 golden_answer、原始答案列表的 JSON 字符串。
- 实施使用 using-git-worktrees；保护其他协作者修改。保持公共评测、其他方法、公共模型及预算配置；无新训练流程、依赖或第三环境实现。提交分关注点，description/body 使用中文。

---

## 实施基准与文件边界

实施前读取 `/Users/guagua/.agents/skills/using-git-worktrees/SKILL.md` 和目标目录适用的 AGENTS.md。旧工作区为 `/Users/guagua/code/MedEvolve/.worktrees/taskcraft`，旧实现固定在 `3cf31128cad45a2b38805db0ce305e18343c357b`。先比较已提交接口；适合时复用，公共层同步困难时新建隔离工作区，仅移植下列方法文件。禁止整分支合并覆盖公共层，禁止把其他人的未提交修改静默当成基准。

```bash
git -C /Users/guagua/code/MedEvolve rev-parse HEAD
git -C /Users/guagua/code/MedEvolve status --short
git -C /Users/guagua/code/MedEvolve worktree list
git -C /Users/guagua/code/TaskCraft rev-parse HEAD
git -C /Users/guagua/code/MedEvolve diff 3cf31128cad45a2b38805db0ce305e18343c357b HEAD -- contracts.py methods/common.py runtime environments experiments/runner.py
```

工作区路径以 using-git-worktrees 返回值为准，后文命令均在该工作区执行。新工作区的移植命令：

```bash
git restore --source=3cf31128cad45a2b38805db0ce305e18343c357b -- methods/taskcraft tests/method/test_taskcraft.py tests/method/test_taskcraft_live.py docs/comparison-experiments/baselines/taskcraft/ADAPTATION.md
```

| 文件 | 最终职责 |
| --- | --- |
| `methods/taskcraft/__init__.py`、`scheduling.py` | 生命周期、四参数入口、种子绑定、分支调度和最终选择 |
| `stages.py`、`verify.py`、`schemas.py`、`prompts.py`、`prompt_text.py` | 源码阶段、内置核验、字段模型及作者提示词 |
| `environment.py`、`ledger.py`、必要的 `agent.py` | 患者只读绑定、协议映射、全部调用记账及源码 Agent 执行边界 |
| `tests/method/test_taskcraft.py`、`test_taskcraft_live.py`、必要的运行层测试 | 分支、协议、计数、随机性及真实机制验收 |
| `experiments/runner.py`、两份 TaskCraft 配置、本目录 `ADAPTATION.md` | 公共分派、冻结配置、源码归属及真实适配结果 |

公共文件只允许增加方法接入所需的兼容接口；接口变更先与公共层负责者对齐，不能替换其当前实现。预计代码及离线检查 1–2 个工作日；真实验收另计，取决于模型和环境服务，不预设成功时长。

## Task 1：只读环境映射与源码归属

**Files:** 修改 `methods/taskcraft/environment.py`、`__init__.py`、`LICENSE-TaskCraft`；修改 `tests/method/test_taskcraft.py`；更新 `docs/comparison-experiments/baselines/taskcraft/ADAPTATION.md` 的来源及环境表。

**Interfaces:** 消费 `environments.get_tools(instance, [patient_id])` 返回的函数工具或 NativeTools；产出 `bind_generation_tools(tools)`，保留 NativeTools 类型识别、原生 GET/FINISH 及公共患者鉴权。未来环境提供同等检索与来源映射，不要求固定工具数量。

- [ ] **1.1 写执行边界失败测试（每个测试 2–5 分钟）。** 在现有测试文件增加以下独立测试；另用已有患者越界 fixture 检查 GET 指向另一患者被拒绝。

```python
import asyncio
from types import SimpleNamespace
import pytest
from environments.medagentbench import NativeTools
from methods.taskcraft.environment import bind_generation_tools


def test_read_tools_do_not_require_nine_names():
    read = SimpleNamespace(name="fhir_observation_search_labs")
    write = SimpleNamespace(name="fhir_medication_request_create")
    assert bind_generation_tools([read, write]) == [read]


def test_native_generation_blocks_post_before_transport(monkeypatch):
    calls = []
    async def execute(self, action):
        calls.append(action)
        return {"kind": "FINISH", "answers": ["ok"]}
    monkeypatch.setattr(NativeTools, "execute", execute)
    original = NativeTools("http://fhir.test/", ["patient-1"], [])
    bound = bind_generation_tools(original)
    assert isinstance(bound, NativeTools)
    rejected = asyncio.run(bound.execute("```tool_code\nPOST http://fhir.test/MedicationRequest\n{}\n```"))
    assert rejected["kind"] == "error"
    assert calls == []
    assert asyncio.run(bound.execute('FINISH(["ok"])'))["answers"] == ["ok"]
    assert calls == ['FINISH(["ok"])']
```

- [ ] **1.2 运行红灯检查（2 分钟）。** `python -m pytest -q tests/method/test_taskcraft.py -k 'nine_names or blocks_post'`。预期旧实现因要求九种工具及原生 POST 透传失败。
- [ ] **1.3 改方法私有绑定（每个改动 2–5 分钟）。** 函数工具先使用 PhysicianBench 已有只读工具声明，允许其中实际开放的非空子集；空集返回 EnvironmentUnavailable。第三环境通过显式方法适配提供只读工具，不凭 `_search` 名称猜权限。原生绑定使用以下结构，工具声明在副本中只保留读取能力；不能修改原对象或患者客户端。

```python
from copy import deepcopy
from environments.medagentbench import NativeTools

class ReadOnlyNativeTools(NativeTools):
    def __init__(self, source):
        super().__init__(source.endpoint, source.patient_ids, [])
        self._source = source
        self.functions = [deepcopy(item) for item in source.functions
                          if item.get("name", "").startswith("GET ")]

    async def execute(self, action):
        normalized = action.replace("```tool_code", "").replace("```", "").strip()
        if not (normalized.startswith("GET ") or
                (normalized.startswith("FINISH(") and normalized.endswith(")"))):
            return {"kind": "error", "request": {"action": action},
                    "reply": "TaskCraft generation permits GET and FINISH only"}
        return await self._source.execute(action)
```

实际 `functions` 从已绑定 source 的 GET 声明复制，保持原声明结构；上述测试的空数组只代表无工具声明的最小输入。沿用公共环境的患者、实例、分页及地址鉴权。核对项目 MIT LICENSE 和移植文件自身的 Apache-2.0 声明，原文保留必要许可与归属，不能仅用根 LICENSE 覆盖文件声明。

- [ ] **1.4 运行绿灯检查（5 分钟）。** 上述测试及 `python -m pytest -q tests/environment/test_patient_boundary.py tests/runtime/test_native.py`。预期方法生成拒绝写入，公共解题原生 POST 测试仍通过。补一项不依赖环境名称的只读工具子集测试，不实现第三环境服务。
- [ ] **1.5 提交（2 分钟）。** 仅暂存本任务方法、测试及来源文档，`git commit -m "feat(taskcraft): 接入患者只读检索边界"`。

## Task 2：源码执行、重试与累计账本

**Files:** 修改 `methods/taskcraft/ledger.py`、`verify.py`、`schemas.py`；按核对结果创建 `agent.py`；确需公共请求接口时修改 `runtime/execution.py` 或 `runtime/native.py` 并增加对应测试；不改变其他调用者默认行为。

**Interfaces:** `Ledger.llm(stage, *, instructions, input_text, output_type, attempts=1)` 返回 RunResult 或 None；`Ledger.agent(stage, *, instructions, input_text, tools, output_type, max_action_steps=20)` 返回同类型。原子核验调用 20，深度 backward/verify 调用 10。每次真实请求和工具调用进入同一 GenerationBudget.records。

- [ ] **2.1 建立源码执行对照测试（每项 2–5 分钟）。** 使用旧测试的 FakeRuntime、completed 和 runtime_config。增加一项首次调用失败、第二次成功的 fixture，断言两次均进入账本；增加评分一次失败而其余有效的情形；增加下面的状态机检查。

```python
# agent.py 的控制函数独立于环境和模型。
from methods.taskcraft.agent import step_kind

def test_source_planning_and_reflection_schedule():
    assert [step_kind(i, 5) for i in range(1, 12)] == [
        "planning", "action", "action", "action", "action",
        "reflection", "action", "action", "action", "action", "reflection",
    ]
```

计数测试必须证明：首次规划的 facts/plan 请求、反思的 score/facts/plan 请求不增加 action step，但增加 model_turns；一个 action 响应的多个工具调用分别增加 tool_calls；达到 action 上限后源码 provide_final_answer 的请求仍计费，累计预算不足时禁止该请求。不能只测 `min(10, remaining_turns)`。

- [ ] **2.2 运行红灯检查并逐函数核对（每项 2–5 分钟）。** 阅读作者 `ToolCallingAgent._run`、`planning_step`、`reflection_step`、`step`、`provide_final_answer`，以及 `agent_tools.SearchAgent.forward`/`VerifyAgent.forward`。将实际提示模板、调用顺序、异常分支、最终步骤计数加入 ADAPTATION 对应表。预期旧实现缺少 planning/reflection，误以 model_turns 为 agent_step_number，VerifyAgent 无外层三次重试。
- [ ] **2.3 移植最小执行控制及重试（按函数切成 2–5 分钟动作）。** 保留作者循环和模板，只替换模型请求与患者工具执行端；不用整个 oagents 的爬虫、视觉工具或新增依赖。控制函数如下：

```python
def step_kind(step: int, reflection_interval: int = 5) -> str:
    if step == 1:
        return "planning"
    if step % reflection_interval == 1:
        return "reflection"
    return "action"
```

`agent.py` 提供 `run_search_agent(*, ledger, stage, instructions, input_text, tools, output_type, max_action_steps)`。其请求端使用 Ledger 的单次调用入口；Agent 记忆和工具结果按作者函数映射，所有 action、planning、reflection、forced_final 独立写入 trace。函数工具与原生协议只在动作编码及终答解析处不同。FINISH 答案数组映射 AnswerList；单个关系对象映射 BackwardContext。调用者传入的参考答案不得进入检索 Agent 输入。

Ledger 的重试是源调用点参数，不能全局设为 3：工具函数通常 3，VerifyAgent 无工具作答及每个评分请求 1；宽度 retry_predict 的 3 次还要保留调用方外层 3 次。终止判定只看真实累计剩余模型预算或明确的总工具预算阻断，局部 action 上限、格式失败不能把整批标成累计耗尽。单次调用必须完整写入公共账本后才重试。

```python
# Ledger.llm 的重试控制；每次 self.budget.call_llm 自行记账。
async def llm(self, stage, *, instructions, input_text, output_type, attempts=1):
    last = None
    for attempt in range(attempts):
        if self.budget.remaining().max_model_turns == 0:
            self.exhausted = True
            return last
        last = await self.budget.call_llm(
            f"{stage}/attempt-{attempt + 1}", instructions=instructions,
            input_text=input_text, output_type=output_type)
        if last.status == "completed":
            return last
    self.exhausted = self.budget.remaining().max_model_turns == 0
    return last
```

VerifyAgent 的三次外层重试重新走无工具作答、评分、工具作答、评分；`llm_score>=1` 立即返回 error，不重试来绕过该判据。评分按源码有效结果求平均；无法比较的评分走源码错误/外层重试分支。序列化 trace 保留模型响应、工具输入输出、步骤类型及真实计数，不以工具调用数量重建 Agent 记忆。

- [ ] **2.4 运行绿灯及差异审查（每组 5 分钟）。** `python -m pytest -q tests/method/test_taskcraft.py tests/runtime/test_execution.py tests/runtime/test_native.py`。核对作者步骤控制与适配实现逐项对应；公共累计预算能在 planning/reflection/action/forced_final 任一请求前截断。若现有依赖无法保留这些行为，先提交具体接口缺口供用户选择，禁止静默用普通 Runner 代替并宣称源码验收完成。
- [ ] **2.5 提交（2 分钟）。** 方法执行与必要公共接口分开提交：`feat(taskcraft): 保留源码检索步骤与重试记账`；确有公共接口改动时单独 `feat(runtime): 提供检索单步执行接口`，后者默认兼容原调用者。

## Task 3：三个阶段与答案映射

**Files:** 修改 `methods/taskcraft/stages.py`、`verify.py`、`prompts.py`、`prompt_text.py`、`schemas.py`；修改 `tests/method/test_taskcraft.py`。

**Interfaces:** 保留 `Candidate(question, origin, parents, locator, detail)` 及 `Pipeline.atomic()/depth(atomic)/width(atomic)`；各返回 list[Candidate]。原子池由 atomic 一次产生，扩展只读该池。TaskCraftParameters 保留源码参数，删除阶段份额及预留字段，增加明确的 Agent 20/10/10 限制参数。

- [ ] **3.1 增加源码分支红灯测试（每项 2–5 分钟）。** 复用 FakeRuntime 分别给出 llm_score 0/1/2、agent_score 0/1/2，断言原子零分允许、深度非正分拒绝、宽度只保留 state=1 且 score<1。扩展 fixture 返回 `SupersetCheck(new_query="invalid")` 时必须拒绝。多宽度组使用不同答案并过滤首组，断言第二组仍使用自身答案和稳定 index；至少 11 组验证源码 batch_size=10。

```python
import json
from methods.taskcraft.stages import Pipeline

def test_width_answer_is_lossless_json():
    item = {"original_questions": ["Q1", "Q2"],
            "golden_answers": ["5.9 mmol/L", "含换行\n和引号\"的原答案"]}
    assert json.loads(Pipeline._merged_answer(item)) == item["golden_answers"]


def test_depth_answer_is_the_original_final_answer():
    initial = {"initial_answer": "5.9 mmol/L"}
    chain = [{"query": "先查哪个标识", "answer": "Observation/77"}]
    assert Pipeline._chain_answer(initial, chain) == "5.9 mmol/L"
```

- [ ] **3.2 运行红灯检查（2 分钟）。** `python -m pytest -q tests/method/test_taskcraft.py`。记录已存在的接口失败与本任务预期行为失败；不能把旧 worktree 的 share/guard/优先级断言当作目标规范。
- [ ] **3.3 对照作者函数修正阶段（每个函数或分支 2–5 分钟）。** atomic 保持结论→ID/R/A→clean_qa→VerifyAgent 顺序；深度保持 valid_hop 初值 1、反向关系、辅助核验、合并判定及上限；宽度保持实际含 golden_answer 的模型输入、2–3 父题、分批和内外重试。父索引保持源码的分组顺序，不排序；按源码处理组大小和重复索引，越界及格式错误按模型输出失败记账，不静默修剪成新组或增加筛选阈值。

```python
# 深度字段判断，以及两个最终答案映射。
if superset.new_query.strip().lower() == "invalid":
    return None

@staticmethod
def _chain_answer(initial: dict, chain: list[dict]) -> str:
    return initial["initial_answer"]

@staticmethod
def _merged_answer(item: dict) -> str:
    return json.dumps(item["golden_answers"], ensure_ascii=False)

# 宽度评分使用本批通过分解检查的稳定 ID，不索引局部数组。
by_id = {item["index"]: item for item in validated}
for answer in answers:
    item = by_id.get(answer.index)
    if item is None:
        continue
```

宽度 Candidate.detail 保存 original_questions、golden_answers、原始 indexes、全部 parent_types；深度 detail 保存 full_results、solve_trajectory、辅助来源和所有失败尝试。原子 detail 保留原结论、关系、来源及清洗答案。临床标签按 SPEC 映射；证据 statement 保持原记录事实语义，来源归属与公共事实核验分别记录。原生医嘱/给药、规格/剂量、记录时间/事件时间禁止混用。

- [ ] **3.4 核验绿灯与提示词（每组 5 分钟）。** 跑方法测试；逐块比对作者 `prompts/*.yaml` 与 prompt_text 原文，追加仅限医疗材料、字段、工具、患者范围和协议。三个缺陷修复分别有测试，所有阈值均使用 0–2 原始评分。用实际 source_id 核对深度辅助事实与宽度父证据；不增加 TaskCraft 专属模型事实审查。
- [ ] **3.5 提交（2 分钟）。** 缺陷修复独立 `fix(taskcraft): 修正源码字段与宽度对应索引`；答案和阶段适配独立 `feat(taskcraft): 对齐三阶段答案与筛选规则`。每次提交包含对应测试且能独立运行。

## Task 4：随机调度与公共实验接入

**Files:** 创建 `methods/taskcraft/scheduling.py`；修改 `stages.py`、`__init__.py`、`experiments/runner.py`、`tests/method/test_taskcraft.py`、`tests/experiment/test_closed_loop.py`；创建 `configs/taskcraft-physicianbench.json`、`configs/taskcraft-medagentbench.json`。

**Interfaces:** `derive_seed(experiment_seed:int, environment:str, patient_id:str, purpose:str)->int`；`bind_generator(random_seed:int)` 返回四参数 async callable。内部 generate_tasks 的可选 keyword 参数承载实验种子，RuntimeConfig 不增加字段。Pipeline 保存分支和选择两个独立随机流。

- [ ] **4.1 写随机性失败测试（每项 2–5 分钟）。** 同输入种子相同、purpose 改变种子不同、执行不改变 random.getstate；用跟踪 Pipeline 子类返回固定原子对象并记录 depth/width 入参，断言二者获得同一原子池。给固定合格候选列表断言选题可重复、最多两题、临床配额生效，改变 origin 标签不改变选中身份。

```python
import random
from methods.taskcraft.scheduling import derive_seed

def test_seed_streams_are_stable_and_private():
    before = random.getstate()
    args = (42, "PhysicianBench", "patient-1")
    assert derive_seed(*args, "branch-order") == derive_seed(*args, "branch-order")
    assert derive_seed(*args, "branch-order") != derive_seed(*args, "output-selection")
    assert random.getstate() == before
```

- [ ] **4.2 运行红灯（2 分钟）。** `python -m pytest -q tests/method/test_taskcraft.py -k 'seed or selection or independent'`，预期旧实现缺少稳定种子且按来源优先。
- [ ] **4.3 实现调度与绑定（每个函数 2–5 分钟）。** 使用下列稳定编码；候选池固定为 atomic、depth、width 的各自生成顺序，独立随机流 shuffle，再按公共字段、患者来源及临床题型配额接纳；不参考内置评分或公共评测结果。

```python
import hashlib
import json
import random
from collections import Counter

def derive_seed(experiment_seed, environment, patient_id, purpose):
    payload = json.dumps([experiment_seed, environment, patient_id, purpose],
                         ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest(), "big")

def select_candidates(candidates, directions, source_ids, seed):
    shuffled = list(candidates)
    random.Random(seed).shuffle(shuffled)
    remaining = Counter(direction.task_type for direction in directions)
    selected = []
    for candidate in shuffled:
        q = candidate.question
        if not q.instruction.strip() or not q.reference_answer.strip():
            continue
        if q.decision_point is not None or not q.evidence:
            continue
        if any(e.source_id not in source_ids or not e.statement.strip() for e in q.evidence):
            continue
        if remaining[q.task_type] <= 0:
            continue
        selected.append(candidate)
        remaining[q.task_type] -= 1
        if len(selected) == 2:
            break
    return selected
```

Pipeline.run 先 atomic；局部 Random(branch_seed).shuffle(["depth","width"]) 后分别调用对应方法。每条分支保存启动、完成、跳过及累计截断记录。原子池不足时宽度记录 insufficient_parents；总预算耗尽时后续分支记录 cumulative_budget_exhausted，不自动补题。最终 pool 永远包括已生成候选；预算耗尽的公共 output 为 None，但完整 pool 仍在 trace。

```python
# __init__.py：公共调用仍是四个位置参数。
def bind_generator(random_seed):
    async def bound(trajectory, environment, directions, runtime):
        return await generate_tasks(trajectory, environment, directions, runtime,
                                    random_seed=random_seed)
    return bound

# runner.py 的 generator 解析之后，只绑定 taskcraft。
if config.method == "taskcraft":
    generator = importlib.import_module("methods.taskcraft").bind_generator(config.random_seed)
```

`_with_pipeline` 保存候选完整 question.model_dump、origin、parents、locator、detail，以及实验种子、派生种子、分支顺序、筛选和最终选择标识。校验 trajectory/environment 一致；环境创建后无论阶段错误、解析失败还是预算截断都释放实例。

配置分别从实施时已提交的 physicianbench-comparison.json、medagentbench-flash.json 复制；仅改变 method、run_id、output_dir。模型、患者、数据、预算、方向和评测保持公共值。现有 1 model turn/0 tool calls 的配置不能支持完整机制验收，正式比较照实返回预算截断；不偷偷扩大 TaskCraft 预算。

- [ ] **4.4 运行绿灯及公共回归（每组 5 分钟）。** `python -m pytest -q tests/method/test_taskcraft.py tests/experiment/test_closed_loop.py experiments/test_common.py`。验证绑定正确传递 random_seed、其他方法入口不变、评测解题输入不含 reference_answer 或内部核验反馈。两份配置用 ExperimentConfig.model_validate_json 校验，再比较除 method/run_id/output_dir 外的字段完全相等。
- [ ] **4.5 提交（2 分钟）。** `feat(taskcraft): 接入固定种子调度与公共实验`；配置独立 `chore(taskcraft): 添加双环境比较配置`。

## Task 5：双环境真实验收与交付

**Files:** 修改 `tests/method/test_taskcraft_live.py`；完成本目录 `ADAPTATION.md`；真实产物写入新运行目录，禁止覆盖已有结果。无需修改其他方法规范。

**Interfaces:** 真实机制输入来自同一公共 PatientTrajectory；RunResult 及公共评测入口沿用当前公共规范。机制验收配置与正式比较配置分开保存，均记录完整模型、预算、源码和实现提交。

- [ ] **5.1 运行离线基线（5 分钟）。** 执行方法、环境、运行层、实验和受影响评测测试；只在新改动或失败说明有需要时扩展检查。记录与实施基准相比新增的失败。确认两个服务可创建和释放实例、数据版本固定、配置凭据变量存在；不输出密钥，不由源码默认模型触发调用。
- [ ] **5.2 更新真实验收入口（每个断言 2–5 分钟）。** live 测试从显式配置读取环境与模型，取消硬编码付费模型和地址。默认跳过真实调用；显式启用后断言真实阶段成功产物包含父题、原始答案、可回查证据、完整工具结果和成本。status=completed 或零输出不算阶段成功。

```python
# 在真实测试取得 RunResult 后，对 pipeline trace 执行阶段断言。
pipeline = next(item for item in result.trace if item.get("stage") == "taskcraft/pipeline")
assert pipeline["author_commit"] == "1e405e0b02c4738654e2d885b16677b46c9fe969"
for origin in ("atomic", "depth", "width"):
    candidates = [item for item in pipeline["pool"] if item["origin"] == origin]
    assert candidates, f"{origin} has no successful real candidate"
    assert all(item["question"]["reference_answer"] for item in candidates)
    assert all(item["question"]["evidence"] for item in candidates)
    if origin != "atomic":
        assert all(item["parents"] for item in candidates)
assert result.usage["model_turns"] > 0
assert result.usage["tool_calls"] > 0
```

- [ ] **5.3 分别执行两个环境机制验收（每次查看产物 5 分钟）。** 使用用户授权的现有模型；机制预算须明确标注，不能替代公共预算比较结果。核对 atomic、depth、width 各一份真实成功样本及其证据；缺阶段保留失败原因，不挑选成功结果冒充完整验收。遇到环境不具备真实关系检索，记录具体工具或数据缺口并停止该项验收。
- [ ] **5.4 执行冻结配置的公共实验并审查产物（每个环境一次）。** 使用 `python -m experiments.run --config configs/taskcraft-physicianbench.json` 和相应 medagentbench 配置。分别报告各阶段候选数、最终题数、公共事实核验、质量评价、解题/完成状态、失败类别、生成及评测成本；缺失 token 为未知，重试和失败成本全部保留。公开预算不够时记录真实截断，不以额外机制预算输出代替。
- [ ] **5.5 提交与交付（5 分钟）。** ADAPTATION 只写最终真实结果：源码对应表、原提示词归属、参数、三项修复、环境/协议映射、Agent 差异、随机调度、完整测试及运行目录。`git commit -m "docs(taskcraft): 记录源码对应与双环境验收结果"`；按 using-git-worktrees 的集成流程交付隔离分支，不自动合并或删除工作区。

## 完成条件

离线检查通过、源码对应明确、两个环境的机制验收和公共联调有真实产物，才可标记完整实施完成。额外源码缺陷、无法保持的 Agent 核心行为或缺失环境能力须具体列出并同步用户；模拟分支成功只证明控制逻辑，不能充当真实算法效果证据。
