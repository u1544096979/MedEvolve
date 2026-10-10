# 生成失败隔离 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把生成流程中「单题或单位置校验失败」从作废整个患者改为丢弃出错的那一道，其余照常产出。

**Architecture:** Workflow 的三处失败点统一改为「写入 `drops` 清单后继续」，只有全部位置都失败才判患者失败；SelfGen 的草稿校验改为逐题校验逐题丢弃，已通过题按位置合并存档版本而不是比对文本。

**Tech Stack:** Python 3.12、pytest、pytest-asyncio、pydantic v2

## Global Constraints

- 不修改任何提示词文本（`methods/workflow/prompts.py`、`methods/selfgen/prompts.py`）。
- 不修改任何判定口径（`citations_supported` 等判据的语义）。
- 降级只针对「输出内容不满足校验」；LLM 调用失败、预算耗尽、全部位置失败、review 判定自相矛盾仍整批失败。
- 止损后的运行状态仍为 `completed`，止损动作必须写进 trace。
- 全部既有测试通过：`python -m pytest tests/method tests/experiment -q`

---

### Task 1: Workflow locate 剔除无效 source

**Files:**
- Modify: `methods/workflow/__init__.py`（`_selected_focuses`、`_finish`、`generate_tasks`）
- Test: `tests/method/test_workflow.py`

**Interfaces:**
- Consumes: `PatientTrajectory.sources`、`FocusSelection`
- Produces:
  - `_selected_focuses(selection, directions, trajectory) -> tuple[list[ClinicalFocus], list[dict]]`，drops 元素形如 `{"focus": int, "reason": str, "source_ids": list[str]}`
  - `_finish(budget, selected, focuses, *, output=None, status=None, error=None, drops=None) -> RunResult`
  - trace 新增条目 `{"type": "drops", "drops": [...]}`，位置在 `{"type": "selection", ...}` 之前

- [ ] **Step 1: 改写既有测试为期望行为**

在 `tests/method/test_workflow.py` 中改写三处：

```python
def test_selected_focuses_match_pool_and_reject_duplicates():
    pool = directions_pool()
    selection = pool_selection()

    focuses, drops = workflow._selected_focuses(selection, pool, patient())

    assert [focus.direction_index for focus in focuses] == [0, 3]
    assert focuses == selection.focuses
    assert drops == []
    duplicate = selection.model_copy(update={"focuses": [selection.focuses[0]] * 2})
    with pytest.raises(ValueError):
        workflow._selected_focuses(duplicate, pool, patient())
```

把 `test_selected_focuses_reject_three_focuses_missing_source_and_blank_topic` 改名为 `test_selected_focuses_reject_three_focuses_and_blank_topic`，并删掉其中的缺失 source 用例：

```python
def test_selected_focuses_reject_three_focuses_and_blank_topic():
    pool = directions_pool()
    selection = pool_selection()
    third = dict(direction_index=2, clinical_focus="检查选择", source_ids=["Observation/o1"])
    cases = [
        selection.model_copy(update={"focuses": [*selection.focuses, third]}),
        selection.model_copy(update={"focuses": [
            selection.focuses[0],
            selection.focuses[1].model_copy(update={"clinical_focus": "   "}),
        ]}),
    ]

    for case in cases:
        with pytest.raises(ValueError):
            workflow._selected_focuses(case, pool, patient())
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/method/test_workflow.py -q -k selected_focuses`
Expected: FAIL，`ValueError: too many values to unpack`

- [ ] **Step 3: 实现剔除逻辑**

替换 `methods/workflow/__init__.py` 中的 `_selected_focuses`：

```python
def _selected_focuses(selection: FocusSelection, directions: list[GenerationDirection],
                      trajectory: PatientTrajectory) -> tuple[list[ClinicalFocus], list[dict]]:
    """Validate the located task types, dropping focuses whose sources are unknown."""
    output_constraints(directions, candidate_pool=True)
    focuses = selection.focuses
    if len(focuses) != 2 or len({focus.direction_index for focus in focuses}) != 2:
        raise ValueError("locate must select two distinct task types")
    known = {source.source_id for source in trajectory.sources}
    kept: list[ClinicalFocus] = []
    drops: list[dict] = []
    for focus in focuses:
        if not 0 <= focus.direction_index < len(directions):
            raise ValueError("focus index is outside candidate pool")
        if not focus.clinical_focus.strip():
            raise ValueError("focus requires a topic and evidence sources")
        valid = [source_id for source_id in focus.source_ids if source_id in known]
        unknown = [source_id for source_id in focus.source_ids if source_id not in known]
        if unknown:
            drops.append({"focus": focus.direction_index, "reason": "unknown source",
                          "source_ids": unknown})
        if not valid:
            drops.append({"focus": focus.direction_index, "reason": "focus has no valid source",
                          "source_ids": list(focus.source_ids)})
            continue
        kept.append(focus.model_copy(update={"source_ids": valid}))
    if not kept:
        raise ValueError("no focus survives source validation")
    return kept, drops
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/method/test_workflow.py -q -k selected_focuses`
Expected: PASS

- [ ] **Step 5: 写端到端测试**

追加到 `tests/method/test_workflow.py`：

```python
@pytest.mark.asyncio
async def test_locate_with_one_unknown_source_still_generates(monkeypatch):
    selection = pool_selection()
    located = selection.model_copy(update={"focuses": [
        selection.focuses[0],
        selection.focuses[1].model_copy(update={"source_ids": ["MedicationRequest/missing"]})]})
    call = substitute([completed(located.model_dump(mode="json")),
                       completed({"questions": [first_candidate()]}),
                       completed(reviews(kept=(0,), count=1))])
    monkeypatch.setattr("runtime.call_llm", call)

    result = await workflow.generate_tasks(patient(), environment(), directions_pool(), runtime())

    assert result.status == "completed"
    assert call.await_count == 3
    assert [entry for entry in result.trace if entry.get("type") == "drops"] == [
        {"type": "drops",
         "drops": [{"focus": 3, "reason": "unknown source",
                    "source_ids": ["MedicationRequest/missing"]}]}]
    assert result.trace[-1] == selection_output()
```

- [ ] **Step 6: 运行测试确认失败**

Run: `python -m pytest tests/method/test_workflow.py -q -k one_unknown_source`
Expected: FAIL，`AssertionError`（trace 中还没有 drops 条目）

- [ ] **Step 7: 让 `_finish` 记录 drops**

替换 `_finish`：

```python
def _finish(budget: GenerationBudget, selected: list[GenerationDirection],
            focuses: list[ClinicalFocus], *, output: TaskBatch | None = None,
            status: str | None = None, error: str | None = None,
            drops: list[dict] | None = None) -> RunResult:
    """Close the run, recording dropped candidates before the located selection."""
    result = budget.finish(selected, output=output, status=status, error=error)
    entries = [{"type": "drops", "drops": drops}] if drops else []
    entries.append(_selection_entry(focuses))
    return result.model_copy(update={"trace": [*result.trace, *entries]})
```

- [ ] **Step 8: 让 `generate_tasks` 传递 drops**

替换 `generate_tasks` 中从 locate 到 generate 循环结束的片段：

```python
    try:
        focuses, drops = _selected_focuses(FocusSelection.model_validate(located.output),
                                           directions, trajectory)
    except Exception as exc:
        return summarize_results(budget.records, status="failed",
                                 error=f"locate: {type(exc).__name__}: {exc}")
```

并把循环内两处 `_finish(...)` 调用补上 `drops=drops`，即：

```python
        if validated.status != "completed":
            return _finish(budget, selected, focuses, status=validated.status,
                           error=f"{stage}: {validated.error}", drops=drops)
```

```python
        if any(evidence.source_id not in known
               for question in batch.questions for evidence in question.evidence):
            return _finish(budget, selected, focuses, status="failed",
                           error=f"{stage}: unknown evidence source", drops=drops)
```

- [ ] **Step 9: 运行全量方法测试**

Run: `python -m pytest tests/method/test_workflow.py -q`
Expected: PASS

- [ ] **Step 10: 提交**

```bash
git add methods/workflow/__init__.py tests/method/test_workflow.py
git commit -m "feat(workflow): locate 剔除无效来源后继续生成

locate 引用不存在的 source_id 时不再作废整个患者，改为剔除无效
来源并用剩余来源继续；两个题型都无效才判该患者失败。止损记录写入
trace 的 drops 条目。"
```

---

### Task 2: Workflow 生成位置级隔离

**Files:**
- Modify: `methods/workflow/__init__.py`（`generate_tasks` 的生成循环与候选为空分支）
- Test: `tests/method/test_workflow.py`

**Interfaces:**
- Consumes: Task 1 的 `_finish(..., drops=)`、`drops` 列表
- Produces: 生成循环内的 drops 元素形如 `{"stage": str, "reason": str}`，其中 `stage` 为 `"generate_0"` / `"generate_1"`

- [ ] **Step 1: 改写既有测试**

替换 `tests/method/test_workflow.py` 的 `test_invalid_node_output_fails_stage_before_check`：

```python
@pytest.mark.asyncio
@pytest.mark.parametrize("stage_index,payload,expected_type", [
    (1, {"questions": [first_candidate(), first_candidate()]}, "药物处方"),
    (1, {"questions": [{**first_candidate(), "instruction": "   "}]}, "药物处方"),
    (1, {"questions": [{**first_candidate(),
                        "evidence": [{"source_id": "Observation/missing", "statement": "缺失"}]}]},
     "药物处方"),
    (2, {"questions": [second_candidate(), second_candidate()]}, "诊断与解释"),
    (2, {"questions": [candidate("治疗规划", "调整治疗", "Observation/o1", "158/96")]},
     "诊断与解释"),
])
async def test_invalid_node_output_drops_only_that_position(monkeypatch, stage_index, payload,
                                                            expected_type):
    results = [completed(located()), completed({"questions": [first_candidate()]}),
               completed({"questions": [second_candidate()]}),
               completed(reviews(kept=(0,), count=1))]
    results[stage_index] = completed(payload)
    call = substitute(results)
    monkeypatch.setattr("runtime.call_llm", call)

    result = await workflow.generate_tasks(patient(), environment(), directions_pool(), runtime())

    assert result.status == "completed"
    assert call.await_count == 4
    questions = TaskBatch.model_validate(result.output).questions
    assert [question.task_type for question in questions] == [expected_type]
    drops = [entry for entry in result.trace if entry.get("type") == "drops"]
    assert [entry["drops"][0]["stage"] for entry in drops] == [f"generate_{stage_index - 1}"]
    assert result.trace[-1]["type"] == "internal_check"
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/method/test_workflow.py -q -k drops_only_that_position`
Expected: FAIL，`assert result.status == "completed"` 得到 `"failed"`

- [ ] **Step 3: 重写生成循环**

把 `generate_tasks` 中的生成循环与空候选分支替换为：

```python
    candidates: list[GeneratedQuestion] = []
    failed_positions = 0
    for index, focus in enumerate(kept):
        direction = directions[focus.direction_index]
        stage = f"generate_{index}"
        generated = await budget.call_llm(
            stage,
            instructions=f"{GENERATE_BY_TYPE[direction.task_type]}\n"
                         + output_constraints([direction], max_questions=1),
            input_text=json.dumps(
                {
                    "trajectory": _trajectory_input(trajectory),
                    "focus": focus.model_dump(mode="json"),
                    "direction": direction.model_dump(mode="json"),
                },
                ensure_ascii=False,
            ),
            output_type=TaskBatch,
        )
        validated = validate_task_batch(generated, [direction], max_questions=1)
        if validated.status != "completed":
            failed_positions += 1
            drops.append({"stage": stage, "reason": validated.error or validated.status,
                          "status": validated.status})
            continue
        batch = TaskBatch.model_validate(validated.output)
        unknown = [evidence.source_id for question in batch.questions
                   for evidence in question.evidence if evidence.source_id not in known]
        if unknown:
            failed_positions += 1
            drops.append({"stage": stage, "reason": "unknown evidence source",
                          "source_ids": unknown})
            continue
        candidates.extend(batch.questions)

    if not candidates:
        if failed_positions == len(kept):
            exhausted = any(drop.get("status") == "budget_exhausted" for drop in drops)
            return _finish(budget, selected, focuses,
                           status="budget_exhausted" if exhausted else "failed",
                           error="every generation position failed", drops=drops)
        return _finish(budget, selected, focuses, output=TaskBatch(questions=[]), drops=drops)
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/method/test_workflow.py -q -k drops_only_that_position`
Expected: PASS

- [ ] **Step 5: 补一个「两个位置都失败仍整批失败」的测试**

```python
@pytest.mark.asyncio
async def test_all_positions_failed_still_fails_the_patient(monkeypatch):
    broken = {**first_candidate(), "instruction": "   "}
    call = substitute([completed(located()), completed({"questions": [broken]}),
                       completed({"questions": [broken]})])
    monkeypatch.setattr("runtime.call_llm", call)

    result = await workflow.generate_tasks(patient(), environment(), directions_pool(), runtime())

    assert result.status == "failed"
    assert result.output is None
    assert result.error == "every generation position failed"
    assert call.await_count == 3
```

- [ ] **Step 6: 运行测试确认通过**

Run: `python -m pytest tests/method/test_workflow.py -q -k all_positions_failed`
Expected: PASS

- [ ] **Step 7: 修正 `test_failed_stage_stops_flow_and_keeps_cost`**

provider 失败在 locate、generate、check 三个阶段仍然整批失败，因此该测试原有的 4 个参数化用例全部保持通过、无需删减，只需在其后补一个覆盖生成位置 provider 失败的用例：

保留原有参数化测试，并在其上加一条 `result.error` 断言：该断言钉住「失败阶段本身报错」这条分支，
否则删掉 `check` 阶段的致命分支后其余断言仍会通过（后续校验会把同一个错误重新包装一次）。

```python
@pytest.mark.asyncio
@pytest.mark.parametrize("stage_index,expected_calls", [(0, 1), (1, 2), (2, 3), (3, 4)])
async def test_failed_stage_stops_flow_and_keeps_cost(monkeypatch, stage_index, expected_calls):
    results = [completed(located()), completed({"questions": [first_candidate()]}),
               completed({"questions": [second_candidate()]}), completed(reviews())]
    results[stage_index] = RunResult(
        trace=[{"raw": "provider error"}], status="failed", error="RuntimeError: provider unavailable",
        usage={"model_turns": 1, "tool_calls": 0, "prompt_tokens": 10, "completion_tokens": 5},
    )
    call = substitute(results)
    monkeypatch.setattr("runtime.call_llm", call)

    result = await workflow.generate_tasks(patient(), environment(), directions_pool(), runtime())

    assert result.status == "failed"
    assert result.output is None
    # Pins the fatal branch of the failing stage: the provider error is reported verbatim
    # instead of being re-wrapped by the following validation step.
    assert result.error == f"{['locate', 'generate_0', 'generate_1', 'check'][stage_index]}: " \
                           "RuntimeError: provider unavailable"
    assert result.usage["model_turns"] == expected_calls
    assert call.await_count == expected_calls
    if expected_calls == 1:
        assert result.trace[-1]["stage"] == "locate"
    else:
        assert result.trace[-1] == selection_output()
```

- [ ] **Step 8: 运行全量方法测试**

Run: `python -m pytest tests/method/test_workflow.py -q`
Expected: PASS

- [ ] **Step 9: 提交**

```bash
git add methods/workflow/__init__.py tests/method/test_workflow.py
git commit -m "feat(workflow): 生成位置独立成败

某个生成位置校验失败时只丢弃该位置的候选，不再连带丢弃另一个位置
已经生成的题；两个位置都失败才判该患者失败。"
```

---

### Task 3: Workflow check 按已判结果收敛

**Files:**
- Modify: `methods/workflow/__init__.py`（`_decisions`、`generate_tasks` 的 check 段）
- Test: `tests/method/test_workflow.py`

**Interfaces:**
- Consumes: Task 1/2 的 `drops` 列表
- Produces: `_decisions(review: BatchReview, questions: list[GeneratedQuestion]) -> tuple[list[dict], list[dict]]`，第一个返回值为 decisions（覆盖全部候选，未判到的 `keep=False`），第二个为 drops，元素形如 `{"index": int, "reason": str}`

- [ ] **Step 1: 改写既有测试**

替换 `test_duplicate_or_missing_verdicts_fail_instead_of_filtering`：

```python
@pytest.mark.asyncio
@pytest.mark.parametrize("verdicts,expected_kept", [
    ("duplicate", [0]),
    ("duplicate_reworded", [0]),
    ("conflicting", []),
    ("conflicting_three", []),
    ("missing", [0]),
    ("out_of_range", [0]),
])
async def test_unjudged_candidates_are_dropped_not_fatal(monkeypatch, verdicts, expected_kept):
    review = reviews()
    replacement = {
        "duplicate": {"reviews": [review["reviews"][0], review["reviews"][0]]},
        "duplicate_reworded": {"reviews": [review["reviews"][0],
                                          {**review["reviews"][0], "reason": "证据充分"}]},
        "conflicting": {"reviews": [review["reviews"][0],
                                    {**review["reviews"][0], "clarity": False}]},
        "conflicting_three": {"reviews": [review["reviews"][0],
                                          {**review["reviews"][0], "clarity": False},
                                          review["reviews"][0]]},
        "missing": {"reviews": [review["reviews"][0]]},
        "out_of_range": {"reviews": [review["reviews"][0], {**review["reviews"][1], "index": 5}]},
    }[verdicts]
    call = substitute([completed(located()), completed({"questions": [first_candidate()]}),
                       completed({"questions": [second_candidate()]}), completed(replacement)])
    monkeypatch.setattr("runtime.call_llm", call)

    result = await workflow.generate_tasks(patient(), environment(), directions_pool(), runtime())

    assert result.status == "completed"
    assert call.await_count == 4
    assert result.trace[-1]["kept"] == expected_kept
    assert len(result.trace[-1]["decisions"]) == 2
    drops = [entry for entry in result.trace if entry.get("type") == "drops"]
    assert drops and any(item["reason"] == "candidate was not reviewed"
                         for item in drops[0]["drops"])
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/method/test_workflow.py -q -k unjudged_candidates`
Expected: FAIL，`assert result.status == "completed"` 得到 `"failed"`

- [ ] **Step 3: 重写 `_decisions`**

```python
def _verdict(item: QuestionReview) -> tuple[bool, bool, bool]:
    """The three judgements, without the prose that only explains them."""
    return (item.clarity, item.answer_alignment, item.citations_supported)


def _decisions(review: BatchReview,
               questions: list[GeneratedQuestion]) -> tuple[list[dict], list[dict]]:
    """Judge every candidate once, dropping the ones the check did not report."""
    judged: dict[int, QuestionReview] = {}
    conflicting: set[int] = set()
    drops: list[dict] = []
    for item in review.reviews:
        if item.index >= len(questions):
            drops.append({"index": item.index, "reason": "review index is outside the candidates"})
            continue
        if item.index in conflicting:
            # Once two verdicts disagree, later ones cannot restore trust.
            continue
        previous = judged.get(item.index)
        if previous is None:
            judged[item.index] = item
        elif _verdict(previous) != _verdict(item):
            # Two conflicting verdicts for one candidate leave no judgement to trust.
            judged.pop(item.index)
            conflicting.add(item.index)
    decisions: list[dict] = []
    for index in range(len(questions)):
        item = judged.get(index)
        if item is None:
            reason = ("candidate was reviewed with conflicting verdicts" if index in conflicting
                      else "candidate was not reviewed")
            drops.append({"index": index, "reason": reason})
            decisions.append({"index": index, "keep": False, "clarity": False,
                              "answer_alignment": False, "citations_supported": False,
                              "reason": reason})
            continue
        decisions.append({
            "index": index,
            "keep": item.clarity and item.answer_alignment and item.citations_supported,
            "clarity": item.clarity,
            "answer_alignment": item.answer_alignment,
            "citations_supported": item.citations_supported,
            "reason": item.reason,
        })
    return decisions, drops
```

- [ ] **Step 4: 让 check 段使用新的返回值**

把 `generate_tasks` 中 check 之后的分支替换为：

```python
    if checked.status != "completed":
        return _finish(budget, selected, focuses, status=checked.status,
                       error=f"check: {checked.error}", drops=drops)
    try:
        decisions, check_drops = _decisions(BatchReview.model_validate(checked.output), candidates)
    except Exception as exc:
        return _finish(budget, selected, focuses, status="failed",
                       error=f"check: {type(exc).__name__}: {exc}", drops=drops)
    drops.extend(check_drops)

    kept_questions = [question for question, decision in zip(candidates, decisions) if decision["keep"]]
    result = _finish(budget, selected, focuses, output=TaskBatch(questions=kept_questions),
                     drops=drops)
    return result.model_copy(
        update={
            "trace": [
                *result.trace,
                {
                    "type": "internal_check",
                    "decisions": decisions,
                    "kept": [d["index"] for d in decisions if d["keep"]],
                    "dropped": [d["index"] for d in decisions if not d["keep"]],
                },
            ]
        }
    )
```

- [ ] **Step 5: 运行测试确认通过**

Run: `python -m pytest tests/method/test_workflow.py -q`
Expected: PASS

- [ ] **Step 6: 补一个「review 输出无法解析仍整批失败」的测试**

```python
@pytest.mark.asyncio
async def test_unparsable_review_still_fails_the_patient(monkeypatch):
    call = substitute([completed(located()), completed({"questions": [first_candidate()]}),
                       completed({"questions": [second_candidate()]}),
                       completed({"reviews": "not-a-list"})])
    monkeypatch.setattr("runtime.call_llm", call)

    result = await workflow.generate_tasks(patient(), environment(), directions_pool(), runtime())

    assert result.status == "failed"
    assert result.output is None
    assert result.error.startswith("check: ")
```

- [ ] **Step 7: 运行测试确认通过**

Run: `python -m pytest tests/method/test_workflow.py -q -k unparsable_review`
Expected: PASS

- [ ] **Step 8: 提交**

```bash
git add methods/workflow/__init__.py tests/method/test_workflow.py
git commit -m "feat(workflow): 内部检查按已判结果收敛

检查模型漏判候选时不再作废整批，改为收取已判为保留的候选、把未判
到的候选按丢弃处理并记入 trace；检查输出无法解析时仍整批失败。"
```

---

### Task 4: SelfGen 草稿逐题校验

**Files:**
- Modify: `methods/selfgen/__init__.py`（`validate_draft`、`generate_tasks`）
- Test: `tests/method/test_selfgen.py`

**Interfaces:**
- Consumes: `validate_task_batch`（`methods.common`）、`PatientTrajectory.sources`
- Produces:
  - `validate_draft(value, trajectory, directions) -> tuple[DirectedBatch, list[dict]]`，drops 元素形如 `{"index": int, "reason": str}`
  - 新增内部辅助 `note(kind, **payload)` 与 `closing(result)`，把 `notes` 追加到任意 `RunResult.trace` 末尾

- [ ] **Step 1: 改写既有测试**

在 `tests/method/test_selfgen.py` 中替换 `test_source_and_passed_candidate_faults` 的 `missing_source` 与 `missing_evidence` 两个用例：

```python
@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ["missing_source", "missing_evidence", "wrong_type", "read_only"])
async def test_bad_draft_question_is_dropped_and_the_rest_continues(monkeypatch, fault):
    original = draft()
    changed = deepcopy(original)
    if fault == "missing_source":
        changed["batch"]["questions"][0]["evidence"][0]["source_id"] = "Observation/other"
    elif fault == "missing_evidence":
        changed["batch"]["questions"][0]["evidence"] = []
    elif fault == "wrong_type":
        changed["batch"]["questions"][0]["task_type"] = TYPES[2]
    else:
        changed["batch"]["questions"][0]["execution_requirement"] = "write"
    result, calls, _ = await run_script(monkeypatch, [changed, review([True])])

    assert result.status == "completed" and len(calls) == 2
    assert result.output == {"questions": [changed["batch"]["questions"][1]]}
    assert result.trace[-1]["type"] == "drops"
    assert result.trace[-1]["drops"][0]["index"] == 0
```

保留 `changed_passed` 与 `deleted_passed` 两个用例，改为断言运行不再整批失败（内容回填由 Task 6 覆盖）：

```python
@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ["changed_passed", "deleted_passed"])
async def test_passed_candidate_is_not_fatal(monkeypatch, fault):
    original = draft()
    changed = deepcopy(original)
    if fault == "changed_passed":
        changed["batch"]["questions"][0]["reference_answer"] = "改动已通过答案"
    else:
        changed["batch"]["questions"] = changed["batch"]["questions"][1:]
        changed["direction_indices"] = [1]
    result, calls, _ = await run_script(monkeypatch,
        [original, review([True, False]), changed, review([True, True])])

    assert result.status == "completed"
    assert len(calls) == 4
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/method/test_selfgen.py -q -k "bad_draft_question or passed_candidate_is_not_fatal"`
Expected: FAIL，`assert result.status == "completed"` 得到 `"failed"`

- [ ] **Step 3: 重写 `validate_draft`**

替换 `methods/selfgen/__init__.py` 中的 `validate_draft`：

```python
def validate_draft(value, trajectory, directions):
    """Keep every candidate whose own fields hold; drop the ones that do not."""
    current = DirectedBatch.model_validate(value)
    slots = current.direction_indices
    questions = current.batch.questions
    allowed = {source.source_id for source in trajectory.sources}
    kept_slots: list[int] = []
    kept_questions: list[GeneratedQuestion] = []
    drops: list[dict] = []
    used: set[int] = set()
    for position, question in enumerate(questions):
        slot = slots[position] if position < len(slots) else None
        if slot is None or slot not in (0, 1):
            drops.append({"index": position,
                          "reason": "direction index is missing or outside the two positions"})
            continue
        if slot in used:
            drops.append({"index": position, "reason": "direction index is duplicated"})
            continue
        probe = validate_task_batch(
            RunResult(output={"questions": [question.model_dump(mode="json")]},
                      status="completed", trace=[], usage={}),
            [directions[slot]], max_questions=1)
        if probe.status != "completed":
            drops.append({"index": position, "reason": probe.error or "output rejected"})
            continue
        if not question.evidence or any(e.source_id not in allowed for e in question.evidence):
            drops.append({"index": position,
                          "reason": "candidate evidence is absent or outside patient sources"})
            continue
        used.add(slot)
        kept_slots.append(slot)
        kept_questions.append(question)
    return (DirectedBatch(batch=TaskBatch(questions=kept_questions),
                          direction_indices=kept_slots), drops)
```

在 `methods/selfgen/__init__.py` 顶部把 `TaskBatch` 的导入改为同时导入题目类型：

```python
from contracts import (
    EnvironmentConfig,
    GeneratedQuestion,
    GenerationDirection,
    PatientTrajectory,
    RunResult,
    RuntimeConfig,
    TaskBatch,
)
```

- [ ] **Step 4: 运行 `validate_draft` 相关测试确认通过**

Run: `python -m pytest tests/method/test_selfgen.py -q -k bad_draft_question`
Expected: PASS

- [ ] **Step 5: 在 `generate_tasks` 中接住 drops**

在 `generate_tasks` 中，`budget = GenerationBudget(runtime, environment.name)` 之后加入：

```python
    notes: list[dict] = []

    def note(kind: str, **payload) -> None:
        notes.append({"type": kind, **payload})

    def closing(result: RunResult) -> RunResult:
        if not notes:
            return result
        return result.model_copy(update={"trace": [*result.trace, *notes]})
```

把草稿段的调用替换为：

```python
        run = await call("candidate_generation", DRAFT + constraints, context, DirectedBatch)
        if run.status != "completed":
            return failed_run(run)
        current, draft_drops = validate_draft(run.output, trajectory, directions)
        if draft_drops:
            note("drops", stage="candidate_generation", drops=draft_drops)
        if not current.batch.questions:
            return closing(budget.finish(directions, output=current.batch))
```

并把函数末尾 `except (ValueError, TypeError)` 分支与 `final_filter` 分支的返回改为 `closing(...)`：

```python
                record(budget, "final_filter", {"review_number": number,
                    "kept_indices": passed, "feedback": review.model_dump(mode="json"),
                    "output": final.model_dump(mode="json")})
                return closing(budget.finish(directions, output=final))
```

```python
    except (ValueError, TypeError) as exc:
        record(budget, "validation_failed", {"error": str(exc)}, status="failed", error=str(exc))
        return closing(budget.finish(directions, status="failed", error=str(exc)))
```

- [ ] **Step 6: 让 `validate_revision` 只保留「不得新增方向」**

把 `validate_revision` 替换为：

```python
def validate_revision(previous, revised):
    """A revision may drop or keep directions, never introduce a new one."""
    before = set(previous.direction_indices)
    if not set(revised.direction_indices) <= before:
        raise ValueError("Revision added a direction")
```

并把调用处改为：

```python
            revised, refine_drops = validate_draft(run.output, trajectory, directions)
            validate_revision(current, revised)
            if refine_drops:
                note("drops", stage=f"refine/{number}", drops=refine_drops)
            current = revised
```

- [ ] **Step 7: 运行测试确认通过**

Run: `python -m pytest tests/method/test_selfgen.py -q -k passed_candidate_is_not_fatal`
Expected: PASS

- [ ] **Step 8: 运行全量 selfgen 测试**

Run: `python -m pytest tests/method/test_selfgen.py -q`
Expected: PASS（`test_revision_cannot_restore_deleted_direction` 保持不变并通过）

- [ ] **Step 9: 提交**

```bash
git add methods/selfgen/__init__.py tests/method/test_selfgen.py
git commit -m "feat(selfgen): 草稿校验改为逐题丢弃

草稿中出现方向索引不匹配、题型越界、引用越界或字段不合规时只丢弃
该题，其余题目照常进入自审；止损记录写入 trace 的 drops 条目。"
```

---

### Task 5: SelfGen 自审按已判结果收敛

**Files:**
- Modify: `methods/selfgen/__init__.py`（`validate_review`、`generate_tasks`）
- Test: `tests/method/test_selfgen.py`

**Interfaces:**
- Produces: `validate_review(value) -> BatchReview`（只拒绝自相矛盾的判定）；`passed_indices(review, size) -> tuple[list[int], list[dict]]`，未判到的候选记为未通过并进入 drops

- [ ] **Step 1: 改写既有测试**

替换 `tests/method/test_selfgen.py` 的 `test_bad_feedback_does_not_trigger_repair`：

```python
@pytest.mark.asyncio
@pytest.mark.parametrize("bad", [
    {"questions": [dict(candidate_index=0, passed=True, issues=[])]},
    {"questions": [dict(candidate_index=0, passed=True, issues=[])] * 2},
    {"questions": [dict(candidate_index=i, passed=True, issues=[]) for i in (0, 2)]},
])
async def test_unjudged_candidates_are_treated_as_failed(monkeypatch, bad):
    result, calls, _ = await run_script(monkeypatch, [draft(), bad, draft(version="一"),
                                                     review([True, True])])

    assert result.status == "completed"
    assert len(calls) == 4
    reasons = [item["reason"] for entry in result.trace if entry.get("type") == "drops"
               for item in entry["drops"]]
    assert "candidate was not reviewed" in reasons

@pytest.mark.asyncio
@pytest.mark.parametrize("bad", [
    {"questions": [dict(candidate_index=i, passed=True, issues=["有问题"]) for i in (0, 1)]},
    {"questions": [dict(candidate_index=i, passed=False, issues=[]) for i in (0, 1)]},
])
async def test_self_contradicting_review_still_fails(monkeypatch, bad):
    result, calls, _ = await run_script(monkeypatch, [draft(), bad])

    assert result.status == "failed" and result.output is None
    assert len(calls) == 2
```

同时修正 `test_empty_fault_and_budget` 中的 `[draft(), {"questions": []}]` 用例：

```python
    ([draft(), {"questions": []}, draft(version="一"), review([True, True])], 6, "completed", 4),
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/method/test_selfgen.py -q -k "unjudged_candidates or self_contradicting"`
Expected: FAIL，`unjudged_candidates` 得到 `"failed"`

- [ ] **Step 3: 重写 `validate_review`**

```python
def validate_review(value):
    """Reject self-contradicting verdicts; leave coverage gaps to the caller."""
    review = BatchReview.model_validate(value)
    for item in review.questions:
        if any(not issue.strip() for issue in item.issues):
            raise ValueError("Review issue must be nonempty")
        if item.passed != (not item.issues):
            raise ValueError("Review verdict conflicts with issues")
    return review
```

- [ ] **Step 4: 加 `passed_indices` 辅助**

```python
def passed_indices(review, size):
    """Indices the review passed; every candidate it did not judge counts as failed."""
    judged: dict[int, QuestionReview] = {}
    drops: list[dict] = []
    for item in review.questions:
        if not 0 <= item.candidate_index < size:
            drops.append({"index": item.candidate_index, "reason": "review index is outside the candidates"})
            continue
        if item.candidate_index in judged:
            drops.append({"index": item.candidate_index, "reason": "candidate was reviewed twice"})
            continue
        judged[item.candidate_index] = item
    passed = sorted(index for index, item in judged.items() if item.passed)
    for index in range(size):
        if index not in judged:
            drops.append({"index": index, "reason": "candidate was not reviewed"})
    return passed, drops
```

- [ ] **Step 5: 接进自审循环**

把 `generate_tasks` 中自审段替换为：

```python
            run = await call(f"self_review/{number}", REVIEW, body, BatchReview)
            if run.status != "completed":
                return failed_run(run)
            review = validate_review(run.output)
            passed, review_drops = passed_indices(review, len(current.batch.questions))
            if review_drops:
                note("drops", stage=f"self_review/{number}", drops=review_drops)
            if len(passed) == len(current.batch.questions) or number == 3:
```

- [ ] **Step 6: 运行测试确认通过**

Run: `python -m pytest tests/method/test_selfgen.py -q`
Expected: PASS

- [ ] **Step 7: 提交**

```bash
git add methods/selfgen/__init__.py tests/method/test_selfgen.py
git commit -m "feat(selfgen): 自审未覆盖的候选按未通过处理

自审未逐题覆盖时不再作废整批，已判为通过的照收，未判到的按未通过
处理并记入 trace；判定自相矛盾仍整批失败。"
```

---

### Task 6: SelfGen 已通过题按位置合并

**Files:**
- Modify: `methods/selfgen/__init__.py`（新增 `merge_passed`、`generate_tasks` 的修订段）
- Test: `tests/method/test_selfgen.py`

**Interfaces:**
- Consumes: Task 4 的 `validate_draft`、Task 5 的 `passed_indices`
- Produces: `merge_passed(previous: DirectedBatch, passed: list[int], revised: DirectedBatch) -> tuple[DirectedBatch, list[int]]`，第二个返回值为被回填为存档版本的方向位置

- [ ] **Step 1: 写失败测试**

追加到 `tests/method/test_selfgen.py`：

```python
@pytest.mark.asyncio
async def test_passed_question_is_restored_from_the_archive(monkeypatch):
    original = draft()
    changed = deepcopy(original)
    changed["batch"]["questions"][0]["reference_answer"] = "改动已通过答案"
    changed["batch"]["questions"][0]["evidence"][0]["statement"] = "改动已通过证据"
    result, calls, _ = await run_script(monkeypatch,
        [original, review([True, False]), changed, review([True, True])])

    assert result.status == "completed"
    assert result.output == original["batch"]
    assert len(calls) == 4
    merges = [entry for entry in result.trace if entry.get("type") == "revision_merge"]
    assert merges == [{"type": "revision_merge", "round": 1, "restored": [0]}]


@pytest.mark.asyncio
async def test_passed_question_deleted_by_revision_is_restored(monkeypatch):
    original = draft()
    changed = deepcopy(original)
    changed["batch"]["questions"] = changed["batch"]["questions"][1:]
    changed["direction_indices"] = [1]
    result, calls, _ = await run_script(monkeypatch,
        [original, review([True, False]), changed, review([True, True])])

    assert result.status == "completed"
    assert result.output == original["batch"]
    merges = [entry for entry in result.trace if entry.get("type") == "revision_merge"]
    assert merges == [{"type": "revision_merge", "round": 1, "restored": [0]}]
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/method/test_selfgen.py -q -k restored_from_the_archive`
Expected: FAIL，`result.output` 里第 0 题是 `"改动已通过答案"`

- [ ] **Step 3: 实现合并**

在 `methods/selfgen/__init__.py` 中新增：

```python
def merge_passed(previous, passed, revised):
    """Restore every passed direction from the archived batch, whatever the revision returned."""
    before = dict(zip(previous.direction_indices, previous.batch.questions, strict=False))
    after = dict(zip(revised.direction_indices, revised.batch.questions, strict=False))
    restored: list[int] = []
    for index in passed:
        if index >= len(previous.direction_indices):
            continue
        slot = previous.direction_indices[index]
        if slot not in after or after[slot] != before[slot]:
            restored.append(slot)
        after[slot] = before[slot]
    order = [slot for slot in previous.direction_indices if slot in after]
    order += [slot for slot in revised.direction_indices if slot in after and slot not in order]
    return (DirectedBatch(batch=TaskBatch(questions=[after[slot] for slot in order]),
                          direction_indices=order), restored)
```

- [ ] **Step 4: 接进修订段**

把修订段替换为：

```python
            run = await call(f"refine/{number}", REFINE + constraints,
                {**body, "feedback": review.model_dump(mode="json")}, DirectedBatch)
            if run.status != "completed":
                return failed_run(run)
            revised, refine_drops = validate_draft(run.output, trajectory, directions)
            validate_revision(current, revised)
            revised, restored = merge_passed(current, passed, revised)
            if refine_drops:
                note("drops", stage=f"refine/{number}", drops=refine_drops)
            if restored:
                note("revision_merge", round=number, restored=restored)
            current = revised
            if not current.batch.questions:
                return closing(budget.finish(directions, output=current.batch))
```

- [ ] **Step 5: 运行测试确认通过**

Run: `python -m pytest tests/method/test_selfgen.py -q -k "restored_from_the_archive or deleted_by_revision"`
Expected: PASS

- [ ] **Step 6: 运行全量测试**

Run: `python -m pytest tests/method tests/experiment -q`
Expected: PASS

- [ ] **Step 7: 提交**

```bash
git add methods/selfgen/__init__.py tests/method/test_selfgen.py
git commit -m "feat(selfgen): 已通过题按位置合并存档版本

修订阶段不再比对模型重抄的已通过题是否逐字段一致，改为按方向位置
用存档版本回填，回填动作记入 trace 的 revision_merge 条目。"
```

---

### Task 7: 单批重跑对比验收

**Files:**
- 无需修改代码
- 产出：`docs/generation-failure-isolation/acceptance.md`

**Interfaces:**
- Consumes: 前六个任务的代码改动
- Produces: 验收结论

- [ ] **Step 1: 记录改动前的基线**

Run:

```bash
cd /home/gwh/code/MedEvolve
.venv/bin/python - <<'EOF'
import json
for path in [
    "outputs/nuc-full-experiment-20261007/full-quarter/gpt-5.4-nano-db788da5/medagentbench/workflow/batch-01/attempt-01/summary.json",
    "outputs/nuc-full-experiment-20261007/full-quarter/gpt-5.4-nano-db788da5/physicianbench/selfgen/batch-01/attempt-01/summary.json",
]:
    ev = json.load(open(path))["evaluation"]
    print(path.split("/full-quarter/")[1], ev["generation_statuses"], ev["actual_questions"])
EOF
```

Expected: Workflow 批为 `{"completed": 56, "failed": 8}` 与 `8`；SelfGen 批 `failed` 为 `29`、`actual_questions` 为 `11`。

- [ ] **Step 2: 由基线配置派生新的批次配置**

两条基线配置位于：

- `.matrix-configs/gpt-5.4-nano-db788da5/medagentbench/workflow/batches/full-medagentbench-workflow-batch-01.json`
- `.matrix-configs/gpt-5.4-nano-db788da5/physicianbench/selfgen/batches/full-physicianbench-selfgen-batch-01.json`

分别复制为 `outputs/nuc-full-experiment-20261007/failure-isolation/workflow-batch-01.json` 与 `selfgen-batch-01.json`，只改两个字段：`run_id` 加后缀 `-isolation`，`output_dir` 指向 `outputs/nuc-full-experiment-20261007/failure-isolation/<模型>/<环境>/<方法>/batch-01`。其余字段（模型、环境、患者列表、`random_seed`）保持不变。

- [ ] **Step 3: 重跑两个批次**

```bash
cd /home/gwh/code/MedEvolve
set -a; . /home/gwh/.config/medevolve/api.env; set +a
.venv/bin/python -u -m scripts.run_experiment_matrix \
  --config outputs/nuc-full-experiment-20261007/failure-isolation/workflow-batch-01.json
.venv/bin/python -u -m scripts.run_experiment_matrix \
  --config outputs/nuc-full-experiment-20261007/failure-isolation/selfgen-batch-01.json
```

Expected: 两条命令均打印 `{"status": "completed", ...}`。

- [ ] **Step 4: 对比并确认验收条件**

```bash
.venv/bin/python - <<'EOF'
import json
for path in [
    "outputs/nuc-full-experiment-20261007/failure-isolation/gpt-5.4-nano-db788da5/medagentbench/workflow/batch-01/attempt-01/summary.json",
    "outputs/nuc-full-experiment-20261007/failure-isolation/gpt-5.4-nano-db788da5/physicianbench/selfgen/batch-01/attempt-01/summary.json",
]:
    ev = json.load(open(path))["evaluation"]
    print(path.split("/failure-isolation/")[1], ev["generation_statuses"], ev["actual_questions"])
    print("  errors:", [e["error"][:60] for e in ev["generation_errors"]])
EOF
```

Expected:

- Workflow 批的 `failed` 只对应 `every generation position failed` 与 `no focus survives source validation`
- SelfGen 批的 `failed` 为 0 或个位数
- 两批 `actual_questions` 均不低于 Step 1 的基线

- [ ] **Step 5: 核对 trace 留痕数量**

```bash
.venv/bin/python - <<'EOF'
import json
for path in [
    "outputs/nuc-full-experiment-20261007/failure-isolation/gpt-5.4-nano-db788da5/medagentbench/workflow/batch-01/attempt-01/runs.json",
    "outputs/nuc-full-experiment-20261007/failure-isolation/gpt-5.4-nano-db788da5/physicianbench/selfgen/batch-01/attempt-01/runs.json",
]:
    drops = merges = 0
    for run in json.load(open(path)):
        for entry in (run.get("result") or {}).get("trace") or []:
            if entry.get("type") == "drops":
                drops += len(entry["drops"])
            if entry.get("type") == "revision_merge":
                merges += 1
    print(path.split("/failure-isolation/")[1], "drops:", drops, "revision_merge:", merges)
EOF
```

Expected: Workflow 批的 `drops` 数量等于基线 8 次失败所对应的位置数；SelfGen 批的 `revision_merge` 次数等于基线中 `Revision changed or removed a passed candidate` 的次数（该批为 13）。

- [ ] **Step 6: 写验收记录并提交**

把基线与新批次的 `failed`、`actual_questions`、`drops` 与 `revision_merge` 计数写入 `docs/generation-failure-isolation/acceptance.md`，然后：

```bash
git add docs/generation-failure-isolation/acceptance.md
git commit -m "docs(generation): 记录失败隔离的单批重跑验收结果"
```
