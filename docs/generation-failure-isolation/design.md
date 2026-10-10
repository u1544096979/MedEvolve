# 生成失败隔离设计

## 1. 背景与现象

实验用两种方法从病历生成考题，每位患者目标 2 题。两种方法的流程都是「LLM 调用 + 校验」串联：

```
Workflow: locate → generate_0 → generate_1 → check → 输出
SelfGen:  candidate_generation → self_review/1 → refine/1 → self_review/2 → refine/2 → self_review/3 → final_filter
```

当前实现在任一校验不通过时把整个患者作废，因此「某一道题不合格」会升级成「这位患者不能出题」。Nano 在两个环境都严重低产。

全季度产物统计（`outputs/nuc-full-experiment-20261007/full-quarter/`）：

| 方法 | 患者运行 | 失败运行 | 候选/草稿 | 保留 | 整批失败丢弃 |
|---|---:|---:|---:|---:|---:|
| Workflow | 312 | 22 | 594 | 285 | 15 |
| SelfGen | 312 | 109 | 609 | 231 | 208 |

SelfGen 丢弃的 208 道草稿中有 50 道已经通过自审。失败集中在 Nano：SelfGen 的 109 次失败中 Nano 占 75 次。

SelfGen 的 109 次失败按校验项分布：

| 校验项 | 次数 |
|---|---:|
| 修订改写或删除了已通过的题 | 45 |
| 方向索引与题目数量不匹配 | 18 |
| 题型超出指定方向 | 14 |
| 题目改了自己的方向 | 13 |
| execution_requirement 不是 read_only | 7 |
| 引用缺失或超出患者来源 | 7 |
| 自审未覆盖全部候选 | 2 |
| 其他（空字段、新增方向、schema） | 3 |

## 2. 修复模式

### 模式 A：单题失败降级为丢弃，不中断批次

统一规则：校验不通过时丢弃出错的题或位置，其余继续；只有整批都不合格才判该患者失败。

#### Workflow 三处

**A1. locate 引用无效 source**

`_selected_focuses` 校验每个 focus 的 `source_ids` 是否都在轨迹来源内，任一不在即抛错。案例 14 例（Nano 9、Mini 5），全部为模型编造的 ID，且每例只有 1–2 个 ID 无效，其余 5–12 个有效。

例：MedAgentBench `S3084624` 引用 14 个 source_id，仅 `Condition/39763` 不存在（该患者记录中是 `Condition/39762` 与 `Condition/39764`），该患者产出 0 题。该例未发生输入截断，因此不是裁剪造成的。

改为：剔除无效 ID 后继续；某个 focus 剔空则丢弃该 focus；两个 focus 都无效才判该患者失败。

**A2. 单题生成失败连坐**

`generate_0` / `generate_1` 任一校验失败即 `_finish(status="failed")`，另一位置已经生成的题一并丢失。案例 3 例，丢 5 题。

例：PhysicianBench `MRN9988355352`，`generate_0` 已产出一道合格候选，`generate_1` 返回空 instruction/answer，该患者产出 0 题。

改为：位置级隔离。失败的位置记入 `drops`，成功的位置照常进入 check；两个位置都失败才判该患者失败。

**A3. check 漏判候选**

`_decisions` 要求 reviews 恰好覆盖每个候选一次，不满足即抛错。案例 5 例，丢 10 题。其中 4 例为漏判（模型只返回 index 0 的一条 review），1 例为返回的 JSON 被截断。

例：PhysicianBench `MRN7805745017`，两道候选均已生成，检查只返回 1 条 review，两道候选全部丢失。

改为：已判为保留的照常收取；未判到的候选按丢弃处理，并在 trace 中记录其 index。

#### SelfGen 五处

`validate_draft` 目前整体校验后抛错，五类问题各自作废整批。改为逐题校验、逐题丢弃：某题不合格就同时丢掉它的 `direction_indices` 与题目；全部丢完则返回空批次。

**A4. 方向索引与题目数量不匹配**（18 次）
`direction_indices` 与 `questions` 长度不一致，或索引重复。改为：丢掉没有配对索引的题，以及重复索引中后出现的题。

**A5. 题型超出指定方向**（14 次）
题目的 `task_type` 不在指定方向内，或同题型数量超出该方向的题数。改为：丢掉越界的题。

**A6. 题目改了自己的方向**（13 次）
题目的 `task_type` 与其 `direction_index` 指向的方向不一致。改为：丢掉该题。

**A7. execution_requirement 不是 read_only**（7 次）
改为：丢掉该题。不自动改写该字段，避免放行未经模型确认的输出。

**A8. 引用缺失或超出患者来源**（7 次）
题目的 `evidence` 为空，或引用了不在患者来源内的 source_id。改为：丢掉该题。

**A9. 自审未覆盖全部候选**（2 次）
`validate_review` 要求 review 覆盖每个候选一次。改为与 A3 一致：已判为通过的照收，未判到的按未通过处理。

### 模式 B：SelfGen 已通过题合并，不比对

`validate_revision` 要求模型在修订阶段重抄的已通过题与上一版逐字段完全相同，任一差异即抛错。案例 45 次运行，丢 90 道草稿，其中 45 道已通过自审。

例：PhysicianBench `evaluation/generate/22`，已通过的 slot 1 证据声明由「2022-12-25T22:44:00 Sapovirus RNA 检测：Not Detected。」变为「2022-12-25T22:44:00 Sapovirus RNA 检测结果为 Not Detected。」，参考答案结论「中等风险（中危）」前后逐字相同，整批仍然失败。

改为：不再比对。最终题按位置合并——已通过位置取存档版本，其余位置取模型修订版本。

### 保留整批失败的情形

降级只针对「输出内容不满足校验」。以下情形仍按现状整批失败：

- LLM 调用本身失败或预算耗尽
- locate 的两个 focus 都无效
- generate 的两个位置都失败
- check 调用本身失败（无输出）
- SelfGen 草稿的全部题目被丢弃（视为空批次正常结束，不是失败）
- SelfGen 的 review 判定自相矛盾（`passed` 与 `issues` 不一致）

## 3. 契约与测试影响

### 3.1 RunResult 状态语义

止损后的运行状态仍为 `completed`（确实产出了结果）。所有止损动作写进 trace，用于区分干净产出与抢救产出：

- Workflow：`drops` 记录被丢弃的位置、候选 index 与原因
- SelfGen：`drops` 记录被丢弃的题目索引与原因，`revision_merge` 记录被回填为存档版本的位置

### 3.2 需重写的既有测试

以下测试把整批失败编码为契约，需改为断言止损后继续：

| 测试 | 受影响的用例 |
|---|---|
| `tests/method/test_workflow.py::test_selected_focuses_reject_three_focuses_missing_source_and_blank_topic` | 缺失 source 一项 |
| `tests/method/test_workflow.py::test_invalid_node_output_fails_stage_before_check` | 全部 6 个用例 |
| `tests/method/test_workflow.py::test_duplicate_or_missing_verdicts_fail_instead_of_filtering` | 全部 3 个用例 |
| `tests/method/test_selfgen.py::test_source_and_passed_candidate_faults` | `missing_source`、`missing_evidence` |
| `tests/method/test_selfgen.py::test_bad_feedback_does_not_trigger_repair` | 前 3 个用例（覆盖不全）；后 2 个（判定自相矛盾）保持不变 |
| `tests/method/test_selfgen.py::test_empty_fault_and_budget` | `[draft(), {"questions": []}]` 一项 |

`test_failed_stage_stops_flow_and_keeps_cost`（provider 失败与预算）、`test_revision_cannot_restore_deleted_direction`、`test_empty_fault_and_budget` 的其余用例保持不变。

## 4. 验收标准

### 4.1 单元测试

- 注入无效 source、单位置生成失败、check 漏判：断言运行仍为 `completed`，产出为存活候选，trace 含对应丢弃记录
- 注入 A4–A8 五类草稿问题：断言只丢弃出错题目，其余题目照常进入自审
- 注入改写已通过题：断言产出中该位置等于存档版本，trace 含回填记录
- provider 失败、预算耗尽、两个 focus 都无效、两个位置都失败、review 判定自相矛盾：断言仍整批失败

### 4.2 单批重跑对比

对 `gpt-5.4-nano / medagentbench / workflow` 与 `gpt-5.4-nano / physicianbench / selfgen` 各重跑一批，对比：

- Workflow 的 `generation_statuses.failed` 只剩「两个 focus 都无效」与「两个位置都失败」的患者
- SelfGen 的失败运行从 75 次（Nano）降到 5 次以内
- `actual_questions` 不低于修改前
- 新增 trace 记录数与预期丢弃数一致
