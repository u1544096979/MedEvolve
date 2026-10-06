# 内部检查的记录表示

## 问题

内部检查原先用 `trajectory.sources[i].statement`（环境原始 FHIR JSON）比对证据声明，而定位与两个生成节点只拿到 `trajectory.text`（精简后的轨迹正文）与 `source_id`。检查端与生成端看到的不是同一份记录，生成端写出的声明在轨迹正文里成立，却被按原始 JSON 判为"引用与记录不符"而丢弃。

证据：`outputs/workflow-medagentbench-validation-20261006-01`，两道候选题被全部 drop，理由均为"数据和记录不一致或缺失"。

## 设计

内部检查的证据映射取轨迹正文中该 source_id 的记录内容，与生成端使用同一份表示。

- `methods/workflow/__init__.py` 提供 `_record_blocks(trajectory)`：按 `"\n\n"` 切分 `text`，与 `sources` 顺序配对，返回 `{source_id: 记录正文块}`。
- 检查阶段的证据映射使用该函数的返回值，替换原来的 `source.statement`。映射缺失时取到 `None`，按提示词既有规则"source_id 缺失记录时视为引用不成立"处理，不新增分支。
- `methods/workflow/prompts.py` 的 `INTERNAL_CHECK`：记录来源改述为"该 source_id 在患者轨迹正文中的记录内容（与生成阶段同一份）"；判定末尾补充"不得因缺少 id、meta、resourceType 等元数据字段判引用不成立"。

生成端输入、检查输出结构（`BatchReview` 的三个布尔）与公共层、公共评测均不变。轨迹截断不属于本设计范围。

## 验收

- `tests/method/test_workflow.py` 与 `tests/experiment/test_workflow.py` 全部通过（fixture 与 payload 断言改为轨迹正文表示）。
- `internal_check` 的 `dropped` 不再出现因记录表示不同导致的"引用与记录不符"理由。
