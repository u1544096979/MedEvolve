# TaskCraft 医疗适配说明

## 1. 源码基线与许可

- 参考仓库：`/Users/guagua/code/TaskCraft`，提交 `1e405e0b02c4738654e2d885b16677b46c9fe969`。
- 论文：[TaskCraft: Automated Generation of Agentic Tasks](https://arxiv.org/abs/2506.10055)。
- 许可：仓库 LICENSE 为 MIT（Copyright (c) 2025 OPPO），`taskcraft/gen_*.py` 文件头标注 Apache License 2.0。提示词副本与许可文本保存在 `methods/taskcraft/LICENSE-TaskCraft`。
- 提示词基线：`methods/taskcraft/prompt_text.py` 保存 `taskcraft/prompts/*.yaml` 的逐字文本，`SOURCES` 给出每个提示词的来源文件与键名。
- 模块归属：`methods/taskcraft/` 与方法测试 `tests/method/test_taskcraft.py`、`tests/method/test_taskcraft_live.py`。

## 2. 阶段映射

| 源码 | 模块 | 说明 |
|---|---|---|
| `gen_atomic_task.py:get_candidate_tasks`、`get_initial_conclusion` | `stages.Pipeline.atomic` | 结论抽取，按 `chunk_size` 切分后逐块调用 |
| `gen_atomic_task.py:generate_initial_question`、`clean_qa` | `stages.Pipeline._initial_question` | 由 ID/R/A 出题，再用 `refined_answer` 作为参考答案 |
| `gen_atomic_task.py:generate_initial_questions`、`filter_questions`、`filter_atomic_tasks` | `stages.Pipeline.atomic` 的生成与核验循环 | `max_candidate_atomic` 上限、内置过滤分支 |
| `gen_depth_based_task.py:DepthExtend.backward` | `stages.Pipeline._backward` | 反向检索 superset、`is_superset_valid` 校验、生成新问题 |
| `gen_depth_based_task.py:DepthExtend.verify`、`process_single_task`、`depth_extend` | `stages.Pipeline._extend_depth`、`_merge` | 逐跳核验、合并、`valid_hop` 与 `max_hops` 控制 |
| `gen_width_based_task.py:width_extend`、`check_queries` | `stages.Pipeline.width`、`_check_merged` | 合并问题、可分解性检查、无工具作答检查 |
| `agent_tools.py:VerifyAgent.forward`、`recall_score` | `verify.verify_query`、`score_recall` | 无工具回答判原子性 + 病历工具 Agent 回答评分 |
| `agent_tools.py:SearchAgent` 描述 | `prompts.MEDICAL_RECORD_AGENT` | 检索范围由网络改为当前患者病历 |
| `agent_tools.py:BaseAgent.capture_trajectory` | `verify._trajectory` | 阶段内保留紧凑轨迹；完整运行记录在公共 `RunResult.trace` |
| `utils.py:run_llm_prompt`、`safe_json_loads` | 公共 `runtime.call_llm`、`run_agent` | 结构化输出替代文本 JSON 解析与重试 |
| `data_processor.py:split_text_intelligently` | `stages.split_text` | 同分隔符优先级（`\n\n`、`\n`、`.`、空格、字符），无重叠 |
| `gen_atomic_task.py:gen_atomic_tasks`、`read_indentifier_language`、`polish_readed_context` | 无对应 | 入口由公共 `generate_tasks` 承担；文件、图片与 URL 读取不适用 |

阶段顺序：原子 → 宽度 → 深度。源码三个入口是独立脚本，没有编排器；宽度先于深度执行，使单次调用的合并机制先于多跳 Agent 检索完成。

## 3. 工具、输入与证据口径

| 源码 | 适配后 |
|---|---|
| `CrawlerSearchTool`、`CrawlerReadTool`、`CrawlerArchiveSearchTool`、`VisualInspectorTool` | 当前患者 9 个只读 FHIR 工具：`fhir_condition_search_problems`、`fhir_observation_search_labs`、`fhir_observation_search_vitals`、`fhir_patient_search_demographics`、`fhir_procedure_search_orders`、`fhir_medication_request_search_orders`、`fhir_document_reference_search_clinical_notes`、`fhir_service_request_search`、`fhir_observation_search_social_history` |
| 写工具（`fhir_*_create`、`fhir_*_create_message`） | 生成阶段不绑定，留给评测阶段解题 Agent |
| 文档文件、PDF、网页 | `PatientTrajectory.text`，按记录时间排序的记录块 |
| `content_identifier`（文献或网页标题） | 病历记录块标签 `[记录时间] 资源类型/资源ID`（`prepare_trajectory` 的格式） |
| `golden_answer` | 原子题为 `clean_qa` 的 `refined_answer`；扩展题见第 4 节 |
| — | 结论抽取额外返回 `source_ids`；映射为 `Evidence(source_id, statement=结论)`，来源不在当前轨迹内即丢弃该候选 |

病历工具按 `patient`/`subject` 参数鉴权，本次开放的患者标识随 Agent 说明一起下发，避免使用记录正文中的其他标识。

## 4. 适配差异与源码缺陷对照

1. 原子过滤判据。源码 `filter_question`（`gen_atomic_task.py:418-444`）仅在核验返回 `error` 时拒绝，更严格的 `llm_score`、`agent_score` 判断在 `425-433` 被注释掉。适配保持该分支：无工具回答满分或核验出错才拒绝，`agent_score` 只记录不用于拒绝。患者私有事实本就不在模型自身的知识内，该判据在病历域很少触发，实际筛选由「病历 Agent 能否作答」承担。
2. superset 校验。源码把返回的 JSON 字典与字符串 `"invalid"` 比较（`gen_depth_based_task.py:67`），字典与该字符串不相等。适配按提示词声明的 `{"new_query": "valid"|"invalid"}` 判断，仅显式 `invalid` 时拒绝。
3. 宽度结果组装。源码最终映射（`gen_width_based_task.py:234-241`）引用循环外的 `golden_answers`、`original_questions`，与当前分组不一致。适配按每题自身的答案与子问题组装。
4. 宽度合并输入。源码把 `golden_answer` 一并送入合并提示词（`gen_width_based_task.py:184-188`，与注释 `without golden_answers` 不一致）。适配保持源码行为；合并结果由后续可分解性与作答检查把关。
5. 参考答案组织。`TaskBatch` 只有单个 `reference_answer`。深度题按源码 `solve_trajectory` 顺序写入「最终答案 + 子问题与中间答案（自外向内）」；宽度题写入「合并题参考答案：各子问题 → 答案」。子问题内容与顺序取自对应源码阶段，不另行改写。
6. 题型。源码没有题型采样。出题提示词追加题型约束，取值限于 `directions` 中的题型；扩展题继承父题题型，跨题型合并取第一父题的题型。批量输出按方向各取一题，同一方向内优先顺序为宽度、深度、原子。
7. 预算边界。`max_candiated_conclusions`、`max_candidate_atomic`、`extended_attempts`、`max_hops`、`num_parallel_predictions`、`merge_attempts`、`query_attempts` 以 `TaskCraftParameters` 显式给出。`atomic_budget_share` 限制原子阶段使用的生成预算比例，`extension_guard_turns` 控制剩余轮数不足时不再发起新的扩展跳，使源码循环在共享预算内结束。
8. Agent 步数。源码 `max_step`（10/20）是单次 Agent 调用的步数上限，公共 `run_agent` 只接受累计剩余预算，无法在单次调用内设置。适配以累计工具预算替代，实测单次反向检索可达 48 轮模型调用。
9. 评分采样。`recall_score` 的 3 次并行判定在公共层顺序执行（`judge_samples=3`），判定提示词与评分区间不变，成本逐次计入累计预算。
10. 无工具回答路径。源码该路径不传 system 指令，公共 `call_llm` 固定携带 system 消息，适配使用最小指令（`LLM_ONLY_INSTRUCTIONS`）保持「无工具、仅自身知识」的语义。
11. 提示词。源码正文逐字保留，医疗适配以追加约束块实现；正文中的文献与网页示例保留用于说明字段格式，追加块声明本任务领域是病历记录。
12. `clean_qa` 输入由 Python 字典的 f-string 改为 JSON 文本，字段不变。
13. 患者标识。病历工具按参数鉴权，本次开放的患者标识写入 Agent 说明。

## 5. 预算要求与实测用量

`TaskCraftParameters` 默认值取自源码签名：`chunk_size=16384`、`max_candidate_conclusions=20`、`max_candidate_atomic=10`、`judge_samples=3`、`question_attempts=3`、`query_attempts=3`、`merge_attempts=5`、`extended_attempts=4`、`max_hops=2`、`width_attempts=3`；适配新增 `atomic_budget_share=0.5`、`extension_guard_turns=25`。`max_candidate_atomic` 与生成预算共同决定单批候选数，小样本运行按预算取 3。

实测（本机 Docker，种子 `fhir-full:v1`，患者 `MRN4460967357`，`gpt-4o-mini`，生成预算 150 轮 / 150 次工具）：

| 项 | 数值 |
|---|---|
| 生成状态 | completed，148 模型轮 / 130 次工具 / 5.85M tokens |
| 分阶段模型轮数 | 原子 1+3+3+36，宽度 9，深度反向检索 96 |
| 产出 | 1 题（宽度合并题），证据 `source_id` 全部属于当前患者 |
| 公共评测 | 核验 `valid=true`，四维质量全 yes，解题未完成 |
| 运行目录 | `outputs/taskcraft-live-20261003-094909` |

配置要求：生成预算低于该量级时，原子阶段的内置核验无法完成，输出为空批，失败原因保留在 `trace`。深度阶段依赖公共层提供单次 Agent 调用的步数上限；解题 Agent 在同类大 snapshot 患者上出现过 131k token 超过 128k 上下文，需在公共层收窄工具返回量。

验证方式：`pytest tests/method/test_taskcraft.py`（替身运行层，覆盖阶段机制与共享边界）；`tests/method/test_taskcraft_live.py` 为可选端到端检查，配置 `MEDEVOLVE_TASKCRAFT_MANAGER`、`MEDEVOLVE_TASKCRAFT_PATIENT`、`MEDEVOLVE_TASKCRAFT_BUDGET` 后运行。

## 6. 未映射能力与影响

1. PDF、图片、URL 读取与多模态分析（`read_indentifier_language`、`polish_readed_context`、`select_images`、`analyze_image`、`download_if_url`）。输入固定为已准备的 `PatientTrajectory`，不影响原子、深度、宽度与内置核验的执行。
2. 语言判定与翻译（`langid`、`trans_language`）。`trans_language` 在源码主流程中未被调用；题目与答案语言跟随病历文本。
3. `langchain` 的 `RecursiveCharacterTextSplitter`。改为同分隔符优先级的贪心切分，`chunk_overlap` 为 0。
4. `oagents` 框架、爬虫密钥与 `.env` 检查（`check_envs`）、本地 Python 执行器与 E2B。由公共 `run_agent`、`call_llm` 与病历工具替代。
5. `augmented_question_prompt` 在源码深度阶段被载入但未使用，适配同样不调用；`score_prompt_both`、`acc_prompt_single`、`extract_answer`、`run_llm_msg` 属于源码未接入主流程的路径。
