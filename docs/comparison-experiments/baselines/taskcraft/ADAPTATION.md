# TaskCraft 医疗适配说明

## 1. 源码基线与许可

- 参考仓库：`/Users/guagua/code/TaskCraft`，提交 `1e405e0b02c4738654e2d885b16677b46c9fe969`。
- 论文：[TaskCraft: Automated Generation of Agentic Tasks](https://arxiv.org/abs/2506.10055)。
- 许可与归属：参考仓库 LICENSE 为 MIT（Copyright (c) 2025 OPPO）；被移植的 `taskcraft/gen_atomic_task.py`、`gen_depth_based_task.py`、`gen_width_based_task.py`、`agent_tools.py`、`utils.py`、`data_processor.py` 文件头声明 Apache License 2.0 并列有作者。两份许可声明与这些文件头的作者归属保存在 `methods/taskcraft/LICENSE-TaskCraft`。
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
| `agent_tools.py:VerifyAgent.forward`、`recall_score` | `verify.verify_query`、`score_recall` | 无工具回答判原子性 + 病历工具 Agent 回答评分；外层重试 3 次重走整段流程，评分取有效采样平均 |
| `agent_tools.py:SearchAgent` 描述 | `prompts.MEDICAL_RECORD_AGENT` | 检索范围由网络改为当前患者病历 |
| `oagents/agents.py:ToolCallingAgent._run`、`step`、`provide_final_answer` | `agent.run_search_agent`、`step_kind` | 第 1 步规划、每 `reflection_interval=5` 步反思、action 上限后仍发起强制终答；步骤类型与真实计数写入 trace |
| `oagents/agents.py:planning_step`、`reflection_step`（`prompts_type=w_think_reflect`） | `agent._SearchRun.planning`、`reflection` | 首次规划 2 次请求、反思 3 次请求（PRM 评分、事实更新、计划更新），两者都不占 action step |
| `oagents/memory.py:TaskStep/PlanningStep/ReflectionStep/ActionStep.to_messages`、`write_memory_to_string_messages` | `agent._memory_messages`、`agent._memory_string` | 记忆与工具结果按作者字段映射；反思的 PRM 使用字符串形式（含 512 字符截断） |
| `oagents/agent_types.py:handle_agent_output_types`、`toolcalling_agent.yaml:final_answer` | `agent._finish`、`agent._finish_output` | FINISH 答案数组映射 `AnswerList`/`SolverAnswer`，单对象映射声明的输出类型；`final_answer` 是虚拟工具，不进入患者工具列表 |
| `agent_tools.py:SearchAgent.forward(max_retries=3)` | `ledger.agent` | 检索 Agent 单次调用由调用方外层重试；`max_step` 映射为 `max_action_steps`（原子核验 20、深度反向与深度核验 10） |
| `utils.py:run_llm_prompt`、`safe_json_loads` | 公共 `runtime.call_llm` + `ledger.llm` | 结构化输出替代文本 JSON 解析；`max_retries` 映射为调用点参数 `attempts`，每次尝试自行记账后才重试 |
| `agent_tools.py:BaseAgent.capture_trajectory` | `verify._trajectory` | 按记忆步骤类型映射轨迹；原始模型响应、工具输入输出与每步计数保留在公共 `RunResult.trace` |
| `data_processor.py:split_text_intelligently` | `stages.split_text` | 同分隔符优先级（`\n\n`、`\n`、`.`、空格、字符），无重叠 |
| `gen_atomic_task.py:gen_atomic_tasks`、`read_indentifier_language`、`polish_readed_context` | 无对应 | 入口由公共 `generate_tasks` 承担；文件、图片与 URL 读取不适用 |

阶段顺序：原子 → 宽度 → 深度。源码三个入口是独立脚本，没有编排器；宽度先于深度执行，使单次调用的合并机制先于多跳 Agent 检索完成。

## 3. 工具、输入与证据口径

| 源码 | 适配后 |
|---|---|
| `CrawlerSearchTool`、`CrawlerReadTool`、`CrawlerArchiveSearchTool`、`VisualInspectorTool` | 当前患者只读 FHIR 工具声明，绑定取其中的非空子集：`fhir_condition_search_problems`、`fhir_observation_search_labs`、`fhir_observation_search_vitals`、`fhir_patient_search_demographics`、`fhir_procedure_search_orders`、`fhir_medication_request_search_orders`、`fhir_document_reference_search_clinical_notes`、`fhir_service_request_search`、`fhir_observation_search_social_history` |
| 写工具（`fhir_*_create`、`fhir_*_create_message`） | 生成阶段不绑定，留给评测阶段解题 Agent |
| 文档文件、PDF、网页 | `PatientTrajectory.text`，按记录时间排序的记录块 |
| `content_identifier`（文献或网页标题） | 病历记录块标签 `[记录时间] 资源类型/资源ID`（`prepare_trajectory` 的格式） |
| `golden_answer` | 原子题为 `clean_qa` 的 `refined_answer`；扩展题见第 4 节 |
| — | 结论抽取额外返回 `source_ids`；映射为 `Evidence(source_id, statement=结论)`，来源不在当前轨迹内即丢弃该候选 |

病历工具按 `patient`/`subject` 参数鉴权，本次开放的患者标识随 Agent 说明一起下发，避免使用记录正文中的其他标识。

MedAgentBench 使用原生文本动作协议，两种协议共用 `methods/taskcraft/environment.py` 的 `bind_generation_tools`：

| 项 | PhysicianBench | MedAgentBench |
|---|---|---|
| 工具绑定 | 方法声明的只读工具白名单与当前环境所提供工具名的交集，允许非空子集；白名单外的工具不绑定，交集为空视为环境不可用 | `ReadOnlyNativeTools` 只读视图：不修改原 `NativeTools` 对象及其患者客户端，只复制其中的 GET 工具声明 |
| 检索动作 | 函数调用，参数 `count`、`page_limit` | `GET <FHIR 接口路径>`，返回量用 `_count`，患者范围由公共环境的实例、分页与地址鉴权承担 |
| 写入动作 | 不绑定 `fhir_*_create` 类工具 | 放行清单只有 GET 与 FINISH，POST 等写动作在执行路径返回错误，不进入传输层 |
| 核验 Agent 输出 | `AnswerList` | `FINISH([...])` 答案列表，经 `SolverAnswer` 承载后解析 |
| 反向检索输出 | `BackwardContext` 对象 | 同样要求 `FINISH` 返回单个对象 |

只读绑定取本方法声明的只读工具白名单与当前环境所提供工具名的交集；白名单外的工具不绑定，工具数量不限；新环境需以这些只读工具名（或显式方法适配）提供检索接口，交集为空时返回 `EnvironmentUnavailable`；不凭 `_search` 之类的名称推断。检索能力与证据来源的映射不依赖固定工具数量，具备同等检索接口与来源映射的环境可直接接入。

## 4. 适配差异与源码缺陷对照

1. 原子过滤判据。源码 `filter_question`（`gen_atomic_task.py:418-444`）仅在核验返回 `error` 时拒绝，更严格的 `llm_score`、`agent_score` 判断在 `425-433` 被注释掉。适配保持该分支：无工具回答满分或核验出错才拒绝，`agent_score` 只记录不用于拒绝。患者私有事实本就不在模型自身的知识内，该判据在病历域很少触发，实际筛选由「病历 Agent 能否作答」承担。
2. superset 校验。源码把返回的 JSON 字典与字符串 `"invalid"` 比较（`gen_depth_based_task.py:67`），字典与该字符串不相等。适配按提示词声明的 `{"new_query": "valid"|"invalid"}` 判断，仅显式 `invalid` 时拒绝。
3. 宽度结果组装。源码最终映射（`gen_width_based_task.py:234-241`）引用循环外的 `golden_answers`、`original_questions`，与当前分组不一致。适配按每题自身的答案与子问题组装。
4. 宽度合并输入。源码把 `golden_answer` 一并送入合并提示词（`gen_width_based_task.py:184-188`，与注释 `without golden_answers` 不一致）。适配保持源码行为；合并结果由后续可分解性与作答检查把关。
5. 参考答案组织。`TaskBatch` 只有单个 `reference_answer`。深度题按源码 `solve_trajectory` 顺序写入「最终答案 + 子问题与中间答案（自外向内）」；宽度题写入「合并题参考答案：各子问题 → 答案」。子问题内容与顺序取自对应源码阶段，不另行改写。
6. 题型。源码没有题型采样。出题提示词追加题型约束，取值限于 `directions` 中的题型；扩展题继承父题题型，跨题型合并取第一父题的题型。批量输出按方向各取一题，同一方向内优先顺序为宽度、深度、原子。
7. 预算边界。`max_candiated_conclusions`、`max_candidate_atomic`、`extended_attempts`、`max_hops`、`num_parallel_predictions`、`merge_attempts`、`query_attempts` 以 `TaskCraftParameters` 显式给出。`atomic_budget_share` 限制原子阶段使用的生成预算比例，`extension_guard_turns` 控制剩余轮数不足时不再发起新的扩展跳，使源码循环在共享预算内结束。
8. Agent 步数与计数。源码 `ToolCallingAgent._run` 以 action step 计数（`max_step` 为 20，深度反向与深度核验为 10），第 1 步先规划、之后每 5 步反思；规划与反思不增加 action step，但每次都增加模型请求。适配保留该循环：`Ledger.agent(..., max_action_steps=)` 给出单次调用上限，规划与反思按 `run_search_agent` 的步骤调度执行，`usage.action_steps` 记录真实 action step 数，`verify.verify_query` 的 `agent_step_number` 取该值（不再取模型请求数）。源码 `traj_step_num()` 返回的 `self.step_number` 在终止时又递增一次，即实际 action step 数加一（达到上限时为 `max_step+1`），该串值与 `usage.step_number` 同时保留在运行结果中。
9. 动作编码。公共单次调用入口返回结构化 JSON，无法使用 SDK 的工具调用字段；一次 action 的编码是 `{"tool_calls": [...], "final_answer": ...}`（原生协议为 `{"action": "GET …"}`）。源码 `step` 的 JSON 回退分支（agents.py:1507-1511）只支持一个 `{"name", "arguments"}` 对象，SDK 分支也只执行 `tool_calls[0]`（agents.py:1534）；适配把同一次响应里的多个工具调用依次执行、逐个计入 `tool_calls`，使台账计数与真实调用一致。
10. 工具调用响应里的终答字段。源码在 `tool_calls` 非空时不解析消息内容，只执行 `tool_calls[0]` 并继续循环（agents.py:1500-1536）；内容解析与终答回退都只在 `tool_calls` 为空时发生（agents.py:1507-1528）。适配同样只在同一次响应不含工具调用时接受 `final_answer` 作为终答；该响应带工具调用时，工具调用全部执行并继续下一步。
11. 无工具调用的回退。源码 `step` 在既无工具调用、内容也不是 JSON 动作时，把模型输出直接当作终答文本；结构化输出下没有对应形态，适配将该情形记为生成失败（`action response carries no tool call`），由调用方的外层重试处理。
12. 终答解析。`final_answer` 的参数是答案 JSON；达到 action 上限后的 `provide_final_answer` 请求按源码要求返回 `<think>/<answer>`（`AgentThought`），再由 `agent._finish` 把 `answer` 解析为声明的输出结构，对应 `SearchAgent.forward` 的 `safe_json_loads`。原始 `oagents` 运行在此处返回文本，公共运行层需要 `output_type` 结构。
13. 工具调用记账。每个模型请求与每次工具调用都进入同一 `GenerationBudget.records`：工具调用单独记一条 `model_turns=0`、`tool_calls=1` 的记录，使累计工具预算真实递减，并在达到上限时按总工具预算阻断。工具调用不发起模型请求，该记录同时声明 `runs_without_known_tokens=0`，不把工具调用计为缺少 token 的运行。
14. Agent 记忆与模板渲染。规划与反思的请求把消息列表展平为单一输入文本（公共单次调用入口只接受 `instructions` 与 `input_text`），字段与顺序保持作者 `to_messages` 口径；`managed_agents` 为空，模板中的对应 Jinja 分支渲染为空文本，工具清单由 Jinja 循环改为预渲染。提示词逐字取自 `oagents/prompts/w_think_reflect/toolcalling_agent.yaml`。
15. 评分采样与失败样本。`recall_score` 的 3 次并行判定在公共层顺序执行（`judge_samples=3`），判定提示词与评分区间不变，成本逐次计入累计预算。单个样本的请求失败或输出无法比较时不计入平均值，其余有效样本继续求平均（源码在并行分支里会把整个评分置为 None）；全部无效时返回 None，无工具回答分支按源码的比较异常路径重试，病历工具回答分支按源码直接带回空分数，由原子过滤或深度扩展的对应分支处理。
16. 无工具回答路径。源码该路径不传 system 指令，公共 `call_llm` 固定携带 system 消息，适配使用最小指令（`LLM_ONLY_INSTRUCTIONS`）保持「无工具、仅自身知识」的语义。
17. 提示词。源码正文逐字保留，医疗适配以追加约束块实现；正文中的文献与网页示例保留用于说明字段格式，追加块声明本任务领域是病历记录。
18. `clean_qa` 输入由 Python 字典的 f-string 改为 JSON 文本，字段不变。
19. 患者标识。病历工具按参数鉴权，本次开放的患者标识写入 Agent 说明。
20. 原生协议分派。MedAgentBench 的工具列表不是函数工具，按名字过滤会得到空列表。生成阶段检测原生工具对象后按只读视图绑定，放行清单为 GET 与 FINISH；核验与反向检索的输出形态改为 `FINISH` 约定，机制分支与 PhysicianBench 一致。
21. 原生接口范围。MedAgentBench 声明含 6 个 GET 与 3 个 POST，没有病历文书检索接口；依赖文书内容的结论在该环境下候选更少，机制不变。

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

MedAgentBench 实测（同一脚本，患者 `S6534835`，同样预算）：

| 项 | 数值 |
|---|---|
| 轨迹 | 686 条记录、191064 字符；容器峰值 3.36 GiB / 4 GiB |
| 生成状态 | completed，70 模型轮 / 18 次工具调用 |
| 分阶段模型轮数 | 原子 1+3+3+27，宽度 2，深度 34 |
| 产出 | 1 题（原子题，证据 `MedicationRequest/780913`） |
| 公共评测 | 核验 `valid=true`；解题与评测模型调用出现 175k token 超过 128k 上下文 |
| 运行目录 | `outputs/taskcraft-live-20261004-040400` |

配置要求：生成预算低于该量级时，原子阶段的内置核验无法完成，输出为空批，失败原因保留在 `trace`。深度阶段按源码 `max_step=10` 限制单次检索的 action step 数，规划与反思的模型请求由累计预算承担；解题 Agent 在同类大 snapshot 患者上出现过 131k token 超过 128k 上下文，需在公共层收窄工具返回量。

验证方式：`pytest tests/method/test_taskcraft.py`（替身运行层，覆盖阶段机制与共享边界）；`tests/method/test_taskcraft_live.py` 为可选端到端检查，配置 `MEDEVOLVE_TASKCRAFT_MANAGER`、`MEDEVOLVE_TASKCRAFT_PATIENT`、`MEDEVOLVE_TASKCRAFT_BUDGET` 后运行。

## 6. 未映射能力与影响

1. PDF、图片、URL 读取与多模态分析（`read_indentifier_language`、`polish_readed_context`、`select_images`、`analyze_image`、`download_if_url`）。输入固定为已准备的 `PatientTrajectory`，不影响原子、深度、宽度与内置核验的执行。
2. 语言判定与翻译（`langid`、`trans_language`）。`trans_language` 在源码主流程中未被调用；题目与答案语言跟随病历文本。
3. `langchain` 的 `RecursiveCharacterTextSplitter`。改为同分隔符优先级的贪心切分，`chunk_overlap` 为 0。
4. `oagents` 框架、爬虫密钥与 `.env` 检查（`check_envs`）、本地 Python 执行器与 E2B。由公共 `run_agent`、`call_llm` 与病历工具替代。
5. `augmented_question_prompt` 在源码深度阶段被载入但未使用，适配同样不调用；`score_prompt_both`、`acc_prompt_single`、`extract_answer`、`run_llm_msg` 属于源码未接入主流程的路径。
