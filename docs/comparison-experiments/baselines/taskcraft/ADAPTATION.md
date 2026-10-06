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
| `gen_depth_based_task.py:DepthExtend.verify`、`process_single_task`、`depth_extend` | `stages.Pipeline._extend_depth`、`_merge`、`_solve_trajectory` | 逐跳核验、合并、`valid_hop` 与 `max_hops` 控制；参考答案取原始最终答案，子问题与中间结果进入 `solve_trajectory` |
| `gen_width_based_task.py:width_extend`、`check_queries` | `stages.Pipeline.width`、`_merge_groups`、`_check_merged`、`_check_batch` | 合并问题、按 10 组一批的可分解性检查与无工具作答检查 |
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
| — | `scheduling.py`、`Pipeline.run`、`Pipeline.select` | 分支调度与最终两题选题；源码三个入口分别独立运行，没有编排器与选题规则，两项都不对应任何源码入口 |
| `gen_atomic_task.py:gen_atomic_tasks`、`read_indentifier_language`、`polish_readed_context` | 无对应 | 入口由公共 `generate_tasks` 承担；文件、图片与 URL 读取不适用 |

阶段顺序：原子先生成并核验；深度与宽度随后按实验种子决定的顺序执行，两者读取同一原子池，互不消费对方输出。源码三个入口是独立脚本，没有编排器，也没有最终选题规则；分支调度与选题是公共对比实验的接入约定，不是作者行为。

随机调度与选题。每患者种子为实验种子、环境名、患者标识与用途（`branch-order`、`output-selection`）的 SHA-256 稳定摘要，不使用内建 `hash()`，也不使用进程级随机状态。`["depth", "width"]` 由分支流置换；候选池在选题前固定为 atomic、depth、width 的各自生成顺序，选题流独立洗牌该列表，再按非空 `instruction` 与 `reference_answer`、`decision_point` 为 null、证据 `source_id` 在当前患者轨迹内且 `statement` 非空、以及请求的临床题型配额接纳，最多两题；不设生成来源配额，不按内置评分或公共评测结果排序。固定种子使同一候选列表与配置下的调度与选题可复现，不会让模型重新生成相同的文本。所有阶段共用 generator 总预算：原子池不足时宽度记 `insufficient_parents`，预算耗尽时后续分支记 `cumulative_budget_exhausted`，不补题；此时公共输出为 `None`，完整候选池与筛选记录仍保留在 trace 中。

## 3. 工具、输入与证据口径

| 源码 | 适配后 |
|---|---|
| `CrawlerSearchTool`、`CrawlerReadTool`、`CrawlerArchiveSearchTool`、`VisualInspectorTool` | 当前患者只读 FHIR 工具声明，绑定取其中的非空子集：`fhir_condition_search_problems`、`fhir_observation_search_labs`、`fhir_observation_search_vitals`、`fhir_patient_search_demographics`、`fhir_procedure_search_orders`、`fhir_medication_request_search_orders`、`fhir_document_reference_search_clinical_notes`、`fhir_service_request_search`、`fhir_observation_search_social_history` |
| 写工具（`fhir_*_create`、`fhir_*_create_message`） | 生成阶段不绑定，留给评测阶段解题 Agent |
| 文档文件、PDF、网页 | `PatientTrajectory.text`，按记录时间排序的记录块 |
| `content_identifier`（文献或网页标题） | 病历记录块标签 `[记录时间] 资源类型/资源ID`（`prepare_trajectory` 的格式） |
| `golden_answer` | 原子题为 `clean_qa` 的 `refined_answer`；扩展题见第 4 节 |
| — | 结论抽取额外返回 `source_ids`；映射为 `Evidence(source_id, statement=结论)`，来源不在当前轨迹内即丢弃该候选 |
| — | 深度辅助核验的 superset 关系事实同样绑定真实来源：`Evidence(source_id, statement="identifier：relation")`，来源取自反向检索返回的记录块标识，与父题证据合并去重 |
| — | 宽度合并题的父证据取各父题自身证据的并集，不重新归因；两阶段都不新增模型事实审查环节 |

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

前三项是源码缺陷的对照修复，其后各项是适配口径与协议映射。

1. 原子过滤判据。源码 `filter_question`（`gen_atomic_task.py:418-444`）仅在核验返回 `error` 时拒绝，更严格的 `llm_score`、`agent_score` 判断在 `425-433` 被注释掉。适配保持该分支：无工具回答评分达到 1 分及以上或核验出错才拒绝，`agent_score` 只记录不用于拒绝。患者私有事实本就不在模型自身的知识内，该判据在病历域很少触发，实际筛选由「病历 Agent 能否作答」承担。
2. superset 校验。源码把返回的 JSON 字典与字符串 `"invalid"` 比较（`gen_depth_based_task.py:67`），字典与该字符串不相等。适配按提示词声明的 `{"new_query": "valid"|"invalid"}` 判断，仅显式 `invalid` 时拒绝。
3. 宽度结果组装。源码最终映射（`gen_width_based_task.py:234-241`）引用循环外的 `golden_answers`、`original_questions`，与当前分组不一致；评分又用响应里的稳定 ID 直接索引过滤后的局部列表（`gen_width_based_task.py:252-253`）。适配按每题自身的答案、原始问题与原始索引组装，评分按稳定 ID 在已通过分解检查的集合里查找；父题索引保持模型给定的顺序与重复项，越界或缺少父题的分组按模型输出失败丢弃整次响应重试（`gen_width_based_task.py:190-200` 的异常分支），不修剪成更小的组。
4. 宽度合并输入。源码把 `golden_answer` 一并送入合并提示词（`gen_width_based_task.py:184-188`，与注释 `without golden_answers` 不一致）。适配保持源码行为；合并结果由后续可分解性与作答检查把关。
5. 参考答案组织。`TaskBatch` 只有单个 `reference_answer`。深度题的参考答案是源码 `solve_trajectory` 的原始最终答案（`full_results[-1]["answer"]`，即父题答案）；子问题、中间标识及其答案与 Agent 轨迹进入 `solve_trajectory` 和阶段 trace，不拼进参考答案。宽度题按每组自身的原始答案列表无损序列化（`json.dumps(..., ensure_ascii=False)`），顺序与重复项与分组一致。`Candidate.detail` 保存阶段来源：原子题为原结论、R、患者来源与清洗答案；宽度题为 `original_questions`、`golden_answers`、原始 `indexes` 与全部 `parent_types`；深度题为 `full_results`、`solve_trajectory`、辅助来源与失败尝试。
6. 题型。源码没有题型采样。出题提示词追加题型约束，取值限于 `directions` 中的题型；扩展题继承父题题型，跨题型合并取第一父题的题型。批量输出的选题规则见第 2 节：候选池顺序固定，选题流独立洗牌后按临床题型配额接纳，最多两题。
7. 预算边界与 Agent 步数。`max_candiated_conclusions`、`max_candidate_atomic`、`judge_samples`、`question_attempts`、`query_attempts`、`merge_attempts`、`extended_attempts`、`max_hops`、`width_attempts` 以 `TaskCraftParameters` 显式给出，源码 SearchAgent 的 `max_step` 以 `atomic_max_action_steps=20`、`depth_backward_max_action_steps=10`、`depth_verify_max_action_steps=10` 给出。阶段不设份额或预留轮数：共享生成预算是唯一上限，原子阶段逐个结论走到核验或预算耗尽，扩展阶段逐跳执行到 `max_hops` 或预算耗尽。
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
22. 强制终答的答案字段类型。达到 action step 上限后的终答请求按源码要求返回 ` thinking/<answer>`，适配用 `FinalThought(think: str, answer: any)` 承载：`answer` 对应源码 `FinalAnswerTool` 的 `inputs = {"answer": {"type": "any"}}` 与 `output_type = "any"`，可以是任意 JSON 值，`think` 仍为字符串；`agent._finish` 在源码 `safe_json_loads` 的位置把字符串解析为 JSON，再映射到本次调用声明的输出结构。JSON 对象与 JSON 文本两种形态都接受。
23. 规划与反思的答案字段类型。规划（事实、计划）与反思（事实、计划）的 `answer` 在源码里是 `extract_think_and_answer` 从 ` thinking/<answer>` 标签抽出的自由文本，不解析 JSON；适配用 `AgentThought(think: str, answer: str)` 承载，保持文本口径。PhysicianBench 的 `gpt-4o-mini` 在规划请求里把任务要求的 JSON 直接放进 `answer`（反向检索任务给出 `{"identifier", "relation"}`；核验任务因 `AUGMENTED_QUERY` 要求 `answer_list` 键而给出 `{"answer_list": [...]}`），`AgentThought` 校验失败，整次检索以 `ValidationError: ... Input should be a valid string` 终止。这是 PhysicianBench 机制验收深度阶段没有候选的直接原因（证据见 5.1 节）；MedAgentBench 的深度阶段在同一适配下产出 6 个候选。

## 5. 预算口径与真实运行结果

`TaskCraftParameters` 默认值取自源码签名：`chunk_size=16384`、`max_candidate_conclusions=20`、`max_candidate_atomic=10`、`judge_samples=3`、`question_attempts=3`、`query_attempts=3`、`merge_attempts=5`、`extended_attempts=4`、`max_hops=2`、`width_attempts=3`；Agent 步数上限为 `atomic_max_action_steps=20`、`depth_backward_max_action_steps=10`、`depth_verify_max_action_steps=10`。`max_candidate_atomic` 与共享生成预算共同决定单批候选数。

机制验收与公共比较各用一组配置，预算口径不混用：

| 用途 | 配置 | generator 预算 | 来源 |
|---|---|---|---|
| 机制验收 | `configs/taskcraft-mechanism-physicianbench.json` | 800 轮 / 800 次工具 | 由 `configs/taskcraft-physicianbench.json` 复制，只改 `run_id`、`output_dir`、该预算 |
| 机制验收 | `configs/taskcraft-mechanism-medagentbench.json` | 800 轮 / 800 次工具 | 由 `configs/taskcraft-medagentbench.json` 复制，只改同三处 |
| 公共比较 | `configs/taskcraft-physicianbench.json` | 8 轮 / 40 次工具 | 冻结公共值，未改动 |
| 公共比较 | `configs/taskcraft-medagentbench.json` | 1 轮 / 0 次工具 | 冻结公共值，未改动 |

800/800 是机制预算，用于让作者的阶段重试参数（`question_attempts=3`、`extended_attempts=4`、`max_hops=2`）与 `max_action_steps` 有空间跑完；它不进入公共比较口径。

真实验收入口：

```
MEDEVOLVE_TASKCRAFT_LIVE_CONFIG=<机制配置> pytest tests/method/test_taskcraft_live.py -q -s
```

未给出配置路径时该测试跳过，不创建实例、不发起调用。一次运行把 `report.json`（状态、用量、候选数、阶段记录、选题）、`pipeline.json`（完整候选池、派生种子、分支顺序、筛选记录）、`result.json`（运行状态与完整 trace）、`config.json` 写入配置的 `output_dir`；目录已有产物时拒绝写入。阶段判据为：`atomic`、`depth`、`width` 各有至少一个带非空参考答案、可回查证据（`source_id` 属于当前患者轨迹）的候选，非原子候选必须带父题，且真实用量非零；`completed` 加空批不算阶段成功。

### 5.1 PhysicianBench 机制验收

运行目录 `outputs/taskcraft-mechanism-physicianbench-rerun`，患者 `MRN4460967357`，种子 `fhir-full:v1`，`gpt-4o-mini`，机制预算 800 轮 / 800 次工具，墙钟 14 分 45 秒。

| 项 | 数值 |
|---|---|
| 生成状态 | completed；262 模型轮 / 26 次工具调用 / 469 017 tokens（输入 438 291，输出 30 726；按 gpt-4o-mini 公开价估算约 $0.084） |
| 候选数 | atomic 3，width 2，depth 0 |
| 分支顺序 | `["width", "depth"]`（派生种子决定） |
| 原子阶段 | 4 块文本、54 708 字符；chunk 0 与 chunk 1 共抽到 46 条结论，生成 10 题，核验接纳 3 题；1 次出题因模型返回空 `task_type` 被拒；7 次核验失败（3 次规划 `answer` 类型不符、4 次 `action response carries no tool call`） |
| 宽度阶段 | 合并 2 组，可分解性检查保留 2 组，产出 2 题，父题各 2 道 |
| 深度阶段 | 3 道父题各 4 次尝试全部失败，反向检索 12 次尝试：6 次规划 `answer` 类型不符、2 次 `APITimeoutError`、1 次 `APIConnectionError`、1 次 `action response carries no tool call`、2 次 superset 校验拒绝；`depth/result.extended=0` |
| 候选证据 | 原子题 `Condition/2565`、`Condition/2567`、`Condition/2571`；宽度题分别合并 `Condition/2571`+`Condition/2567` 与 `Condition/2567`+`Condition/2565`，各带两道父题 |
| 最终选题 | 1 题（请求方向为药物处方与诊断与解释；候选均为诊断与解释，宽度题按 `task_type_quota_exhausted` 未进入） |
| 无已知 token 的请求 | 3 次（2 次 `APITimeoutError`、1 次 `APIConnectionError`），计入 `runs_without_known_tokens` |
| 提交基线 | trace 内 `author_commit=1e405e0b02c4738654e2d885b16677b46c9fe969` |

阶段结论：`atomic` 与 `width` 产出带参考答案、可回查证据与父题的真实候选；`depth` 没有候选。

深度阶段没有候选的位置是规划请求，不是强制终答。用同一份配置的 provider 复现（`max_action_steps=1`、空工具列表、反向检索任务文本），失败记录是 `probe/backward/planning/facts/attempt-1`，报错与验收记录逐字一致：

```
ValidationError: 1 validation error for AgentThought
answer
  Input should be a valid string [type=string_type,
  input_value={'identifier': 'SupersetI...lation': 'is a part of'}, input_type=dict]
```

同一次复现把任务文本换成不含 JSON 结构的提问时，记录里出现 `probe/final/forced_final/attempt-1` 且请求本身通过校验，说明强制终答的类型口径已与源码一致，瓶颈在规划。核验任务里 `answer_list` 形态的对象来自 `AUGMENTED_QUERY` 明确要求的 JSON 键，同样落在规划请求上。

同一代码路径的另一次运行 `outputs/taskcraft-mechanism-physicianbench`（墙钟 16 分 35 秒）结果相同：completed，261 模型轮 / 17 次工具调用 / 447 995 tokens，atomic 4、width 1、depth 0；深度阶段 16 次反向检索尝试里 12 次是同一个 `AgentThought.answer` 类型错误，2 次 superset 校验拒绝，1 次 `action response carries no tool call`，1 次成功但后续核验 `llm_score=0.0`。

第三次运行 `outputs/taskcraft-mechanism-physicianbench-attempt1` 保留为失败记录：原子 40 条结论的 `source_ids` 全为空，证据门槛拒绝全部 10 题，`atomic.accepted=0`，整批 `status=failed`（227 轮 / 16 次工具 / 332 106 tokens）。该次运行里 16 次工具调用有 5 次返回 `PatientAccessError: Patient 4460967357 is outside the allowed patient scope`（模型把 `MRN4460967357` 写成 `4460967357`）；结论里 `source_ids` 由模型给出，这一轮的模型没有按追加约束填写。

### 5.2 MedAgentBench 机制验收

运行目录 `outputs/taskcraft-medagentbench`，患者 `S2874099`，种子 `jyxsu6/medagentbench@sha256:3fb83d7ed71c5476f9eb6212bd440a909ef7505922bbc757dc488a8fc0701966`，`deepseek-v4.1-flash`，机制预算 800 轮 / 800 次工具。

| 项 | 数值 |
|---|---|
| 生成状态 | completed；686 模型轮 / 187 次工具调用 / 4 115 580 tokens（输入 3 091 613，输出 1 023 967），墙钟 2 小时 11 分 |
| 候选数 | atomic 10，depth 6，width 4 |
| 轨迹 | 898 条来源、234 000 字符，按 `chunk_size=16384` 分为 15 块 |
| 原子阶段 | chunk 0 抽到 75 条结论，生成 10 题，核验接纳 10 题，证据为 `Procedure/726139` 等 10 条真实记录 |
| 深度阶段 | 31 次反向检索：12 次成功、14 次 superset 校验拒绝、5 次硬失败；核验 12 次；6 道父题扩展成功，每道带一个父题，证据为父题证据与 superset 关系来源的合并 |
| 宽度阶段 | 合并检查保留 4 组，产出 4 题，父题数 2–3 |
| 最终选题 | 1 题（请求方向为药物处方与诊断与解释，候选均为诊断与解释，另一方向的配额没有候选） |
| 过程异常 | 5 次反向检索硬失败：2 次模型返回 `{"name": "final_answer", ...}` 而非 `{"action": ...}`，3 次收到 `PatientAccessError: Only FHIR search URLs are supported`；1 次核验返回 `LLM can solve this task`（作者的无法工具作答过滤分支）；均按作者重试参数处理 |
| 提交基线 | trace 内 `author_commit=1e405e0b02c4738654e2d885b16677b46c9fe969` |

阶段结论：`atomic`、`depth`、`width` 均产出带参考答案、可回查证据与父题的真实候选。

### 5.3 冻结公共比较

两份公共配置按提交状态原样运行：`python -m experiments.run --config <公共配置>`。

| 项 | PhysicianBench | MedAgentBench |
|---|---|---|
| 运行目录 | `outputs/taskcraft-physicianbench` | `outputs/taskcraft-medagentbench` |
| 请求题数 / 生成次数 | 2 / 1 | 2 / 1 |
| 生成状态 | budget_exhausted，用满 8 轮 | budget_exhausted，用满 1 轮 |
| 停止位置 | 原子结论抽取完成，随后 7 次出题因空 `task_type` 被拒，第 8 轮预算耗尽；原子池为空，两个分支记 `cumulative_budget_exhausted` | 第 1 轮预算耗尽，轨迹已就绪 |
| 最终题数 | 0 | 0 |
| 公共核验 / 质量评价 / 解题完成 | 无记录（未出题，未发起评测） | 无记录 |
| 失败类别 | 无 | 无 |
| 生成成本 | 8 轮 / 0 次工具 / 18 067 tokens | 1 轮 / 0 次工具 / 27 088 tokens（输入 8 114，输出 18 974） |
| 评测成本 | 0 | 0 |

两个环境的公共预算都不足以走完原子阶段，公共输出为题数 0 的真实截断，未以机制预算的输出代替。

### 5.4 离线检查

```
pytest -q tests/method tests/runtime tests/experiment tests/environment tests/evaluation -W error
→ 143 passed, 4 skipped

pytest -q experiments -W error
→ 4 passed, 3 failed
  FAILED experiments/test_common.py::test_unimplemented_method_is_explicit[workflow]
  FAILED experiments/test_common.py::test_unimplemented_method_is_explicit[selfgen]
  FAILED experiments/test_common.py::test_unimplemented_method_is_explicit[envexplore]
```

`experiments` 的三项失败来自 venv 的 editable 安装把主检出 `/Users/guagua/code/MedEvolve/methods` 命名空间并入本工作区：主检出里已有 `workflow`、`selfgen`、`envexplore` 三个方法包，于是「未实现方法应显式报错」的断言不成立。它们与本方法无关，也不随本次改动变化。未给出 `MEDEVOLVE_TASKCRAFT_LIVE_CONFIG` 时 `tests/method/test_taskcraft_live.py` 跳过且不发起调用。


## 6. 未映射能力与影响

1. PDF、图片、URL 读取与多模态分析（`read_indentifier_language`、`polish_readed_context`、`select_images`、`analyze_image`、`download_if_url`）。输入固定为已准备的 `PatientTrajectory`，不影响原子、深度、宽度与内置核验的执行。
2. 语言判定与翻译（`langid`、`trans_language`）。`trans_language` 在源码主流程中未被调用；题目与答案语言跟随病历文本。
3. `langchain` 的 `RecursiveCharacterTextSplitter`。改为同分隔符优先级的贪心切分，`chunk_overlap` 为 0。
4. `oagents` 框架、爬虫密钥与 `.env` 检查（`check_envs`）、本地 Python 执行器与 E2B。由公共 `run_agent`、`call_llm` 与病历工具替代。
5. `augmented_question_prompt` 在源码深度阶段被载入但未使用，适配同样不调用；`score_prompt_both`、`acc_prompt_single`、`extract_answer`、`run_llm_msg` 属于源码未接入主流程的路径。
