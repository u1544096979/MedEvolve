# 共享层缺陷清单

在 EnvExplore 双环境真实运行中暴露的公共层问题，影响所有方法（主方法与各对比方法共用环境层、运行层与核验层）。EnvExplore 的实现本身未引入这些缺陷。运行产物位于 `outputs/envexplore-v2-acceptance/`。

## 待修复

### 3. 公共核验出现假拒绝

- 现象：核验把与声明一致的记录判为矛盾，据此拒绝整题。
- 证据：PhysicianBench 题 `...:evaluation:0:1` 被拒理由为 `Procedure/2517` 记录 `MR BRAIN WO IV CONTRAST` 与声明“无静脉对比”矛盾，而 `WO IV CONTRAST` 即“无静脉对比”（`physicianbench-envexplore-deepseek`）。
- 影响：有效题目被丢弃，通过率被低估，错误理由写入运行产物。
- 位置：`evaluation/` 的核验提示词与判定流程。
- 建议：判定矛盾前按字面解释医学缩写，并要求引用原始字段与取值。

### 4. MedAgentBench 生成材料超出常规上下文

- 现象：生成阶段一次性发送“完整轨迹正文 + 全部原始来源”，材料随患者记录数线性增长。
- 证据：患者 `S2874099` 正文 234,000 字符、898 条来源 527,192 字符，请求体 898,874 字符约 30 万 token，gpt-4o-mini 返回 400；其中 `meta` 管理字段占 17%，键名与结构字符约 15%。对照 PhysicianBench 同等位置患者 116 条来源、约 3.7 万 token。
- 影响：MedAgentBench 只能使用百万级上下文模型，成本高且无法在小模型上运行；主方法使用同一份材料，仅因配置大上下文模型未暴露。
- 位置：`environments/__init__.py` 的 `prepare_trajectory` 与生成输入组装。
- 建议：去除 `meta` 等管理字段、按需裁剪来源，或接入轨迹压缩。

## 已修复

环境探索既不探测也不停止：探索 Agent 要么重复同一批查询直到耗尽预算，要么完全不调用工具直接写出指导；PhysicianBench 运行 30 轮、66 次调用、仅 2 种不同参数、模型文本 0 条，以 `MaxTurnsExceeded` 结束，累计 1,579,360 input tokens（`physicianbench-envexplore-r5000`），换用 deepseek 后为 1 轮、0 次工具调用（`physicianbench-envexplore-deepseek`）。

- 修复：`methods/medevolve/prompts.py` 的 `ENVIRONMENT_EXPLORATION` 增加探索规则，要求交付前实际调用工具、相同工具与参数的查询只执行一次、被裁剪时缩小范围而非重复原查询、信息足够即交付，并要求指导区分已验证能力、调用失败与未验证能力；双环境共用同一份提示词，预算与原生动作格式不变。
- 验证：`tests/method/test_envexplore_environment.py` 断言 PhysicianBench 与 MedAgentBench 两条通道都把 `ENVIRONMENT_EXPLORATION` 作为 `instructions` 传入，预算、释放与 FINISH 格式测试通过；提示词只能约束模型行为，尚无真实运行证据，仍待实跑确认模型遵守这些规则。

工具返回截断产出非法 JSON：截断在 `json.dumps` 之后按字符数切，模型收到的是被截断的 JSON 字符串加 `… [result truncated]`，无法解析；PhysicianBench 运行 54 条工具返回中 27 条被截断，最长 10020 字符（`physicianbench-envexplore-acceptance`）。

- 修复：`environments/__init__.py` 的 FHIR 工具包装改为调用 `environments/tool_results.py`，先按完整记录裁剪数据再序列化，超限返回合法 JSON 与 `truncation` 标记，写入响应与超限错误同样受限。
- 验证：`tests/environment/test_tool_results.py` 与 `tests/environment/test_patient_boundary.py` 覆盖 500、5000、10000、200000 四种上限、单条超大记录、写入响应、超限错误与低于下限的配置；`PYTHONPATH="$PWD" python -m pytest -q` 382 passed, 4 skipped。

原生通道缺少交付说明：协议只有 GET、POST、FINISH 三种动作，未说明结论需作为 `FINISH` 的文本项提交，模型提交 `FINISH([])` 导致指导为空或解题答案为空。

- 修复：`runtime/native.py` 对 `EnvironmentExploration` 与 `SolverAnswer` 分别追加提交说明，解题路径与 PhysicianBench 通道不变。
- 验证：MedAgentBench 复跑后 2/2 题核验通过，解题提交真实答案，完成判定由 0/2 转为 1/2。

工具声明未覆盖参数范围，且单次环境错误否决整轮：工具声明只列出部分筛选参数，模型据此补充标准 FHIR 参数后被服务拒绝；一次失败会使该轮整体记为环境错误。MedAgentBench 解题轮中模型发出 `Condition?patient=...&status=active` 得到 400，去掉 `status` 重试成功并提交答案，但该轮仍记为 `failed / environment_error`，完成判定未产出（`medagentbench-envexplore-smallpatient-solverfix`）。

- 修复：`environments/medagentbench-functions.json` 只改描述不改参数表：Condition 改称按 patient 与 category 查询、MedicationRequest 改称按 patient 与 category 或 date 查询，Patient 去掉未声明的 `careProvider` 说法，各 GET 声明注明只支持表内参数、服务拒绝参数时修正后重试。`environments/medagentbench.py` 把 GET 的 HTTP 400 记为该轮可恢复的 `tool_error` 并回传状态码与服务原文，其余 HTTP 状态以及连接失败、超时仍为环境错误。`runtime/native.py` 新增 `pending_parameter_error`：可恢复错误只进 `tool_output` 记录，成功 GET 清除、POST 不清除，合法 FINISH 时未修正则记 `failed / tool_error`，真实环境失败与原有预算法则不变。
- 验证：离线证据为 `tests/environment/test_medagentbench.py` 的分类矩阵（400 三种响应体、401、403、404、503、ConnectError、ReadTimeout）与声明结构测试，以及 `tests/runtime/test_native.py` 的参数错误恢复、400 后直接 FINISH、400 后预算耗尽、400 后 POST 再 FINISH、503 后成功 GET 再 FINISH 五条序列；观察到的失败请求形状为 `Condition?patient=...&status=active` 返回 400。上述序列用模拟传输，恢复后的端到端完成评价与真实服务返回的错误原文仍需一次实跑确认。
