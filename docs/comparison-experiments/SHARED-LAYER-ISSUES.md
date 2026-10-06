# 共享层缺陷清单

在 EnvExplore 双环境真实运行中暴露的公共层问题，影响所有方法（主方法与各对比方法共用环境层、运行层与核验层）。EnvExplore 的实现本身未引入这些缺陷。运行产物位于 `outputs/envexplore-v2-acceptance/`。

## 待修复

### 1. 工具返回截断产出非法 JSON

- 现象：截断在 `json.dumps` 之后按字符数切，模型收到的是被截断的 JSON 字符串加 `… [result truncated]`，无法解析。
- 证据：PhysicianBench 运行 54 条工具返回中 27 条被截断，最长 10020 字符（`physicianbench-envexplore-acceptance`）。
- 影响：模型将残缺结果视为读取失败并重复同一查询，是探索阶段反复查询与上下文膨胀的助推因素。
- 位置：`environments/__init__.py` 的 FHIR 工具包装。
- 建议：先按记录条数与字段裁剪数据，再序列化，保证返回始终是合法 JSON。

### 2. 环境探索既不探测也不停止

- 现象：探索 Agent 要么重复同一批查询直到耗尽预算，要么完全不调用工具直接写出指导。
- 证据：PhysicianBench 运行 30 轮、66 次调用、仅 2 种不同参数、模型文本 0 条，以 `MaxTurnsExceeded` 结束，累计 1,579,360 input tokens（`physicianbench-envexplore-r5000`）；换用 deepseek 后为 1 轮、0 次工具调用（`physicianbench-envexplore-deepseek`）。
- 影响：正式配置下 PhysicianBench 在探索阶段终止，实验无法继续；指导可能包含未验证的能力声明。
- 位置：`methods/medevolve/prompts.py` 的 `ENVIRONMENT_EXPLORATION`。
- 建议：明确“同一查询只做一次”与“确认能力后立即交付指导”，并设定最少探测要求。

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

### 5. 工具声明未覆盖参数范围，且单次环境错误否决整轮

- 现象：工具声明只列出部分筛选参数，模型据此补充标准 FHIR 参数后被服务拒绝；一次失败会使该轮整体记为环境错误。
- 证据：MedAgentBench 解题轮中模型发出 `Condition?patient=...&status=active` 得到 400，去掉 `status` 重试成功并提交答案，但该轮仍记为 `failed / environment_error`，完成判定未产出（`medagentbench-envexplore-smallpatient-solverfix`）。
- 影响：模型有能力完成任务仍被判失败，完成率被低估。
- 位置：`environments/medagentbench-functions.json` 的函数声明与 `runtime/native.py` 的运行状态判定。
- 建议：补全各查询支持的筛选参数，或对未知参数返回明确提示并允许重试后计入成功。

## 已修复

原生通道缺少交付说明：协议只有 GET、POST、FINISH 三种动作，未说明结论需作为 `FINISH` 的文本项提交，模型提交 `FINISH([])` 导致指导为空或解题答案为空。

- 修复：`runtime/native.py` 对 `EnvironmentExploration` 与 `SolverAnswer` 分别追加提交说明，解题路径与 PhysicianBench 通道不变。
- 验证：MedAgentBench 复跑后 2/2 题核验通过，解题提交真实答案，完成判定由 0/2 转为 1/2。
