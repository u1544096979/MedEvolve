# FHIR-AgentBench 本地环境接入设计

## 1. 数据准备

MedEvolve 使用 MIMIC-IV Clinical Database Demo on FHIR v2.1.0 的原始患者数据，在本地 FHIR 服务上执行新任务。环境提供查询工具和原始证据读取；实验开始前整理患者轨迹，供现有方法生成任务。任务生成、公共核验、解题、反馈与策略进化沿用当前项目的接口。

数据来源为 [PhysioNet 官方 Demo v2.1.0](https://physionet.org/content/mimic-iv-fhir-demo/2.1.0/)。保存发布版本、原始文件与校验值，保留原始资源 ID、字段、时间、编码和引用关系。患者身份使用 `Patient.id`，其他患者标识作为原始字段保留。数据文件、数据库种子和实验结果存放在版本库外。

原生查询行为以 FHIR-AgentBench 提交 `bbb42909a5a7eb907d1cd91f72a560729e7037ea` 为依据：

- [部署说明](https://github.com/glee4810/FHIR-AgentBench/blob/bbb42909a5a7eb907d1cd91f72a560729e7037ea/README.md)
- [FHIR 客户端](https://github.com/glee4810/FHIR-AgentBench/blob/bbb42909a5a7eb907d1cd91f72a560729e7037ea/fhir_client.py)
- [资源检索工具](https://github.com/glee4810/FHIR-AgentBench/blob/bbb42909a5a7eb907d1cd91f72a560729e7037ea/tools/resource_tools.py)
- [多轮资源检索 Agent](https://github.com/glee4810/FHIR-AgentBench/blob/bbb42909a5a7eb907d1cd91f72a560729e7037ea/agent/multi_turn_resource_agent.py)

官方提供连接 Google Cloud Healthcare API 的客户端和工具，本地数据服务采用 HAPI FHIR R4 与 PostgreSQL。官方题目、参考答案和证据标签不进入新任务生成输入；本次验收对象是本地环境与新任务执行。

导入按资源引用关系分批执行，使用携带原始 ID 的写入请求。逐条检查批量请求的结果，失败批次不能计入成功数量。导入完成后，按资源类型比较源文件与服务端的数量和 ID，并读取代表性关联记录。验证通过后生成数据库种子，保存数据校验值、HAPI 与 PostgreSQL 镜像摘要、资源数量及数据库备份校验值。

## 2. 本地服务

### 实例与数据隔离

每个 `FHIR-AgentBench` 实验实例由一套 HAPI 服务、PostgreSQL 数据库、私有网络和独立数据库卷组成。实例从同一份已验证数据库种子恢复，返回本机可访问的 FHIR 地址。种子保持只读，实例之间不共享可写数据库。

本地种子目录包含 `manifest.json` 和 `database.dump`。manifest 保存固定镜像摘要、数据版本、源数据与备份校验值、按类型统计的资源数量。`EnvironmentConfig.seed_data` 对本环境表示管理服务所在主机上的种子目录绝对路径；其他环境继续使用现有镜像标识。

管理器对 `FHIR-AgentBench` 使用成对容器的生命周期，返回现有 `EnvironmentInstance`，其 `endpoint` 仍为原生 FHIR 地址。启动顺序为创建独立卷与网络、启动数据库、恢复种子、启动 HAPI、检查就绪。恢复过程失败时不返回实例。

就绪检查同时确认 FHIR R4 能力声明、种子患者数量，以及一条固定原始资源可读取。启动超时或检查失败时回收本次创建的容器、卷与网络。释放接口按实例标识回收整套资源，可重复调用；仅操作带本项目管理标签的资源。

### 本机运行约束

FHIR 接口绑定本机回环地址，数据库仅在实例私有网络内开放。实验串行运行，同一时刻最多一套实验环境。FHIR-AgentBench 的 HAPI 与 PostgreSQL 总内存预算为 4 GB，初始分配为 HAPI 3 GB、PostgreSQL 1 GB；实际启动、恢复和查询能力以真实运行验收为准。内存不足记录为环境故障，调整预算需要重新确认。

HAPI、PostgreSQL 和种子必须使用相互匹配的固定版本。Docker 镜像需支持本机 ARM64 或经过本机验证；种子清单记录实际镜像摘要，启动不使用浮动标签解析新版本。数据导入和种子创建是独立准备步骤，普通实例启动只恢复种子。

## 3. 执行工具

Agent 使用官方多轮资源检索路径的两个工具：

| 工具 | 输入 | 返回 |
| --- | --- | --- |
| `get_resources_by_patient_fhir_id` | 资源类型、患者 FHIR ID | 按资源类型分组的患者记录 |
| `get_resources_by_resource_id` | 资源类型、资源 ID | 按资源类型分组的匹配记录 |

支持类型沿用官方声明：Patient、Encounter、Condition、MedicationRequest、Procedure、Observation、MedicationAdministration、Location、Specimen、Medication。工具仅执行读取，采用现有 SDK 工具调用流程，输出接入公共调用预算、trace 和错误分类。

适配保留工具名、参数和分组返回结构。客户端连接当前实例地址，取消 Google Cloud 身份与固定端点依赖。按 ID 查询使用本地 FHIR R4 支持的 `_id`，跟随服务端分页链接取得完整结果。官方资源客户端移除返回记录的 `text`、`meta` 字段，该工具输出行为保留；原始证据读取保留完整记录。

患者查询限制在实验开放的患者集合内。按 ID 读取时核对记录所属患者；药品、地点等共享资源通过开放患者的原始引用确认可访问关系。分页和资源引用始终指向当前实例。缓存按实例隔离，释放后失效。

工具返回不直接截断记录或字符串。完整分页读取的结果若超过当前模型输入预算，运行明确记为预算或上下文失败，不能把部分数据冒充完整返回，也不能据此宣称任务难度提升。首版不增加官方工具之外的检索或计算接口。

## 4. 患者轨迹

### 原始材料与公共结构

`prepare_trajectory` 输出现有 `PatientTrajectory`。`patient_id` 使用 FHIR 资源 ID；`sources` 按 `ResourceType/id` 保存完整原始 JSON；`text` 呈现原始记录的日期、就诊关联及临床字段。证据回查通过相同 ID 读取本地服务中的原始记录。

轨迹读取覆盖患者实际存在的临床资源，包括执行工具支持的类型，以及数据中的 MedicationStatement、MedicationDispense 等原始记录。公共材料读取不因此扩大解题 Agent 的官方工具能力。出题所需事实必须能经执行工具读取或从其返回记录推导；只能由公共整理层读取的记录不能成为解题时唯一可用的证据。

优先读取患者完整关联资源；若服务不支持完整患者读取，则逐类型读取全部分页并补齐药品、地点、标本等关联记录。按资源 ID 去重，保留原始患者引用、就诊引用和资源间引用，核对完整性后才产生轨迹。

### 就诊组织与时间

轨迹按“患者 → 就诊 → 原始记录”组织。就诊来源按 `identifier.system` 中的 `encounter-hosp`、`encounter-ed`、`encounter-icu` 区分；`class` 原值同时保留，不能只凭 `class` 判断就诊来源。医院就诊可能带有 AMB、EMER 或 OBSENC 等不同 class 值。

保留 `partOf` 父子关系。急诊和 ICU 子就诊展示在对应医院就诊下，仍保留各自 ID、起止时间及原始关联记录；统计医院就诊次数时只计算 `encounter-hosp` 记录，不将子就诊重复计为医院就诊。子就诊时间可能早于父就诊开始时间，按原始关系和时间呈现，不改写为父就诊时间。

记录依据原始 `encounter` 或 `context` 引用归组，属于子就诊的记录保留子就诊归属。缺少直接就诊引用时保留原有 `derivedFrom`、`hasMember`、`partOf`、`specimen` 等关联，可通过这些引用呈现关联记录；无法关联的记录单独呈现，不按时间接近程度推断就诊归属。

| 资源 | 时间读取规则 |
| --- | --- |
| Encounter | 保留 `period.start` 与 `period.end` |
| Observation | 保留 `effectiveDateTime` 或 `effectivePeriod`；`issued` 单列为发布时间；`valueDateTime` 保留为测量值所描述的日期 |
| MedicationRequest、MedicationAdministration | 分别保留 `authoredOn`、`effectiveDateTime` 或 `effectivePeriod` |
| Procedure、Specimen | 分别保留 `performedDateTime` 或 `performedPeriod`、`collection.collectedDateTime` 或 `collection.collectedPeriod` |
| Condition、MedicationStatement、MedicationDispense | 按原始字段保留 onset、recorded、effective、dateAsserted、whenPrepared、whenHandedOver 等实际存在的时间，并标明字段来源 |

排序依据对应资源的原始事件时间，时间相同按资源类型与 ID 稳定排序。不同时间字段含义不能混用；没有事件时间时明确标记缺失，仅显示所属就诊的起止时间作为背景，不把就诊时间当作事件发生时间。Condition 或 MedicationDispense 仅有关联就诊时，记录置于该就诊的无事件时间分组，不补齐诊断或发药时间。无法解析的时间保留原值并计入整理统计，不使用数据库更新时间补齐。

Observation 的记录时间、发布时间和日期型测量值分别呈现。例如，置管日期记录的 `effectiveDateTime` 表示本次观察的时间，`valueDateTime` 表示记录所描述的置管日期；二者可不同。MedicationStatement 的 `dateAsserted` 是用药信息被记录或陈述的时间，不自动作为用药开始时间。

日期保留数据中的去标识值，药物医嘱、实际给药、发药和用药陈述分别呈现。药物表示同时支持 `medicationCodeableConcept` 内的名称与编码，以及 `medicationReference` 指向的 Medication 记录。被引用的 Medication 保留在关联材料中；药名读取其原始名称标识，保留完整编码，不假定 `code` 中必有可读药名。大小写和空白归一仅用于匹配，不覆盖原始值。

整理层不生成临床因果、疗效判断或未经记载的病程事件。现有轨迹压缩作为方法侧能力使用，原始轨迹与来源仍是事实依据。

### 现有方法接入

环境公共入口复用 `create_environment`、`get_tools`、`prepare_trajectory`、`read_evidence`、`read_state`、`release_environment`。适配属于 `environments/` 与 `server/`，所有方法共用同一实现，不修改生成算法或方法入口。

`read_state` 返回患者标识、环境名称和只读操作语义；完成判断依据 solver 查询 trace、答案和原始证据，不要求写后资源变化。公共核验按 ID 读取原始记录，并使用现有统一评测规则。

`decision_point` 为 null。相对时间问题在题目中给出明确时间背景，不宣称环境过滤了未来记录。独立评测患者不进入数据探索或策略学习；解题输入只含题目、患者身份和必要背景，参考答案、出题证据列表及缓存轨迹不传给解题 Agent。

## 5. 验收

1. **数据与本地服务**：导入清单与实际资源数量、ID 一致；FHIR R4 能力声明可读取；跨多页的真实患者查询完整返回；药品、标本和就诊引用可解析。
2. **实例生命周期**：串行创建、查询、释放两套实例，确认种子恢复一致、数据卷独立；失败启动可清理；释放后容器、卷与网络无遗留。记录启动时间、种子恢复时间及内存使用。
3. **原生工具与访问边界**：在同一本地实例上比较官方工具与适配工具的记录 ID、临床字段和返回结构；覆盖 Patient、Encounter、Observation、MedicationRequest、MedicationAdministration 及关联 Medication。越界患者不可读；大量记录的预算失败明确报告。
4. **患者轨迹与证据**：选择具有多次就诊、父子就诊和缺失时间记录的真实材料，核对轨迹 ID、时间字段、关联关系及原始证据回查；报告各类型读取量、缺失时间量和无法解析时间量。正确与错误证据均经过公共核验。
5. **任务执行与进化**：使用现有已授权模型生成新题，串行验证公共核验与独立 Agent 解题，再运行两轮反馈与策略更新及独立患者评测。报告实际生成、核验通过、解题完成数量、失败类别与调用成本；展示一条题目、实际工具调用和证据链。环境故障与模型故障单列，小样本不作为进化收益成立的依据。
