# FHIR-AgentBench 本地环境接入实施计划

> **执行要求：** 使用 executing-plans 按任务顺序实施；实施前使用 using-git-worktrees 建立独立 worktree。每项任务完成验证后提交，再进入下一项。

**目标：** 在本机启动装有 MIMIC-IV-FHIR Demo 数据的独立 FHIR 实例，让现有 MedEvolve Agent 通过官方查询工具执行新任务，并提供可回查证据的患者轨迹。

**架构：** HAPI FHIR R4 与 PostgreSQL 成对构成一个实例，从已验证的数据库种子恢复。官方两个资源检索工具适配本地地址与患者范围；患者轨迹组织和原始证据读取接入现有公共环境入口。方法算法、任务结构和公共评测接口沿用现有实现。

**技术栈：** Python 3.10+、现有 httpx／FastAPI／OpenAI Agents SDK、Docker、HAPI FHIR R4、PostgreSQL 16。设计依据为同目录 `design.md`。

## 全局约束

- 项目为 `/Users/guagua/code/MedEvolve`；实施 worktree 从执行时的当前分支建立，兼容正在集成的公共接口，不覆盖其他会话的未提交改动。
- 原始数据为 `/Users/guagua/data/mimic-iv-fhir-demo/2.1.0/fhir/`，已有 100 名患者、928,935 条资源；32 个发布文件校验通过。实施仍校验输入清单，不重复下载。
- 种子放在 `/Users/guagua/data/mimic-iv-fhir-demo/2.1.0/seed/`；数据库备份、患者材料、凭据与运行结果不进入 Git。
- 官方源码固定为 `bbb42909a5a7eb907d1cd91f72a560729e7037ea`。SDK 解题只提供其两个资源检索工具和官方十种资源类型。
- 实验串行，同一时刻最多一套环境；HAPI 3 GB、PostgreSQL 1 GB，总内存预算 4 GB。已有实验占用本机时等待其释放，不停止其他会话的容器。
- 保留现有 `EnvironmentConfig`、`EnvironmentInstance`、`PatientTrajectory`、`Evidence` 及方法入口。`decision_point=null`。
- 模型使用现有已授权 DeepSeek V4.1 Flash 和运行时凭据，凭据不写配置或日志。环境与模型故障单列。
- 按关注点使用中文 Conventional Commit description 与 body。只提交本任务文件和审阅过的必要公共接口改动。

## 文件与职责

| 文件 | 职责 |
| --- | --- |
| `server/fhir_agentbench.py` | 种子校验、双容器创建、恢复、就绪与释放 |
| `scripts/fhir_agentbench_data.py` | 原始数据校验、导入、全量 ID 核对、种子备份 |
| `environments/fhir_agentbench.py` | 官方工具绑定、受限读取、共享资源关联 |
| `environments/fhir_agentbench_trajectory.py` | 原始材料读取、就诊关系与时间呈现 |
| 现有公共入口、配置与测试 | 环境分派、真实运行与关键机制验证 |

## 任务 1：准备本地服务与数据库种子

**文件：** 新建 `scripts/fhir_agentbench_data.py`、`server/fhir_agentbench.py`、`tests/environment/test_fhir_agentbench_seed.py`。

**接口：**

```python
# server/fhir_agentbench.py
def load_seed(path: str) -> dict: ...

# scripts/fhir_agentbench_data.py
def source_inventory(data_dir: Path) -> dict[str, set[str]]: ...
async def import_resources(endpoint: str, data_dir: Path) -> None: ...
async def verify_inventory(endpoint: str, expected: dict[str, set[str]]) -> None: ...
```

`load_seed` 返回 manifest，字段固定为 `schema_version=1`、`data_version`、`hapi_image`、`postgres_image`、`database_sha256`、`source_sha256`、`resource_counts`、`probe_source_id`、`probe_sha256`。数据库文件为目录下的 `database.dump`。镜像必须使用 `repository@sha256:digest`；数据版本必须为 `2.1.0`，患者数为 100。

- [ ] **1.1：添加批量导入失败测试。** 构造 HTTP 200 的 transaction-response，其中某条 entry 为 `400 Bad Request`；导入器必须抛出该资源 ID 对应的异常。再测试重复资源 ID、缺失患者和服务端额外 ID，防止仅凭 HTTP 状态或数量误报成功。

```python
def transaction_entry(resource):
    source_id = f"{resource['resourceType']}/{resource['id']}"
    return {"resource": resource,
            "request": {"method": "PUT", "url": source_id}}

def require_transaction_success(bundle):
    if bundle.get("resourceType") != "Bundle" or bundle.get("type") != "transaction-response":
        raise ValueError("Expected transaction-response")
    for entry in bundle.get("entry", []):
        status = entry.get("response", {}).get("status", "")
        if status.split(" ", 1)[0] not in {"200", "201"}:
            raise ValueError(f"Import failed: {status}; {entry}")
```

导入还须断言响应 entry 数等于提交数，并将失败 entry 与对应请求的 `ResourceType/id` 关联。运行 `python -m pytest tests/environment/test_fhir_agentbench_seed.py -q`，先确认测试揭示未实现行为。

- [ ] **1.2：实现输入校验和按依赖导入。** 用 gzip 逐行读取 NDJSON，按 resourceType 合并分文件数据，构建 ID 集合。按照 Organization、Location、Patient、Encounter、Medication、Specimen、Condition、MedicationRequest、MedicationAdministration、MedicationStatement、MedicationDispense、Procedure、Observation 顺序，以每批 200 条 transaction PUT 导入。遇到悬空引用时保留原始引用；本地准备配置关闭写入引用完整性检查和外部 profile 自动校验，不删除引用或临床字段。一个批次验证成功后才计入导入量。

```python
async def submit_batch(client, endpoint, batch):
    payload = {"resourceType": "Bundle", "type": "transaction",
               "entry": [transaction_entry(r) for r in batch]}
    response = await client.post(endpoint.rstrip("/") + "/", json=payload)
    response.raise_for_status()
    result = response.json()
    require_transaction_success(result)
    if len(result["entry"]) != len(batch):
        raise ValueError("Transaction response length mismatch")
```

- [ ] **1.3：实现本地准备 CLI。** `python scripts/fhir_agentbench_data.py prepare --data-dir <path> --seed-dir <path>` 校验发布 SHA256SUMS，确认种子目录为空或未完成，创建带管理标签的准备容器与私有网络，导入后按每种资源的完整分页返回 ID 集合比较源数据，读取固定 Patient 与一条 Observation 探针。验证通过后执行 PostgreSQL `pg_dump -Fc --no-owner --no-acl` 保存备份，再写入 manifest。失败时不写成功清单；已有成功种子不覆盖。

首次准备镜像固定为：

```text
hapiproject/hapi@sha256:4d59628d42b3556b2dcd274d26054a43ba53697ebdae0fd71e6a5475065ed22d
postgres@sha256:6b0faa930dc29e5af5af9b50081ad496fad106b85a88ad66aea01f11d4494733
```

以上是官方仓库的 ARM64 镜像，分别对应 HAPI `v8.12.0-2` 和 PostgreSQL `16-bookworm`。准备步骤显式拉取摘要并确认镜像架构；创建实例阶段只使用已加载镜像。PostgreSQL 数据卷挂载 `/var/lib/postgresql/data`。HAPI 设置 R4、PostgreSQL JDBC URL、驱动与 PostgreSQL Hibernate 方言；通过能力声明确认配置生效。准备环境与普通实例使用相同镜像和配置。

HAPI 的容器环境设置如下，数据库主机名由当前实例的数据库容器名替换，密码来自内存变量：

```text
SPRING_DATASOURCE_URL=jdbc:postgresql://<当前实例数据库容器名>:5432/medevolve
SPRING_DATASOURCE_USERNAME=medevolve
SPRING_DATASOURCE_PASSWORD=<当前实例随机密码>
SPRING_DATASOURCE_DRIVERCLASSNAME=org.postgresql.Driver
SPRING_JPA_PROPERTIES_HIBERNATE_DIALECT=ca.uhn.fhir.jpa.model.dialect.HapiFhirPostgresDialect
HAPI_FHIR_FHIR_VERSION=R4
HAPI_FHIR_ENFORCE_REFERENTIAL_INTEGRITY_ON_WRITE=false
HAPI_FHIR_VALIDATION_REQUESTS_ENABLED=false
HAPI_FHIR_VALIDATION_RESPONSES_ENABLED=false
JAVA_TOOL_OPTIONS=-XX:MaxRAMPercentage=60.0
```

这两个运行时值由创建函数传入，不要求操作者手工填写。属性名称以固定镜像的配置确认，启动后用导入、关联读取与数据库持久化验证实际生效；检查配置依据为 [HAPI 官方服务项目](https://github.com/hapifhir/hapi-fhir-jpaserver-starter)。

数据库凭据在准备进程内随机生成，经容器环境传入；固定 `POSTGRES_USER=medevolve`、`POSTGRES_DB=medevolve`，备份不含属主或 ACL。不要写凭据到 manifest。HAPI 仅映射 `127.0.0.1` 的动态端口，数据库不映射主机端口。

- [ ] **1.4：串行运行真实准备。** 先检查受管容器列表；存在其他活动实验时等待。执行下列命令，保存按类型资源统计与导入耗时到 seed 目录；最后删除准备容器、卷与网络。

```bash
python scripts/fhir_agentbench_data.py prepare --data-dir /Users/guagua/data/mimic-iv-fhir-demo/2.1.0/fhir --seed-dir /Users/guagua/data/mimic-iv-fhir-demo/2.1.0/seed
python -m pytest tests/environment/test_fhir_agentbench_seed.py -q
```

通过条件：服务端 ID 集合与全部源文件一致，资源总数为 928,935，种子备份与 manifest 校验通过，准备资源已清理。导入或内存限制不能满足时记录具体失败，不生成伪成功种子。

- [ ] **1.5：提交。** `feat(fhir-agentbench): 添加本地数据导入与种子准备`。

## 任务 2：接入独立实例管理

**文件：** 修改 `server/fhir_agentbench.py`、`server/docker.py`、`server/app.py`、`environments/__init__.py`；新建 `tests/environment/test_fhir_agentbench_lifecycle.py`。

**接口：**

```python
class FhirAgentBenchManager:
    def __init__(self, command, public_host: str, bind_host: str): ...
    async def create(self, config: EnvironmentConfig) -> EnvironmentInstance: ...
    async def release(self, instance_id: str) -> None: ...
```

`command` 注入现有 `docker_command`；数据库恢复命令另用 asyncio 子进程文件 stdin 流传输备份，避免把二进制备份装入普通字符串命令结果。数据库备份／恢复等待上限 1,800 秒，服务启动等待上限 300 秒，超时终止子进程并回收实例。普通短 Docker 命令保持现有 60 秒限制。

- [ ] **2.1：增加生命周期失败测试。** fake Docker 后端必须记录 HAPI、数据库、卷与网络四类资源；模拟数据库恢复失败、FHIR 不就绪、只创建数据库即失败。每种失败后断言本次受管资源归零，已有无关容器仍存在。释放两次均成功。运行 `python -m pytest tests/environment/test_fhir_agentbench_lifecycle.py -q`，确认当前单容器管理器不满足要求。

- [ ] **2.2：实现生命周期与分派。** 资源命名使用同一个 UUID：HAPI `medevolve-<id>`、数据库 `medevolve-<id>-db`、卷 `medevolve-<id>-data`、网络 `medevolve-<id>-net`。所有资源带 `medevolve.managed=true`、环境名与实例 ID 标签，操作前验证标签。数据库使用新密码，新卷恢复 `database.dump`；manifest 中的镜像摘要与本地镜像一致后才启动。

```python
# server/docker.py: create 的环境分派，放在原单容器路径之前。
if config.name == "FHIR-AgentBench":
    return await self._fhir_agentbench().create(config)

# release 按已标记资源识别环境；部分启动失败由专用管理器直接清理。
if labels.get("medevolve.environment") == "FHIR-AgentBench":
    await self._fhir_agentbench().release(instance_id)
    return
```

`_fhir_agentbench()` 构造专用 manager，不复制 PhysicianBench／MedAgentBench 管理逻辑。公共环境名称检查增加 `FHIR-AgentBench`；`seed_data` 按设计表示种子目录。release 在 HAPI 不存在但数据库、卷或网络残留时仍检查并回收整套资源。创建异常后以 `asyncio.shield` 保护清理，清理失败连同原始错误保留。

```python
def _fhir_agentbench(self) -> FhirAgentBenchManager:
    return FhirAgentBenchManager(docker_command, self.public_host, self.bind_host)
```

- [ ] **2.3：调整请求等待与真实恢复验证。** 现有环境客户端 create 超时 240 秒不足以覆盖数据库恢复。仅对本环境使用 2,160 秒请求上限，其他环境维持原值；长等待异步执行并保持进度可见。release 为本环境设置 180 秒。就绪检查比较 R4、Patient 总数 100 与探针原始临床字段校验值；探针校验去掉服务器生成的 `meta` 后与种子一致。

- [ ] **2.4：串行验证两次实例。** 创建第一个实例，读取固定 Patient 与 Observation，记录数据库卷，释放；创建第二个实例并检查相同数据、不同卷。测试进程有写权限，仅在第一个隔离数据库内修改测试记录，以确认第二次恢复回到种子；Agent 工具始终只读。释放后核对带该 UUID 标签的四类资源均不存在。

运行关键回归：

```bash
python -m pytest tests/environment/test_lifecycle.py tests/environment/test_fhir_agentbench_lifecycle.py -q
```

创建与释放仍通过现有管理服务 `/instances` API；不增加独立环境管理 API。

- [ ] **2.5：提交。** `feat(environment): 接入 FHIR-AgentBench 本地实例管理`。

## 任务 3：接入官方检索工具

**文件：** 新建 `environments/fhir_agentbench.py`、`tests/environment/test_fhir_agentbench_tools.py`；修改 `environments/__init__.py`，必要时在 `environments/fhir.py` 增加本环境所需读取能力。

**接口：**

```python
SUPPORTED_TYPES = ("Patient", "Encounter", "Condition", "MedicationRequest",
                   "Procedure", "Observation", "MedicationAdministration",
                   "Location", "Specimen", "Medication")
def get_native_tools(instance: EnvironmentInstance, patient_ids: list[str]) -> list[Tool]: ...
async def read_patient_resources(instance: EnvironmentInstance, patient_id: str) -> list[dict]: ...
async def read_source(instance: EnvironmentInstance, patient_id: str, source_id: str) -> dict: ...
```

- [ ] **3.1：添加真正影响执行的工具测试。** 使用 httpx MockTransport 返回两页 Observation、一个关联 Medication 和患者 B 的记录。断言两页均返回，`text/meta` 从工具输出移除而原始证据仍保留，空结果为 `{}`；患者 B 和未关联共享药品被拒绝。Patient 按 `_id` 查询，不使用 `patient=`；共享 Location/Medication 按 ID 读取，不添加无效患者参数。

- [ ] **3.2：复用官方名称、参数和输出。** 两个工具签名保持如下，封装为 SDK `function_tool(strict_mode=False)`。本地查询 ID 使用 `_id`，名称和参数不变；SDK 异步绑定传输，沿用公共 trace 与预算记录。工具失败继续走既有运行错误通道，不返回看似成功的空结果。

```python
async def get_resources_by_patient_fhir_id(resource_type: str, patient_fhir_id: str) -> dict:
    records = await scoped_reader.by_patient(resource_type, patient_fhir_id)
    return grouped_tool_output(records)

async def get_resources_by_resource_id(resource_type: str, resource_id: str) -> dict:
    records = await scoped_reader.by_id(resource_type, resource_id)
    return grouped_tool_output(records)

def grouped_tool_output(records):
    result = {}
    for record in records:
        cleaned = {k: v for k, v in record.items() if k not in {"text", "meta"}}
        result.setdefault(record["resourceType"], []).append(cleaned)
    return result
```

`scoped_reader` 在工具闭包内绑定当前 instance 与开放患者集合，不设置进程级全局 FHIR 地址。按患者工具的 Medication/Location 访问返回由患者记录引用的支持资源；它属于本地关联访问适配，真实验收必须单列与官方 `patient=` 查询方式的差异。按 ID 工具的共享资源也只允许已有患者引用可达的记录。验证所有分页链接留在实例内。

空患者参数按开放集合查询；空资源 ID 按开放集合读取对应类型，不能展开全库。资源类型必须来自官方列表，批量资源 ID 每个单独验证访问范围。首版关闭官方磁盘缓存，防止实例间共享结果；不增加新工具、参数或计算能力。

- [ ] **3.3：接公共工具与证据入口。**

```python
# environments/__init__.py: get_tools
if instance.environment == "FHIR-AgentBench":
    from .fhir_agentbench import get_native_tools
    return get_native_tools(instance, patient_ids)

# read_evidence: 使用同一环境的受限原始资源读取。
if instance.environment == "FHIR-AgentBench":
    from .fhir_agentbench import read_source
    return await read_source(instance, patient_id, source_id)
```

get_tools 必须仍先检查非空 decision_point 并拒绝。不复用 PhysicianBench 的 10,000 字符截断。完整结果过大时保留真实模型上下文失败，明确归类；不修改结果以掩盖超限。

- [ ] **3.4：真实工具对照。** 同一实例、同一患者，在测试进程内分别调用固定版本的官方工具和本地适配，比较返回 ID 与临床字段。官方客户端仅替换地址／认证及 `id`→`_id`；Patient 和共享资源查询适配差异逐项验证。使用一名患者的小结果验证 SDK function calling，再验证大结果不会静默截断。运行：

```bash
python -m pytest tests/environment/test_fhir_agentbench_tools.py tests/environment/test_patient_boundary.py -q
```

- [ ] **3.5：提交。** `feat(environment): 接入 FHIR-AgentBench 原生检索工具`。

## 任务 4：整理完整患者轨迹

**文件：** 新建 `environments/fhir_agentbench_trajectory.py`、`tests/environment/test_fhir_agentbench_trajectory.py`；修改 `environments/__init__.py`；扩展任务 3 的公共材料读取。

**接口：**

```python
def build_trajectory(patient_id: str, resources: list[dict]) -> PatientTrajectory: ...
def encounter_kind(resource: dict) -> str | None: ...
def time_fields(resource: dict) -> dict: ...
async def prepare(instance: EnvironmentInstance, patient_id: str) -> PatientTrajectory: ...
```

- [ ] **4.1：添加时间与父子就诊测试。** 以下记录构成最小 fixture；断言医院次数为 1、ICU 保留父引用、日期型测量值不覆盖观察时间、无时间诊断不被补日期，Evidence.statement 解码后与输入 JSON 相同。

```python
hospital = {"resourceType": "Encounter", "id": "h",
            "subject": {"reference": "Patient/a"},
            "class": {"code": "AMB"},
            "identifier": [{"system": "http://mimic.mit.edu/fhir/mimic/identifier/encounter-hosp"}],
            "period": {"start": "2136-04-10T20:33:00-04:00", "end": "2136-05-02T16:35:00-04:00"}}
icu = {"resourceType": "Encounter", "id": "i",
       "subject": {"reference": "Patient/a"}, "partOf": {"reference": "Encounter/h"},
       "identifier": [{"system": "http://mimic.mit.edu/fhir/mimic/identifier/encounter-icu"}],
       "period": {"start": "2136-04-22T18:01:00-04:00"}}
diagnosis = {"resourceType": "Condition", "id": "c",
             "subject": {"reference": "Patient/a"}, "encounter": {"reference": "Encounter/h"}}
observation = {"resourceType": "Observation", "id": "o",
               "subject": {"reference": "Patient/a"}, "encounter": {"reference": "Encounter/i"},
               "effectiveDateTime": "2136-04-22T18:47:00-04:00",
               "valueDateTime": "2136-03-27T18:47:00-04:00"}

def test_distinct_observation_times():
    fields = time_fields(observation)
    assert fields["effectiveDateTime"] == observation["effectiveDateTime"]
    assert fields["valueDateTime"] == observation["valueDateTime"]
    assert time_fields(diagnosis) == {}
```

再加入直接药物编码与 Medication 引用两种记录、缺少就诊的微生物子结果。断言原始 derivedFrom/hasMember/specimen 链保留，关联来源可回查，不依据日期接近程度猜测就诊。

- [ ] **4.2：实现完整材料读取。** `read_patient_resources` 优先使用完整患者关联读取；只在 400、404、405、501 明确不支持时逐类型分页回退。回退包含 Patient、Encounter、Condition、Observation、MedicationRequest、MedicationAdministration、MedicationStatement、MedicationDispense、Procedure、Specimen；读取前核对服务端实际支持的搜索参数。扫描引用后按 ID 补齐 Medication、Location、Organization 和需要的临床关联记录，临床资源必须属于当前患者。所有读取结果按 `ResourceType/id` 去重，原始 JSON 保留。

- [ ] **4.3：实现轨迹呈现。** encounter_kind 按 identifier 来源判断；医院按原始起始时间排序，子就诊沿 partOf 呈现；医院记录的 class 不决定来源。将患者原始信息、就诊原始信息、归属事件与无时间记录分段放入 `text`，每条记录都有 source_id 与原始时间字段。关联多条记录时只显示引用，不生成临床因果。

```python
def encounter_kind(resource):
    known = {"encounter-hosp", "encounter-ed", "encounter-icu"}
    for identifier in resource.get("identifier", []):
        kind = identifier.get("system", "").rsplit("/", 1)[-1]
        if kind in known:
            return kind
    return None

def time_fields(resource):
    keys = ("effectiveDateTime", "effectivePeriod", "issued", "valueDateTime",
            "authoredOn", "performedDateTime", "performedPeriod", "onsetDateTime",
            "onsetPeriod", "recordedDate", "dateAsserted", "whenPrepared", "whenHandedOver")
    fields = {key: resource[key] for key in keys if key in resource}
    if resource.get("resourceType") == "Encounter" and "period" in resource:
        fields["period"] = resource["period"]
    for key in ("collectedDateTime", "collectedPeriod"):
        if key in resource.get("collection", {}):
            fields["collection." + key] = resource["collection"][key]
    return fields
```

事件排序使用资源种类对应的事件字段，Observation 用 effective、医嘱用 authoredOn、给药用 effective、操作用 performed、标本用 collected。Condition 无 onset 和发药无 whenHandedOver 时归无事件时间组。MedicationStatement 仅有 dateAsserted 时标为陈述时间，不冒充用药开始。原始日期、时区和字段来源保留；解析失败计数并保留原值，不统一套用固定时区。

- [ ] **4.4：接 prepare_trajectory 并验证真实患者。** 公共入口仅增加本环境分派，现有轨迹压缩和生成方法继续使用相同结构。患者 `28dcf33b-0c52-587f-83ad-2a3270976719` 的服务读取应匹配本地文件：8 条 Encounter（5 hosp、2 ed、1 icu）、1,992 条 Observation、224 条 MedicationRequest、22 条 MedicationAdministration、209 条 MedicationDispense、31 条 MedicationStatement、76 条 Condition、28 条 Procedure、130 条 Specimen。另有 1 条 Patient 和按引用取得的共享资源，不混入上述临床计数。

逐条核对上述患者 sources ID 集合与源文件，读取代表性 evidence 比较原始 JSON（允许服务器生成 meta 不同）。验证 76 条 Condition 没有被补入诊断时间，直接药品编码与 73 个 Medication 引用均保留。读取计数、无时间计数和解析失败计数保存到运行输出。

```bash
python -m pytest tests/environment/test_fhir_agentbench_trajectory.py tests/environment/test_fhir_agentbench_tools.py -q
```

- [ ] **4.5：提交。** `feat(environment): 添加 MIMIC 患者轨迹整理`。

## 任务 5：公共任务执行与真实验收

**文件：** 修改 `environments/__init__.py`；新建 `configs/fhir-agentbench-flash.json`、`tests/environment/test_fhir_agentbench_live.py`、`tests/integration/test_fhir_agentbench_pipeline.py`。必要的 SDK 错误分类兼容修改属于 `runtime/execution.py`，不修改方法算法。

**接口：** 保持现有公共环境与 `python -m experiments.run --config <file>` 入口。FHIR-AgentBench 的 `read_state` 返回现有 JsonObject，至少包含 patient_id、environment、`operation_semantics={"read_only": true}`；任务完成根据答案与检索 trace 判断，不重复读取全量病历作为写后状态。

- [ ] **5.1：验证公共评测输入隔离。** 使用固定轨迹与生成结果跑公共核验 fixture，断言正确 source_id 可读、错误 ID 明确失败；solver 输入不含参考答案、evidence 列表和缓存轨迹。验证 read_state 不使用 PhysicianBench 可写资源筛选。测试继承现有 public evaluator 的核验→质量→解题→完成链，不为本环境建立单独判分算法。

- [ ] **5.2：添加配置。** 从当前 `configs/medagentbench-flash.json` 复制公共模型与预算设置，环境改为 FHIR-AgentBench，manager_url 仍为 `http://127.0.0.1:18084`，seed_data 为本地 seed 目录。生成配置不得包含凭据。

```python
config["environment"] = {
    "name": "FHIR-AgentBench",
    "manager_url": "http://127.0.0.1:18084",
    "seed_data": "/Users/guagua/data/mimic-iv-fhir-demo/2.1.0/seed",
}
config["run_id"] = "fhir-agentbench-flash"
config["output_dir"] = "outputs/fhir-agentbench-flash"
```

患者从实际 Patient 资源 ID 生成确定性排序，以种子 42 选取两名进化患者和一名独立评测患者，保证不重叠；优先选择具有至少两条 encounter-hosp 且 Observation 数较少的患者，以降低首次联调成本。保存实际选取 ID 至配置。两轮各使用一名进化患者，每名生成两个位置；独立评测两题。方向保持现有五类题型，题目以该环境真实只读能力可完成。

Solver、environment_explorer 与 generator 配置沿用已授权 DeepSeek V4.1 Flash，明确设置提供方支持的 context_window/max_output_tokens；如果真实长材料超过预算，保存失败记录，不改变病历或放宽核验。轨迹压缩只用于现有数据探索入口。

- [ ] **5.3：运行环境测试与必要回归。** 真实环境测试通过显式开关 `MEDEVOLVE_FHIR_AGENTBENCH_SEED` 启用，在 finally 中释放实例。测试环境创建、患者轨迹、两个 SDK 工具、证据回查和状态读取；不调用模型。

```bash
python -m pytest tests/environment tests/runtime/test_execution.py tests/integration/test_fhir_agentbench_pipeline.py -q
MEDEVOLVE_FHIR_AGENTBENCH_SEED=/Users/guagua/data/mimic-iv-fhir-demo/2.1.0/seed python -m pytest tests/environment/test_fhir_agentbench_live.py -q
```

测试同时保留 PhysicianBench 与 MedAgentBench 的访问与生命周期行为。现有其他会话改动导致的失败单独定位，不通过还原其文件解决。

- [ ] **5.4：串行运行新任务小样本与两轮闭环。** 先检查管理服务能创建新环境且无其他活动实例，使用现有实验入口，不启动第二套相同端口服务。首次仅跑一名独立患者的两题以验证整个执行链，再启动两轮完整配置；输入配置和失败输出均保存到 outputs。凭据从现有进程环境读取。

```bash
python -m experiments.run --config configs/fhir-agentbench-flash.json
```

核对输出的实际生成数、逐条和整题核验通过数、解题完成数、独立评测数、反馈与更新结果、模型和环境失败、调用及 token 用量。展示一条真实题目、solver 的实际检索与原始证据；将摘要结果写入本次运行目录，不把中间调试记录混入设计文档。三个连续修复均失败时停止并报告待验证假设。

- [ ] **5.5：提交与集成。** 分开提交公共环境接入／配置和必要测试，中文主题分别为 `feat(environment): 接入 FHIR-AgentBench 公共评测`、`test(fhir-agentbench): 添加真实环境与任务执行验证`。完成规定检查后审阅整个 diff，将本任务 worktree 的提交集成回执行时的项目分支，不提交或覆盖其他会话改动。

## 验收对应关系

| 设计验收 | 计划任务 |
| --- | --- |
| 数据完整性与本地服务 | 1.1–1.4 |
| 实例恢复、隔离与清理 | 2.1–2.4 |
| 原生工具与患者范围 | 3.1–3.4 |
| 就诊、时间、药物与原始证据 | 4.1–4.4 |
| 公共核验、真实解题及两轮进化 | 5.1–5.4 |
