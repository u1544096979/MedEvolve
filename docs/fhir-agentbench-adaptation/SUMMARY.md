# FHIR-AgentBench 本地环境

在本机 Docker 上运行 MIMIC-IV Clinical Database Demo on FHIR v2.1.0，通过官方多轮资源检索工具执行
医疗任务。方法算法、任务结构与公共评测接口沿用项目现有实现。

## 1. 环境概况与组成

| 项目 | 取值 |
| --- | --- |
| 数据 | MIMIC-IV Clinical Database Demo on FHIR v2.1.0，100 名患者、928,935 条资源、13 种类型 |
| FHIR 服务 | `hapiproject/hapi@sha256:4d59628d42b3556b2dcd274d26054a43ba53697ebdae0fd71e6a5475065ed22d`（R4，v8.12.0-2） |
| 数据库 | `postgres@sha256:6b0faa930dc29e5af5af9b50081ad496fad106b85a88ad66aea01f11d4494733`（16.15） |
| 内存预算 | HAPI 3 GB、PostgreSQL 1 GB |
| 种子目录 | `/Users/guagua/data/mimic-iv-fhir-demo/2.1.0/seed/`（`manifest.json` + `database.dump`） |
| 实验配置 | `configs/fhir-agentbench-flash.json` |

一个实例由 HAPI 容器、PostgreSQL 容器、私有网络与独立数据卷组成，从同一份种子恢复，实例之间不共享可写数据库。
FHIR 接口只映射本机回环地址，数据库只在实例私有网络内开放。

| 文件 | 职责 |
| --- | --- |
| `scripts/fhir_agentbench_data.py` | 源数据校验、按依赖分批导入、全量 ID 核对、`pg_dump` 种子备份 |
| `server/fhir_agentbench.py` | 种子清单契约（`load_seed`）、固定镜像与内存常量、双容器实例生命周期 |
| `environments/fhir_agentbench.py` | 两个官方检索工具、患者范围与共享资源可达性、原始资源读取 |
| `environments/fhir_agentbench_trajectory.py` | 就诊组织、时间字段呈现、证据来源、`PatientTrajectory` 输出 |
| `tests/environment/test_fhir_agentbench_*.py` | 种子、生命周期、工具、轨迹、真实环境测试 |
| `tests/integration/test_fhir_agentbench_pipeline.py` | 公共评测链路的输入隔离与只读状态语义 |

## 2. 使用方式

```bash
# 1. 准备种子（一次；已有 manifest.json 时不会覆盖）
python scripts/fhir_agentbench_data.py prepare \
  --data-dir /Users/guagua/data/mimic-iv-fhir-demo/2.1.0/fhir \
  --seed-dir /Users/guagua/data/mimic-iv-fhir-demo/2.1.0/seed

# 2. 启动环境管理服务
MEDEVOLVE_PUBLIC_HOST=127.0.0.1 MEDEVOLVE_BIND_HOST=127.0.0.1 \
  python -m uvicorn server.app:create_app --factory --host 127.0.0.1 --port 18084

# 3. 运行实验
python -m experiments.run --config configs/fhir-agentbench-flash.json

# 4. 真实环境测试（创建并释放一套实例，不调用模型）
MEDEVOLVE_FHIR_AGENTBENCH_SEED=/Users/guagua/data/mimic-iv-fhir-demo/2.1.0/seed \
  python -m pytest tests/environment/test_fhir_agentbench_live.py -q
```

实验串行运行，同一时刻最多一套实例。内存上限可用 `MEDEVOLVE_FHIR_HAPI_MEMORY` 与
`MEDEVOLVE_FHIR_PG_MEMORY` 覆盖。

## 3. 环境契约

| 公共入口 | 本环境行为 |
| --- | --- |
| `create_environment` / `release_environment` | 创建或回收整套资源，`endpoint` 为原生 FHIR 地址；释放可重复调用 |
| `get_tools` | 返回两个官方检索工具；非空 `decision_point` 被拒绝 |
| `prepare_trajectory` | 患者 → 就诊 → 原始记录的 `PatientTrajectory`，`sources` 保存完整原始 JSON |
| `read_evidence` | 按 `ResourceType/id` 读取原始记录，保留服务端 `meta` |
| `read_state` | `{"patient_id", "environment", "operation_semantics": {"read_only": true}}` |

工具名、参数名、参数 schema 与 description 与官方提交 `bbb42909a5a7eb907d1cd91f72a560729e7037ea`
逐字一致，返回按 `resourceType` 分组、移除 `text` 与 `meta` 的记录。支持类型为官方十种：
Patient、Encounter、Condition、MedicationRequest、Procedure、Observation、
MedicationAdministration、Location、Specimen、Medication。完整分页结果不截断。

`decision_point` 为 `null`。`MedicationStatement`、`MedicationDispense` 等公共层可读的记录在轨迹里带
`[工具不可读]` 标记，不构成解题工具可取的证据。

## 4. 验收结果

| 验收项 | 实测 |
| --- | --- |
| 数据完整性 | 服务端 13 种类型的 ID 集合与源文件逐一相等；总数 928,935 |
| 种子 | `database.dump` 259,064,396 字节，sha256 `8ef259ca…`；探针 `Observation/004cbddb…` 去 `meta` 后与源文件一致 |
| 实例生命周期 | 创建 114.1–126.9 秒（种子恢复约 102 秒）；HAPI 0.87–1.14 GiB / 3 GiB，PostgreSQL 0.25–0.33 GiB / 1 GiB |
| 隔离与清理 | 实例间数据卷独立、写入不串实例；释放 0.7 秒且可重复，带标签资源归零 |
| 工具对照 | 7 个临床类型 ID 集合与照搬官方源码的实现完全一致；单次最大返回 1,992 条 / 1,938,615 字符，未截断 |
| 患者轨迹 | 患者 `28dcf33b…` 2,824 条 sources 与递归引用闭包完全相等；就诊 8 条（医院 5 / 急诊 2 / ICU 1）；76 条 Condition 无补写时间 |
| 任务执行 | 两轮进化与独立评测跑通：生成、公共核验、质量评价、独立解题、完成复核、反馈与策略更新全部产出结果 |
| 环境与工具故障 | 完整运行中为 0；模型侧故障单列记录 |

## 5. 与官方的差异与已知限制

本地 FHIR R4 与 Google Cloud Healthcare API 的检索行为差异，逐项适配并已实测：

| 查询 | 官方写法 | 本地结果 |
| --- | --- | --- |
| `Patient?patient=<id>` | 可用 | 400；改用 `_id` |
| `Location?patient=` / `Medication?patient=` | 可用 | 400；按开放患者的引用可达性读取 |
| `<类型>?id=<id>` | 可用 | 400；改用 `_id` |

已知限制：

- **模型上下文上限 1,048,576 token**（实测约 2.611 字符/token）。数据探索一次交出全部进化患者，
  两名最小候选患者的完整轨迹合计约 1,438,703 token 超过上限；`configs/fhir-agentbench-flash.json`
  据此选择轨迹较小的患者。
- **工具不可读证据**。公共核验按被引记录自身的文本判定，而生成模型仍会写出事实位于相邻记录
  （药名在 `Medication`、就诊类型在 `Encounter`）的断言，这类题目会被核验拒绝。轨迹中的
  `[工具不可读]` 标记只暴露边界，强制约束需要生成或核验提示词侧处理。
- 镜像为 arm64；更换平台需要重新确认摘要。
