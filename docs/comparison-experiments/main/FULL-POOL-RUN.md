# 主方法全患者池三环境运行

## 1. 目标与范围

主方法 `medevolve` 在 PhysicianBench、MedAgentBench、FHIR-AgentBench 三个环境的患者池上完成全量任务生成与公共评测。进化口径与对比实验一致：每环境 2 名进化患者各 2 轮，冻结策略后对剩余患者生成任务并评测。

本轮不做初始策略与最终策略的前后对照，不运行除主方法外的其他方法。

## 2. 患者池与划分

| 环境 | 患者池 | 进化患者 | 评测患者 |
| --- | ---: | ---: | ---: |
| PhysicianBench | 108 | 2 | 其余 |
| MedAgentBench | 100 | 2 | 其余 |
| FHIR-AgentBench | 100 | 2 | 其余 |

划分规则：患者 ID 排序后按固定种子 42 打乱，取前 2 名进入进化组，其余按顺序切成每批 8 名的评测批。名单固定写入运行配置，运行中不再改变。

### 2.1 上下文筛查

生成是每患者一次模型调用，输入为完整轨迹正文与原始记录，因此患者是否可用取决于生成模型的实际上下文预算。

预算计算：`上下文上限 − 最大输出 − 指令与 schema 预留`，全部模型角色使用 gpt-4o-mini，上下文上限 128000 token、最大输出 16384 token、指令预留 4096 token，可用输入预算 107520 token。

- FHIR-AgentBench 在源数据上离线估算：逐患者汇总其临床记录字节数，加按引用可达的共享资源（Medication、Location、Specimen），再叠加轨迹正文开销，按实测 2.611 字符/token 换算。
- PhysicianBench 与 MedAgentBench 通过环境逐患者完整读取，按实际载荷字节换算。

超出预算的患者不进入运行，记录为 `context_exceeded`，不更换模型、不压缩轨迹、不改写提示词。筛查产出每环境可用名单与跳过名单。

## 3. 运行结构

每环境分两阶段，避免单次长跑不可续跑：

1. **进化运行**：2 名进化患者、2 轮生成与评测、逐轮反馈与策略更新，产出 `final_strategy.json` 与 `guidance.json`。
2. **批量运行**：每批最多 8 名评测患者，通过 `guidance_path` 读取进化运行的缓存指导，不再执行数据探索与环境探索，不再进化，直接使用冻结策略生成并评测。

同一环境内串行执行；不同环境可并行。批量运行之间互不依赖，单批失败只重跑该批。

## 4. 部署与并发

运行位置为 NUC（`szu-hosp-nuc`，10.7.0.15）：Docker、环境管理服务与实验客户端都在 NUC，患者轨迹不经过 WireGuard。

部署步骤：

1. 停止 NUC 上运行旧代码的 manager 进程与旧的环境容器（`medagentbench-fhir`、`fhir-agentbench-fhir`、`fhir-agentbench-db`），容器只停不删，可随时重启。
2. 将当前分支同步到 `/home/gwh/code/MedEvolve`，复用该目录已有的 venv 与依赖。
3. 在 NUC 用 amd64 镜像重建 FHIR-AgentBench 种子：导入 MIMIC-IV FHIR Demo 2.1.0 全量资源并生成 `database.dump` 与 manifest。Mac 上的种子镜像是 arm64 摘要，NUC 不能复用。
4. 以 `MEDEVOLVE_PUBLIC_HOST=10.7.0.15`、`MEDEVOLVE_BIND_HOST=0.0.0.0`、`MEDEVOLVE_CONTAINER_MEMORY=4g` 启动 manager。

并发与内存上限：

| 项目 | 取值 |
| --- | --- |
| 并行运行数 | 3（每环境 1 个），运行内串行 |
| 峰值实例数 | 3（PhysicianBench 4 GB、MedAgentBench 4 GB、FHIR-AgentBench 3 GB + 1 GB） |
| 峰值内存 | 约 12 GB；NUC 物理 30 GB，保留旧容器时仍有余量 |
| 每次释放后核对 | 带 `medevolve.managed=true` 标签的容器与卷为 0 |

## 5. 模型与预算

全部模型角色使用 OpenLux 的 `gpt-4o-mini`，认证只从 `OPENLUX_API_KEY` 环境变量读取，不写入任何文件。本轮不使用 DeepSeek V4.1 Flash，也不配置 `trajectory_compressor` 角色。

| 阶段 | 角色 | 模型 | 预算 |
| --- | --- | --- | --- |
| 数据探索 | data_explorer | gpt-4o-mini | 30 次模型调用 / 200 次工具调用 |
| 环境探索 | environment_explorer | gpt-4o-mini | 30 / 200 |
| 任务生成 | generator | gpt-4o-mini | 每患者 1 次模型调用、0 次工具调用 |
| 反馈 | feedback | gpt-4o-mini | 30 / 200 |
| 策略更新 | updater | gpt-4o-mini | 30 / 200 |
| 引用核验 | verifier | gpt-4o-mini | 每条证据 1 次调用 |
| 质量评价 | quality | gpt-4o-mini | 3 裁判 |
| 难度评价 | difficulty | gpt-4o-mini | 3 裁判 |
| 独立解题 | solver | gpt-4o-mini | 30 / 200 |
| 完成评价 | completion | gpt-4o-mini | 30 / 200 |

每名患者每轮最多生成两题，证据不足允许少出题或空列表。

## 6. 时间与成本估算

| 环节 | 单位耗时 | 说明 |
| --- | --- | --- |
| 单题公共评测 | 5~6 分钟 | 2 次实例创建与释放 + 8 次模型调用 |
| 单患者生成 | 约 1 分钟 | 1 次调用，输入为完整轨迹 |
| 患者轨迹读取 | 0.7~2 分钟 | FHIR-AgentBench 每实例恢复种子约 2 分钟 |

| 环境 | 预计题数 | 预计时长 |
| --- | ---: | ---: |
| PhysicianBench | 约 200 | 约 22 小时 |
| MedAgentBench | 约 200 | 约 20 小时 |
| FHIR-AgentBench | 取决于筛查 | 约 4 小时 |

三环境并行时墙钟时间由最慢环境决定，约 22~24 小时。模型用量约 100M token，其中公共评测约占七成。

## 7. 输出与验收

输出目录 `outputs/<run_id>/`，每个运行包含 `config.json`、`manifest.json`、`runs.json`、`trajectories/`、`rounds/` 或 `evaluation/`、`summary.json`。

| 验收项 | 判定 |
| --- | --- |
| 覆盖 | 名单内可用患者全部进入运行；跳过患者数与原因单列 |
| 产出 | 每环境实际生成题数、核验通过率、四维质量、难度分、解题完成率 |
| 失败 | 模型故障、环境故障、证据不足、`context_exceeded` 分开计数，不当作难度 |
| 隔离 | 解题输入不含参考答案、证据列表与完整轨迹 |
| 资源 | 运行结束无带管理标签的容器、卷、网络残留 |
| 成本 | 按阶段记录模型调用、工具调用与 token 用量 |

跨环境汇总读出各运行的 `summary.json`，合并为一张表。筛查结果与最终结果分别记录在筛查报告与结果文档中。

## 8. 实现项

| 项目 | 文件 | 说明 |
| --- | --- | --- |
| 缓存指导复用 | `contracts.py`、`experiments/runner.py` | 新增可选 `guidance_path`；存在时读取缓存指导并跳过数据探索与环境探索 |
| 患者池筛查 | `scripts/pool_screening.py` | 枚举患者、读取或估算载荷规模、按种子 42 划分并输出名单与筛查报告 |
| 运行配置 | `configs/main-pool-*.json` | 每环境 1 份进化配置与若干批配置，患者名单由筛查结果确定 |
| 跨环境汇总 | `scripts/summarize_pool.py` | 读取多个运行的 `summary.json`，输出跨环境结果表 |
| FHIR-AgentBench 配置补齐 | `configs/fhir-agentbench-flash.json` | 补 `difficulty` 角色 |
