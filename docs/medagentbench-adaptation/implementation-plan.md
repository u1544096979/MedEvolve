# MedAgentBench 环境适配实施计划

**执行方式：** 在当前会话按任务顺序执行，使用 executing-plans 技能；实施前使用 using-git-worktrees 技能创建独立 worktree。各步骤以复选框记录，不启动并行代理或并行实验。

**目标：** 在官方 MedAgentBench 环境上运行 MedEvolve 的任务生成、公共核验和两轮策略自进化。

**架构：** 环境层管理官方镜像、患者数据和原生动作；公共运行层根据环境返回的执行上下文选择原生文本循环或已有 SDK 循环。方法入口、共享数据结构和单次生成流程保持现有约定。

**技术栈：** Python 3.10+、现有 OpenAI 客户端、httpx、FastAPI、Docker、pytest；不引入官方 controller/worker 框架的依赖。

## 全局约束

- 设计依据为同目录 `design.md`；官方源码固定在 `99260117137b09f04837a8c18d18a1107efa55ae`，镜像首次获取后记录并固定实际摘要。
- 本机管理服务为 `http://127.0.0.1:18084`；实验串行，单容器内存上限为 4 GB，不运行现有的双实例并行 live 测试。
- 原生 POST 只解析并接受提交，不调用真实 FHIR POST；提交正确与数据库变化分别判断。
- 只使用已授权 OpenCode Go 的 `deepseek-v4.1-flash`；认证读取 `OPENCODE_API_KEY`，凭据不进入文件、日志和提交。
- 生成一次调用、零工具调用；初始探索和策略更新保留；`decision_point` 为 null。
- 公共核验对同环境各方法一致；运行故障不能计为难度提升。
- 原子提交使用中文 Conventional Commit description 和 body。真实结果保存在独立 outputs 目录并在聊天报告，不把失败运行改写为成功。

## 文件与接口

| 文件 | 职责 |
|---|---|
| `server/docker.py` | 扩展种子环境与 Docker 标签分派，复用现有隔离和释放 |
| `environments/medagentbench.py` | 官方声明、原生动作、患者访问范围和协议返回 |
| `environments/medagentbench-functions.json` | 指定提交的原始 `funcs_v1.json`，附源码与许可证出处 |
| `environments/__init__.py` | 公共环境分派、轨迹读取和状态语义 |
| `runtime/native.py`、`runtime/execution.py` | 原生文本循环与已有 SDK 循环的分派 |
| `evaluation/prompts.py`、`methods/medevolve/prompts.py` | 通用任务背景与提交/持久化的判断约束 |
| `configs/medagentbench-flash.json` | 使用真实患者 ID、固定镜像和两轮串行实验配置 |
| `tests/environment/test_medagentbench.py` | 动作语义、来源和患者边界 |
| `tests/runtime/test_native.py` | 原生模型循环、预算、失败与输出映射 |
| `tests/integration/test_medagentbench.py` | 公共入口分派、核验材料和方法隔离 |
| `tests/environment/test_medagentbench_live.py` | 显式开启的单实例官方题验证与资源回收 |

保留 `create_environment`、`get_tools`、`prepare_trajectory`、`read_evidence`、`read_state`、`release_environment` 的调用方式，保留 `run_agent` 的参数名。`contracts.py` 不增加字段。

`NativeTools` 在 `environments/medagentbench.py` 中定义为携带原生执行上下文的 `list[Tool]` 子类，初始化参数为 `endpoint: str`、`patient_ids: list[str]`、`functions: list[JsonObject]`。其列表不包含 SDK FunctionTool，属性提供 endpoint、开放患者和官方声明；`async execute(action: str) -> JsonObject` 执行一个原生动作。运行层先识别这个上下文，再进入现有 SDK 类型检查，不能把空列表当作普通无工具 Agent。

execute 返回 `kind`、`request`、`reply`；FINISH 额外返回 `answers`。`request` 保留原始动作和解析出的 URL、payload；`reply` 保留官方执行器的反馈文本。错误分类通过公共 trace 记录，不能把网络故障写成正常临床结果。

`runtime/native.py` 提供 `async run_native_agent(*, instructions, input_text, tools, output_type, model, budget, client_factory, result_factory) -> RunResult`。前七个参数与公共入口对应；公共入口注入现有 `_client` 和 `_result`，复用认证、OpenCode 会话头与凭据脱敏。

## 任务 1：官方来源与单实例管理

**修改：** `server/docker.py`、`environments/__init__.py`、`tests/environment/test_lifecycle.py`。原生声明文件归入任务 2。

**输入：** `EnvironmentConfig(name, manager_url, seed_data)`。

**输出：** 环境名称为 `MedAgentBench` 的现有 `EnvironmentInstance`；生命周期管理支持该名称，但未完成工具适配的入口明确报错，不能误用 PhysicianBench 工具。

- [ ] 阅读 using-git-worktrees，基于已提交设计创建 `codex/medagentbench-adaptation` 独立 worktree；后续命令使用其路径。
- [ ] 下载指定提交源码至临时参考目录，读取 LICENSE 和原生执行器；记录官方 `refsol.py` 的获取方式，取得文件后验证其来源。无法取得时保留原生交互验收，不报告官方评分通过。
- [ ] 获取官方镜像，记录 `RepoDigests`、镜像 ID、架构和存储方式。检查是否依赖外部数据库或共享可写存储；实际启动不能保证独立种子时停止真实实验并报告具体原因。
- [ ] 扩展 Docker 环境分派，将创建标签从固定 PhysicianBench 改为 `config.name`。保留 `--pull=never`、不可变镜像 ID、动态端口、独立存储、失败清理和 `rm --volumes`。

环境名称检查采用以下集合；此阶段仅开放创建/释放，其他未完成入口仍明确拒绝：

```python
SUPPORTED_DOCKER_ENVIRONMENTS = {"PhysicianBench", "MedAgentBench"}
if config.name not in SUPPORTED_DOCKER_ENVIRONMENTS:
    raise EnvironmentUnavailable(f"{config.name} is unavailable")
```

- [ ] 修改生命周期测试：以 MedAgentBench 验证正确标签、不可变镜像和释放行为；原来使用 MedAgentBench 表示“不支持环境”的测试改用 FHIR-AgentBench。
- [ ] 使用现有 fake Docker backend 增加以下断言并运行 `pytest tests/environment/test_lifecycle.py -q`：

```python
containers, commands = docker_backend
manager = DockerManager("127.0.0.1", "127.0.0.1")
config = EnvironmentConfig(name="MedAgentBench", manager_url="http://manager",
                           seed_data="medagentbench@sha256:seed")
instance = await manager.create(config)
assert instance.environment == "MedAgentBench"
run_command = next(c for c in commands if c[0] == "run")
assert "medevolve.environment=MedAgentBench" in run_command
assert "sha256:seed" in run_command
await manager.release(instance.instance_id)
assert not containers
```

- [ ] 串行创建一个真实实例，检查 `/fhir/metadata`，再释放；记录启动耗时和内存限制下的实际结果。
- [ ] 提交 `feat(environment): 添加 MedAgentBench 实例管理`。

## 任务 2：原生动作与患者数据

**新建：** `environments/medagentbench.py`、`environments/medagentbench-functions.json`、`tests/environment/test_medagentbench.py`。

**修改：** `environments/__init__.py`；仅在实际字段缺失时补充公共临床日期提取。

**输入：** 实例地址、开放患者 ID、官方接口声明与单条动作。

**输出：** `NativeTools`、单动作执行结果、原始患者轨迹与可回查的来源。

- [ ] 原样保存官方 `funcs_v1.json`，保留来源提交及许可证归属；不加入 PhysicianBench 的额外接口。
- [ ] 先编写 fake HTTP 测试，覆盖 Patient 身份定位、Observation 查询、合法/非法 POST、FINISH、患者越界与实例外 URL。

关键 POST 测试使用 httpx MockTransport 收集全部 HTTP 请求，要求：

```python
payload = {"resourceType": "MedicationRequest",
           "subject": {"reference": "Patient/p1"}}
result = await tools.execute(
    "POST " + tools.endpoint + "MedicationRequest\n" + json.dumps(payload)
)
assert result["kind"] == "POST"
assert result["request"]["payload"] == payload
assert "accepted and executed successfully" in result["reply"]
assert http_requests == []
```

测试构造的简短 payload 用于核对官方 JSON 解析行为，不将其作为临床处方正确性的测试。

- [ ] 按指定版本复用去除代码围栏、动作分类、GET 的 `_format=json` 和反馈文本；POST 解析错误返回原生错误反馈，不发 HTTP 请求；FINISH 解析 JSON 列表并保存原文。单轮无合法动作返回失败，不增加自动修复调用。
- [ ] 患者访问限制只收紧实验可见范围：GET URL 必须属于当前实例；Patient 搜索保持原始查询参数并限制返回患者；临床查询检查患者条件及结果归属。POST 保存原始 payload 并检查目标患者，但不新增临床 schema 校验改变官方接受语义。
- [ ] GET 返回原生 Bundle，不转换为 PhysicianBench 的 `entries` 摘要，也不自动添加官方执行器没有的全量翻页行为。数据整理通过公共 FHIRClient 单独读取完整材料。
- [ ] 在 `get_tools` 中返回 NativeTools；在 `prepare_trajectory` 与 `read_evidence` 中允许 MedAgentBench 复用患者范围受限的原始读取。验证 `$everything` 在实际镜像中的支持；未支持时在适配内按 Patient、Encounter、Condition、Observation、MedicationRequest、Procedure、ServiceRequest 检索并跟随分页，不改变解题接口。
- [ ] `read_state` 返回实际资源和原生语义，使用以下固定结构；PhysicianBench 返回结构保持原样：

```python
return {
    "patient_id": task.patient_id,
    "environment": instance.environment,
    "resources": resources,
    "operation_semantics": {
        "post": "accept_request_without_persistence",
        "acceptance_proves_database_write": False,
    },
}
```

- [ ] 以日期顺序、Encounter 引用、MRN/资源 ID 不混用和 `ResourceType/id` 回查断言验证轨迹。材料中不得新增病程总结或推断事实。
- [ ] 运行 `pytest tests/environment/test_medagentbench.py tests/environment/test_patient_boundary.py -q`，随后串行回查一名真实患者的来源记录。
- [ ] 提交 `feat(environment): 添加 MedAgentBench 原生交互与轨迹适配`。

## 任务 3：公共原生模型循环

**新建：** `runtime/native.py`、`tests/runtime/test_native.py`。

**修改：** `runtime/execution.py`，不修改 `call_llm` 的单次生成行为。

**输入：** NativeTools、角色指令、输入材料、模型配置、双预算和最终输出类型。

**输出：** 现有 RunResult，含最终结构化输出、原始模型 trace、动作 trace、tokens 和失败类别。

- [ ] 先使用现有 MockTransport 风格测试模拟模型依次返回 GET、POST、FINISH；验证响应反馈作为 user 消息进入下一轮，不使用 SDK function/tool 消息。
- [ ] 在公共入口的 SDK 检查前添加显式分派：

```python
if isinstance(tools, NativeTools):
    return await run_native_agent(
        instructions=instructions, input_text=input_text, tools=tools,
        output_type=output_type, model=model, budget=budget,
        client_factory=_client, result_factory=_result,
    )
```

- [ ] 原生循环使用 `client.chat.completions.create(model=model.model, messages=messages)`，不传 `tools` 或 JSON response_format。官方交互提示和接口列表放入 user 消息；公共角色职责按 `instructions_role` 保留。明确终止格式为 FINISH，不能叠加 SDK 的“最终只输出 JSON 对象”规则。
- [ ] 每次模型响应先计调用与用量，再解析动作；GET/POST 执行前检查工具预算，FINISH 不计工具调用。输出至少包含 `model_request`、`model_response`、`tool_start`、`tool_output`、`run_error` 或 `tool_budget`。
- [ ] 将 FINISH 答案列表直接序列化为 `SolverAnswer.answer`；EnvironmentExploration 将列表中的指导文本按顺序连接成 guidance。对其他公共输出类型，FINISH 列表必须包含一个匹配该类型的对象，由 `output_type.model_validate` 本地验证；不调用模型修复。

解题最终映射采用：

```python
output = SolverAnswer(
    answer=json.dumps(answers, ensure_ascii=False)
).model_dump(mode="json")
```

- [ ] 验证原生输出不会收到 JSON 格式强制要求，且提交内容可被公共评测的 trace 过滤器读取：

```python
assert result.status == "completed"
assert json.loads(result.output["answer"]) == ["result"]
assert result.usage["model_turns"] == 3
assert result.usage["tool_calls"] == 2
assert len([t for t in result.trace if t["type"] == "tool_output"]) == 2
assert all("tools" not in r and "response_format" not in r for r in model_requests)
assert model_requests[1]["messages"][-1]["role"] == "user"
```

- [ ] 验证零预算不执行操作、最后一轮未 FINISH 为预算耗尽、网络失败保留已完成 trace 和用量、密钥脱敏。GET 请求错误保留原生反馈供下一轮处理；无法完成环境访问的最终结果不能标记 clinical。
- [ ] 运行 `pytest tests/runtime/test_native.py tests/runtime/test_execution.py -q`，确认 PhysicianBench SDK 回归通过。
- [ ] 提交 `feat(runtime): 接入原生文本动作执行循环`。

## 任务 4：任务背景与公共判定

**修改：** `evaluation/prompts.py`、`methods/medevolve/prompts.py`。

**新建：** `tests/integration/test_medagentbench.py`。

**输入：** 原始证据、任务正文、solver trace 和 observed_state。

**输出：** 相同 Verification、QualityReview、CompletionReview 与方法反馈输入。

- [ ] 编写一条提交任务的公共集成用例：以 fake 环境和 fake 模型隔离外部调用，确认 completion 输入保留 POST 的 URL、payload、接受结果和 `operation_semantics`；未带 reference_answer 或 evidence 的 SolverInput 保持原样。
- [ ] 公共完成提示新增通用区分：要求“提交请求”时核对实际请求和内容；要求“数据库创建/修改”时核对资源变化；环境能力说明不能覆盖题目实际要求，不按方法名称放宽判定。
- [ ] 环境指导与通用生成约束明确背景、当前时间和给定规则必须写进 instruction；任务只要求实际开放的操作。将现有“药物处方必须实际开具”表述改为“按环境原生语义明确要求开具或提交处方请求”，提交动作不得声称持久化，不能把处方题改成历史解释题。
- [ ] 不增加 context 字段，不改生成调用数量，不在 methods/medevolve 中按环境名分支；代码检查：

```python
from pathlib import Path
from contracts import GeneratedQuestion

assert "context" not in GeneratedQuestion.model_fields
assert "MedAgentBench" not in Path("methods/medevolve/__init__.py").read_text()
```

- [ ] 运行公共集成、方法与评测测试：`pytest tests/integration/test_medagentbench.py tests/method tests/evaluation -q`。fake 模型用例只验证材料衔接，真实判定另用已授权模型检查正确提交、错误内容、未提交三种情况。
- [ ] 提交 `fix(evaluation): 区分原生提交与数据库写入判定`。

## 任务 5：真实验证、配置与两轮闭环

**新建：** `tests/environment/test_medagentbench_live.py`、`configs/medagentbench-flash.json`。

**输入：** 固定镜像、已授权模型和经过数据检查的真实患者。

**输出：** 串行原生验证记录、小样本结果、两轮闭环与独立患者评测产物。

- [ ] 新 live 测试仅在 `MEDEVOLVE_MEDAGENTBENCH_TEST_IMAGE` 设置时运行；用一个实例检查官方查询题与提交题，finally 中释放。第二个实例必须在第一个释放后创建，核对种子和清理，不调用旧的双实例测试。
- [ ] 官方任务从固定提交 test_data_v2.json 读取：查询选 `task1_1`，提交选 `task5_1`；以实际数据定位其 eval_MRN 对应资源 ID。保留 context、instruction 和官方动作反馈。取得 refsol 时另外运行官方评分；没有评分文件则仅报告交互与原始数据核对结果。
- [ ] 从真实镜像中选定并记录三名数据足够的患者，两名仅用于进化，一名仅用于独立评测。使用原始资源 ID 配置，不用示例 ID 或 MRN 直接代替；无法找到足够材料则报告实际数据缺口，不编造配置。
- [ ] 复制现有 Flash 模型配置，只改运行 ID、环境、固定镜像、患者组和输出目录；两轮各一名进化患者，配置由以下结构确定：

```python
config = json.loads(Path("configs/physicianbench-flash.json").read_text())
config.update(run_id="medagentbench-flash", output_dir="outputs/medagentbench-flash")
config["environment"] = {
    "name": "MedAgentBench", "manager_url": "http://127.0.0.1:18084",
    "seed_data": inspected_image_digest,
}
config["evolution_patients"] = selected_patient_ids[:2]
config["evaluation_patients"] = selected_patient_ids[2:3]
config["round_patients"] = [[p] for p in selected_patient_ids[:2]]
validated = ExperimentConfig.model_validate(config)
Path("configs/medagentbench-flash.json").write_text(
    validated.model_dump_json(indent=2) + "\n", encoding="utf-8"
)
```

`inspected_image_digest` 来自任务 1，`selected_patient_ids` 来自本任务真实数据检查。模型、预算和题型分配保持现有 Flash 配置；若证据不支持某方向，允许少出题。

- [ ] 先验证同一患者单次生成两题和公共评测，人工查看题目、背景、来源、候选答案与核验结果。真实调用使用运行进程中的凭据，不打印或写入 key。
- [ ] 小样本完成且环境执行问题解决后运行：

```bash
python -m experiments.run --config configs/medagentbench-flash.json \
  --run-id medagentbench-flash-closed-loop \
  --output-dir outputs/medagentbench-flash-closed-loop
```

正式目录存在时使用新的时间后缀，不覆盖此前结果。

- [ ] 运行非 live 回归 `pytest tests --ignore=tests/environment/test_live_image.py --ignore=tests/environment/test_medagentbench_live.py -q`。真实测试单独串行执行，确认最终受管实例和数据卷已释放。
- [ ] 核对两份反馈、两份策略、单次生成调用数、真实失败类别和 held-out 隔离；统计生成量、有效量、完成量、模型调用与 tokens，并展示一道人为检查的新任务。所有结果以实际产物为依据。
- [ ] 提交 `test(environment): 添加 MedAgentBench 串行环境验证` 和 `chore(experiment): 添加 MedAgentBench Flash 配置`，分别保持独立关注点；代码与验证完成后将 worktree 的提交集成回当前项目分支。
