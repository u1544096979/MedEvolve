# Task 2 本地 Harness 接入报告

## 交付内容

- `harnesses/medical_bridge.py`：增加只读医疗工具会话、宿主侧身份校验 inspect；只读状态隐藏并拒绝写工具，MedAgentBench 仍保留 GET/FINISH。
- `harnesses/runners/__init__.py`：普通 dict runner 工厂 `RUNNERS` / `create_runner(name, *, config, group_dir, bridge_url)`，惰性导入 OpenClaw 与 Hermes；接受 Task 2 flat config 或 `configs/harness-evolution.json` 的 `models`、`limits`、`local_harnesses`。
- `harnesses/runners/openclaw.py`：受控 OpenClaw Gateway 子进程、独立 state/workspace/端口；直接配置 OpenAI-compatible provider；每题新 session、同题 feedback 沿用 session；>=10 assistant turns 时触发官方 review 并只接受本题开始后新写入的 review 记录；freeze 前停止 Gateway 并导出 skills/memory。
- `harnesses/runners/hermes.py`、`hermes_worker.py`：每次调用使用隔离 Python worker，导入前设置 `HERMES_HOME`；通过 Hermes 官方 registry 注册医疗工具；每题新 session、反馈回合沿用；学习角色可单独配置；冻结保留 skills/memory 读取并拦截写操作。经验 digest 仅比较可导出的 skills/memory 文件。
- 新增 OpenClaw、Hermes worker/runner 和医疗桥文件级测试。

## 配置接口

`create_runner` 必填参数：`name`（openclaw 或 hermes）、`config`、`group_dir`、`bridge_url`。从完整 harness 配置读取：`models.execution`、`models.learning`、`limits.execution.max_model_turns`、`limits.learning.max_model_turns`、`limits.timeout_seconds`、`limits.learning_timeout_seconds`、`local_harnesses.<name>.source_dir`、`program_path`、`python_path`。模型项使用 `model`、`base_url`、`api_key_env`。Flat config 对应 `execution_model`、`learning_model`、`execution_budget`、`learning_budget`、`timeout_seconds`、`learning_timeout_seconds`；可选 `frozen`、`client_id`、`gateway_port`、`review_wait_seconds`。OpenClaw native review 使用前台执行的模型/Provider，Task 5 成本记录应将 learning 阶段模型记为 execution 模型；若 upstream review 不返回 usage，标记未知，不估造 token/cost。Hermes learning 调用使用 `learning_model`。

## 原生 review 判定

OpenClaw 使用 Gateway embedded agent path。上游 `src/skills/workshop/experience-review.ts` 的 review 条件包含 Workshop 可用、模型/session 信息齐全、成功事件、非关闭模式、至少 10 次模型迭代，随后等待空闲窗口。实现读取独立 Gateway state SQLite 的 `config_machine_state` 中 `state_key='skills.curatorState'`，定位 `experienceReviews[sha256Hex(resolve(workspace))]`，并要求 `attemptedAtMs` 晚于本任务开始时间及执行前基线；因此历史 review 不会成为本题成功记录。源码证据位于锁定 checkout `.superpowers/sdd/implementation-plan/upstreams/openclaw/src/skills/workshop/experience-review.ts`、`src/config/types.agents.ts` 和 `src/agents/agent-runner.ts`。Hermes 的 detached review 不能将 usage 可靠关联当前阶段；worker 关闭自动 nudge，以当前显式 feedback 回合训练。

## 提交

- `3168edf feat(harness): 接入OpenClaw本地训练与冻结测试`
- `28616c2 feat(harness): 接入Hermes本地训练与冻结测试`

## 验证和限制

- `/Users/guagua/code/MedEvolve/.venv/bin/python -m pytest tests/harness -q`：123 passed。
- `git diff --check`：通过。
- 真实 OpenClaw Gateway、Hermes upstream Python 环境、外部 provider、模型请求与医疗 live 服务未启动或调用；因此真实依赖运行行为仍未验证。测试使用假命令/进程、假 Hermes agent/registry 和本地假医疗边界。


## Scoped review 修复

- Gateway CLI JSON 保持原始 response envelope；runner 现在优先读取 `result.meta.agentMeta`，并兼容扁平 envelope，提取嵌套 usage、assistantTurns 与 toolCalls。执行和 feedback 共用此解析逻辑；原生 status/error/in_flight 与错误 payload 会归为失败。
- 答案提取支持 Gateway `result.payloads[].text` 和 exec 扁平 `final` / `finalResponse` 字段。
- 测试夹具改为上游 Gateway 的 `status` + `result.payloads` + `result.meta.agentMeta` 形状，并验证 10-turn 原生 review 等待路径及 feedback usage。增加 `agent-exec-result` 扁平 final 和本机 SQLite review state 读取测试。
- review state 源码确认：`src/skills/workshop/collection-review-state.ts` 使用 `sha256Hex(path.resolve(workspaceDir))` 写入 `skills.curatorState.lastResult.experienceReviews[workspaceKey]`；SQLite `config_machine_state.value_json` 保存整份状态。runner 的只读查询及 workspace hash 与该实现一致，且仍要求 review 时间晚于本题开始/基线。
- 验证：`tests/harness/test_local_openclaw.py tests/harness`，125 passed；`git diff --check` 通过。
