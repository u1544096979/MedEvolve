# Task 3 ACE / Reflexion runner 接入报告

## 上游与医疗适配

- ACE 锁定提交：`82709de050e1db6e6ef2f07bcb0393560b94992a`（Apache-2.0，依赖锁 `upstreams/ace/uv.lock`）。隔离 worker 从该 checkout 加载 `Generator.generate()`、`Reflector.reflect()`、`Curator.curate()`；Generator 的 OpenAI-compatible client shim 提供有界医疗工具循环。Reflector 收到题目答案、工具轨迹、完成评审反馈和训练参考答案；Curator 收到反思及任务元数据，并直接采用其返回的官方合并结果。
- Reflexion 锁定提交：`218cf0ef1df84b05ce379dd4a8e47f17766733a0`（MIT；上游 `hotpotqa_runs/requirements.txt` 锁定 LangChain 0.0.162、openai 0.27.4、tiktoken 0.4.0，没有依赖锁）。隔离 worker 使用官方 `ReactReflectAgent.run()`、`reflect()`、`prompt_agent()` 和官方反思 prompt；子类只替换 Search/Lookup 动作步骤为当前医疗工具会话。模型通过标准库 HTTP 调用 OpenAI-compatible API，兼容上游旧版 openai 环境。
- 两个 worker 都检查锁定源码与必需组件；源码缺失或依赖导入失败时返回带 checkout/依赖指引的错误。模型 key 只从配置指定的环境变量读取。真实的上游 Python 环境、模型和医疗服务没有启动。

## Harness 与 Task 5 接口

工厂 `RUNNERS` 是普通 dict，新增 `ace` 和 `reflexion`。两个 runner 实现既有 `start/execute/learn/freeze/close` 接口。ACE 学习阶段按 Reflector 和 Curator 分开写入 `learning_runs`；Reflexion 按反思和跨任务经验提炼分别记录。宿主应以 `learning_runs` 作为阶段明细成本来源，避免再把 `HarnessStep.run` 摘要中的 call count 累加一次。工具/模型调用次数照常记录；服务未返回 usage 时 token 状态为未知。

Reflexion 另提供 `retry_state` 和 `execute_retry(task, new_tool_session)`：完成一次临床失败评审并释放旧医疗实例后，host 调用 `learn()` 保存反思，再检查 `retry_state["ready"]`；如可重试，为同一任务创建新工具会话并交给 `execute_retry()`。默认最多同题重试 2 次，可由 `reflexion_max_retries` 覆盖；冻结执行、成功评审、未知评审和非临床失败不会启动反思重试。每次新任务调用常规 `execute()` 会清空同题反思，保留跨题经验。Host 负责新实例创建、评分与释放。

Reflexion 经验字段为 `id/task_type/tool_names/condition/procedure/pitfall/source_task_id`；按任务类型优先、工具交集数降序、经验 ID 升序取最多 3 条，并遵守 `experience_limit.max_context_tokens`。缺少 task type 时按空类型处理。经验从 `task_metadata[stable_task_key]` 读取，不从 gold 推断。经验提炼是独立的学习调用，要求使用通用流程表述；写入前拒绝含数字、患者 ID 或完整作答文本的候选。快照只导出 `experiences.json`，不包含逐题 scratchpad、答案或反思记录。

ACE playbook 以官方 playbook 结构保存，并让 Curator 使用官方 `apply_curator_operations` 合并一次。其跨题 Curator 输入限于脱敏后的反思与任务元数据；参考答案只传给训练 Reflector，不加入 Generator 或持久材料。

两个 runner 使用配置中的 `task_metadata`，按 `ACERunner.stable_task_key(task)` / `ReflexionRunner.stable_task_key(task)` 生成 key。Task 5 应以该 helper 的相同输入（patient_id、instruction、decision_point）构造元数据映射，不改变 `execute(SolverInput, tool_session)` 接口。

## 验证

- `/Users/guagua/code/MedEvolve/.venv/bin/python -m pytest tests/harness -q`：130 passed。
- 新增 ACE fake-boundary 测试实际调用锁定 Generator、Reflector 和 Curator 两轮；断言医疗 tool call、官方 playbook ID 合并、冻结学习不调用 worker。
- 新增 Reflexion fake-transport 测试实际调用锁定 ReactReflectAgent 的 actor/reflection 和 prompt，断言医疗 action schema、无 gold actor 输入、跨题经验不含患者 ID/逐题答案、后续任务可读取经验、host 新 session retry 与冻结不反思。
- `git diff --check` 与四个 worker/runner 文件的 `py_compile` 通过。
- 当前 MedEvolve `.venv` 未安装 ACE 的 `python-dotenv` 等完整锁定依赖，也没有 Reflexion 的旧版 LangChain；真实隔离环境中的上游 import 和 API 运行尚未验证。测试仅 fake 外部模型/医疗边界，没有付费模型或 live 医疗请求。

## 提交

- ACE：`cb3a4d3 feat(harness): 接入ACE官方策略更新组件`
- Reflexion：本报告所在提交（当前分支 HEAD）。
