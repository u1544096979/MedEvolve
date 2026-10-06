# 环境工具可靠性实施计划

> 执行方式：按任务顺序实施并勾选步骤。使用 `executing-plans` 执行；开始修改代码前使用 `using-git-worktrees` 创建隔离工作区。每个任务完成测试后独立提交。

**目标：** 工具交付合法且受配置约束的 JSON，环境探索按提示词高效收敛，MedAgentBench 参数错误能够修正后正常完成。

**结构：** PhysicianBench 在现有工具绑定入口调用一个小型返回序列化模块。环境探索仅修改共享提示词。MedAgentBench 在 GET 入口分类参数错误，原生运行时保留可恢复错误记录并管理恢复状态。

**技术：** Python 3.10+、json、httpx、Pydantic、pytest、pytest-asyncio；沿用项目依赖。

**设计依据：** `docs/environment-tool-reliability/design.md`。

## 全局约束与准备

- `environment.tool_result_limit` 沿用现有字段，默认 10000，环境探索现有配置 5000；限制单位为字符，包含最终 JSON 全部内容。
- 保留完整记录与原有临床字段；原始查询、患者访问检查和写入语义保持有效。返回裁剪不扩展到 MedAgentBench 原生文本通道。
- 环境探索不增加运行时调用检查、去重器、自动重试或额外模型调用；预算和输出格式保持兼容。
- 参数错误不得静默忽略；真实环境故障不得通过恢复逻辑清除。长病历输入与轨迹压缩不在实施范围内。
- 提交按关注点拆分，description 与 body 使用中文；每次明确列出提交文件。

实施前使用 `using-git-worktrees`，检查当前聊天附属工作区及 Git 状态，从实施时的当前分支提交创建隔离工作区。保留当前工作区其他任务的暂存与未暂存修改，不复制、不覆盖、不提交这些文件。开始每个任务前重新核对目标文件，适配已合入的公共环境接口，不用旧版本覆盖并行任务的实现。

在隔离工作区复用可用的项目测试环境。本文测试命令使用 `python -m pytest`，执行前确认该 Python 能导入现有依赖；不要为本次修复新增依赖。

## 任务 1：PhysicianBench 合法 JSON 裁剪

**文件：** 新增 `environments/tool_results.py`、`tests/environment/test_tool_results.py`；修改 `environments/__init__.py`、`tests/environment/test_patient_boundary.py`、`docs/physicalbench-tool-output/README.md`。

**接口：** 新增 `validate_tool_result_limit(limit: int) -> None` 与 `serialize_tool_result(value: object, limit: int, *, write: bool = False) -> str`。`get_tools` 的参数和返回类型保持现状，`contracts.py` 不增加配置字段。

- [x] **1. 添加行为测试。** 在新增测试文件使用以下测试主体，并导入 json、pytest 和上述两个接口：

```python
@pytest.mark.parametrize("limit", [500, 5000, 10000])
def test_complete_records_within_configured_limit(limit):
    records = [{"id": str(i), "valueString": '化验\\n"结果"' * 40}
               for i in range(100)]
    original = {"entries": records, "total": 100, "pages": 1}
    raw = serialize_tool_result(original, limit)
    result = json.loads(raw)
    assert len(raw) <= limit
    assert result["entries"] == records[:len(result["entries"])]
    assert result["truncation"]["total_records"] == 100
    assert result["truncation"]["returned_records"] == len(result["entries"])
    assert result["truncation"]["truncated"] is True
    assert original["entries"] == records

def test_oversized_first_record_is_not_partially_returned():
    result = json.loads(serialize_tool_result(
        {"entries": [{"id": "large", "valueString": "x" * 10000}],
         "total": 1, "pages": 1}, 500))
    assert result["entries"] == []
    assert result["truncation"]["returned_records"] == 0
    assert result["truncation"]["total_records"] == 1

def test_small_result_preserves_data_and_exact_boundary():
    value = {"entries": [{"id": "a"}], "total": 1, "pages": 1}
    raw = serialize_tool_result(value, 500)
    assert json.loads(raw) == value
    large = {"entries": [{"id": "a", "valueString": "x" * 500}]}
    full = serialize_tool_result(large, 10000)
    assert serialize_tool_result(large, len(full)) == full

def test_oversized_write_preserves_success_without_resource_fragment():
    raw = serialize_tool_result(
        {"resourceType": "MedicationRequest", "id": "created",
         "note": [{"text": "x" * 10000}]}, 500, write=True)
    result = json.loads(raw)
    assert len(raw) <= 500
    assert result["status"] == "succeeded"
    assert "note" not in result
    assert "Do not repeat" in result["truncation"]["message"]

def test_limit_too_small_is_rejected():
    with pytest.raises(ValueError):
        validate_tool_result_limit(1)
```

在现有 `test_tool_result_limit_truncates_fhir_results` 中替换尾缀断言：对默认、500、200000 三种上限解析返回 JSON，分别检查长度、裁剪计数、记录完整性和全量结果。增加一次 `get_tools(instance(), ["a"], tool_result_limit=1)` 的拒绝测试，断言 HTTP 请求列表为空。

- [x] **2. 运行新增与受影响测试，确认行为缺口。**

```bash
python -m pytest tests/environment/test_tool_results.py tests/environment/test_patient_boundary.py -q
```

预期：新增接口尚不存在或旧返回无法解析导致失败；记录失败位置，不修改患者边界测试的原有要求。

- [x] **3. 添加最小序列化实现并接入绑定入口。** 新模块仅负责长度验证、记录选择和序列化，可采用以下完整核心实现：

```python
import json

MIN_TOOL_RESULT_LIMIT = 256

def _dump(value):
    return json.dumps(value, ensure_ascii=False, default=str,
                      separators=(",", ":"))

def validate_tool_result_limit(limit: int) -> None:
    if limit < MIN_TOOL_RESULT_LIMIT:
        raise ValueError("tool_result_limit must be at least 256 characters")

def serialize_tool_result(value: object, limit: int, *, write: bool = False) -> str:
    validate_tool_result_limit(limit)
    raw = _dump(value)
    if len(raw) <= limit:
        return raw
    if isinstance(value, dict) and isinstance(value.get("entries"), list):
        records = value["entries"]
        message = ("One record exceeds the limit; narrow the query."
                   if records and len(_dump(records[0])) > limit else
                   "Response incomplete; narrow the query.")
        base = {key: item for key, item in value.items() if key != "entries"}

        def render(count):
            return _dump({**base, "entries": records[:count],
                          "truncation": {"truncated": True,
                              "total_records": len(records),
                              "returned_records": count, "message": message}})

        if len(render(0)) > limit:
            base = {}
        low, high = 0, len(records)
        while low < high:
            middle = (low + high + 1) // 2
            if len(render(middle)) <= limit:
                low = middle
            else:
                high = middle - 1
        if records and low == 0:
            message = "One record exceeds the limit; narrow the query."
        result = render(low)
    else:
        result = _dump({"status": "succeeded" if write else "response_omitted",
                        "truncation": {"truncated": True,
                            "message": ("Write succeeded; response exceeds limit. Do not repeat."
                                        if write else
                                        "One record exceeds the limit; narrow the query.")}})
    if len(result) > limit:
        raise ValueError("tool_result_limit cannot contain the truncation metadata")
    return result
```

256 是可容纳上述最小提示的配置下限，不改变默认值。在 `get_tools` 的 PhysicianBench 分支创建工具前调用 `validate_tool_result_limit`；MedAgentBench 分支保持现状。绑定调用改为：

```python
value = await native(client, *args, **kwargs)
return serialize_tool_result(
    value, tool_result_limit, write=native.__name__ in {
        "fhir_communication_create_message", "fhir_service_request_create",
        "fhir_medication_request_create", "fhir_appointment_create",
    },
)
```

保持访问检查先于返回裁剪。错误返回也经过长度限制，使用结构化 `error`；过长错误用固定短提示替代，不能硬切。`EnvironmentUnavailable` 与 `UnsupportedDecisionPoint` 继续抛出。补充长错误响应测试，验证其 JSON 合法且长度受限。

- [x] **4. 运行任务测试并更新说明。** 使用步骤 2 的命令，预期全部通过。将 `docs/physicalbench-tool-output/README.md` 写为当前规则：可配置上限、完整记录、裁剪字段、最小配置值及单条和写入超限行为。
- [x] **5. 独立提交。**

```bash
git add environments/tool_results.py environments/__init__.py tests/environment/test_tool_results.py tests/environment/test_patient_boundary.py docs/physicalbench-tool-output/README.md
git commit -m 'fix(environment): 按完整记录限制工具 JSON 返回'
```

## 任务 2：环境探索提示词

**文件：** 修改 `methods/medevolve/prompts.py`。现有验证入口为 `tests/method/test_envexplore_environment.py`、`tests/runtime/test_native.py`、`experiments/test_envexplore.py`。

**接口：** `ENVIRONMENT_EXPLORATION: str` 保持原名称，`explore_environment` 和 `EnvironmentExploration` 接口不变。

- [x] **1. 修改共享提示词。** 在现有能力和患者边界规则后加入：

```text
交付指导前必须实际调用本次环境工具，不能仅凭函数清单认定能力已验证。
相同工具与相同参数的查询只执行一次；参数错误后允许修正重试，
允许缩小筛选范围、读取已有下一页，以及写入后的必要回读。
返回被裁剪时缩小查询范围，不重复原查询获取缺失内容。
确认已有信息足以说明工具参数、返回结构、能力与限制后立即交付，
不要求逐个验证全部工具或全部患者。
指导中明确区分已验证能力、调用失败与未验证能力；
提交被接受不能证明数据库持久化成功。
```

保留最终 JSON 和原生 FINISH 规则，不改变配置预算或原生动作格式。
- [x] **2. 检查两条探索通道。** 使用现有共享探索测试确认 PhysicianBench 和 MedAgentBench 均接收同一个 `ENVIRONMENT_EXPLORATION`；复用已有测试，在调用参数检查中增加 `instructions` 与该常量相等的断言。仅验证接线，不新增词句快照测试或模拟模型遵守提示词的测试。
- [x] **3. 运行兼容性测试。**

```bash
python -m pytest tests/method/test_envexplore_environment.py tests/runtime/test_native.py experiments/test_envexplore.py -q
```

预期：共享探索、预算、释放和 FINISH 格式测试全部通过。提示词无法保证真实模型绝不重复；本任务不通过额外运行时机制扩大约束。
- [x] **4. 独立提交。**

```bash
git add methods/medevolve/prompts.py tests/method/test_envexplore_environment.py
git commit -m 'fix(envexplore): 明确工具验证与探索停止规则'
```

## 任务 3：MedAgentBench 参数说明与错误恢复

**文件：** 修改 `environments/medagentbench-functions.json`、`environments/medagentbench.py`、`runtime/native.py`、`tests/environment/test_medagentbench.py`、`tests/runtime/test_native.py`。

**接口：** `NativeTools.execute(action: str) -> JsonObject` 保持 GET/POST/FINISH 结构。可恢复的参数错误返回 `failure_category="tool_error"`、`status_code=400`；错误详情放在 `reply`。成功 GET 与真实环境故障保持原有接口。

- [x] **1. 添加分类与恢复测试。** 环境层采用 `httpx.MockTransport`，逐例返回：明确说明未知查询参数的 GET 400、明确说明筛选参数值无效的 GET 400、无参数错误信息的 GET 400、401、403、404、503，以及 ConnectError 和 ReadTimeout。前两类断言为 `tool_error`，其他类断言为 `environment_error`；患者越界继续使用现有访问拒绝测试。

参数错误响应样例：

```python
httpx.Response(400, json={
    "resourceType": "OperationOutcome",
    "issue": [{"severity": "error", "code": "invalid",
               "diagnostics": "Unknown search parameter status"}],
})
```

在运行时测试中复用 `install_transport`、`native_tools` 与 `MODEL`，添加以下测试主体：

```python
@pytest.mark.asyncio
async def test_parameter_error_recovered_before_finish(monkeypatch):
    install_transport(monkeypatch, [
        {"role": "assistant", "content": "GET Condition?patient=p1&status=active"},
        {"role": "assistant", "content": "GET Condition?patient=p1"},
        {"role": "assistant", "content": 'FINISH(["result"])'},
    ])
    tools = native_tools()

    async def execute(action):
        if "status=active" in action:
            return {"kind": "GET", "request": {"action": action},
                    "reply": "HTTP 400: Unknown search parameter status; correct parameters.",
                    "failure_category": "tool_error", "status_code": 400}
        if action.startswith("FINISH("):
            return {"kind": "FINISH", "request": {"action": action},
                    "reply": "FINISH", "answers": ["result"]}
        return {"kind": "GET", "request": {"action": action},
                "reply": "query succeeded", "data": {"entry": []}}

    monkeypatch.setattr(tools, "execute", execute)
    result = await run_agent(instructions="Solve", input_text="patient", tools=tools,
        output_type=SolverAnswer, model=MODEL,
        budget=Budget(max_model_turns=3, max_tool_calls=2))
    assert result.status == "completed"
    outputs = [item["output"] for item in result.trace if item["type"] == "tool_output"]
    assert outputs[0]["failure_category"] == "tool_error"
    assert outputs[1]["data"] == {"entry": []}
    assert result.usage["tool_calls"] == 2
```

另加“400 后直接 FINISH”“400 后预算耗尽”“503 后成功 GET 再 FINISH”三条序列，分别要求未恢复失败、预算耗尽和环境失败；保留已有无效 FINISH 和异常脱敏测试。
- [x] **2. 运行测试，确认旧分类导致失败。**

```bash
python -m pytest tests/environment/test_medagentbench.py tests/runtime/test_native.py -q
```

- [x] **3. 校准声明并实现分类。** Condition 描述改为 patient 与 category，不补入 status。检查 MedicationRequest 等 GET 描述与参数表的矛盾，只保留参数表明确声明的范围，不新增未经验证的参数。增加清单说明：查询只使用声明参数，服务拒绝参数时可修正重试。保持九个函数和现有 POST 字段。

现有 `test_functions_are_unmodified_official_declaration` 依赖整文件哈希。改为验证九个函数、Condition 与 MedicationRequest 的 GET 参数集合，以及 POST 参数结构，保留上游版本和许可证出处；不要通过更新哈希继续声称声明未经修改。模块说明明确本地声明与错误分类适配。

将 GET 的 HTTPStatusError 单独处理，分类核心如下；`_is_parameter_error` 在同一模块定义：

```python
except httpx.HTTPStatusError as exc:
    response = exc.response
    recoverable = response.status_code == 400 and _is_parameter_error(response)
    return {
        "kind": "GET", "request": request,
        "reply": (f"HTTP {response.status_code}: {response.text}. "
                  + ("Correct the query parameters and retry."
                     if recoverable else "FHIR request failed.")),
        "status_code": response.status_code,
        "failure_category": "tool_error" if recoverable else "environment_error",
    }
```

`_is_parameter_error` 检查 OperationOutcome 的 `issue[].diagnostics` 与 `issue[].details.text`；非 JSON 时检查响应文本。只接受含 `parameter`、`search parameter` 或 `query parameter` 且明确含 unknown、unsupported、unrecognized、invalid、not supported 或 not recognized 的错误描述，匹配不区分大小写。没有这种明确诊断的 400 保持环境错误。ConnectError、ReadTimeout 和其他异常沿用环境错误。不要向服务发验证或重试请求。

原生运行时增加一个 `pending_parameter_error` 标志，初始为 False：

```python
if result.get("failure_category") == "tool_error":
    pending_parameter_error = True
elif result.get("kind") == "GET" and not result.get("failure_category"):
    pending_parameter_error = False
```

可恢复错误通过现有 `tool_output` 留痕，不追加终止性的 `run_error`，不设置 `env_failed`。合法 FINISH 时先保持现有环境失败优先判断，再检查 pending 标志：未恢复则返回 failed 和 tool_error，已恢复则按现有接口完成。真实环境失败标记不因成功 GET 清除；POST 不清除待恢复查询错误。预算耗尽和无效 FINISH 沿用现有规则，不新增查询去重或自动重试。
- [x] **4. 运行步骤 2 的测试并检查边界。** 预期参数错误恢复序列正常完成，原有环境故障、患者访问、原生 POST 和 FINISH 测试仍通过。
- [x] **5. 独立提交。**

```bash
git add environments/medagentbench-functions.json environments/medagentbench.py runtime/native.py tests/environment/test_medagentbench.py tests/runtime/test_native.py
git commit -m 'fix(medagentbench): 校准查询说明并支持参数错误恢复'
```

## 任务 4：完成评价回归与交付

**文件：** 修改 `tests/evaluation/test_evaluator.py`；更新本计划的完成勾选。生产评测逻辑没有已证实缺陷时不修改 `evaluation/evaluator.py`。

**接口：** 消费任务 3 的 completed `RunResult` 与包含参数错误、成功 GET 的 `tool_output` 执行记录；评测继续使用原有 `read_state` 和 `CompletionReview`。

- [x] **1. 添加恢复后的评价接线测试。** 复用现有 `harness`，保留 verifier、quality 和 completion 的原有模拟实现，只替换 solver：

```python
def test_recovered_parameter_error_reaches_completion(harness, monkeypatch):
    h = harness
    h.state = {"prescriptions": [{"id": "written"}]}
    runtime_module = sys.modules["runtime"]

    async def recovered_solver(**kwargs):
        h.model_calls.append(kwargs)
        return RunResult(output={"answer": "处方已创建"}, usage={}, status="completed",
            trace=[
                {"type": "tool_output", "name": "native_action", "output": {
                    "kind": "GET", "failure_category": "tool_error",
                    "status_code": 400, "reply": "Unknown search parameter status"}},
                {"type": "tool_output", "name": "native_action", "output": {
                    "kind": "GET", "reply": "query succeeded", "data": {"entry": []}}},
            ])

    monkeypatch.setattr(runtime_module, "run_agent", recovered_solver)
    result = evaluate(h)
    assert result.status == "completed"
    assert result.completion.completed is True
    assert h.state_reads
    completion_call = next(c for c in h.model_calls
                           if c["output_type"] is CompletionReview)
    payload = json.loads(completion_call["input_text"])
    assert payload["solver_trace"][0]["output"]["failure_category"] == "tool_error"
    assert payload["solver_trace"][1]["output"]["data"] == {"entry": []}
```

此测试验证恢复状态与错误记录进入评测，不证明任意 FINISH 或任意操作已完成；完成真假仍使用现有状态判定测试覆盖。
- [x] **2. 运行相关离线回归。**

```bash
python -m pytest tests/environment/test_tool_results.py tests/environment/test_patient_boundary.py tests/environment/test_medagentbench.py tests/runtime tests/method/test_envexplore_environment.py tests/method/test_envexplore.py tests/evaluation tests/integration experiments/test_envexplore.py experiments/test_envexplore_configs.py -q
git diff --check
```

预期：上述受影响路径通过，无格式错误。测试使用本地模拟，不调用付费模型、不启动远程环境。提示词真实效果需要运行记录才能判断；本次交付不能声称已经证明真实模型完全遵循提示词。
- [x] **3. 核对最终改动并提交。** 确认 config 上限仍传入工具绑定，未增加病历压缩或查询宽容逻辑，提交内容不包含其他工作区改动。

```bash
git add tests/evaluation/test_evaluator.py docs/environment-tool-reliability/implementation-plan.md
git commit -m 'test(evaluation): 验证参数错误恢复后的完成评价'
```

- [ ] **4. 交付实施结果。** 汇报实际通过的测试和四类提交；根据实施会话中用户确认的交付方式处理 worktree 与分支。
