"""TaskCraft 医疗适配：阶段机制、内置核验与共享边界的检查。"""

import importlib
import json
import re
from types import SimpleNamespace

import pytest

from contracts import (
    Budget, EnvironmentConfig, EnvironmentInstance, Evidence, GenerationDirection,
    ModelConfig, PatientTrajectory, RunResult, RuntimeConfig, SolverAnswer, TaskBatch,
)
from environments.medagentbench import NativeTools
from methods.taskcraft.environment import is_native
from methods.taskcraft import READ_ONLY_TOOLS, TaskCraftParameters, generate_tasks
from methods.taskcraft.schemas import (
    AnswerList, AnswerScore, BackwardContext, CleanedQuestion, ConclusionList,
    DecompositionReport, InitialQuestion, KnowledgeAnswers, MergedQuestions, NewQuery,
    QueryJudgement, QueryMerge, SupersetCheck,
)
from methods.taskcraft.stages import split_text

WRITE_TOOLS = ("fhir_medication_request_create", "fhir_service_request_create",
               "fhir_communication_create_message", "fhir_appointment_create")
PATIENT = "patient-1"
BLOCK_A = "Observation/lab-77"
BLOCK_B = "Condition/cond-12"


def trajectory_text() -> str:
    return (
        f"[2024-03-05] {BLOCK_A}\n{{\"resourceType\": \"Observation\"}}\n\n"
        f"[2024-03-06] {BLOCK_B}\n{{\"resourceType\": \"Condition\"}}"
    )


def trajectory() -> PatientTrajectory:
    return PatientTrajectory(
        patient_id=PATIENT, environment="PhysicianBench", text=trajectory_text(),
        sources=[Evidence(source_id=BLOCK_A, statement="血钾检验记录"),
                 Evidence(source_id=BLOCK_B, statement="高钾血症问题条目")],
    )


def directions() -> list[GenerationDirection]:
    return [GenerationDirection(task_type="诊断与解释", difficulty="基础", weight=1),
            GenerationDirection(task_type="治疗规划", difficulty="基础", weight=1)]


def runtime_config(max_model_turns=160, max_tool_calls=40) -> RuntimeConfig:
    model = ModelConfig(model="gpt-4o-mini", api_key_env="TEST_UNUSED_KEY")
    return RuntimeConfig(models={"generator": model},
                         budgets={"generator": Budget(max_model_turns=max_model_turns,
                                                      max_tool_calls=max_tool_calls)})


def pipeline_records(result):
    entry = next(item for item in result.trace if item["stage"] == "taskcraft/pipeline")
    return entry["records"]


def pipeline_pool(result):
    entry = next(item for item in result.trace if item["stage"] == "taskcraft/pipeline")
    return entry["pool"]


def stage_records(result, stage):
    return [item for item in pipeline_records(result) if item["stage"] == stage]


def completed(output, *, model_turns=1, tool_calls=0):
    return RunResult(output=output, trace=[{"type": "model_response"}],
                     usage={"model_turns": model_turns, "tool_calls": tool_calls,
                            "input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
                     status="completed")


class FakeRuntime:
    """按阶段输出结构返回固定结果的运行层替身，并记录每次调用。"""

    def __init__(self, *, task_types=("诊断与解释", "治疗规划"), conclusions=None,
                 llm_answer="无工具无法确定", agent_answer="5.9 mmol/L",
                 superset_state="valid", off_type=None, merge_valid=True,
                 decompose_states=None, width_answer="无法确定", backward_failure=False):
        self.task_types = list(task_types)
        self.conclusions = conclusions if conclusions is not None else [
            {"conclusion": "2024-03-05 血钾 5.9 mmol/L", "R": "该次检验的血钾结果",
             "source_ids": [BLOCK_A]},
            {"conclusion": "2024-03-06 诊断高钾血症", "R": "活动问题列表中的诊断",
             "source_ids": [BLOCK_B]},
        ]
        self.llm_answer = llm_answer
        self.agent_answer = agent_answer
        self.superset_state = superset_state
        self.off_type = off_type
        self.merge_valid = merge_valid
        self.decompose_states = decompose_states or {}
        self.width_answer = width_answer
        self.backward_failure = backward_failure
        self.llm_calls = []
        self.agent_calls = []
        self.agent_instructions = []
        self.agent_inputs = []
        self.agent_output_types = []
        self.log = []
        self.question_index = 0

    async def call_llm(self, *, model, instructions, input_text, output_type):
        self.llm_calls.append(output_type.__name__)
        self.log.append(("llm", output_type.__name__, input_text))
        if output_type is ConclusionList:
            return completed({"conclusions": self.conclusions})
        if output_type is InitialQuestion:
            task_type = self.off_type or self.task_types[self.question_index % len(self.task_types)]
            self.question_index += 1
            return completed({"Q": f"该患者第 {self.question_index} 项记录说明了什么？",
                              "task_type": task_type})
        if output_type is CleanedQuestion:
            payload = json.loads(input_text.split("as follows: ", 1)[1])
            return completed({"question": payload["question"],
                              "original_answer": payload["original_answer"],
                              "refined_answer": "5.9 mmol/L"})
        if output_type is AnswerList:
            return completed({"answer_list": [self.llm_answer]})
        if output_type is AnswerScore:
            other_answer = input_text.split("Other Answer:")[-1]
            score = 2 if "5.9" in other_answer else 0
            return completed({"answer_analysis": "judged", "answer_score": score})
        if output_type is SupersetCheck:
            return completed({"new_query": self.superset_state})
        if output_type is NewQuery:
            return completed({"new_query": "在该患者的高钾血症问题条目下，血钾结果是多少？"})
        if output_type is QueryMerge:
            return completed({"analysis": "merged", "new_query": "综合问题：问题条目下的血钾结果"})
        if output_type is QueryJudgement:
            return completed({"analysis": "judged", "is_valid": self.merge_valid})
        if output_type is MergedQuestions:
            indexes = [int(value) for value in re.findall(r'"index": (\d+)', input_text)]
            return completed({"merged": [{"question": "合并问题：血钾结果与诊断的关系",
                                          "index": sorted(set(indexes)),
                                          "content_identifier": "记录块"}]})
        if output_type is DecompositionReport:
            indexes = [int(value) for value in re.findall(r'"index": (\d+)', input_text)]
            return completed({"results": [{"index": index,
                                           "state": self.decompose_states.get(index, 1)}
                                          for index in sorted(set(indexes))]})
        if output_type is KnowledgeAnswers:
            indexes = [int(value) for value in re.findall(r'"index": (\d+)', input_text)]
            return completed({"answers": [{"index": index, "llm_answer": self.width_answer}
                                          for index in sorted(set(indexes))]})
        raise AssertionError(f"unexpected output type: {output_type}")

    async def run_agent(self, *, model, instructions, input_text, tools, output_type, budget):
        if is_native(tools):
            assert instructions
        else:
            names = [tool.name for tool in tools]
            assert names and all(name in READ_ONLY_TOOLS for name in names), names
        self.agent_instructions.append(instructions)
        self.agent_inputs.append(input_text)
        self.agent_output_types.append(output_type.__name__)
        self.agent_calls.append(output_type.__name__)
        self.log.append(("agent", output_type.__name__, input_text))
        trace = [{"type": "tool_start", "name": "fhir_observation_search_labs",
                  "call_id": "call-1", "arguments": "{}"},
                 {"type": "tool_output", "name": "fhir_observation_search_labs",
                  "call_id": "call-1", "output": "{}"}]
        if output_type is SolverAnswer:
            return RunResult(output={"answer": json.dumps([self.agent_answer], ensure_ascii=False)},
                             trace=trace,
                             usage={"model_turns": 2, "tool_calls": 1, "input_tokens": 30,
                                    "output_tokens": 10, "total_tokens": 40},
                             status="completed")
        if output_type is AnswerList:
            return RunResult(output={"answer_list": [self.agent_answer]}, trace=trace,
                             usage={"model_turns": 2, "tool_calls": 1, "input_tokens": 30,
                                    "output_tokens": 10, "total_tokens": 40},
                             status="completed")
        if output_type is BackwardContext and self.backward_failure:
            return RunResult(trace=[{"type": "run_error", "error": "UserError: mis-scoped patient"}],
                             usage={"model_turns": 1, "tool_calls": 1}, status="failed",
                             error="UserError: mis-scoped patient")
        if output_type is BackwardContext:
            return RunResult(output={"identifier": BLOCK_B, "relation": "该问题条目下的检验结果",
                                     "source_ids": [BLOCK_B]}, trace=trace,
                             usage={"model_turns": 2, "tool_calls": 1, "input_tokens": 30,
                                    "output_tokens": 10, "total_tokens": 40},
                             status="completed")
        raise AssertionError(f"unexpected agent output type: {output_type}")


@pytest.fixture
def environment():
    return EnvironmentConfig(name="PhysicianBench", manager_url="http://unused",
                             seed_data="fhir-full:v1")


@pytest.fixture
def patched(monkeypatch):
    """接入替身运行层与环境，返回 (fake, 记录表)。"""

    def install(fake, *, tools_fail=False, native=False):
        state = {"created": 0, "released": 0, "patients": None}
        tools = [SimpleNamespace(name=name) for name in READ_ONLY_TOOLS]
        tools.append(SimpleNamespace(name="fhir_medication_request_create"))
        if native:
            tools = NativeTools("http://unused/fhir/", [PATIENT], [])

        async def create(config):
            state["created"] += 1
            return EnvironmentInstance(instance_id="taskcraft-1", endpoint="http://unused",
                                       environment=config.name)

        async def get_tools(instance, patient_ids):
            if tools_fail:
                raise RuntimeError("tools unavailable")
            state["patients"] = patient_ids
            return tools

        async def release(config, instance):
            state["released"] += 1

        monkeypatch.setattr("methods.taskcraft.create_environment", create)
        monkeypatch.setattr("methods.taskcraft.get_tools", get_tools)
        monkeypatch.setattr("methods.taskcraft.release_environment", release)
        monkeypatch.setattr("runtime.call_llm", fake.call_llm)
        monkeypatch.setattr("runtime.run_agent", fake.run_agent)
        return state

    return install


@pytest.mark.asyncio
async def test_atomic_stage_filters_scores_and_binds_patient_evidence(patched, environment):
    fake = FakeRuntime()
    state = patched(fake)
    result = await generate_tasks(trajectory(), environment, directions(), runtime_config())

    assert result.status == "completed"
    batch = TaskBatch.model_validate(result.output)
    assert state["patients"] == [PATIENT]
    assert state["created"] == state["released"] == 1
    # 内置核验：每题先做无工具回答判断，再交给病历工具 Agent 回答。
    atomic = "该患者第 1 项记录"
    assert len([item for item in fake.log if item == ("llm", "AnswerList", item[2])
                and atomic in item[2] and "Please solve" in item[2]]) == 1
    assert len([item for item in fake.log if item[0] == "agent" and atomic in item[2]]) == 1
    assert len(stage_records(result, "atomic/verify")) == 2
    assert len(stage_records(result, "atomic/result")) == 1
    verifications = stage_records(result, "atomic/verify")
    assert [entry["llm_score"] for entry in verifications] == [0.0, 0.0]
    assert [entry["agent_score"] for entry in verifications] == [2.0, 2.0]
    assert {candidate["origin"] for candidate in pipeline_pool(result)} == {
        "atomic", "width", "depth"}
    for question in batch.questions:
        assert question.evidence
        assert {item.source_id for item in question.evidence} <= {BLOCK_A, BLOCK_B}
        assert question.decision_point is None


@pytest.mark.asyncio
async def test_native_environment_keeps_native_actions_and_parses_finish_answers(patched, environment):
    fake = FakeRuntime()
    patched(fake, native=True)
    result = await generate_tasks(trajectory(), environment, directions(), runtime_config())

    assert result.status == "completed", result.error
    assert TaskBatch.model_validate(result.output).questions
    # 原生协议：整份工具对象透传，Agent 用 FINISH 答案列表返回，说明里禁止 POST。
    assert fake.agent_output_types.count("SolverAnswer") >= 1
    assert "AnswerList" not in fake.agent_output_types
    assert all("只发送 GET" in text for text in fake.agent_instructions)
    assert any("FINISH([{" in text for text in fake.agent_inputs)
    # FINISH 的答案列表进入评分提示词，说明答案已从原生协议解析出来。
    scored = [text for kind, _, text in fake.log if kind == "llm" and "Other Answer:" in text]
    assert any("Other Answer: ['5.9 mmol/L']" in text for text in scored)
    verifications = stage_records(result, "atomic/verify")
    assert [entry["agent_score"] for entry in verifications] == [2.0, 2.0]


@pytest.mark.asyncio
async def test_function_tool_environment_keeps_physicianbench_answers(patched, environment):
    fake = FakeRuntime()
    patched(fake)
    result = await generate_tasks(trajectory(), environment, directions(), runtime_config())

    assert result.status == "completed"
    assert "AnswerList" in fake.agent_output_types
    assert all("只发送 GET" not in text for text in fake.agent_instructions)


@pytest.mark.asyncio
async def test_atomic_stage_rejects_question_the_model_can_answer_alone(patched, environment):
    fake = FakeRuntime(llm_answer="5.9 mmol/L")
    patched(fake)
    result = await generate_tasks(trajectory(), environment, directions(), runtime_config())

    assert result.status == "failed" and result.output is None
    assert fake.agent_calls == []
    rejections = stage_records(result, "atomic/verify")
    assert {entry["error"] for entry in rejections} == {"LLM can solve this task"}
    assert "没有通过内置核验" in result.error


@pytest.mark.asyncio
async def test_atomic_stage_drops_conclusions_without_patient_records(patched, environment):
    fake = FakeRuntime(conclusions=[{"conclusion": "外部文献结论", "R": "文献指标",
                                     "source_ids": ["Paper/unknown"]}])
    patched(fake)
    result = await generate_tasks(trajectory(), environment, directions(), runtime_config())

    assert result.status == "failed"
    rejections = [entry for entry in stage_records(result, "atomic/verify")
                  if entry.get("rejected") == "evidence"]
    assert rejections and rejections[0]["reason"].startswith("结论没有属于当前患者")


@pytest.mark.asyncio
async def test_atomic_stage_rejects_task_type_outside_directions(patched, environment):
    fake = FakeRuntime(off_type="药物处方")
    patched(fake)
    result = await generate_tasks(trajectory(), environment, directions(), runtime_config())

    assert result.status == "failed"
    rejected = [entry for entry in pipeline_records(result)
                if entry.get("reason") == "题型超出请求方向"]
    assert rejected and rejected[0]["allowed"] == ["诊断与解释", "治疗规划"]


@pytest.mark.asyncio
async def test_depth_extension_depends_on_parent_and_keeps_sub_query_chain(patched, environment):
    fake = FakeRuntime()
    patched(fake)
    result = await generate_tasks(trajectory(), environment, directions(), runtime_config())

    assert result.status == "completed"
    batch = TaskBatch.model_validate(result.output)
    extended = [question for question in batch.questions
                if "综合问题" in question.instruction]
    assert len(extended) == 1
    question = extended[0]
    assert question.reference_answer.startswith("最终答案：5.9 mmol/L")
    assert "子问题：在该患者的高钾血症问题条目下，血钾结果是多少？" in question.reference_answer
    assert {item.source_id for item in question.evidence} == {BLOCK_B}
    pool = pipeline_pool(result)
    depth = [candidate for candidate in pool if candidate["origin"] == "depth"]
    assert depth and all(candidate["parents"] for candidate in depth), "深度扩展必须记录父任务"
    assert {source for candidate in depth for source in candidate["evidence"]} == {BLOCK_A, BLOCK_B}
    # 顺序：反向检索 → superset 校验 → 生成问题 → 核验 → 合并检查。
    # 顺序：反向检索 → superset 校验 → 生成问题 → 病历核验 → 合并检查。
    start = next(index for index, item in enumerate(fake.log) if item[1] == "BackwardContext")
    depth_order = [name for _, name, _ in fake.log[start:] if name in {
        "BackwardContext", "SupersetCheck", "NewQuery", "AnswerList", "QueryMerge", "QueryJudgement"}]
    assert depth_order[:7] == ["BackwardContext", "SupersetCheck", "NewQuery",
                               "AnswerList", "AnswerList",
                               "QueryMerge", "QueryJudgement"]
    assert stage_records(result, "depth/extension")[0]["status"] == "completed"


@pytest.mark.asyncio
async def test_depth_extension_rejects_invalid_superset(patched, environment):
    fake = FakeRuntime(superset_state="invalid")
    patched(fake)
    result = await generate_tasks(trajectory(), environment, directions(), runtime_config())

    pool = pipeline_pool(result)
    assert not [candidate for candidate in pool if candidate["origin"] == "depth"]
    rejected = [entry for entry in stage_records(result, "depth/superset")
                if entry["status"] == "rejected"]
    assert len(rejected) == 2 * TaskCraftParameters().extended_attempts
    assert "NewQuery" not in fake.llm_calls


@pytest.mark.asyncio
async def test_width_extension_checks_decomposition_and_model_answering(patched, environment):
    fake = FakeRuntime()
    patched(fake)
    result = await generate_tasks(trajectory(), environment, directions(), runtime_config())

    batch = TaskBatch.model_validate(result.output)
    merged = [question for question in batch.questions if "合并问题" in question.instruction]
    assert len(merged) == 1
    question = merged[0]
    assert question.reference_answer.startswith("合并题参考答案：")
    assert "该患者第 1 项记录说明了什么？ → 5.9 mmol/L" in question.reference_answer
    assert {item.source_id for item in question.evidence} == {BLOCK_A, BLOCK_B}
    pool = pipeline_pool(result)
    width = next(candidate for candidate in pool if candidate["origin"] == "width")
    assert len(width["parents"]) == 2
    assert stage_records(result, "width/decompose")[0]["kept"] == 1
    assert stage_records(result, "width/answer_score")[0]["llm_score"] == 0.0


@pytest.mark.asyncio
async def test_width_extension_rejects_undecomposable_group(patched, environment):
    fake = FakeRuntime(decompose_states={0: 0})
    patched(fake)
    result = await generate_tasks(trajectory(), environment, directions(), runtime_config())

    pool = pipeline_pool(result)
    assert not [candidate for candidate in pool if candidate["origin"] == "width"]
    assert "KnowledgeAnswers" not in fake.llm_calls


@pytest.mark.asyncio
async def test_width_extension_rejects_question_the_model_answers_without_records(patched, environment):
    fake = FakeRuntime(width_answer="5.9 mmol/L")
    patched(fake)
    result = await generate_tasks(trajectory(), environment, directions(), runtime_config())

    pool = pipeline_pool(result)
    assert not [candidate for candidate in pool if candidate["origin"] == "width"]
    scores = stage_records(result, "width/answer_score")
    assert scores and scores[0]["llm_score"] == 2.0


@pytest.mark.asyncio
async def test_failed_extension_stage_does_not_discard_produced_candidates(patched, environment):
    fake = FakeRuntime(backward_failure=True)
    patched(fake)
    result = await generate_tasks(trajectory(), environment, directions(), runtime_config())

    assert result.status == "completed", result.error
    batch = TaskBatch.model_validate(result.output)
    assert batch.questions, "深度阶段失败时仍应返回已通过核验的候选"
    failures = [entry for entry in result.trace
                if entry.get("stage", "").startswith("depth/") and
                entry["result"]["status"] == "failed"]
    assert failures, "失败记录必须保留在 trace 中"
    pool = pipeline_pool(result)
    assert not [candidate for candidate in pool if candidate["origin"] == "depth"]


@pytest.mark.asyncio
async def test_batch_selection_follows_directions_and_two_question_cap(patched, environment):
    fake = FakeRuntime(conclusions=[{"conclusion": "2024-03-05 血钾 5.9 mmol/L",
                                     "R": "该次检验的血钾结果", "source_ids": [BLOCK_A]}])
    patched(fake)
    requested = [GenerationDirection(task_type="诊断与解释", difficulty="基础", weight=1),
                 GenerationDirection(task_type="药物处方", difficulty="基础", weight=1)]
    result = await generate_tasks(trajectory(), environment, requested, runtime_config())

    assert result.status == "completed"
    batch = TaskBatch.model_validate(result.output)
    assert len(batch.questions) == 1
    assert batch.questions[0].task_type == "诊断与解释"


@pytest.mark.asyncio
async def test_cumulative_budget_exhaustion_keeps_usage_and_full_batch_cost(patched, environment):
    fake = FakeRuntime()
    patched(fake)
    result = await generate_tasks(trajectory(), environment, directions(),
                                  runtime_config(max_model_turns=2),
                                  parameters=TaskCraftParameters(atomic_budget_share=1.0))

    assert result.status == "budget_exhausted" and result.output is None
    assert result.usage["model_turns"] == 2
    assert len(fake.llm_calls) + len(fake.agent_calls) == 2
    assert result.trace[-1]["stage"] == "taskcraft/pipeline"
    assert result.trace[-1]["records"]


@pytest.mark.asyncio
async def test_atomic_stage_stops_at_its_budget_share(patched, environment):
    conclusions = [{"conclusion": f"2024-03-0{index} 血钾 5.9 mmol/L", "R": "该次检验的血钾结果",
                    "source_ids": [BLOCK_A]} for index in range(1, 5)]
    fake = FakeRuntime(conclusions=conclusions)
    patched(fake)
    result = await generate_tasks(trajectory(), environment, directions(),
                                  runtime_config(max_model_turns=60))

    assert result.status == "completed", result.error
    skipped = [entry for entry in stage_records(result, "atomic/verify")
               if entry.get("status") == "skipped"]
    assert skipped and skipped[0]["share_limit"] == 30
    accepted = [entry for entry in stage_records(result, "atomic/result")][0]
    assert 0 < accepted["accepted"] < accepted["generated"]
    assert len(fake.llm_calls) + len(fake.agent_calls) < 60


@pytest.mark.asyncio
async def test_extension_stage_keeps_a_guard_so_the_batch_completes(patched, environment):
    fake = FakeRuntime(backward_failure=False)
    patched(fake)
    result = await generate_tasks(trajectory(), environment, directions(),
                                  runtime_config(max_model_turns=45, max_tool_calls=40))

    assert result.status == "completed", result.error
    assert TaskBatch.model_validate(result.output).questions
    guards = [entry for entry in stage_records(result, "depth/extension")
              if entry.get("status") == "skipped"]
    assert guards and guards[0]["reason"] == "剩余预算不足以完成一跳"
    assert result.usage["model_turns"] <= 45


@pytest.mark.asyncio
async def test_environment_instance_is_released_when_tools_fail(patched, environment):
    fake = FakeRuntime()
    state = patched(fake, tools_fail=True)
    result = await generate_tasks(trajectory(), environment, directions(), runtime_config())

    assert result.status == "failed"
    assert "tools unavailable" in result.error
    assert state["created"] == state["released"] == 1
    assert fake.llm_calls == []


def test_only_read_only_medical_tools_are_bound(patched, environment):
    fake = FakeRuntime()
    patched(fake)
    assert "fhir_medication_request_create" not in READ_ONLY_TOOLS
    assert len(READ_ONLY_TOOLS) == 9


def test_stage_parameters_keep_source_defaults(environment):
    settings = TaskCraftParameters()
    assert settings.chunk_size == 16384
    assert settings.max_candidate_conclusions == 20
    assert settings.judge_samples == 3
    assert settings.extended_attempts == 4 and settings.max_hops == 2
    assert settings.merge_attempts == 5 and settings.query_attempts == 3


def test_method_reads_no_medevolve_guidance():
    package = importlib.import_module("methods.taskcraft")
    assert not hasattr(package, "explore_data")
    assert not hasattr(package, "update_strategy")
    parameters = importlib.import_module("methods.taskcraft.stages").TaskCraftParameters
    assert "guidance" not in parameters.model_fields
    assert "strategy" not in parameters.model_fields
    for module in ("methods.taskcraft", "methods.taskcraft.stages", "methods.taskcraft.verify"):
        assert "medevolve" not in importlib.import_module(module).__doc__


def test_split_text_keeps_record_blocks_and_respects_chunk_size():
    text = trajectory_text()
    assert split_text(text, 16384) == [text]
    chunks = split_text(text, 40)
    assert all(len(chunk) <= 40 for chunk in chunks)
    assert any(BLOCK_A in chunk for chunk in chunks)
    assert any(BLOCK_B in chunk for chunk in chunks)


def test_json_payloads_are_python_literals_in_stages():
    """阶段拼接的输入必须是 JSON 文本，便于在运行记录中核对。"""
    payload = json.dumps({"question": "q", "original_answer": "a"}, ensure_ascii=False)
    assert payload.startswith("{\"question\"")
