"""TaskCraft 医疗适配：阶段机制、内置核验与共享边界的检查。"""

import asyncio
import importlib
import json
import random
import re
from collections import Counter
from types import SimpleNamespace

import pytest

from contracts import (
    Budget, EnvironmentConfig, EnvironmentInstance, Evidence, GeneratedQuestion,
    GenerationDirection, ModelConfig, PatientTrajectory, RunResult, RuntimeConfig,
    SolverAnswer, TaskBatch,
)
from environments import get_tools
from environments.errors import EnvironmentUnavailable, PatientAccessError
from environments.medagentbench import NativeTools
from methods.common import GenerationBudget, summarize_results
from methods.taskcraft.agent import run_search_agent, step_kind
from methods.taskcraft.environment import bind_generation_tools, is_native
from methods.taskcraft import READ_ONLY_TOOLS, TaskCraftParameters, generate_tasks
from methods.taskcraft.ledger import Ledger
from methods.taskcraft import prompts
from methods.taskcraft.scheduling import derive_seed, select_candidates
from methods.taskcraft.schemas import (
    AgentThought, AnswerList, AnswerScore, BackwardContext, CleanedQuestion,
    ConclusionList, DecompositionReport, InitialQuestion, KnowledgeAnswers,
    MergedQuestions, NativeAction, NewQuery, QueryJudgement, QueryMerge,
    SearchAction, SupersetCheck, TrajectoryScore,
)
from methods.taskcraft.verify import score_recall, verify_query
from methods.taskcraft.stages import Candidate, Pipeline, split_text
from tests.environment.test_patient_boundary import native_fhir  # noqa: F401

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


class FakeTool:
    """替身病历检索工具：保留工具名并记录每次调用参数。"""

    def __init__(self, name):
        self.name = name
        self.description = f"{name} 病历检索"
        self.params_json_schema = {"type": "object", "properties": {}}
        self.calls = []

    async def on_invoke_tool(self, context, arguments):
        self.calls.append((context.tool_name, arguments))
        return json.dumps({"resourceType": "Observation", "id": "lab-77"})


class FakeRuntime:
    """按阶段输出结构返回固定结果的运行层替身，并记录每次调用。"""

    AGENT_TOOL = "fhir_observation_search_labs"
    BACKWARD_MARK = "Conduct divergent searches"
    FINAL_OBJECT_MARK = '"identifier"'

    def __init__(self, *, task_types=("诊断与解释", "治疗规划"), conclusions=None,
                 llm_answer="无工具无法确定", agent_answer="5.9 mmol/L",
                 superset_state="valid", off_type=None, merge_valid=True,
                 decompose_states=None, width_answer="无法确定", backward_failure=False,
                 agent_action_rounds=1, tool_calls_per_action=1, fail_agent_runs=0,
                 score_failures=(), final_answer_with_tools=False, backward_failures=0,
                 final_answer_as_object=False, planning_answer_as_object=False):
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
        # 前几次反向检索失败，用于验证失败尝试进入 trace。
        self.backward_failures = 0
        self.backward_runs = 0
        # 首次检索前发出几次工具调用；检索步数达到 agent_action_rounds 后给出答案。
        self.agent_action_rounds = agent_action_rounds
        self.tool_calls_per_action = tool_calls_per_action
        self.fail_agent_runs = fail_agent_runs
        self.score_failures = set(score_failures)
        # 工具调用轮次里是否同时给出 final_answer 字段。
        self.final_answer_with_tools = final_answer_with_tools
        # 强制终答的 answer 是直接给出 JSON 值还是序列化后的 JSON 文本。
        self.final_answer_as_object = final_answer_as_object
        # 规划的 answer 是自由文本还是任务要求的 JSON 对象。
        self.planning_answer_as_object = planning_answer_as_object
        self.planning_answer_object = {"identifier": BLOCK_B, "relation": "该问题条目下的检验结果"}
        self.backward_failures = backward_failures
        self.llm_calls = []
        self.agent_calls = []
        self.agent_instructions = []
        self.agent_inputs = []
        self.agent_output_types = []
        self.llm_instructions = []
        self.agent_steps = {}
        self.agent_runs = 0
        self.score_requests = 0
        self.action_requests = []
        self.log = []
        self.question_index = 0

    def _task(self, input_text):
        """取记忆里的任务文本：任务块之后是规划、反思与 action 块。"""
        head = re.split(r"\n\n(?=\[FACTS RESULT LIST\]:|\[HISTORY TRAJECTORY ANALYSIS\]:|"
                        r"Calling tool)", input_text, maxsplit=1)[0]
        return head

    def _final_answer(self, text, *, as_list: bool):
        """按调用方声明的输出结构给出答案：text 用于识别单对象关系题。"""
        if self.FINAL_OBJECT_MARK in text:
            answer = {"identifier": BLOCK_B, "relation": "该问题条目下的检验结果",
                      "source_ids": [BLOCK_B]}
            return [answer] if as_list else answer
        return [self.agent_answer] if as_list else {"answer_list": [self.agent_answer]}

    def _agent_action(self, instructions, input_text, *, native):
        """返回一次 action 响应；None 表示本次检索运行失败。"""
        task = self._task(input_text)
        if self.backward_failure and self.BACKWARD_MARK in task:
            return None
        if self.BACKWARD_MARK in task:
            self.backward_runs += 1
            if self.backward_runs <= self.backward_failures:
                return None
        served = self.agent_steps.get(task, 0)
        if served == 0:
            self.agent_runs += 1
            if self.agent_runs <= self.fail_agent_runs:
                return None
        self.agent_steps[task] = served + 1
        if served < self.agent_action_rounds:
            if native:
                return {"action": "GET http://unused/fhir/Observation?code=2823-3"}
            return {"tool_calls": [{"name": self.AGENT_TOOL,
                                    "arguments": {"patient": PATIENT}}
                                   for _ in range(self.tool_calls_per_action)],
                    "final_answer": ({"answer_list": ["工具调用响应里的终答"]}
                                     if self.final_answer_with_tools else None)}
        if native:
            answers = self._final_answer(instructions + input_text, as_list=True)
            return {"action": f"FINISH({json.dumps(answers, ensure_ascii=False)})"}
        return {"tool_calls": [],
                "final_answer": self._final_answer(instructions, as_list=False)}

    def _model_response(self, output_type, payload):
        """与 runtime.call_llm 同一口径：返回前按本次请求声明的结构校验。"""
        try:
            output = output_type.model_validate(payload).model_dump(mode="json")
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            return RunResult(trace=[{"type": "run_error", "error": error}],
                             usage={"model_turns": 1, "tool_calls": 0},
                             status="failed", error=error)
        return completed(output)

    def _thought(self, think, text):
        """源码 extract_think_and_answer 的 <think>/<answer>；answer 也常带任务要求的 JSON 对象。"""
        answer = self.planning_answer_object if self.planning_answer_as_object else text
        return self._model_response(AgentThought, {"think": think, "answer": answer})

    async def call_llm(self, *, model, instructions, input_text, output_type):
        self.llm_calls.append(output_type.__name__)
        self.llm_instructions.append(instructions)
        self.log.append(("llm", output_type.__name__, input_text))
        if output_type in (SearchAction, NativeAction):
            native = output_type is NativeAction
            self.agent_calls.append(output_type.__name__)
            self.agent_instructions.append(instructions)
            self.agent_inputs.append(input_text)
            self.agent_output_types.append(output_type.__name__)
            self.action_requests.append(input_text)
            self.log.append(("agent", output_type.__name__, input_text))
            action = self._agent_action(instructions, input_text, native=native)
            if action is None:
                return RunResult(trace=[{"type": "run_error",
                                         "error": "AgentGenerationError: action response missing"}],
                                 usage={"model_turns": 1, "tool_calls": 0},
                                 status="failed",
                                 error="AgentGenerationError: action response missing")
            return completed(action)
        if "An agent tried to answer a user query" in instructions:
            # 源码 provide_final_answer 把答案放在 answer 字段；源码 FinalAnswerTool 的
            # answer 类型是 any，因此这里既可能是 JSON 值，也可能是序列化后的 JSON 文本。
            answer = self._final_answer(instructions + input_text, as_list=False)
            payload = {"think": "按记忆给出答案",
                       "answer": (answer if self.final_answer_as_object
                                  else json.dumps(answer, ensure_ascii=False))}
            return self._model_response(output_type, payload)
        if output_type is AgentThought:
            if "Below I will present you a task." in input_text:
                return self._thought("先核对该片段已知事实",
                                     "### 1. Facts given in the task\n病历片段")
            if "You are a world expert at making efficient plans" in input_text:
                return self._thought("先检索后作答", "1. 用病历工具检索该记录")
            if "We previously constructed a fact list" in input_text:
                return self._thought("保留已学到的事实",
                                     "### 1. Facts given in the task\n病历片段")
            if "You are now tasked with updating the action plan" in input_text:
                return self._thought("继续检索", "1. 继续用病历工具检索")
            raise AssertionError(f"unexpected AgentThought input: {input_text[:80]}")
        if output_type is TrajectoryScore:
            return completed({"analysis": "轨迹仍在推进", "score": 8})
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
            self.score_requests += 1
            if self.score_requests in self.score_failures:
                return RunResult(trace=[{"type": "run_error", "error": "model_error: judge failed"}],
                                 usage={"model_turns": 1, "tool_calls": 0}, status="failed",
                                 error="model_error: judge failed")
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
        raise AssertionError("检索 Agent 不再走公共一次性 Agent 运行入口")


class WidthRuntime(FakeRuntime):
    """宽度阶段替身：按给定分组返回合并题，并按各组答案区分无工具作答的评分。"""

    def __init__(self, groups, *, states=None, width_answers=None, **kwargs):
        super().__init__(**kwargs)
        self.groups = [list(group) for group in groups]
        self.states = dict(states or {})
        self.width_answers = dict(width_answers or {})

    async def call_llm(self, *, model, instructions, input_text, output_type):
        if output_type is MergedQuestions:
            self.llm_calls.append(output_type.__name__)
            self.log.append(("llm", output_type.__name__, input_text))
            return completed({"merged": [
                {"question": f"合并问题 {position}", "index": list(group),
                 "content_identifier": "记录块"}
                for position, group in enumerate(self.groups)]})
        if output_type is DecompositionReport:
            self.llm_calls.append(output_type.__name__)
            self.log.append(("llm", output_type.__name__, input_text))
            indexes = [int(value) for value in re.findall(r'"index": (\d+)', input_text)]
            return completed({"results": [{"index": index, "state": self.states.get(index, 1)}
                                          for index in indexes]})
        if output_type is KnowledgeAnswers:
            self.llm_calls.append(output_type.__name__)
            self.log.append(("llm", output_type.__name__, input_text))
            indexes = [int(value) for value in re.findall(r'"index": (\d+)', input_text)]
            return completed({"answers": [
                {"index": index, "llm_answer": self.width_answers.get(index, "无法确定")}
                for index in indexes]})
        if output_type is AnswerScore:
            self.score_requests += 1
            self.log.append(("llm", output_type.__name__, input_text))
            golden, other = input_text.rsplit("Golden Answer:", 1)[-1].split("Other Answer:")
            value = other[0].strip().strip("[]").strip().strip("'\"")
            return completed({"answer_analysis": "judged",
                              "answer_score": 2 if value and value in golden else 0})
        return await super().call_llm(model=model, instructions=instructions,
                                      input_text=input_text, output_type=output_type)


def atomic_pool(count, *, task_types=None):
    """构造可直接驱动扩展阶段的原子候选；参考答案逐题不同以便区分分组来源。"""
    types = list(task_types or [])
    return [Candidate(
        question=GeneratedQuestion(
            instruction=f"该患者第 {index} 项记录说明了什么？",
            task_type=types[index % len(types)] if types else "诊断与解释",
            reference_answer=f"答案 {index}",
            evidence=[Evidence(source_id=BLOCK_A, statement=f"第 {index} 项记录")]),
        origin="atomic", locator=BLOCK_A) for index in range(count)]


def stage_pipeline(monkeypatch, fake, *, tools=None):
    """用替身运行层直接驱动一个阶段，便于核对候选的 detail 与阶段记录。"""
    monkeypatch.setattr("runtime.call_llm", fake.call_llm)
    return Pipeline(trajectory(),
                    tools if tools is not None else [FakeTool(name) for name in READ_ONLY_TOOLS],
                    directions(), TaskCraftParameters(), Ledger(GenerationBudget(runtime_config())))


def width_pipeline(fake, monkeypatch, *, turns=160):
    monkeypatch.setattr("runtime.call_llm", fake.call_llm)
    budget = GenerationBudget(runtime_config(max_model_turns=turns))
    return Pipeline(trajectory(), [], directions(), TaskCraftParameters(), Ledger(budget))


@pytest.fixture
def environment():
    return EnvironmentConfig(name="PhysicianBench", manager_url="http://unused",
                             seed_data="fhir-full:v1")


@pytest.fixture
def patched(monkeypatch):
    """接入替身运行层与环境，返回 (fake, 记录表)。"""

    def install(fake, *, tools_fail=False, native=False):
        state = {"created": 0, "released": 0, "patients": None, "actions": []}
        tools = [FakeTool(name) for name in READ_ONLY_TOOLS]
        tools.append(FakeTool("fhir_medication_request_create"))
        if native:
            tools = NativeTools("http://unused/fhir/", [PATIENT], [])

            async def execute(self, action):
                normalized = action.replace("```tool_code", "").replace("```", "").strip()
                state["actions"].append(normalized)
                if normalized.startswith("FINISH("):
                    return {"kind": "FINISH",
                            "answers": json.loads(normalized[len("FINISH("):-1])}
                return {"kind": "GET", "request": {"action": action}, "reply": "{}"}

            monkeypatch.setattr(NativeTools, "execute", execute)

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
        state["tools"] = tools
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
    # 内置核验：每题先做无工具回答判断，再交给病历工具 Agent 检索并作答。
    atomic = "该患者第 1 项记录"
    assert len([item for item in fake.log if item == ("llm", "AnswerList", item[2])
                and atomic in item[2] and "Please solve" in item[2]]) == 1
    # 源码循环：首个 action 发出检索工具调用，第二个 action 用 final_answer 给出答案。
    actions = [item for item in fake.log if item[0] == "agent" and atomic in item[2]]
    assert len(actions) == 2
    assert "Observation:" not in actions[0][2] and "Observation:" in actions[1][2]
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
    state = patched(fake, native=True)
    result = await generate_tasks(trajectory(), environment, directions(), runtime_config())

    assert result.status == "completed", result.error
    assert TaskBatch.model_validate(result.output).questions
    # 原生协议：工具对象透传只读视图，Agent 用 FINISH 答案列表返回，说明里禁止 POST。
    assert fake.agent_output_types.count("NativeAction") >= 1
    assert "SearchAction" not in fake.agent_output_types
    assert all("只发送 GET" in text for text in fake.agent_instructions)
    assert any("FINISH([" in text for text in fake.agent_inputs)
    assert any(action.startswith("GET ") for action in state["actions"])
    # FINISH 的答案列表进入评分提示词，说明答案已从原生协议解析出来。
    scored = [text for kind, _, text in fake.log if kind == "llm" and "Other Answer:" in text]
    assert any("Other Answer: ['5.9 mmol/L']" in text for text in scored)
    verifications = stage_records(result, "atomic/verify")
    assert [entry["agent_score"] for entry in verifications] == [2.0, 2.0]


@pytest.mark.asyncio
async def test_function_tool_environment_keeps_physicianbench_answers(patched, environment):
    fake = FakeRuntime()
    state = patched(fake)
    result = await generate_tasks(trajectory(), environment, directions(), runtime_config())

    assert result.status == "completed"
    assert "SearchAction" in fake.agent_output_types
    assert all("只发送 GET" not in text for text in fake.agent_instructions)
    # 函数工具协议：Agent 直接调用病历工具，参数保留患者范围。
    calls = [call for tool in state["tools"] for call in tool.calls]
    assert calls and all(name in READ_ONLY_TOOLS for name, _ in calls)
    assert any(PATIENT in arguments for _, arguments in calls)


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
    assert TaskBatch.model_validate(result.output).questions
    pool = pipeline_pool(result)
    atomic = [entry for entry in pool if entry["origin"] == "atomic"]
    # 每个深度题都挂在某道原子题下；第二道原子题的证据是 BLOCK_B。
    depth = [entry for entry in pool if entry["origin"] == "depth"]
    assert len(depth) == len(atomic) == 2
    for entry in depth:
        parent_entry = next(item for item in atomic
                            if item["question"]["instruction"] == entry["parents"][0])
        assert {record["source_id"] for record in parent_entry["question"]["evidence"]} <= {
            record["source_id"] for record in entry["question"]["evidence"]}, "深度题必须继承父题证据"
    depth_entry = next(entry for entry in depth
                       if entry["parents"][0] == atomic[1]["question"]["instruction"])
    question = GeneratedQuestion.model_validate(depth_entry["question"])
    assert question.reference_answer == "5.9 mmol/L"
    assert {item.source_id for item in question.evidence} == {BLOCK_B}
    parent = atomic[1]
    # 子问题与中间标识属于 trace，不拼进最终答案。
    completed = next(entry for entry in stage_records(result, "depth/extension")
                     if entry["status"] == "completed"
                     and [item["answer"] for item in entry["chain"]] == [parent["locator"]])
    assert [item["query"] for item in completed["chain"]] == [
        "在该患者的高钾血症问题条目下，血钾结果是多少？"]
    assert {record["source_id"] for entry in depth
            for record in entry["question"]["evidence"]} == {BLOCK_A, BLOCK_B}
    # 顺序：反向检索 → superset 校验 → 生成问题 → 病历核验 → 合并检查。
    start = next(index for index, item in enumerate(fake.log) if item[0] == "llm"
                 and "Conduct divergent searches" in item[2])
    depth_order = [name for kind, name, _ in fake.log[start:] if kind == "llm" and name in {
        "SearchAction", "SupersetCheck", "NewQuery", "AnswerList", "QueryMerge", "QueryJudgement"}]
    # 反向检索的两个 action step → superset → 新问题 → 无工具作答 → 核验检索 → 合并判定。
    assert depth_order[:7] == ["SearchAction", "SearchAction", "SupersetCheck", "NewQuery",
                               "AnswerList", "SearchAction", "SearchAction"]
    assert "QueryMerge" in depth_order[7:] and "QueryJudgement" in depth_order[7:]
    assert stage_records(result, "depth/extension")[0]["status"] == "completed"


@pytest.mark.parametrize("state", ["invalid", " invalid ", "INVALID"])
@pytest.mark.asyncio
async def test_depth_superset_check_reads_the_new_query_field(patched, environment, state):
    """superset 校验读取返回对象的 new_query 字段，只有显式 invalid 才拒绝扩展。"""
    fake = FakeRuntime(superset_state=state)
    patched(fake)
    result = await generate_tasks(trajectory(), environment, directions(), runtime_config())

    pool = pipeline_pool(result)
    assert not [candidate for candidate in pool if candidate["origin"] == "depth"]
    rejected = [entry for entry in stage_records(result, "depth/superset")
                if entry["status"] == "rejected"]
    assert len(rejected) == 2 * TaskCraftParameters().extended_attempts
    # 记录里带回被判定的 superset 与关系，证明读到的是校验结果本身而不是整段响应。
    assert rejected[0]["identifier"] == BLOCK_B and rejected[0]["relation"]
    assert "NewQuery" not in fake.llm_calls


@pytest.mark.asyncio
async def test_width_extension_checks_decomposition_and_model_answering(patched, environment):
    fake = FakeRuntime()
    patched(fake)
    result = await generate_tasks(trajectory(), environment, directions(), runtime_config())

    pool = pipeline_pool(result)
    width = next(candidate for candidate in pool if candidate["origin"] == "width")
    question = GeneratedQuestion.model_validate(width["question"])
    assert "合并问题" in question.instruction
    assert json.loads(question.reference_answer) == ["5.9 mmol/L", "5.9 mmol/L"]
    assert {item.source_id for item in question.evidence} == {BLOCK_A, BLOCK_B}
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
async def test_width_group_keeps_its_own_answers_and_stable_index(monkeypatch):
    """首组在分解检查被淘汰后，保留组仍按自身答案与稳定 index 评分。"""
    fake = WidthRuntime(groups=[[0, 1], [2, 3], [4, 5]], states={0: 0, 2: 0})
    pipeline = width_pipeline(fake, monkeypatch)

    candidates = await pipeline.width(atomic_pool(6))

    assert [candidate.question.instruction for candidate in candidates] == ["合并问题 1"]
    # 第二组是响应里的最后一项；保留组不能借用它的原始问题与答案。
    scored = next(text for kind, name, text in fake.log if name == "AnswerScore")
    assert "['答案 2', '答案 3']" in scored
    assert "答案 4" not in scored
    step2 = next(text for kind, name, text in fake.log if name == "KnowledgeAnswers")
    assert [int(value) for value in re.findall(r'"index": (\d+)', step2)] == [1]


@pytest.mark.asyncio
async def test_width_out_of_range_parent_index_is_a_model_output_failure(monkeypatch):
    """越界父题索引按模型输出失败记账并重试，不把分组修剪成更小的组。"""
    fake = WidthRuntime(groups=[[0, 9]])
    pipeline = width_pipeline(fake, monkeypatch)

    candidates = await pipeline.width(atomic_pool(4))

    assert candidates == []
    failures = [entry for entry in pipeline.records if entry["stage"] == "width/merge"
                and entry["status"] == "failed"]
    assert len(failures) == TaskCraftParameters().width_attempts
    assert "IndexError" in failures[0]["error"]
    assert "DecompositionReport" not in fake.llm_calls
    assert pipeline.failure == "宽度扩展没有生成合并问题"


@pytest.mark.asyncio
async def test_width_keeps_duplicate_parent_indexes_in_model_order(monkeypatch):
    """父题索引保持模型给定的顺序与重复项，不排序也不去重。"""
    fake = WidthRuntime(groups=[[2, 0, 2]], width_answers={0: "无法确定"})
    pipeline = width_pipeline(fake, monkeypatch)

    candidates = await pipeline.width(atomic_pool(3))

    assert len(candidates) == 1
    step1 = next(text for kind, name, text in fake.log if name == "DecompositionReport")
    assert step1.count("该患者第 2 项") == 2 and step1.count("该患者第 0 项") == 1
    assert step1.index("该患者第 2 项") < step1.index("该患者第 0 项")
    scored = next(text for kind, name, text in fake.log if name == "AnswerScore")
    assert scored.count("答案 2") == 2 and scored.count("答案 0") == 1


@pytest.mark.asyncio
async def test_width_checks_merged_groups_in_batches_of_ten(monkeypatch):
    """至少 11 组合并题按源码 batch_size=10 分两批，各批保留自身答案。"""
    groups = [[index, index + 1] for index in range(0, 22, 2)]
    fake = WidthRuntime(groups=groups,
                        width_answers={index: "无法确定" for index in range(len(groups))})
    pipeline = width_pipeline(fake, monkeypatch)

    candidates = await pipeline.width(atomic_pool(22))

    assert len(candidates) == 11
    for group, candidate in zip(groups, candidates):
        assert json.loads(candidate.question.reference_answer) == [f"答案 {index}" for index in group]
    decompose = [text for kind, name, text in fake.log if name == "DecompositionReport"]
    assert len(decompose) == 2
    assert [len(re.findall(r'"complex_question"', text)) for text in decompose] == [10, 1]


@pytest.mark.asyncio
async def test_width_detail_keeps_parent_questions_answers_indexes_and_types(monkeypatch):
    """宽度候选保存各父题原文、答案、原始索引与全部题型；跨题型取首父题标签。"""
    fake = WidthRuntime(groups=[[2, 1, 0]], width_answers={0: "无法确定"})
    pipeline = width_pipeline(fake, monkeypatch)

    candidates = await pipeline.width(atomic_pool(3, task_types=["治疗规划", "诊断与解释"]))

    detail = candidates[0].detail
    assert detail["original_indexes"] == [2, 1, 0]
    assert detail["original_questions"] == [
        f"该患者第 {index} 项记录说明了什么？" for index in (2, 1, 0)]
    assert detail["golden_answers"] == ["答案 2", "答案 1", "答案 0"]
    assert detail["parent_types"] == ["治疗规划", "诊断与解释", "治疗规划"]
    assert candidates[0].question.task_type == "治疗规划"
    mixed = [entry for entry in pipeline.records if entry["stage"] == "width/task_type"]
    assert mixed and mixed[0]["parent_types"] == ["治疗规划", "诊断与解释", "治疗规划"]


@pytest.mark.asyncio
async def test_atomic_detail_keeps_conclusion_relation_sources_and_clean_answer(monkeypatch):
    """原子候选保留原结论、关系、患者来源与清洗后的答案。"""
    fake = FakeRuntime()
    pipeline = stage_pipeline(monkeypatch, fake)

    candidates = await pipeline.atomic()

    assert len(candidates) == 2
    assert candidates[0].detail == {
        "conclusion": "2024-03-05 血钾 5.9 mmol/L", "relation": "该次检验的血钾结果",
        "source_ids": [BLOCK_A], "refined_answer": "5.9 mmol/L", "task_type": "诊断与解释"}


@pytest.mark.asyncio
async def test_atomic_zero_score_tool_answer_is_still_accepted(monkeypatch):
    """原子过滤只按核验出错判断：工具作答得 0 分但核验无出错时仍接纳候选。"""
    fake = FakeRuntime(agent_answer="无法确定")
    pipeline = stage_pipeline(monkeypatch, fake)

    candidates = await pipeline.atomic()

    checks = [entry for entry in pipeline.records if entry["stage"] == "atomic/verify"]
    assert [entry["agent_score"] for entry in checks] == [0.0, 0.0]
    assert [entry["error"] for entry in checks] == [None, None]
    assert [candidate.question.instruction for candidate in candidates] == [
        "该患者第 1 项记录说明了什么？", "该患者第 2 项记录说明了什么？"]


@pytest.mark.asyncio
async def test_depth_rejects_hop_whose_tool_answer_scores_zero(monkeypatch):
    """深度核验的工具作答评分非正时换下一跳，不为该父题生成扩展候选。"""
    fake = FakeRuntime(agent_answer="无法确定")
    pipeline = stage_pipeline(monkeypatch, fake)

    extended = await pipeline.depth(atomic_pool(1))

    attempts = TaskCraftParameters().extended_attempts
    assert extended == []
    checks = [entry for entry in pipeline.records if entry["stage"] == "depth/verify"]
    assert [entry["agent_score"] for entry in checks] == [0.0] * attempts
    assert [entry["error"] for entry in checks] == [None] * attempts
    outcomes = [entry for entry in pipeline.records if entry["stage"] == "depth/extension"]
    assert [entry["status"] for entry in outcomes] == ["failed"]
    assert [item["reason"] for item in outcomes[0]["failed_attempts"]] == ["verify"] * attempts


@pytest.mark.asyncio
async def test_depth_detail_keeps_results_trajectory_auxiliary_sources_and_failures(monkeypatch):
    """深度 detail 保存 full_results、solve_trajectory、辅助来源与失败尝试。"""
    fake = FakeRuntime(backward_failures=1)
    pipeline = stage_pipeline(monkeypatch, fake)
    atomic = await pipeline.atomic()

    extended = await pipeline.depth([atomic[0]])

    assert len(extended) == 1
    detail = extended[0].detail
    assert detail["failed_attempts"] == [{"hop": 1, "reason": "backward"}]
    assert [item["query"] for item in detail["full_results"][1::2]] == [
        "在该患者的高钾血症问题条目下，血钾结果是多少？"]
    assert detail["full_results"][-1]["answer"] == "5.9 mmol/L"
    assert detail["solve_trajectory"]["golden_answer"] == "5.9 mmol/L"
    assert detail["solve_trajectory"]["trajectory"][-1] == {
        "sub_query": atomic[0].question.instruction, "sub_answer": "5.9 mmol/L",
        "sub_trajectory": None}
    assert detail["superset_source_ids"] == [BLOCK_B]
    assert detail["relations"] == [{"identifier": BLOCK_B, "relation": "该问题条目下的检验结果",
                                    "source_ids": [BLOCK_B]}]
    assert {item.source_id for item in extended[0].question.evidence} == {BLOCK_A, BLOCK_B}


@pytest.mark.asyncio
async def test_depth_answer_is_the_original_final_answer():
    initial = {"initial_answer": "5.9 mmol/L"}
    chain = [{"query": "先查哪个标识", "answer": "Observation/77"}]
    assert Pipeline._chain_answer(initial, chain) == "5.9 mmol/L"


def test_width_answer_is_lossless_json():
    item = {"original_questions": ["Q1", "Q2"],
            "golden_answers": ["5.9 mmol/L", "含换行\n和引号\"的原答案"]}
    assert json.loads(Pipeline._merged_answer(item)) == item["golden_answers"]


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
                                  runtime_config(max_model_turns=2))

    assert result.status == "budget_exhausted" and result.output is None
    assert result.usage["model_turns"] == 2
    assert len(fake.llm_calls) + len(fake.agent_calls) == 2
    assert result.trace[-1]["stage"] == "taskcraft/pipeline"
    assert result.trace[-1]["records"]


@pytest.mark.asyncio
async def test_stages_use_the_shared_budget_as_their_only_limit(patched, environment):
    """原子与扩展阶段不再按阶段份额或预留轮数截断，跑满共享生成预算才停。"""
    conclusions = [{"conclusion": f"2024-03-{index:02d} 血钾 5.9 mmol/L", "R": "该次检验的血钾结果",
                    "source_ids": [BLOCK_A]} for index in range(1, 9)]
    fake = FakeRuntime(conclusions=conclusions)
    patched(fake)
    result = await generate_tasks(trajectory(), environment, directions(), runtime_config())

    assert result.status == "budget_exhausted"
    assert result.usage["model_turns"] == 160
    atomic = stage_records(result, "atomic/result")[0]
    assert atomic["generated"] == atomic["accepted"] == 8
    assert not [entry for entry in pipeline_records(result)
                if entry.get("reason") in {"原子阶段预算份额已用尽", "剩余预算不足以完成一跳"}]


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


def test_read_tools_do_not_require_nine_names():
    read = SimpleNamespace(name="fhir_observation_search_labs")
    write = SimpleNamespace(name="fhir_medication_request_create")
    assert bind_generation_tools([read, write]) == [read]


def test_read_only_subset_binds_without_environment_name():
    """只读绑定取自显式声明：任意非空子集可用，空集拒绝，不凭名称推断权限。"""
    declared = [SimpleNamespace(name="fhir_observation_search_vitals"),
                SimpleNamespace(name="fhir_patient_search_demographics")]
    unlisted = [SimpleNamespace(name="audit_search_probe"),
                SimpleNamespace(name="fhir_medication_request_create")]
    assert bind_generation_tools(declared + unlisted) == declared
    with pytest.raises(EnvironmentUnavailable):
        bind_generation_tools(unlisted)


def test_native_generation_blocks_post_before_transport(monkeypatch):
    calls = []

    async def execute(self, action):
        calls.append(action)
        return {"kind": "FINISH", "answers": ["ok"]}

    monkeypatch.setattr(NativeTools, "execute", execute)
    original = NativeTools("http://fhir.test/", ["patient-1"], [])
    bound = bind_generation_tools(original)
    assert isinstance(bound, NativeTools)
    rejected = asyncio.run(bound.execute(
        "```tool_code\nPOST http://fhir.test/MedicationRequest\n{}\n```"))
    assert rejected["kind"] == "error"
    assert calls == []
    assert asyncio.run(bound.execute('FINISH(["ok"])'))["answers"] == ["ok"]
    assert calls == ['FINISH(["ok"])']


@pytest.mark.asyncio
async def test_native_generation_binds_declared_reads_and_allows_only_get_and_finish(monkeypatch):
    calls = []

    async def execute(self, action):
        calls.append(action)
        return {"kind": "GET", "request": {"action": action}, "reply": "recorded"}

    monkeypatch.setattr(NativeTools, "execute", execute)
    declarations = [{"name": "GET {api_base}/Observation", "parameters": {}},
                    {"name": "POST {api_base}/MedicationRequest", "parameters": {}}]
    source = NativeTools("http://fhir.test/", ["patient-1"], declarations)
    bound = bind_generation_tools(source)

    assert [item["name"] for item in bound.functions] == ["GET {api_base}/Observation"]
    assert bound.functions[0] is not declarations[0]
    assert source.functions == declarations

    unfenced = "POST http://fhir.test/MedicationRequest\n{}"
    blocked = await bound.execute(unfenced)
    assert blocked["kind"] == "error"
    assert blocked["request"] == {"action": unfenced}
    assert await bound.execute("GET {api_base}/Observation") == {
        "kind": "GET", "request": {"action": "GET {api_base}/Observation"},
        "reply": "recorded"}
    assert calls == ["GET {api_base}/Observation"]


@pytest.mark.asyncio
async def test_generation_binding_keeps_the_patient_read_boundary(native_fhir):
    instance = EnvironmentInstance(instance_id="instance-one",
                                   endpoint="http://instance-one/fhir/",
                                   environment="MedAgentBench")
    bound = bind_generation_tools(await get_tools(instance, ["a"]))

    with pytest.raises(PatientAccessError):
        await bound.execute("GET Patient/b")
    with pytest.raises(PatientAccessError):
        await bound.execute("GET Observation?patient=b")
    assert (await bound.execute("GET Observation?patient=a"))["kind"] == "GET"
    assert len(native_fhir[1]) == 1


def test_stage_parameters_keep_source_defaults(environment):
    settings = TaskCraftParameters()
    assert settings.chunk_size == 16384
    assert settings.max_candidate_conclusions == 20
    assert settings.max_candidate_atomic == 10
    assert settings.judge_samples == 3
    assert settings.question_attempts == 3 and settings.width_attempts == 3
    assert settings.extended_attempts == 4 and settings.max_hops == 2
    assert settings.merge_attempts == 5 and settings.query_attempts == 3
    # 内层重试：utils.run_llm_prompt(max_retries=3) 与 retry_predict(MAX_RETRIES=3)。
    assert settings.request_attempts == 3
    # 源码 SearchAgent 的 max_step：原子核验 20，深度反向与深度核验各 10。
    assert settings.atomic_max_action_steps == 20
    assert settings.depth_backward_max_action_steps == 10
    assert settings.depth_verify_max_action_steps == 10
    # 阶段份额与预留轮数已取消，共享生成预算是唯一上限。
    assert "atomic_budget_share" not in TaskCraftParameters.model_fields
    assert "extension_guard_turns" not in TaskCraftParameters.model_fields


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


# --------------------------------------------------------------- 源码执行与重试
def agent_ledger(monkeypatch, fake, *, turns=100, tool_calls=40):
    """直接驱动一次检索 Agent：替身运行层 + 公共累计预算 + 替身病历工具。"""
    monkeypatch.setattr("runtime.call_llm", fake.call_llm)
    budget = GenerationBudget(runtime_config(max_model_turns=turns, max_tool_calls=tool_calls))
    tools = [FakeTool(name) for name in READ_ONLY_TOOLS]
    return Ledger(budget), budget, tools


def failed_run(error="model_error: format"):
    return RunResult(trace=[{"type": "run_error", "error": error}],
                     usage={"model_turns": 1, "tool_calls": 0}, status="failed", error=error)


def test_source_planning_and_reflection_schedule():
    assert [step_kind(i, 5) for i in range(1, 12)] == [
        "planning", "action", "action", "action", "action",
        "reflection", "action", "action", "action", "action", "reflection",
    ]


@pytest.mark.asyncio
async def test_ledger_retry_bills_every_attempt(monkeypatch):
    """首次调用失败、第二次成功：两次真实请求都进入同一账本。"""
    calls = []

    async def call_llm(*, model, instructions, input_text, output_type):
        calls.append(output_type.__name__)
        if len(calls) == 1:
            return failed_run()
        return completed({"answer_list": ["5.9 mmol/L"]})

    monkeypatch.setattr("runtime.call_llm", call_llm)
    ledger = Ledger(GenerationBudget(runtime_config()))
    run = await ledger.llm("width/answer", instructions="i", input_text="t",
                           output_type=AnswerList, attempts=2)

    assert run.status == "completed"
    assert [stage for stage, _ in ledger.budget.records] == [
        "width/answer/attempt-1", "width/answer/attempt-2"]
    assert [record.status for _, record in ledger.budget.records] == ["failed", "completed"]
    assert ledger.exhausted is False
    assert ledger.used_turns() == 2


class FlakyRuntime(FakeRuntime):
    """指定结构的前若干次请求失败：用于核对源码内层重试的账单。"""

    def __init__(self, failures=(), **kwargs):
        super().__init__(**kwargs)
        self.failures = Counter(failures)
        self.attempts = Counter()

    async def call_llm(self, *, model, instructions, input_text, output_type):
        self.attempts[output_type.__name__] += 1
        if self.failures[output_type.__name__] >= self.attempts[output_type.__name__]:
            self.llm_calls.append(output_type.__name__)
            return failed_run()
        return await super().call_llm(model=model, instructions=instructions,
                                      input_text=input_text, output_type=output_type)


def stage_attempt_stages(pipeline, stage):
    return [name for name, _ in pipeline.ledger.budget.records
            if name.startswith(f"{stage}/attempt-")]


def two_attempts(stage):
    return [f"{stage}/attempt-1", f"{stage}/attempt-2"]


@pytest.mark.asyncio
async def test_depth_requests_retry_inside_the_source_loops(monkeypatch):
    """源码 depth 请求带 max_retries=3：首次失败后同一次调用内再试，两次都进入账本。"""
    fake = FlakyRuntime(failures=("SupersetCheck", "NewQuery", "QueryMerge", "QueryJudgement"))
    pipeline = stage_pipeline(monkeypatch, fake)

    backward = await pipeline._backward(prompts.depth_prompts(), BLOCK_A)
    merged = await pipeline._merge(prompts.depth_prompts(), "综合问题", "子问题", BLOCK_A, "5.9 mmol/L")

    assert backward["query"] == "在该患者的高钾血症问题条目下，血钾结果是多少？"
    assert merged == "综合问题：问题条目下的血钾结果"
    assert stage_attempt_stages(pipeline, "depth/superset") == two_attempts("depth/superset")
    assert stage_attempt_stages(pipeline, "depth/query") == two_attempts("depth/query")
    assert stage_attempt_stages(pipeline, "depth/merge") == two_attempts("depth/merge")
    assert stage_attempt_stages(pipeline, "depth/merge_judge") == two_attempts("depth/merge_judge")
    # 内层重试成功即结束外层循环，不叠加重试次数。
    assert [record.status for name, record in pipeline.ledger.budget.records
            if name.startswith("depth/superset/") ] == ["failed", "completed"]


@pytest.mark.asyncio
async def test_width_requests_retry_inside_the_source_loops(monkeypatch):
    """源码 retry_predict 的 MAX_RETRIES=3：宽度三处请求各自内层重试并逐次记账。"""
    fake = FlakyRuntime(failures=("MergedQuestions", "DecompositionReport", "KnowledgeAnswers"))
    pipeline = width_pipeline(fake, monkeypatch)

    candidates = await pipeline.width(atomic_pool(4))

    assert [candidate.question.instruction for candidate in candidates] == ["合并问题：血钾结果与诊断的关系"]
    assert stage_attempt_stages(pipeline, "width/merge") == two_attempts("width/merge")
    assert stage_attempt_stages(pipeline, "width/decompose") == two_attempts("width/decompose")
    assert stage_attempt_stages(pipeline, "width/answer") == two_attempts("width/answer")
    assert [record.status for name, record in pipeline.ledger.budget.records
            if name.startswith("width/merge/")] == ["failed", "completed"]


@pytest.mark.asyncio
async def test_format_failure_does_not_mark_the_batch_exhausted(monkeypatch):
    """格式失败只结束本次调用，账本仍有剩余预算时不标成累计耗尽。"""
    async def call_llm(*, model, instructions, input_text, output_type):
        return failed_run("model_error: invalid json")

    monkeypatch.setattr("runtime.call_llm", call_llm)
    ledger = Ledger(GenerationBudget(runtime_config()))
    run = await ledger.llm("atomic/clean", instructions="i", input_text="t",
                           output_type=AnswerList, attempts=3)

    assert run.status == "failed"
    assert len(ledger.budget.records) == 3
    assert ledger.exhausted is False
    assert ledger.remaining_turns() == 160 - 3


@pytest.mark.asyncio
async def test_score_samples_skip_a_failed_judgement(monkeypatch):
    """一次评分失败，其余有效采样继续求平均。"""
    fake = FakeRuntime(score_failures={1})
    monkeypatch.setattr("runtime.call_llm", fake.call_llm)
    ledger = Ledger(GenerationBudget(runtime_config()))

    score = await score_recall(ledger, "5.9 mmol/L", ["5.9 mmol/L"],
                               samples=3, stage="atomic/verify/llm_score")

    assert score == 2.0
    assert fake.score_requests == 3
    assert [record.status for _, record in ledger.budget.records] == [
        "failed", "completed", "completed"]


@pytest.mark.asyncio
async def test_search_agent_keeps_planning_reflection_and_action_steps_apart(monkeypatch):
    """规划与反思增加模型轮数，但不增加 action step；真实计数进入账本。"""
    fake = FakeRuntime(agent_action_rounds=6)
    ledger, budget, tools = agent_ledger(monkeypatch, fake)
    run = await run_search_agent(
        ledger=ledger, stage="depth/backward", instructions="只使用当前患者的病历工具",
        input_text="该患者 2024-03-05 的血钾结果是多少？", tools=tools,
        output_type=AnswerList, max_action_steps=20)

    assert run.status == "completed", run.error
    assert AnswerList.model_validate(run.output).answer_list == ["5.9 mmol/L"]
    assert run.usage["action_steps"] == 7
    assert run.usage["tool_calls"] == 6
    # 规划 2 次请求 + 第 6 步反思 3 次请求 + 7 个 action step 各 1 次请求。
    assert run.usage["model_turns"] == 12
    steps = [(item["type"], item.get("step")) for item in run.trace
             if item["type"] in {"planning", "reflection", "action"}]
    assert steps[0] == ("planning", 1) and steps[1] == ("action", 1)
    assert ("reflection", 6) in steps
    assert [kind for kind, _ in steps].count("action") == 7
    assert fake.agent_runs == 1


@pytest.mark.parametrize("as_object", [False, True])
@pytest.mark.asyncio
async def test_planning_answer_keeps_author_text_form(monkeypatch, as_object):
    """源码 planning_step 是自由文本补全：answer 里的 JSON 对象按其文本进入记忆与计划。"""
    fake = FakeRuntime(planning_answer_as_object=as_object)
    ledger, budget, tools = agent_ledger(monkeypatch, fake)
    run = await run_search_agent(
        ledger=ledger, stage="depth/backward", instructions="只使用当前患者的病历工具",
        input_text="该患者 2024-03-05 的血钾结果是多少？", tools=tools,
        output_type=AnswerList, max_action_steps=20)

    assert run.status == "completed", run.error
    facts_text = "### 1. Facts given in the task\n病历片段"
    plan_text = "1. 用病历工具检索该记录"
    object_text = json.dumps(fake.planning_answer_object, ensure_ascii=False)
    expected = (object_text, object_text) if as_object else (facts_text, plan_text)
    planning = next(item for item in run.trace if item["type"] == "planning")
    assert (planning["facts"], planning["plan"]) == expected
    assert planning["facts_think"] == "先核对该片段已知事实"


@pytest.mark.asyncio
async def test_action_response_with_several_tool_calls_counts_each(monkeypatch):
    """一个 action 响应的多个工具调用分别计数，并各自进入公共账本。"""
    fake = FakeRuntime(tool_calls_per_action=2)
    ledger, budget, tools = agent_ledger(monkeypatch, fake)
    run = await run_search_agent(
        ledger=ledger, stage="depth/backward", instructions="只使用当前患者的病历工具",
        input_text="该患者 2024-03-05 的血钾结果是多少？", tools=tools,
        output_type=AnswerList, max_action_steps=20)

    assert run.status == "completed", run.error
    assert run.usage["tool_calls"] == 2
    assert [item["type"] for item in run.trace].count("tool_start") == 2
    tool_records = [record for _, record in budget.records if record.usage["tool_calls"] == 1]
    assert len(tool_records) == 2
    assert [item["name"] for item in run.trace if item["type"] == "tool_start"] == [
        FakeRuntime.AGENT_TOOL, FakeRuntime.AGENT_TOOL]
    called = next(tool for tool in tools if tool.name == FakeRuntime.AGENT_TOOL)
    assert len(called.calls) == 2


@pytest.mark.asyncio
async def test_action_response_with_tools_and_final_answer_keeps_the_loop(monkeypatch):
    """同一响应同时带工具调用与 final_answer 时，源码继续循环并执行工具调用。"""
    fake = FakeRuntime(agent_action_rounds=2, final_answer_with_tools=True)
    ledger, budget, tools = agent_ledger(monkeypatch, fake)
    run = await run_search_agent(
        ledger=ledger, stage="depth/backward", instructions="只使用当前患者的病历工具",
        input_text="该患者 2024-03-05 的血钾结果是多少？", tools=tools,
        output_type=AnswerList, max_action_steps=20)

    assert run.status == "completed", run.error
    assert AnswerList.model_validate(run.output).answer_list == ["5.9 mmol/L"]
    assert run.usage["action_steps"] == 3
    assert run.usage["tool_calls"] == 2
    tool_events = [item for item in run.trace if item["type"] == "tool_start"]
    assert [item["name"] for item in tool_events] == [FakeRuntime.AGENT_TOOL] * 2
    # 前两步执行工具调用且没有收尾；只有第 3 步（无工具调用的响应）给出终答。
    early = [item for item in run.trace if item["type"] == "action" and item["step"] < 3]
    assert [item.get("action_output") for item in early] == [None, None]
    assert all(item["observations"] is not None for item in early)


@pytest.mark.asyncio
async def test_tool_records_do_not_inflate_runs_without_known_tokens(monkeypatch):
    """工具调用记录没有 token 字段，但不计入 runs_without_known_tokens。"""
    fake = FakeRuntime()
    ledger, budget, tools = agent_ledger(monkeypatch, fake)
    await ledger.llm("atomic/clean", instructions="i", input_text="t",
                     output_type=AnswerList)
    ledger.tool("depth/backward/tool", name="fhir_observation_search_labs",
                arguments={"patient": PATIENT}, output="{}")

    usage = summarize_results(budget.records).usage
    assert usage["model_turns"] == 1
    assert usage["tool_calls"] == 1
    assert usage["runs_without_known_tokens"] == 0


@pytest.mark.asyncio
async def test_post_cap_provide_final_answer_is_billed(monkeypatch):
    """达到 action 上限后源码仍发起 provide_final_answer，该请求计费。"""
    fake = FakeRuntime(agent_action_rounds=99)
    ledger, budget, tools = agent_ledger(monkeypatch, fake)
    run = await run_search_agent(
        ledger=ledger, stage="depth/backward", instructions="只使用当前患者的病历工具",
        input_text="该患者 2024-03-05 的血钾结果是多少？", tools=tools,
        output_type=AnswerList, max_action_steps=2)

    assert run.status == "completed", run.error
    assert run.usage["action_steps"] == 2
    assert run.usage["model_turns"] == 5          # 规划 2 + action 2 + 强制终答 1
    assert [item["kind"] for item in run.trace if item["type"] == "final_answer"] == ["forced_final"]
    assert AnswerList.model_validate(run.output).answer_list == ["5.9 mmol/L"]
    # 局部 action 上限不等于累计预算耗尽。
    assert ledger.exhausted is False
    assert [stage for stage, _ in budget.records][-1] == "depth/backward/forced_final/attempt-1"


@pytest.mark.parametrize("as_object", [False, True])
@pytest.mark.asyncio
async def test_forced_final_keeps_author_answer_typing_for_depth_backward(monkeypatch, as_object):
    """源码 FinalAnswerTool 的 answer 为 any：终答是 JSON 对象或 JSON 文本都要落到 BackwardContext。"""
    fake = FakeRuntime(agent_action_rounds=99, final_answer_as_object=as_object)
    ledger, budget, tools = agent_ledger(monkeypatch, fake)
    run = await run_search_agent(
        ledger=ledger, stage="depth/backward", instructions="只使用当前患者的病历工具",
        input_text=('Conduct divergent searches from {"identifier": "Observation/lab-77"} '
                    "以找出更宽的问题条目。"),
        tools=tools, output_type=BackwardContext, max_action_steps=2)

    assert run.status == "completed", run.error
    assert [item["kind"] for item in run.trace if item["type"] == "final_answer"] == ["forced_final"]
    # 强制终答的 answer 落到本次调用声明的 BackwardContext 上。
    context = BackwardContext.model_validate(run.output)
    assert context.identifier == BLOCK_B
    assert context.relation == "该问题条目下的检验结果"
    assert context.source_ids == [BLOCK_B]


@pytest.mark.asyncio
async def test_forced_final_answer_is_blocked_by_cumulative_budget(monkeypatch):
    """累计预算不足时禁止源码的 provide_final_answer 请求。"""
    fake = FakeRuntime(agent_action_rounds=99)
    ledger, budget, tools = agent_ledger(monkeypatch, fake, turns=4)
    run = await run_search_agent(
        ledger=ledger, stage="depth/backward", instructions="只使用当前患者的病历工具",
        input_text="该患者 2024-03-05 的血钾结果是多少？", tools=tools,
        output_type=AnswerList, max_action_steps=2)

    assert run.status == "budget_exhausted"
    assert run.usage["model_turns"] == 4
    assert run.output is None
    assert not [item for item in run.trace if item.get("kind") == "forced_final"]
    assert "forced_final" not in "".join(stage for stage, _ in budget.records)
    assert ledger.exhausted is True


@pytest.mark.asyncio
async def test_verify_agent_retries_the_whole_sequence_and_reports_action_steps(monkeypatch):
    """VerifyAgent 外层重试重新走一遍流程；步数取真实 action step 数。"""
    fake = FakeRuntime(fail_agent_runs=1)
    ledger, budget, tools = agent_ledger(monkeypatch, fake)
    outcome = await verify_query(
        ledger, "该患者 2024-03-05 的血钾结果是多少？", "5.9 mmol/L", tools,
        samples=3, stage="atomic/verify", instructions="只使用当前患者的病历工具")

    assert outcome.error is None
    assert outcome.llm_score == 0.0 and outcome.agent_score == 2.0
    assert outcome.llm_answer == ["无工具无法确定"]
    assert outcome.agent_answer == ["5.9 mmol/L"]
    assert outcome.agent_step_number == 2
    assert [item["name"] for item in outcome.agent_trajectory] == [
        "facts", "plan", "action", "action"]
    assert [record.status for _, record in budget.records].count("failed") == 1


@pytest.mark.asyncio
async def test_verify_stops_without_retry_when_the_model_answers_alone(monkeypatch):
    """无工具回答评分达到 1 分及以上立即拒绝，不用外层重试绕过该判据。"""
    fake = FakeRuntime(llm_answer="5.9 mmol/L")
    ledger, budget, tools = agent_ledger(monkeypatch, fake)
    outcome = await verify_query(
        ledger, "该患者 2024-03-05 的血钾结果是多少？", "5.9 mmol/L", tools,
        samples=3, stage="atomic/verify", instructions="只使用当前患者的病历工具")

    assert outcome.error == "LLM can solve this task"
    assert fake.agent_calls == []
    assert fake.score_requests == 3
    assert len(budget.records) == 4


@pytest.mark.asyncio
async def test_verify_keeps_the_source_branch_when_the_agent_score_is_unavailable(monkeypatch):
    """Agent 评分样本全部无效时按源码分支返回空分数，不用外层重试重走流程。"""
    fake = FakeRuntime(score_failures={4, 5, 6})
    ledger, budget, tools = agent_ledger(monkeypatch, fake)
    outcome = await verify_query(
        ledger, "该患者 2024-03-05 的血钾结果是多少？", "5.9 mmol/L", tools,
        samples=3, stage="atomic/verify", instructions="只使用当前患者的病历工具")

    assert outcome.error is None
    assert outcome.llm_score == 0.0
    assert outcome.agent_score is None
    assert outcome.agent_answer == ["5.9 mmol/L"]
    assert outcome.agent_step_number == 2
    # 外层没有重试：无工具作答只请求一次。
    assert fake.llm_calls.count("AnswerList") == 1


def test_agent_prompts_render_without_jinja_leftovers():
    """检索 Agent 的提示词按 SearchAgent 的实际取值渲染：工具清单就位、无未替换的占位符。"""
    from methods.taskcraft import agent

    catalog = "- fhir_observation_search_labs: 检索当前患者的检验记录"
    variables = {"task": "任务文本", "answer_facts": "已知事实", "facts_update": "更新后事实",
                 "tools": catalog, "trajectory_score": 8, "trajectory_analysis": "分析",
                 "remaining_steps": 3, "standard_answer": None}
    rendered = {name: agent._populate(text, variables)
                for name, text in agent.AGENT_PROMPTS.items()}
    rendered["final_answer.note"] = agent._populate(agent.FUNCTION_ACTION_NOTE,
                                                   {"output_schema": '{"type": "object"}'})
    rendered["native_action.note"] = agent._populate(agent.NATIVE_ACTION_NOTE,
                                                    {"output_schema": '{"type": "object"}'})

    for name, text in rendered.items():
        assert "{{" not in text and "{%" not in text, name
        assert "managed_agents" not in text, name
    assert catalog in rendered["system_prompt"]
    assert catalog in rendered["planning.initial_plan"]
    assert catalog in rendered["reflection.update_plan_post_messages"]
    assert "任务文本" in rendered["planning.initial_facts"]
    assert "任务文本" in rendered["final_answer.post_messages"]
    assert '{"type": "object"}' in rendered["final_answer.note"]
    # 源码模板里的单花括号占位符保持逐字（Jinja 不替换），行为与作者一致。
    assert "{language}" in rendered["system_prompt"]
    assert "{remaining_steps}" in rendered["reflection.update_plan_post_messages"]


# --------------------------------------------------------------- 固定种子调度与选题
def qualified_candidates(count, *, origins=None, task_types=("诊断与解释", "治疗规划")):
    """构造全部通过公共字段、患者来源与决策点校验的候选，来源标签可替换。"""
    labels = list(origins) if origins is not None else ["atomic"] * count
    types = list(task_types)
    return [Candidate(
        question=GeneratedQuestion(
            instruction=f"该患者第 {index} 项记录说明了什么？",
            task_type=types[index % len(types)],
            reference_answer=f"答案 {index}",
            evidence=[Evidence(source_id=BLOCK_A, statement=f"第 {index} 项记录")]),
        origin=labels[index], locator=BLOCK_A)
        for index in range(count)]


class TrackingPipeline(Pipeline):
    """记录分支入参的 Pipeline：原子阶段返回固定对象，分支不回写原子池。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.atomic_candidates = atomic_pool(3, task_types=["诊断与解释", "治疗规划"])
        self.branch_inputs = {}

    async def atomic(self):
        return self.atomic_candidates

    async def depth(self, atomic):
        self.branch_inputs["depth"] = atomic
        return [Candidate(
            question=GeneratedQuestion(instruction="深度问题", task_type="诊断与解释",
                                       reference_answer="深度答案",
                                       evidence=[Evidence(source_id=BLOCK_A, statement="深度记录")]),
            origin="depth", parents=["父题"], locator=BLOCK_A)]

    async def width(self, atomic):
        self.branch_inputs["width"] = atomic
        return [Candidate(
            question=GeneratedQuestion(instruction="宽度问题", task_type="治疗规划",
                                       reference_answer="宽度答案",
                                       evidence=[Evidence(source_id=BLOCK_B, statement="宽度记录")]),
            origin="width", parents=["父题"], locator=BLOCK_B)]


class RaisingDepthPipeline(Pipeline):
    """深度分支抛错的 Pipeline：原子阶段产出固定候选，用于核对失败时的候选池。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.atomic_candidates = atomic_pool(3, task_types=["诊断与解释", "治疗规划"])

    async def atomic(self):
        return self.atomic_candidates

    async def width(self, atomic):
        return []

    async def depth(self, atomic):
        raise RuntimeError("深度分支执行失败")


def scheduling_pipeline(random_seed):
    return TrackingPipeline(trajectory(), [], directions(), TaskCraftParameters(),
                            Ledger(GenerationBudget(runtime_config())), random_seed=random_seed)


def test_seed_streams_are_stable_and_private():
    before = random.getstate()
    args = (42, "PhysicianBench", "patient-1")

    assert derive_seed(*args, "branch-order") == derive_seed(*args, "branch-order")
    assert derive_seed(*args, "branch-order") != derive_seed(*args, "output-selection")
    assert derive_seed(*args, "branch-order") != derive_seed(43, args[1], args[2], "branch-order")
    assert derive_seed(*args, "branch-order") != derive_seed(args[0], args[1], "patient-2", "branch-order")
    assert derive_seed(*args, "branch-order") != derive_seed(args[0], "MedAgentBench", args[2],
                                                            "branch-order")
    assert random.getstate() == before


def test_selection_leaves_the_global_random_state_untouched():
    before = random.getstate()
    selected = select_candidates(qualified_candidates(6), directions(), {BLOCK_A}, 42)

    assert len(selected) == 2
    assert random.getstate() == before


@pytest.mark.asyncio
async def test_branches_share_one_atomic_pool_with_a_seeded_order():
    pipeline = scheduling_pipeline(7)
    await pipeline.run()

    atomic = pipeline.atomic_candidates
    assert pipeline.branch_inputs["depth"] is atomic
    assert pipeline.branch_inputs["width"] is atomic
    assert sorted(pipeline.branch_order) == ["depth", "width"]
    assert [entry["stage"] for entry in pipeline.records if entry.get("status") == "started"] == [
        f"{pipeline.branch_order[0]}/branch", f"{pipeline.branch_order[1]}/branch"]
    # 候选池固定为 atomic、depth、width 的各自生成顺序，与分支执行顺序无关。
    assert [candidate.origin for candidate in pipeline.pool] == [
        "atomic", "atomic", "atomic", "depth", "width"]


@pytest.mark.asyncio
async def test_branch_order_is_reproducible_for_one_seed():
    first = scheduling_pipeline(5)
    await first.run()
    second = scheduling_pipeline(5)
    await second.run()

    assert first.branch_order == second.branch_order
    assert tuple(first.branch_order) in {("depth", "width"), ("width", "depth")}


@pytest.mark.asyncio
async def test_exhausted_budget_skips_the_remaining_branches_without_adding_tasks():
    pipeline = scheduling_pipeline(7)
    pipeline.ledger.exhausted = True
    await pipeline.run()

    assert pipeline.branch_inputs == {}
    assert [entry["reason"] for entry in pipeline.records if entry.get("status") == "skipped"] == [
        "cumulative_budget_exhausted", "cumulative_budget_exhausted"]
    assert len(pipeline.pool) == 3, "预算耗尽不丢弃已生成的候选"


@pytest.mark.asyncio
async def test_width_skips_with_a_real_reason_when_the_atomic_pool_is_short():
    pipeline = scheduling_pipeline(7)
    pipeline.atomic_candidates = atomic_pool(1)
    await pipeline.run()

    skipped = [entry for entry in pipeline.records if entry.get("reason") == "insufficient_parents"]
    assert len(skipped) == 1 and skipped[0]["available"] == 1
    assert "width" not in pipeline.branch_inputs
    assert "depth" in pipeline.branch_inputs


@pytest.mark.asyncio
async def test_branch_failure_keeps_generated_candidates_in_the_pool(patched, environment,
                                                                    monkeypatch):
    fake = FakeRuntime()
    patched(fake)
    monkeypatch.setattr("methods.taskcraft.Pipeline", RaisingDepthPipeline)

    result = await generate_tasks(trajectory(), environment, directions(), runtime_config(),
                                  random_seed=7)

    assert result.status == "failed" and "深度分支执行失败" in result.error
    failures = [entry for entry in pipeline_records(result)
                if entry["stage"] == "depth/branch" and entry["status"] == "failed"]
    assert [entry["reason"] for entry in failures] == ["RuntimeError: 深度分支执行失败"], \
        "分支异常必须记录真实原因"
    pool = pipeline_pool(result)
    assert [candidate["origin"] for candidate in pool] == ["atomic"] * 3, \
        "分支抛错时已生成的原子候选必须留在池中"
    assert [candidate["question"]["instruction"] for candidate in pool] == [
        candidate.question.instruction for candidate in atomic_pool(3)]


def test_selection_is_repeatable_capped_and_bound_by_the_clinical_quota():
    candidates = qualified_candidates(6)
    first = select_candidates(candidates, directions(), {BLOCK_A}, 9)
    second = select_candidates(candidates, directions(), {BLOCK_A}, 9)

    assert [candidate.question.instruction for candidate in first] == [
        candidate.question.instruction for candidate in second]
    assert len(first) == 2
    assert Counter(candidate.question.task_type for candidate in first) == Counter(
        direction.task_type for direction in directions())


def test_selection_stops_at_the_direction_quota():
    selected = select_candidates(qualified_candidates(5, task_types=("诊断与解释",)),
                                 directions(), {BLOCK_A}, 3)

    assert [candidate.question.task_type for candidate in selected] == ["诊断与解释"]


def test_selection_ignores_incomplete_candidates_and_foreign_sources():
    qualified = qualified_candidates(2)
    unusable = [
        Candidate(question=GeneratedQuestion(instruction="  ", task_type="诊断与解释",
                                             reference_answer="答案",
                                             evidence=[Evidence(source_id=BLOCK_A, statement="记录")]),
                  origin="atomic", locator=BLOCK_A),
        Candidate(question=GeneratedQuestion(instruction="问题", task_type="治疗规划",
                                             reference_answer="   ",
                                             evidence=[Evidence(source_id=BLOCK_A, statement="记录")]),
                  origin="atomic", locator=BLOCK_A),
        Candidate(question=GeneratedQuestion(instruction="问题", task_type="治疗规划",
                                             reference_answer="答案", evidence=[]),
                  origin="atomic", locator=BLOCK_A),
        Candidate(question=GeneratedQuestion(instruction="问题", task_type="治疗规划",
                                             reference_answer="答案", decision_point="2026-01-01",
                                             evidence=[Evidence(source_id=BLOCK_A, statement="记录")]),
                  origin="atomic", locator=BLOCK_A),
        Candidate(question=GeneratedQuestion(instruction="问题", task_type="治疗规划",
                                             reference_answer="答案",
                                             evidence=[Evidence(source_id="Paper/unknown", statement="记录")]),
                  origin="atomic", locator=BLOCK_A),
        Candidate(question=GeneratedQuestion(instruction="问题", task_type="治疗规划",
                                             reference_answer="答案",
                                             evidence=[Evidence(source_id=BLOCK_A, statement="  ")]),
                  origin="atomic", locator=BLOCK_A),
    ]
    selected = select_candidates([*unusable, *qualified], directions(), {BLOCK_A}, 4)

    assert sorted(candidate.question.instruction for candidate in selected) == sorted(
        candidate.question.instruction for candidate in qualified)


def test_selection_does_not_read_the_generation_origin():
    candidates = qualified_candidates(6)
    relabelled = [Candidate(question=candidate.question, origin=f"origin-{index}",
                            parents=list(candidate.parents), locator=candidate.locator,
                            detail=dict(candidate.detail))
                  for index, candidate in enumerate(candidates)]

    assert [candidate.question.instruction
            for candidate in select_candidates(candidates, directions(), {BLOCK_A}, 21)] == [
        candidate.question.instruction
        for candidate in select_candidates(relabelled, directions(), {BLOCK_A}, 21)]


def test_selection_stream_is_separate_from_the_branch_stream():
    pipeline = scheduling_pipeline(17)
    assert pipeline.derived_seeds["branch-order"] != pipeline.derived_seeds["output-selection"]

    pipeline.pool = qualified_candidates(4)
    first = pipeline.select()
    assert len(first) == len(pipeline.selected) == 2
    assert pipeline.selection_records
    # 分支流与该次选题无关：重新选题结果不变。
    assert [candidate.question.instruction for candidate in pipeline.select()] == [
        candidate.question.instruction for candidate in first]

    other = scheduling_pipeline(17)
    other.pool = qualified_candidates(4)
    assert [candidate.question.instruction for candidate in other.select()] == [
        candidate.question.instruction for candidate in first]


@pytest.mark.asyncio
async def test_pipeline_trace_records_scheduling_and_selection(patched, environment):
    fake = FakeRuntime()
    patched(fake)
    result = await generate_tasks(trajectory(), environment, directions(), runtime_config(),
                                  random_seed=42)

    entry = next(item for item in result.trace if item["stage"] == "taskcraft/pipeline")
    assert entry["experiment_seed"] == 42
    assert set(entry["derived_seeds"]) == {"branch-order", "output-selection"}
    assert sorted(entry["branch_order"]) == ["depth", "width"]
    assert entry["selection"]["selected"] and entry["selection"]["filtering"]
    assert all(item["question"]["instruction"] for item in entry["pool"])
    assert all("detail" in item and "origin" in item and "parents" in item and "locator" in item
               for item in entry["pool"])
    assert json.loads(json.dumps(entry)) == entry, "候选与调度记录必须可 JSON 序列化"
    # 公共 runner 以 model_dump(mode="json") 持久化整条运行结果。
    assert json.loads(result.model_dump_json()) == result.model_dump(mode="json")


@pytest.mark.asyncio
async def test_trajectory_environment_mismatch_fails_before_creating_an_instance(patched, environment):
    fake = FakeRuntime()
    state = patched(fake)
    result = await generate_tasks(trajectory(), environment.model_copy(update={"name": "MedAgentBench"}),
                                  directions(), runtime_config())

    assert result.status == "failed" and result.output is None
    assert "does not match" in result.error
    assert state["created"] == state["released"] == 0
    assert fake.llm_calls == []
