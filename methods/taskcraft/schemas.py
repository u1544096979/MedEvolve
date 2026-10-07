"""各阶段模型输出结构，对应源码提示词声明的 JSON 约定。"""

import json
from typing import Literal

from pydantic import BaseModel, Field, JsonValue, field_validator

__all__ = [
    "AgentThought", "AnswerList", "AnswerScore", "BackwardContext", "CleanedQuestion",
    "Conclusion", "ConclusionList", "DecompositionReport", "DecompositionState",
    "InitialQuestion", "KnowledgeAnswer", "KnowledgeAnswers", "MergedQuestion",
    "MergedQuestions", "NativeAction", "NewQuery", "QueryJudgement", "QueryMerge",
    "SearchAction", "SupersetCheck", "ToolCall", "TrajectoryScore",
]


class Conclusion(BaseModel):
    conclusion: str
    R: str
    # 医疗适配新增：支持该结论的病历记录块标识。
    source_ids: list[str] = Field(default_factory=list)


class ConclusionList(BaseModel):
    conclusions: list[Conclusion]


class InitialQuestion(BaseModel):
    Q: str
    # 医疗适配新增：该题属于哪个请求的题型。
    task_type: str = ""
    execution_requirement: Literal["read_only"] = "read_only"


class CleanedQuestion(BaseModel):
    question: str
    original_answer: str = ""
    refined_answer: str
    execution_requirement: Literal["read_only"] = "read_only"


class AnswerList(BaseModel):
    answer_list: list[str]


class AnswerScore(BaseModel):
    answer_analysis: str = ""
    answer_score: int


class SupersetCheck(BaseModel):
    # 源码 is_superset_valid 提示词声明的键，取值为 valid 或 invalid。
    new_query: str


class NewQuery(BaseModel):
    new_query: str
    execution_requirement: Literal["read_only"] = "read_only"


class QueryMerge(BaseModel):
    analysis: str = ""
    new_query: str
    execution_requirement: Literal["read_only"] = "read_only"


class QueryJudgement(BaseModel):
    analysis: str = ""
    is_valid: bool


class BackwardContext(BaseModel):
    identifier: str
    relation: str
    # 医疗适配新增：支持该上下文与关系的病历记录块标识。
    source_ids: list[str] = Field(default_factory=list)


class MergedQuestion(BaseModel):
    question: str
    index: list[int]
    content_identifier: str = ""
    execution_requirement: Literal["read_only"] = "read_only"


class MergedQuestions(BaseModel):
    merged: list[MergedQuestion]


class DecompositionState(BaseModel):
    index: int
    state: int


class DecompositionReport(BaseModel):
    results: list[DecompositionState]


class KnowledgeAnswer(BaseModel):
    index: int
    llm_answer: str = ""


class KnowledgeAnswers(BaseModel):
    answers: list[KnowledgeAnswer]


def thought_text(value: JsonValue) -> str:
    """源码 extract_think_and_answer 的文本化：字符串保持原样，其余取 JSON 文本。"""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


# 检索 Agent 的请求结构：源码 oagents 的提示词要求 <think>/<answer> 两段。
# 规划、反思与达到 action 上限后的强制终答都走 extract_think_and_answer：两段都是自由文本，
# 不解析 JSON，因此模型把任务要求的对象直接放进 answer 时，按源码取它的 JSON 文本。
class AgentThought(BaseModel):
    think: JsonValue = ""
    answer: JsonValue = ""

    @field_validator("think", "answer")
    @classmethod
    def _as_text(cls, value: JsonValue) -> str:
        return thought_text(value)


# 源码 reflection_step 的 process reward model 输出。
class TrajectoryScore(BaseModel):
    analysis: str = ""
    score: int


# 源码 ToolCall（oagents.agent_types）：工具名加参数。
class ToolCall(BaseModel):
    name: str
    arguments: JsonValue = None


# 函数工具协议的动作编码：一次响应可携带多个工具调用，
# final_answer 对应源码的 FinalAnswerTool 收尾分支。
class SearchAction(BaseModel):
    tool_calls: list[ToolCall] = Field(default_factory=list)
    final_answer: JsonValue = None


# 原生协议的动作编码：GET 检索或 FINISH 收尾的文本动作。
class NativeAction(BaseModel):
    action: str
