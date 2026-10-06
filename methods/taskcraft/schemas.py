"""各阶段模型输出结构，对应源码提示词声明的 JSON 约定。"""

from pydantic import BaseModel, Field, JsonValue

__all__ = [
    "AgentThought", "AnswerList", "AnswerScore", "BackwardContext", "CleanedQuestion",
    "Conclusion", "ConclusionList", "DecompositionReport", "DecompositionState",
    "FinalThought", "InitialQuestion", "KnowledgeAnswer", "KnowledgeAnswers", "MergedQuestion",
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


class CleanedQuestion(BaseModel):
    question: str
    original_answer: str = ""
    refined_answer: str


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


class QueryMerge(BaseModel):
    analysis: str = ""
    new_query: str


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


# 检索 Agent 的请求结构：源码 oagents 的提示词要求 <think>/<answer> 两段，
# 公共运行层使用结构化输出承载同一语义。
class AgentThought(BaseModel):
    think: str = ""
    answer: str = ""


# 达到 action 上限后的强制终答：源码 FinalAnswerTool 的 answer 声明为 any，
# 可以返回任意 JSON 值，SearchAgent.forward 再对文本形态跑 safe_json_loads。
class FinalThought(BaseModel):
    think: str = ""
    answer: JsonValue = None


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
