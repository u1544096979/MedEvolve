"""各阶段模型输出结构，对应源码提示词声明的 JSON 约定。"""

from pydantic import BaseModel, Field

__all__ = [
    "AnswerList", "AnswerScore", "BackwardContext", "CleanedQuestion", "Conclusion",
    "ConclusionList", "DecompositionReport", "DecompositionState", "InitialQuestion",
    "KnowledgeAnswer", "KnowledgeAnswers", "MergedQuestion", "MergedQuestions",
    "NewQuery", "QueryJudgement", "QueryMerge", "SupersetCheck",
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
