"""TaskCraft 内置核验：移植 agent_tools.VerifyAgent（提交 1e405e0）。

源码路径：agent_tools.py:141-243。
每个问题先尝试无工具回答，回答得分达到满分即判为可由模型自身解决；
再交给病历工具 Agent 回答并打分。两个分数都保留。
"""

import json
from dataclasses import dataclass, field

from contracts import SolverAnswer

from .environment import NATIVE_FINISH_ANSWERS, is_native
from .ledger import Ledger
from .prompts import AUGMENTED_QUERY, LLM_ONLY_INSTRUCTIONS, verify_prompts
from .schemas import AnswerList, AnswerScore

__all__ = ["VerifyOutcome", "score_recall", "verify_query"]


@dataclass
class VerifyOutcome:
    query: str
    golden_answer: object
    llm_score: float | None = None
    llm_answer: list[str] | None = None
    agent_score: float | None = None
    agent_answer: list[str] | None = None
    agent_step_number: int | None = None
    agent_trajectory: list[dict] = field(default_factory=list)
    error: str | None = None

    def record(self) -> dict:
        return {
            "query": self.query,
            "llm_score": self.llm_score,
            "llm_answer": self.llm_answer,
            "agent_score": self.agent_score,
            "agent_answer": self.agent_answer,
            "agent_step_number": self.agent_step_number,
            "agent_trajectory": self.agent_trajectory,
            "error": self.error,
        }


def _failure(run) -> str:
    if run is None:
        return "budget exhausted before the call"
    return run.error or run.status


async def score_recall(ledger: Ledger, golden_answer, other_answer, *,
                       samples: int, stage: str) -> float | None:
    """移植 VerifyAgent.recall_score：多次判定取平均，返回 None 表示评分未完成。"""
    prompt = f"""The inputs are as follows:
    Golden Answer: {golden_answer}
    Other Answer: {other_answer}
    """.strip()
    scores = []
    for _ in range(samples):
        run = await ledger.llm(stage, instructions=verify_prompts()["score_prompt_single"],
                              input_text=prompt, output_type=AnswerScore)
        if run is None or run.status != "completed":
            break
        scores.append(AnswerScore.model_validate(run.output).answer_score)
    if not scores:
        return None
    return sum(scores) / len(scores)


def _trajectory(run) -> list[dict]:
    """阶段内保留紧凑轨迹；工具运行的完整记录在公共运行结果 trace 中。"""
    steps = []
    for item in run.trace:
        if item.get("type") == "tool_start":
            steps.append({"type": "tool", "name": item.get("name"),
                          "arguments": item.get("arguments")})
        elif item.get("type") == "run_error":
            steps.append({"type": "run_error", "error": item.get("error")})
    return steps


def _native_answer_list(output: dict) -> list[str]:
    """原生协议把 FINISH 的答案列表序列化到 SolverAnswer.answer。"""
    value = json.loads(SolverAnswer.model_validate(output).answer)
    if not isinstance(value, list):
        raise ValueError("FINISH must carry a JSON list of answers")
    return [str(item) for item in value]


async def verify_query(ledger: Ledger, query: str, golden_answer, tools: list, *,
                       samples: int, stage: str, instructions: str) -> VerifyOutcome:
    outcome = VerifyOutcome(query=query, golden_answer=golden_answer)
    native = is_native(tools)
    augmented = AUGMENTED_QUERY + query
    if native:
        augmented += f"\n{NATIVE_FINISH_ANSWERS}"

    llm_run = await ledger.llm(f"{stage}/llm", instructions=LLM_ONLY_INSTRUCTIONS,
                              input_text=augmented, output_type=AnswerList)
    if llm_run is None or llm_run.status != "completed":
        outcome.error = _failure(llm_run)
        return outcome
    outcome.llm_answer = AnswerList.model_validate(llm_run.output).answer_list
    outcome.llm_score = await score_recall(ledger, golden_answer, outcome.llm_answer,
                                          samples=samples, stage=f"{stage}/llm_score")
    if outcome.llm_score is None:
        outcome.error = "LLM score unavailable"
        return outcome
    # agent_tools.py:184-185：无工具回答已满分，说明该结论不是原子结论。
    if outcome.llm_score >= 1:
        outcome.error = "LLM can solve this task"
        return outcome

    agent_run = await ledger.agent(f"{stage}/agent", instructions=instructions,
                                  input_text=augmented, tools=tools,
                                  output_type=SolverAnswer if native else AnswerList)
    if agent_run is None or agent_run.status != "completed":
        outcome.error = _failure(agent_run)
        return outcome
    try:
        outcome.agent_answer = (_native_answer_list(agent_run.output) if native
                                else AnswerList.model_validate(agent_run.output).answer_list)
    except Exception as exc:
        outcome.error = f"invalid agent answer: {type(exc).__name__}: {exc}"
        return outcome
    outcome.agent_step_number = int(agent_run.usage.get("model_turns") or 0)
    outcome.agent_trajectory = _trajectory(agent_run)
    outcome.agent_score = await score_recall(ledger, golden_answer, outcome.agent_answer,
                                            samples=samples, stage=f"{stage}/agent_score")
    if outcome.agent_score is None:
        outcome.error = "agent score unavailable"
    return outcome
