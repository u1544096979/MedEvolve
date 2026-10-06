"""TaskCraft 内置核验：移植 agent_tools.VerifyAgent（提交 1e405e0）。

源码路径：agent_tools.py:132-189。
每个问题先尝试无工具回答，评分达到 1 分及以上即判为可由模型自身解决；
再交给病历工具 Agent 回答并打分。两个分数都保留，外层重试重新走一遍整个流程。

适配差异：无工具作答与每次评分请求按源码 `max_retries=1` 单次调用，外层重试次数为源码
`forward(max_retries=3)`；三次并行判定在公共层顺序执行，失败采样不计入平均值。
"""

import json
from dataclasses import dataclass, field

from contracts import SolverAnswer

from .environment import NATIVE_FINISH_ANSWERS, is_native
from .ledger import Ledger
from .prompts import AUGMENTED_QUERY, LLM_ONLY_INSTRUCTIONS, verify_prompts
from .schemas import AnswerList, AnswerScore

__all__ = ["VerifyOutcome", "score_recall", "verify_query"]

# VerifyAgent.forward(max_retries=3)：外层重试重新走无工具作答、评分、工具作答、评分。
VERIFY_ATTEMPTS = 3
# VerifyAgent.__init__ 的 run_llm_prompt(max_retries=1)：无工具作答与评分请求各只调用一次。
VERIFY_CALL_ATTEMPTS = 1
# SearchAgent 默认 max_step=20；深度反向与深度核验源码取 10。
ATOMIC_MAX_ACTION_STEPS = 20
DEPTH_BACKWARD_MAX_ACTION_STEPS = 10
DEPTH_VERIFY_MAX_ACTION_STEPS = 10


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
    """移植 VerifyAgent.recall_score：多次判定取有效结果平均，None 表示评分未完成。"""
    prompt = f"""The inputs are as follows:
    Golden Answer: {golden_answer}
    Other Answer: {other_answer}
    """.strip()
    scores = []
    for _ in range(samples):
        run = await ledger.llm(stage, instructions=verify_prompts()["score_prompt_single"],
                               input_text=prompt, output_type=AnswerScore,
                               attempts=VERIFY_CALL_ATTEMPTS)
        if run is None:
            break
        if run.status != "completed":
            continue
        try:
            scores.append(AnswerScore.model_validate(run.output).answer_score)
        except Exception:
            # 无法比较的判定不计入平均值；全部无效时返回 None，由上一级走错误分支。
            continue
    if not scores:
        return None
    return sum(scores) / len(scores)


def _trajectory(run) -> list[dict]:
    """移植 capture_trajectory：按记忆步骤映射，不用工具调用数量重建 Agent 记忆。"""
    steps = []
    for item in run.trace:
        kind = item.get("type")
        if kind == "planning":
            steps.append({"name": "facts", "value": item.get("facts"),
                          "think": item.get("facts_think")})
            steps.append({"name": "plan", "value": item.get("plan"),
                          "think": item.get("plan_think")})
        elif kind == "reflection":
            steps.append({"name": "reflection", "value": item.get("score"),
                          "think": item.get("analysis")})
            steps.append({"name": "facts", "value": item.get("facts"), "think": ""})
            steps.append({"name": "plan", "value": item.get("plan"), "think": ""})
        elif kind == "action":
            steps.append({"name": "action", "value": item.get("model_output"),
                          "obs": item.get("observations"), "think": item.get("model_think", "")})
    return steps


def _native_answer_list(output: dict) -> list[str]:
    """原生协议把 FINISH 的答案列表序列化到 SolverAnswer.answer。"""
    value = json.loads(SolverAnswer.model_validate(output).answer)
    if not isinstance(value, list):
        raise ValueError("FINISH must carry a JSON list of answers")
    return [str(item) for item in value]


def _failed(query: str, golden_answer, error: str) -> VerifyOutcome:
    """源码 forward 的返回形态：出错时只带回 error。"""
    return VerifyOutcome(query=query, golden_answer=golden_answer, error=error)


async def verify_query(ledger: Ledger, query: str, golden_answer, tools: list, *,
                       samples: int, stage: str, instructions: str,
                       max_action_steps: int = ATOMIC_MAX_ACTION_STEPS) -> VerifyOutcome:
    """VerifyAgent.forward：外层重试整段流程，llm_score 达到 1 分及以上立即返回 error。"""
    native = is_native(tools)
    augmented = AUGMENTED_QUERY + query
    if native:
        augmented += f"\n{NATIVE_FINISH_ANSWERS}"

    last_error = None
    for _ in range(VERIFY_ATTEMPTS):
        llm_run = await ledger.llm(f"{stage}/llm", instructions=LLM_ONLY_INSTRUCTIONS,
                                   input_text=augmented, output_type=AnswerList,
                                   attempts=VERIFY_CALL_ATTEMPTS)
        if llm_run is None:
            return _failed(query, golden_answer, _failure(llm_run))
        if llm_run.status != "completed":
            last_error = _failure(llm_run)
            continue
        llm_answer = AnswerList.model_validate(llm_run.output).answer_list
        llm_score = await score_recall(ledger, golden_answer, llm_answer,
                                       samples=samples, stage=f"{stage}/llm_score")
        if llm_score is None:
            # 源码：评分返回 None 时比较运算抛错，走外层重试分支。
            last_error = "LLM score unavailable"
            continue
        # agent_tools.py:154-155：无工具回答评分达到 1 分及以上，说明该结论不是原子结论，立即返回。
        if llm_score >= 1:
            return _failed(query, golden_answer, "LLM can solve this task")

        agent_run = await ledger.agent(f"{stage}/agent", instructions=instructions,
                                       input_text=augmented, tools=tools,
                                       output_type=SolverAnswer if native else AnswerList,
                                       max_action_steps=max_action_steps)
        if agent_run is None:
            return _failed(query, golden_answer, _failure(agent_run))
        if agent_run.status != "completed":
            last_error = _failure(agent_run)
            continue
        try:
            agent_answer = (_native_answer_list(agent_run.output) if native
                            else AnswerList.model_validate(agent_run.output).answer_list)
        except Exception as exc:
            last_error = f"invalid agent answer: {type(exc).__name__}: {exc}"
            continue
        agent_score = await score_recall(ledger, golden_answer, agent_answer,
                                         samples=samples, stage=f"{stage}/agent_score")
        # 源码 agent_tools.py:164-173 直接返回评分结果（包括 None），不重试；
        # 阶段按源码分支处理：原子过滤只看出错，深度扩展把空分数当作未通过。
        return VerifyOutcome(
            query=query, golden_answer=golden_answer, llm_score=llm_score,
            llm_answer=llm_answer, agent_score=agent_score, agent_answer=agent_answer,
            # 真实 action step 数，不是模型请求数。
            agent_step_number=int(agent_run.usage.get("action_steps") or 0),
            agent_trajectory=_trajectory(agent_run),
        )
    return _failed(query, golden_answer, last_error or "verify attempts exhausted")
