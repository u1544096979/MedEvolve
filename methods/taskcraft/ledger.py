"""累计生成预算入口：每个模型调用与每次工具调用都记入公共预算，耗尽后停止发起调用。

源码路径：``utils.py:run_llm_prompt``（``max_retries`` 是调用点参数）、
``agent_tools.SearchAgent.forward``（``max_retries=3``）与
``oagents.agents.ToolCallingAgent``（单次 Agent 调用的 ``max_step`` 上限）。
"""

from contracts import RunResult
from methods.common import GenerationBudget

from .agent import DEFAULT_MAX_ACTION_STEPS, run_search_agent

__all__ = ["Ledger"]


class Ledger:
    """包装 GenerationBudget，记录调用并把预算耗尽暴露给阶段循环。"""

    def __init__(self, budget: GenerationBudget):
        self.budget = budget
        self.exhausted = False

    async def llm(self, stage: str, *, instructions: str, input_text: str, output_type,
                  attempts: int = 1):
        """单次模型请求的重试：每次尝试自行记账后才重试，失败成本保留在公共账本中。"""
        last = None
        for attempt in range(attempts):
            if self.budget.remaining().max_model_turns == 0:
                self.exhausted = True
                return last
            last = await self.budget.call_llm(
                f"{stage}/attempt-{attempt + 1}", instructions=instructions,
                input_text=input_text, output_type=output_type)
            if last.status == "completed":
                return last
        self.exhausted = self.budget.remaining().max_model_turns == 0
        return last

    async def agent(self, stage: str, *, instructions: str, input_text: str, tools: list,
                    output_type, max_action_steps: int = DEFAULT_MAX_ACTION_STEPS):
        """单次检索 Agent 调用：保留源码步骤，内部每个请求各自记账。"""
        if self.exhausted:
            return None
        return self._track(await run_search_agent(
            ledger=self, stage=stage, instructions=instructions, input_text=input_text,
            tools=tools, output_type=output_type, max_action_steps=max_action_steps))

    def tool(self, stage: str, *, name: str, arguments, output) -> None:
        """把一次真实工具调用记入公共账本，使累计工具预算与失败成本可见。"""
        self.budget.records.append((stage, RunResult(
            trace=[{"type": "tool_start", "name": name, "arguments": arguments},
                   {"type": "tool_output", "name": name, "output": output}],
            usage={"model_turns": 0, "tool_calls": 1}, status="completed")))

    def limit_turns(self) -> int:
        return self.budget.limit.max_model_turns

    def used_turns(self) -> int:
        return self.limit_turns() - self.budget.remaining().max_model_turns

    def remaining_turns(self) -> int:
        return self.budget.remaining().max_model_turns

    def remaining_tools(self) -> int:
        return self.budget.remaining().max_tool_calls

    def _track(self, run):
        if run.status == "budget_exhausted":
            self.exhausted = True
        return run
