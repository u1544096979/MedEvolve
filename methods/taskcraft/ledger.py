"""累计生成预算入口：每个模型调用都记入公共预算，耗尽后停止发起调用。"""

from methods.common import GenerationBudget

__all__ = ["Ledger"]


class Ledger:
    """包装 GenerationBudget，记录调用并把预算耗尽暴露给阶段循环。"""

    def __init__(self, budget: GenerationBudget):
        self.budget = budget
        self.exhausted = False

    async def llm(self, stage: str, *, instructions: str, input_text: str, output_type):
        if self.exhausted:
            return None
        return self._track(await self.budget.call_llm(
            stage, instructions=instructions, input_text=input_text, output_type=output_type))

    async def agent(self, stage: str, *, instructions: str, input_text: str, tools: list, output_type):
        if self.exhausted:
            return None
        return self._track(await self.budget.run_agent(
            stage, instructions=instructions, input_text=input_text, tools=tools,
            output_type=output_type))

    def limit_turns(self) -> int:
        return self.budget.limit.max_model_turns

    def used_turns(self) -> int:
        return self.limit_turns() - self.budget.remaining().max_model_turns

    def remaining_turns(self) -> int:
        return self.budget.remaining().max_model_turns

    def _track(self, run):
        if run.status == "budget_exhausted":
            self.exhausted = True
        return run
