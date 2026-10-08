"""Shared contracts between the experiment scheduler and the four harnesses."""

from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel, Field

from contracts import Budget, ModelConfig, RunResult, SolverInput, Task, TaskSplit


__all__ = ["Budget", "GroupKey", "HarnessAdapter", "HarnessLimits", "HarnessName",
           "HarnessStep", "ModelConfig", "RunResult", "Snapshot", "SolverInput",
           "Task", "TaskSplit", "TrainingFeedback"]


HarnessName = Literal["openclaw", "hermes", "ace", "reflexion"]


class GroupKey(BaseModel):
    """One isolated experiment group: environment, harness, source and seed."""

    run_id: str
    environment: str
    harness: HarnessName
    source_method: str
    seed: int


class HarnessLimits(BaseModel):
    """Explicit budgets and caps for one group; no implicit defaults."""

    execution: Budget
    learning: Budget
    max_input_tokens: int = Field(gt=0)
    max_output_tokens: int = Field(gt=0)
    max_total_tokens: int = Field(gt=0)
    timeout_seconds: int = Field(gt=0)
    learning_timeout_seconds: int = Field(gt=0)


class HarnessStep(BaseModel):
    """One learning round, including whether learning was triggered at all."""

    run: RunResult
    learning_runs: list[RunResult]
    learning_status: Literal[
        "updated", "unchanged", "not_triggered", "failed", "budget_exhausted"
    ]


class TrainingFeedback(BaseModel):
    """Review-process feedback for the learning stage; never solver input."""

    task_id: str
    completed: bool | None
    failure_category: str | None
    reason: str
    reference_answer: str
    evidence: list[dict]


class Snapshot(BaseModel):
    """A frozen experience state identified by path and digest."""

    path: Path
    digest: str


class HarnessAdapter(Protocol):
    """Native execution, learning and freezing for one harness.

    Construction receives the model configs, budgets, upstream image lock and
    state paths. Execution receives only solver input and a tool session so that
    reference answers never reach the solving context.
    """

    async def start(self, group: GroupKey, initial: Snapshot) -> None: ...
    async def execute(self, task: SolverInput, tool_session: str) -> RunResult: ...
    async def learn(self, feedback: TrainingFeedback) -> HarnessStep: ...
    async def freeze(self, destination: Path) -> Snapshot: ...
    async def close(self) -> None: ...
