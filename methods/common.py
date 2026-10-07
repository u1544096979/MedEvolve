"""Shared generation constraints, cumulative budgets and run accounting."""

import importlib
import json
from collections import Counter
from typing import Literal, get_args

from pydantic import BaseModel

from contracts import (Budget, GenerationDirection, RunResult, RuntimeConfig, TaskBatch,
                       TaskType)
from environments.capabilities import capability_instructions

RunStatus = Literal["completed", "budget_exhausted", "failed"]

USAGE_FIELDS = ("model_turns", "tool_calls", "input_tokens", "output_tokens", "total_tokens")


def normalize_usage(usage: dict) -> dict:
    result = {"model_turns": usage.get("model_turns", 0),
              "tool_calls": usage.get("tool_calls", 0),
              "input_tokens": usage.get("input_tokens", usage.get("prompt_tokens")),
              "output_tokens": usage.get("output_tokens", usage.get("completion_tokens")),
              "total_tokens": usage.get("total_tokens")}
    if result["total_tokens"] is None and all(result[k] is not None for k in ("input_tokens", "output_tokens")):
        result["total_tokens"] = result["input_tokens"] + result["output_tokens"]
    if "runs_without_known_tokens" in usage:
        result["runs_without_known_tokens"] = usage["runs_without_known_tokens"]
    return result


NUMBER_WORDS = {1: "one", 2: "two"}


def _validate_directions(directions: list[GenerationDirection], *, max_questions: int,
                         candidate_pool: bool) -> None:
    if max_questions not in (1, 2):
        raise ValueError("max_questions must be one or two")
    types = [d.task_type for d in directions]
    if candidate_pool:
        if len(types) != 5 or set(types) != set(get_args(TaskType)):
            raise ValueError("candidate pool must contain each of the five task types once")
    elif len(directions) != max_questions:
        raise ValueError("directions must contain exactly "
                         f"{NUMBER_WORDS[max_questions]} generation positions")


def _selection_policy(directions: list[GenerationDirection] | None,
                      task_types: list[TaskType] | None, *, max_questions: int,
                      candidate_pool: bool) -> Counter | set:
    """Resolve the allowed task types of direction mode or free-selection mode."""
    if task_types is not None:
        if directions is not None:
            raise ValueError("Use directions or task_types, not both")
        expected = set(get_args(TaskType))
        if len(task_types) != len(expected) or set(task_types) != expected:
            raise ValueError("task_types must contain all five task types once")
        return set(task_types)
    if directions is None:
        raise ValueError("directions must contain exactly two generation positions")
    _validate_directions(directions, max_questions=max_questions, candidate_pool=candidate_pool)
    return Counter(d.task_type for d in directions)


def output_constraints(directions: list[GenerationDirection] | None = None, *,
                       task_types: list[TaskType] | None = None, max_questions: int = 2,
                       candidate_pool: bool = False) -> str:
    _selection_policy(directions, task_types, max_questions=max_questions,
                      candidate_pool=candidate_pool)
    fields = (
        f"返回 TaskBatch JSON，questions 最多{max_questions}题；证据不足可少出题或返回空列表。"
        "每题包含 instruction、task_type、reference_answer、evidence、decision_point、execution_requirement。"
        'execution_requirement 必须为 "read_only"。'
        "instruction 和 reference_answer 非空；evidence 逐条包含可定位的 source_id 和事实 statement。"
        "decision_point 必须为 null。不要附加患者、环境、轮次、方法或任务编号。"
    )
    if task_types is not None:
        return fields + (
            "从以下五种题型中自主选择适合资料的题目，两题可同类型，不要求覆盖全部题型："
            + json.dumps(task_types, ensure_ascii=False)
        )
    positions = ("候选题型池覆盖五类题型，每类题型最多一题；" if candidate_pool else
                 f"按 {len(directions)} 个生成位置出题；")
    return fields + positions + (
        "题型数量不能超过指定方向中该题型的数量；题型与难度方向："
        + json.dumps([d.model_dump(mode="json") for d in directions], ensure_ascii=False)
    )


def validate_task_batch(result: RunResult, directions: list[GenerationDirection] | None = None, *,
                        task_types: list[TaskType] | None = None, max_questions: int = 2,
                        candidate_pool: bool = False) -> RunResult:
    allowed = _selection_policy(directions, task_types, max_questions=max_questions,
                                candidate_pool=candidate_pool)
    result = result.model_copy(update={"usage": normalize_usage(result.usage)})
    if result.status == "failed":
        return result.model_copy(update={"output": None})
    if result.status == "budget_exhausted" and result.output is None:
        return result
    try:
        batch = TaskBatch.model_validate(result.output)
        if len(batch.questions) > max_questions:
            raise ValueError(f"TaskBatch exceeds the {max_questions} allowed questions")
        actual = Counter(q.task_type for q in batch.questions)
        if task_types is None:
            if actual - allowed:
                raise ValueError("TaskBatch task types exceed requested directions")
        elif set(actual) - allowed:
            raise ValueError("TaskBatch contains unsupported task types")
        for question in batch.questions:
            if question.decision_point is not None:
                raise ValueError("decision_point must be null until time filtering is supported")
            if question.execution_requirement != "read_only":
                raise ValueError("execution_requirement must be read_only")
            if not question.instruction.strip() or not question.reference_answer.strip():
                raise ValueError("instruction and reference_answer must not be empty")
            if any(not e.source_id.strip() or not e.statement.strip() for e in question.evidence):
                raise ValueError("evidence source_id and statement must not be empty")
        return result.model_copy(update={"output": batch.model_dump(mode="json")})
    except Exception as exc:
        return result.model_copy(update={"output": None, "status": "failed", "error": str(exc),
                                         "trace": [*result.trace, {"type": "output_rejected",
                                                                   "output": result.output, "error": str(exc)}]})


def summarize_results(records: list[tuple[str, RunResult]], *, output: TaskBatch | None = None,
                      status: RunStatus | None = None, error: str | None = None) -> RunResult:
    usages = [normalize_usage(run.usage) for _, run in records]
    usage = {field: (sum(u[field] for u in usages if u[field] is not None)
                     if any(u[field] is not None for u in usages) else None)
             for field in USAGE_FIELDS}
    usage["model_turns"] = usage["model_turns"] or 0
    usage["tool_calls"] = usage["tool_calls"] or 0
    usage["runs_without_known_tokens"] = sum(u.get("runs_without_known_tokens", int(u["total_tokens"] is None)) for u in usages)
    final = records[-1][1] if records else None
    final_status = status or (final.status if final else "completed")
    if output is not None and final_status in {"completed", "budget_exhausted"}:
        final_output = output.model_dump(mode="json")
    elif final_status == "completed" and final is not None:
        final_output = final.output
    else:
        final_output = None
    return RunResult(output=final_output,
                     trace=[{"stage": stage, "result": run.model_dump(mode="json")}
                            for stage, run in records], usage=usage, status=final_status,
                     error=error or (final.error if final else None))


class GenerationBudget:
    """One instance per patient batch; execute sequential stages through this object."""

    def __init__(self, runtime: RuntimeConfig, environment: str | None = None):
        self.model = runtime.models["generator"]
        self.limit = runtime.budgets["generator"]
        self.records: list[tuple[str, RunResult]] = []
        self.environment = environment

    def _instructions(self, instructions: str) -> str:
        from runtime.context import PARTIAL_NOTE

        instructions = f"{instructions}\n{PARTIAL_NOTE}"
        if self.environment is None:
            return instructions
        return f"{instructions}\n\n{capability_instructions(self.environment)}"

    def remaining(self) -> Budget:
        used = summarize_results(self.records).usage
        return Budget(max_model_turns=max(0, self.limit.max_model_turns - used["model_turns"]),
                      max_tool_calls=max(0, self.limit.max_tool_calls - used["tool_calls"]))

    async def _execute(self, stage, function, kwargs, *, agent):
        remaining = self.remaining()
        if remaining.max_model_turns == 0:
            result = RunResult(trace=[], usage={}, status="budget_exhausted",
                               error="Cumulative generator model budget exhausted")
        else:
            try:
                if agent:
                    kwargs["budget"] = remaining
                result = await function(model=self.model, **kwargs)
            except Exception as exc:
                result = RunResult(trace=[], usage={"model_turns": 1, "tool_calls": 0},
                                   status="failed", error=f"{type(exc).__name__}: {exc}")
        result = result.model_copy(update={"usage": normalize_usage(result.usage)})
        self.records.append((stage, result))
        return result

    async def call_llm(self, stage: str, *, instructions: str, input_text: str, output_type: type[BaseModel]) -> RunResult:
        return await self._execute(stage, importlib.import_module("runtime").call_llm,
                                   dict(instructions=self._instructions(instructions), input_text=input_text, output_type=output_type),
                                   agent=False)

    async def run_agent(self, stage: str, *, instructions: str, input_text: str,
                        tools: list, output_type: type[BaseModel]) -> RunResult:
        return await self._execute(stage, importlib.import_module("runtime").run_agent,
                                   dict(instructions=self._instructions(instructions), input_text=input_text,
                                        tools=tools, output_type=output_type), agent=True)

    def finish(self, directions: list[GenerationDirection] | None = None, *,
               task_types: list[TaskType] | None = None, output: TaskBatch | None = None,
               status: RunStatus | None = None, error: str | None = None) -> RunResult:
        return validate_task_batch(summarize_results(self.records, output=output,
                                                     status=status, error=error),
                                   directions, task_types=task_types)
