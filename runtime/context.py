"""公共患者材料选取及模型请求预算；临床记录保持完整。"""

import copy
import json
from functools import lru_cache

import tiktoken

from contracts import ModelConfig, PatientTrajectory
from environments.fitting import fit_trajectory

CONTEXT_DEFAULT = 128_000
OUTPUT_DEFAULT = 16_384
MESSAGE_RESERVE = 1024
PARTIAL_NOTE = "患者材料可能只包含部分记录；未提供的记录不能推断为不存在。"
HISTORY_NOTE = "部分较早的交互因上下文预算未提供；仅依据当前可见材料作答。"


class ContextBudgetError(ValueError):
    pass


def dumps(value) -> str:
    return json.dumps(value, ensure_ascii=False)


@lru_cache(maxsize=32)
def _encoding(model_name: str):
    try:
        return tiktoken.encoding_for_model(model_name)
    except KeyError:
        return None


def count_tokens(text: str, model: ModelConfig) -> int:
    encoding = _encoding(model.model)
    if encoding is None:
        return len(text.encode("utf-8"))
    return len(encoding.encode(text, disallowed_special=()))


def output_limit(model: ModelConfig) -> int:
    return model.max_output_tokens or OUTPUT_DEFAULT


def input_limit(model: ModelConfig) -> int:
    available = (model.context_window or CONTEXT_DEFAULT) - output_limit(model) - MESSAGE_RESERVE
    if available <= 0:
        raise ContextBudgetError("Model context window leaves no input budget after output reservation")
    return available


def budget_entry(model: ModelConfig, tokens: int, **fields) -> dict:
    return {"type": "context_budget", "input_tokens_estimate": tokens,
            "input_limit": input_limit(model), "output_limit": output_limit(model),
            "counting": "tokenizer" if _encoding(model.model) else "utf8_bytes", **fields}


def fit_patient(trajectory: PatientTrajectory, model: ModelConfig,
                reserved_tokens: int = 8192) -> PatientTrajectory:
    reserved_tokens = min(reserved_tokens, input_limit(model) // 2)
    return fit_trajectory(trajectory, input_limit(model) - reserved_tokens,
                          measure=lambda value: count_tokens(dumps(value.model_dump(mode="json")), model))


def prepare_input(input_text: str, model: ModelConfig, *, instructions: str,
                  schema: dict, tools: list | None = None) -> tuple[str, dict]:
    """计入阶段材料后选取病历，已被候选或焦点引用的记录保持可见。"""
    fixed = count_tokens(instructions + dumps(schema) + dumps(tools or []), model)
    available = input_limit(model) - fixed
    original_tokens = count_tokens(input_text, model)
    if original_tokens <= available:
        return input_text, budget_entry(model, fixed + original_tokens, trajectories_fitted=0)
    try:
        body = json.loads(input_text)
    except ValueError:
        raise ContextBudgetError("Fixed stage material exceeds model input budget") from None
    fitted_body = copy.deepcopy(body)
    trajectories = []
    references: set[str] = set()

    def visit(value):
        if isinstance(value, dict):
            if {"patient_id", "environment", "text", "sources"} <= value.keys():
                original = copy.deepcopy(value)
                # Workflow 的 sources 仅带定位编号；裁剪过程中不添加事实。
                parsed = {**original, "sources": [{"statement": "", **source}
                                                  for source in original["sources"]]}
                trajectories.append((value, original, PatientTrajectory.model_validate(parsed)))
                value.update(text="", sources=[], input_truncated=True)
                return
            if isinstance(value.get("source_id"), str):
                references.add(value["source_id"])
            references.update(item for item in value.get("source_ids", []) if isinstance(item, str))
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(fitted_body)
    available -= count_tokens(dumps(fitted_body), model)
    if not trajectories or available <= 0:
        raise ContextBudgetError("Fixed stage material exceeds model input budget")
    costs = [count_tokens(dumps(original), model) for _, original, _ in trajectories]
    allocations = costs.copy()
    remaining = available
    pending = set(range(len(costs)))
    # 短患者完整保留，大患者均分剩余空间，避免反馈材料被一个患者占满。
    while pending:
        share = remaining // len(pending)
        small = {index for index in pending if costs[index] <= share}
        if not small:
            for index in pending:
                allocations[index] = share
            break
        for index in small:
            remaining -= costs[index]
        pending -= small

    for _ in range(8):
        for index, (target, original, trajectory) in enumerate(trajectories):
            source_map = {source["source_id"]: source for source in original["sources"]}

            def material(value):
                return {**original, "text": value.text,
                        "sources": [source_map[source.source_id] for source in value.sources],
                        "input_truncated": value.input_truncated}

            try:
                fitted = fit_trajectory(trajectory, allocations[index],
                    measure=lambda value: count_tokens(dumps(material(value)), model),
                    protected_source_ids=references & source_map.keys())
            except ValueError as exc:
                raise ContextBudgetError(str(exc)) from exc
            target.update(material(fitted))
        candidate = dumps(fitted_body)
        tokens = fixed + count_tokens(candidate, model)
        if tokens <= input_limit(model):
            return candidate, budget_entry(model, tokens, trajectories_fitted=sum(
                target != original for target, original, _ in trajectories))
        allocations = [int(value * 0.9) for value in allocations]
    raise ContextBudgetError("Patient records and required stage material exceed model input budget")


def fit_history(items: list[dict], model: ModelConfig, *, fixed_tokens: int,
                prefix_count: int, group_start) -> tuple[list[dict], int]:
    """保留固定输入与近期完整交互，不拆开工具调用和结果。"""
    limit = input_limit(model) - fixed_tokens
    if count_tokens(dumps(items), model) <= limit:
        return items, 0
    prefix = items[:prefix_count]
    groups: list[list[dict]] = []
    for item in items[prefix_count:]:
        if group_start(item) or not groups:
            groups.append([])
        groups[-1].append(item)
    dropped = 0
    while len(groups) > 1:
        dropped += len(groups.pop(0))
        candidate = prefix + [{"role": "user", "content": HISTORY_NOTE}] + [
            item for group in groups for item in group]
        if count_tokens(dumps(candidate), model) <= limit:
            return candidate, dropped
    raise ContextBudgetError("Current task and latest complete interaction exceed model input budget")
