"""固定种子调度与最终选题：实验种子经稳定摘要派生互不影响的随机流。

作者源码（提交 1e405e0）提供三个互不调用的生成入口，没有分支调度器，也没有最终
两题选题规则；这里的调度与选题是公共对比实验的接入约定，不是作者行为。
"""

import hashlib
import json
import random
from collections import Counter

__all__ = ["BRANCH_ORDER_PURPOSE", "DEFAULT_SEED", "OUTPUT_SELECTION_PURPOSE",
           "derive_seed", "select_candidates"]

# 公共四参数入口未绑定实验种子时使用的固定调度种子，使该入口自身可复现。
DEFAULT_SEED = 0
# 分支顺序与输出选择使用不同 purpose，得到两条互不影响的随机流。
BRANCH_ORDER_PURPOSE = "branch-order"
OUTPUT_SELECTION_PURPOSE = "output-selection"


def derive_seed(experiment_seed: int, environment: str, patient_id: str, purpose: str) -> int:
    """实验种子、环境名、患者标识与用途的稳定摘要；不使用内建 hash()。"""
    payload = json.dumps([experiment_seed, environment, patient_id, purpose],
                         ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest(), "big")


def select_candidates(candidates: list, directions: list, source_ids, seed: int,
                      records: list | None = None) -> list:
    """洗牌固定顺序候选池后按公共字段、患者来源与临床题型配额接纳；最多两题。

    不读生成来源标签、内置评分或公共评测结果。``records`` 传入时逐条记录接纳与拒绝
    的真实原因，顺序与下面的接纳判据一致。
    """
    shuffled = list(candidates)
    random.Random(seed).shuffle(shuffled)
    remaining = Counter(direction.task_type for direction in directions)
    selected = []
    for candidate in shuffled:
        q = candidate.question
        if not q.instruction.strip() or not q.reference_answer.strip():
            reason = "empty_instruction_or_answer"
        elif q.decision_point is not None or not q.evidence:
            reason = "decision_point_or_missing_evidence"
        elif any(e.source_id not in source_ids or not e.statement.strip() for e in q.evidence):
            reason = "evidence_not_retrievable"
        elif remaining[q.task_type] <= 0:
            reason = "task_type_quota_exhausted"
        else:
            selected.append(candidate)
            remaining[q.task_type] -= 1
            if records is not None:
                records.append(_selection_record(candidate, accepted=True))
            if len(selected) == 2:
                break
            continue
        if records is not None:
            records.append(_selection_record(candidate, accepted=False, reason=reason))
    return selected


def _selection_record(candidate, *, accepted: bool, reason: str | None = None) -> dict:
    return {"accepted": accepted, "reason": reason, "origin": candidate.origin,
            "task_type": candidate.question.task_type,
            "instruction": candidate.question.instruction, "locator": candidate.locator}
