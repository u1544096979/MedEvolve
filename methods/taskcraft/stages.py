"""TaskCraft 原子生成、深度扩展、宽度扩展的医疗适配。

源码路径（提交 1e405e0）：
- gen_atomic_task.py: get_candidate_tasks / generate_initial_questions /
  filter_atomic_tasks（原子生成与内置过滤）
- gen_depth_based_task.py: DepthExtend.backward / verify / process_single_task（深度扩展）
- gen_width_based_task.py: width_extend / check_queries（宽度扩展与检查）

适配差异集中记录在本模块与 prompts.py：数据源由文档改为患者轨迹，
检索工具由网络改为当前患者的病历工具，阶段依赖、判断分支与提示词语义保持不变。
"""

import json
import re
from dataclasses import dataclass, field

from contracts import Evidence, GeneratedQuestion, PatientTrajectory
from pydantic import BaseModel

from . import prompts
from .ledger import Ledger
from .schemas import (
    AnswerList, BackwardContext, CleanedQuestion, ConclusionList, DecompositionReport,
    InitialQuestion, KnowledgeAnswers, MergedQuestions, NewQuery, QueryJudgement,
    QueryMerge, SupersetCheck,
)
from .verify import score_recall, verify_query

__all__ = ["Candidate", "Pipeline", "TaskCraftParameters", "split_text"]

# prepare_trajectory 生成的记录块标签：“[记录时间] 资源类型/资源ID”。
_BLOCK = re.compile(r"^\[(?P<time>[^\]]*)\] (?P<source_id>\S+)$", re.MULTILINE)
# data_processor.split_text_intelligently 的分隔符优先级。
_SEPARATORS = ("\n\n", "\n", ".", " ", "")


class TaskCraftParameters(BaseModel):
    """源码默认参数；按 common/SPEC.md 显式配置源码循环与内置核验的规模。"""

    chunk_size: int = 16384               # gen_atomic_tasks(chunk_size=16384)
    max_candidate_conclusions: int = 20   # gen_atomic_tasks(max_candiated_conclusions=20)
    max_candidate_atomic: int = 10        # gen_atomic_tasks(max_candidate_atomic=10)
    # 源码循环与内置核验的预算边界：原子阶段最多使用该比例的生成预算。
    atomic_budget_share: float = 0.5
    # 剩余模型轮数低于该值时不启动新的扩展跳，保证阶段在有界预算内结束。
    extension_guard_turns: int = 25
    judge_samples: int = 3                # VerifyAgent.recall_score(num_parallel_predictions=3)
    question_attempts: int = 3            # gen_atomic_task 各检索/生成步骤的重试次数
    query_attempts: int = 3               # DepthExtend.backward: for _ in range(3)
    merge_attempts: int = 5               # process_single_task: for _ in range(5)
    extended_attempts: int = 4            # depth_extend(extended_attempts=4)
    max_hops: int = 2                     # depth_extend(max_hops=2)
    width_attempts: int = 3               # gen_width_based_task: MAX_RETRIES = 3


@dataclass
class Candidate:
    """候选任务及其来源记录；parents 用于核对扩展是否依赖父任务。"""

    question: GeneratedQuestion
    origin: str
    parents: list[str] = field(default_factory=list)
    locator: str = ""
    detail: dict = field(default_factory=dict)


def split_text(text: str, chunk_size: int) -> list[str]:
    """移植 split_text_intelligently：按分隔符优先级切分，不引入重叠。"""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    return _split(text, _SEPARATORS, chunk_size)


def _split(text: str, separators: tuple[str, ...], chunk_size: int) -> list[str]:
    if not text:
        return []
    if len(text) <= chunk_size or not separators:
        return [text] if len(text) <= chunk_size else _hard_split(text, chunk_size)
    separator, rest = separators[0], separators[1:]
    if separator == "":
        return _hard_split(text, chunk_size)
    chunks, buffer = [], ""
    for part in text.split(separator):
        if not part:
            continue
        joined = f"{buffer}{separator}{part}" if buffer else part
        if len(joined) <= chunk_size:
            buffer = joined
            continue
        if buffer:
            chunks.append(buffer)
            buffer = ""
        if len(part) <= chunk_size:
            buffer = part
        else:
            chunks.extend(_split(part, rest, chunk_size))
    if buffer:
        chunks.append(buffer)
    return chunks


def _hard_split(text: str, chunk_size: int) -> list[str]:
    return [text[index:index + chunk_size] for index in range(0, len(text), chunk_size)]


def _status(run) -> str:
    return run.status if run is not None else "budget_exhausted"


def _error(run) -> str | None:
    if run is None:
        return "cumulative model budget exhausted"
    return run.error or (None if run.status == "completed" else run.status)


class Pipeline:
    """单患者一次生成的阶段编排；阶段顺序为原子、宽度、深度。"""

    def __init__(self, trajectory: PatientTrajectory, tools: list,
                 directions: list, parameters: TaskCraftParameters, ledger: Ledger):
        self.trajectory = trajectory
        self.tools = tools
        self.directions = directions
        self.parameters = parameters
        self.ledger = ledger
        self.allowed_types = list(dict.fromkeys(direction.task_type for direction in directions))
        self.records: list[dict] = []
        self.pool: list[Candidate] = []
        self.failure: str | None = None
        self.source_ids = [source.source_id for source in trajectory.sources]
        self.blocks = {source_id: time for time, source_id in _BLOCK.findall(trajectory.text)}

    def record(self, stage: str, **fields) -> dict:
        # 阶段结构记录挂在 taskcraft/pipeline 条目下，与单次调用记录分层保存。
        entry = {"stage": stage, **fields}
        self.records.append(entry)
        return entry

    def agent_instructions(self) -> str:
        """病历工具只开放当前患者；正文里的资源引用中可能出现其他标识。"""
        return (
            f"{prompts.MEDICAL_RECORD_AGENT}\n"
            f"本次开放的患者标识是 {self.trajectory.patient_id}；"
            "所有病历工具的 patient 或 subject 参数都必须使用该标识，"
            "不要使用正文中出现的其他患者号或资源引用。"
        )

    def locator(self, source_ids: list[str]) -> str:
        labels = []
        for source_id in source_ids:
            time = self.blocks.get(source_id)
            labels.append(f"[{time}] {source_id}" if time else source_id)
        return "；".join(labels) if labels else self.trajectory.patient_id

    def evidence(self, statement: str, source_ids: list[str]) -> tuple[list[Evidence], list[str]]:
        accepted, evidence = [], []
        for source_id in source_ids:
            if source_id in self.source_ids and source_id not in accepted:
                accepted.append(source_id)
                evidence.append(Evidence(source_id=source_id, statement=statement))
        return evidence, accepted

    async def run(self) -> None:
        atomic = await self.atomic()
        width = await self.width(atomic)
        depth = await self.depth(atomic)
        self.pool = [*width, *depth, *atomic]

    # ------------------------------------------------------------------ 原子生成
    async def atomic(self) -> list[Candidate]:
        chunks = split_text(self.trajectory.text, self.parameters.chunk_size)
        self.record("atomic/input", chunks=len(chunks), characters=len(self.trajectory.text))
        conclusions = []
        for index, chunk in enumerate(chunks):
            run = await self.ledger.llm(
                "atomic/conclusions", instructions=prompts.conclusion_extraction_prompt(),
                input_text=f"The document content to be processed is as follows: {chunk}",
                output_type=ConclusionList,
            )
            if run is None or run.status != "completed":
                self.record("atomic/conclusions", chunk=index, status=_status(run), error=_error(run))
                if self.ledger.exhausted:
                    break
                continue
            items = ConclusionList.model_validate(run.output).conclusions
            self.record("atomic/conclusions", chunk=index, status="completed", conclusions=len(items))
            conclusions.extend(items)
            if 0 < self.parameters.max_candidate_conclusions <= len(conclusions):
                break
        if self.parameters.max_candidate_conclusions > 0:
            conclusions = conclusions[:self.parameters.max_candidate_conclusions]
        if not conclusions:
            self.failure = self.failure or "原子阶段没有抽取到候选结论"

        share_limit = int(self.ledger.limit_turns() * self.parameters.atomic_budget_share)
        generated = []
        for conclusion in conclusions:
            if self.ledger.exhausted or self.ledger.used_turns() >= share_limit:
                self.record("atomic/question", status="skipped", reason="原子阶段预算份额已用尽",
                            share_limit=share_limit, used=self.ledger.used_turns())
                break
            question = await self._initial_question(conclusion)
            if question is None:
                continue
            generated.append(question)
            if 0 < self.parameters.max_candidate_atomic <= len(generated):
                break
        if not generated:
            self.failure = self.failure or "原子阶段没有生成候选问题"

        accepted = []
        for item in generated:
            if self.ledger.exhausted or self.ledger.used_turns() >= share_limit:
                self.record("atomic/verify", question=item["question"], status="skipped",
                            reason="原子阶段预算份额已用尽", share_limit=share_limit,
                            used=self.ledger.used_turns())
                continue
            outcome = await verify_query(
                self.ledger, item["question"], item["answer"], self.tools,
                samples=self.parameters.judge_samples, stage="atomic/verify",
                instructions=self.agent_instructions(),
            )
            self.record("atomic/verify", question=item["question"], **outcome.record())
            if outcome.error is not None:
                continue
            evidence, source_ids = self.evidence(item["conclusion"], item["source_ids"])
            if not evidence:
                self.record("atomic/verify", question=item["question"], rejected="evidence",
                            reason="结论没有属于当前患者的记录块标识", reported=item["source_ids"])
                continue
            accepted.append(Candidate(
                question=GeneratedQuestion(
                    instruction=item["question"], task_type=item["task_type"],
                    reference_answer=item["answer"], evidence=evidence,
                ),
                origin="atomic", locator=self.locator(source_ids), detail={"source_ids": source_ids},
            ))
        if not accepted:
            self.failure = self.failure or "原子阶段没有通过内置核验的候选"
        self.record("atomic/result", candidates=len(conclusions), generated=len(generated),
                    accepted=len(accepted))
        return accepted

    async def _initial_question(self, conclusion) -> dict | None:
        locator = self.locator(conclusion.source_ids)
        payload = (f"\nData to be Processed:\nID: {locator}\n"
                   f"R: {conclusion.R}\nA: {conclusion.conclusion}\n")
        run = await self.ledger.llm("atomic/question",
                                   instructions=prompts.question_prompt(self.allowed_types),
                                   input_text=payload, output_type=InitialQuestion)
        if run is None or run.status != "completed":
            self.record("atomic/question", status=_status(run), error=_error(run), conclusion=conclusion.conclusion)
            return None
        parsed = InitialQuestion.model_validate(run.output)
        if parsed.task_type not in self.allowed_types:
            self.record("atomic/question", status="rejected", reason="题型超出请求方向",
                        task_type=parsed.task_type, allowed=self.allowed_types)
            return None
        clean_input = json.dumps({"question": parsed.Q, "original_answer": conclusion.conclusion},
                                 ensure_ascii=False)
        clean = await self.ledger.llm("atomic/clean", instructions=prompts.clean_prompt(),
                                     input_text=f"The data need to be processed is as follows: {clean_input}",
                                     output_type=CleanedQuestion)
        if clean is None or clean.status != "completed":
            self.record("atomic/clean", status=_status(clean), error=_error(clean), question=parsed.Q)
            return None
        cleaned = CleanedQuestion.model_validate(clean.output)
        if not cleaned.question.strip() or not cleaned.refined_answer.strip():
            self.record("atomic/clean", status="rejected", reason="清洗后的问题或答案为空",
                        question=parsed.Q)
            return None
        # 源码 generate_initial_question 返回 clean_qa 的结果：问题用清洗后的文本，答案用 refined_answer。
        return {"question": cleaned.question, "task_type": parsed.task_type,
                "answer": cleaned.refined_answer,
                "conclusion": conclusion.conclusion, "source_ids": list(conclusion.source_ids),
                "locator": locator}

    # ------------------------------------------------------------------ 宽度扩展
    async def width(self, atomic: list[Candidate]) -> list[Candidate]:
        if len(atomic) < 2:
            self.record("width/skip", reason="合并至少需要两道原子题", available=len(atomic))
            return []
        source = [{"index": index, "question": item.question.instruction,
                   "content_identifier": item.locator,
                   "golden_answer": item.question.reference_answer}
                  for index, item in enumerate(atomic)]
        prompt = (f"\n    Here are the base questions to process:\n    "
                  f"{json.dumps(source, ensure_ascii=False)}\n"
                  "    Each dictionary contains: index (unique ID), question (original question), "
                  "and content_identifier (identifier).\n    ")
        merged = None
        for _ in range(self.parameters.width_attempts):
            run = await self.ledger.llm("width/merge", instructions=prompts.width_prompts()["merge_prompt"],
                                       input_text=prompt, output_type=MergedQuestions)
            if run is None or run.status != "completed":
                self.record("width/merge", status=_status(run), error=_error(run))
                if self.ledger.exhausted:
                    return []
                continue
            merged = MergedQuestions.model_validate(run.output).merged
            break
        if merged is None:
            self.failure = self.failure or "宽度扩展没有生成合并问题"
            return []

        merged_results = []
        for item in merged:
            indexes = sorted({index for index in item.index if 0 <= index < len(atomic)})
            if len(indexes) < 2 or not item.question.strip():
                self.record("width/merge", status="rejected", reason="合并分组不合法",
                            index=item.index, question=item.question)
                continue
            parents = [atomic[index] for index in indexes]
            merged_results.append({
                "index": len(merged_results),
                "question": item.question,
                "content_identifier": item.content_identifier,
                "original_questions": [parent.question.instruction for parent in parents],
                "golden_answers": [parent.question.reference_answer for parent in parents],
                "parents": parents,
            })
        self.record("width/merge", status="completed", merged=len(merged_results))
        if not merged_results:
            self.failure = self.failure or "宽度扩展没有合法合并分组"
            return []

        checked = await self._check_merged(merged_results)
        results = []
        for item in checked:
            parents = item["parents"]
            task_type = self._merged_task_type(parents)
            results.append(Candidate(
                question=GeneratedQuestion(
                    instruction=item["question"], task_type=task_type,
                    reference_answer=self._merged_answer(item), evidence=self._merged_evidence(parents),
                ),
                origin="width", parents=[parent.question.instruction for parent in parents],
                detail={"group": [parent.locator for parent in parents], "task_type": task_type},
            ))
        self.record("width/result", checked=len(results))
        return results

    async def _check_merged(self, merged_results: list[dict]) -> list[dict]:
        """移植 check_queries：先查可分解性，再查无工具回答是否已能答对。"""
        step1_input = [{"index": item["index"], "complex_question": item["question"],
                        "original_questions": item["original_questions"]} for item in merged_results]
        prompt1 = ("\n    Here are the base questions to process:\n    "
                   f"{json.dumps(step1_input, ensure_ascii=False, indent=2)}\n"
                   "    Each dictionary contains: index (unique ID), complex_question (original complex question), \n"
                   "    and original_questions (list of original questions).\n    ")
        validated = []
        report = None
        for _ in range(self.parameters.width_attempts):
            run = await self.ledger.llm("width/decompose",
                                       instructions=prompts.width_prompts()["check_prompt_1"],
                                       input_text=prompt1, output_type=DecompositionReport)
            if run is None or run.status != "completed":
                self.record("width/decompose", status=_status(run), error=_error(run))
                if self.ledger.exhausted:
                    return []
                continue
            report = DecompositionReport.model_validate(run.output).results
            break
        if report is None:
            return []
        states = {item.index: item.state for item in report}
        for item in merged_results:
            if states.get(item["index"], 0) == 1:
                validated.append(item)
            else:
                self.record("width/decompose", status="rejected", index=item["index"],
                            question=item["question"])
        self.record("width/decompose", status="completed", kept=len(validated))
        if not validated:
            return []

        step2_input = [{"index": item["index"], "complex_question": item["question"]}
                       for item in validated]
        prompt2 = ("\n    Please answer these research questions:\n    "
                   f"{json.dumps(step2_input, ensure_ascii=False, indent=2)}\n    ")
        answers = None
        for _ in range(self.parameters.width_attempts):
            run = await self.ledger.llm("width/answer",
                                       instructions=prompts.width_prompts()["check_prompt_2"],
                                       input_text=prompt2, output_type=KnowledgeAnswers)
            if run is None or run.status != "completed":
                self.record("width/answer", status=_status(run), error=_error(run))
                if self.ledger.exhausted:
                    return []
                continue
            answers = KnowledgeAnswers.model_validate(run.output).answers
            break
        if answers is None:
            return []

        index = {item["index"]: item for item in validated}
        kept = []
        for answer in answers:
            item = index.get(answer.index)
            if item is None:
                continue
            if self.ledger.exhausted:
                break
            score = await score_recall(self.ledger, item["golden_answers"], answer.llm_answer,
                                      samples=self.parameters.judge_samples, stage="width/answer_score")
            self.record("width/answer_score", index=answer.index, llm_answer=answer.llm_answer,
                        llm_score=score)
            if score is None:
                continue
            # gen_width_based_task.py: 无工具回答已能答对（similarity 达满分）的问题不保留。
            if score < 1:
                kept.append(item)
        self.record("width/answer", status="completed", kept=len(kept))
        return kept

    def _merged_task_type(self, parents: list[Candidate]) -> str:
        types = [parent.question.task_type for parent in parents]
        if len(set(types)) == 1:
            return types[0]
        # 跨题型合并时沿用第一道父题的题型，保证输出题型位于请求方向内。
        self.record("width/task_type", status="inherited", parent_types=types, chosen=types[0])
        return types[0]

    @staticmethod
    def _merged_answer(item: dict) -> str:
        lines = ["合并题参考答案："]
        for index, (question, answer) in enumerate(
                zip(item["original_questions"], item["golden_answers"]), start=1):
            lines.append(f"{index}. {question} → {answer}")
        return "\n".join(lines)

    @staticmethod
    def _merged_evidence(parents: list[Candidate]) -> list[Evidence]:
        evidence, seen = [], set()
        for parent in parents:
            for record in parent.question.evidence:
                if record.source_id not in seen:
                    seen.add(record.source_id)
                    evidence.append(record)
        return evidence

    # ------------------------------------------------------------------ 深度扩展
    async def depth(self, atomic: list[Candidate]) -> list[Candidate]:
        results = []
        for parent in atomic:
            if self.ledger.exhausted:
                break
            if self.ledger.remaining_turns() < self.parameters.extension_guard_turns:
                self.record("depth/extension", status="skipped", reason="剩余预算不足以完成一跳",
                            remaining=self.ledger.remaining_turns(),
                            guard=self.parameters.extension_guard_turns)
                break
            extended = await self._extend_depth(parent)
            if extended is not None:
                results.append(extended)
        self.record("depth/result", extended=len(results))
        return results

    async def _extend_depth(self, parent: Candidate) -> Candidate | None:
        """移植 process_single_task：反向找 superset、核验、合并，逐跳变长。"""
        prompts_depth = prompts.depth_prompts()
        initial = {"initial_query": parent.question.instruction,
                   "initial_answer": parent.question.reference_answer,
                   "initial_identifier": parent.locator}
        chain = []
        full_query = initial["initial_query"]
        last_identifier = parent.locator
        valid_hop = 1
        superset_sources: list[str] = []
        for attempt in range(self.parameters.extended_attempts):
            if self.ledger.exhausted:
                break
            if self.ledger.remaining_turns() < self.parameters.extension_guard_turns:
                self.record("depth/extension", status="skipped", attempt=attempt,
                            reason="剩余预算不足以完成一跳", remaining=self.ledger.remaining_turns())
                break
            backward = await self._backward(prompts_depth, last_identifier)
            if backward is None:
                continue
            now_query = backward["query"]
            verify = await verify_query(self.ledger, now_query, last_identifier, self.tools,
                                        samples=self.parameters.judge_samples, stage="depth/verify",
                                        instructions=self.agent_instructions())
            self.record("depth/verify", attempt=attempt, question=now_query, **verify.record())
            if verify.error is not None or not verify.agent_score:
                continue
            merged_query = await self._merge(prompts_depth, full_query, now_query, last_identifier,
                                             parent.question.reference_answer)
            if merged_query is None:
                continue
            valid_hop += 1
            chain.append({"query": now_query, "answer": last_identifier,
                          "agent_score": verify.agent_score, "llm_score": verify.llm_score})
            superset_sources.extend(backward["source_ids"])
            last_identifier = backward["identifier"]
            full_query = merged_query
            if valid_hop >= self.parameters.max_hops:
                break
        if not chain:
            self.record("depth/extension", status="failed", question=parent.question.instruction,
                        reason="未能在尝试次数内完成扩展")
            return None
        evidence, source_ids = self.evidence(self._chain_statement(initial, chain), superset_sources)
        evidence = parent.question.evidence + [
            record for record in evidence
            if record.source_id not in {item.source_id for item in parent.question.evidence}
        ]
        self.record("depth/extension", status="completed", question=full_query, hops=len(chain),
                    chain=chain)
        return Candidate(
            question=GeneratedQuestion(
                instruction=full_query, task_type=parent.question.task_type,
                reference_answer=self._chain_answer(initial, chain), evidence=evidence,
            ),
            origin="depth", parents=[parent.question.instruction],
            locator=last_identifier,
            detail={"chain": chain, "superset_source_ids": source_ids},
        )

    async def _backward(self, prompts_depth: dict, element: str) -> dict | None:
        """DepthExtend.backward：找 superset、校验关系、生成新问题。"""
        payload = prompts_depth["backward_task_prompt"].replace("{element}", element)
        run = await self.ledger.agent("depth/backward", instructions=self.agent_instructions(),
                                     input_text=payload, tools=self.tools, output_type=BackwardContext)
        if run is None or run.status != "completed":
            self.record("depth/backward", status=_status(run), error=_error(run), element=element)
            return None
        context = BackwardContext.model_validate(run.output)

        check = await self.ledger.llm(
            "depth/superset",
            instructions=prompts_depth["is_superset_valid"],
            input_text=(f"\n                Given superset: {context.identifier}\n\n"
                        f"                Given relationship: {context.relation}\n\n"
                        f"                Given subset: {element}\n                "),
            output_type=SupersetCheck,
        )
        if check is None or check.status != "completed":
            self.record("depth/superset", status=_status(check), error=_error(check))
            return None
        state = SupersetCheck.model_validate(check.output).new_query.strip().lower()
        if state == "invalid":
            self.record("depth/superset", status="rejected", element=element,
                        identifier=context.identifier, relation=context.relation)
            return None

        query = None
        for _ in range(self.parameters.query_attempts):
            generated = await self.ledger.llm(
                "depth/query", instructions=prompts_depth["generate_query_prompt"],
                input_text=(f"\n                Certain answer: {element}\n\n"
                            f"                Identifier: {context.identifier}\n\n"
                            f"                Relationship: {context.relation}\n                "),
                output_type=NewQuery,
            )
            if generated is None or generated.status != "completed":
                self.record("depth/query", status=_status(generated), error=_error(generated))
                if self.ledger.exhausted:
                    return None
                continue
            query = NewQuery.model_validate(generated.output).new_query
            break
        if not query:
            self.record("depth/query", status="failed", element=element, reason="未生成新问题")
            return None
        self.record("depth/backward", status="completed", element=element,
                    identifier=context.identifier, relation=context.relation, query=query)
        return {"query": query, "identifier": context.identifier, "relation": context.relation,
                "source_ids": list(context.source_ids)}

    async def _merge(self, prompts_depth: dict, full_query: str, now_query: str,
                     last_identifier: str, golden_answer: str) -> str | None:
        for _ in range(self.parameters.merge_attempts):
            merged = await self.ledger.llm(
                "depth/merge", instructions=prompts_depth["merge_query_prompt"],
                input_text=(f"\n                        Core query: {full_query}\n"
                            f"                        Auxiliary query: {now_query}\n"
                            f"                        Element to be replaced: {last_identifier} \n"
                            "                        "),
                output_type=QueryMerge,
            )
            if merged is None or merged.status != "completed":
                self.record("depth/merge", status=_status(merged), error=_error(merged))
                if self.ledger.exhausted:
                    return None
                continue
            new_query = QueryMerge.model_validate(merged.output).new_query
            compare = prompts_depth["query_compare_prompt"].replace(
                "{last_identifier}", last_identifier).replace(
                "{new_task}", new_query).replace("{full_query}", full_query).replace(
                "{golden_answer}", golden_answer)
            judged = await self.ledger.llm(
                "depth/merge_judge",
                instructions="按用户消息中的要求比较问题复杂度并输出 JSON。",
                input_text=compare, output_type=QueryJudgement,
            )
            if judged is None or judged.status != "completed":
                self.record("depth/merge_judge", status=_status(judged), error=_error(judged))
                if self.ledger.exhausted:
                    return None
                continue
            judgement = QueryJudgement.model_validate(judged.output)
            self.record("depth/merge_judge", status="completed", is_valid=judgement.is_valid,
                        new_query=new_query, analysis=judgement.analysis)
            if judgement.is_valid:
                return new_query
        self.record("depth/merge", status="failed", reason="合并问题未通过语义检查",
                    core_query=full_query, auxiliary_query=now_query)
        return None

    @staticmethod
    def _chain_statement(initial: dict, chain: list[dict]) -> str:
        parts = [f"{item['query']}：{item['answer']}" for item in chain]
        return "；".join([initial["initial_query"], *parts])

    @staticmethod
    def _chain_answer(initial: dict, chain: list[dict]) -> str:
        lines = [f"最终答案：{initial['initial_answer']}", "",
                 "支撑该答案的子问题与中间结果（自外向内）："]
        for index, item in enumerate(reversed(chain), start=1):
            lines.append(f"{index}. 子问题：{item['query']} → 中间答案：{item['answer']}")
        lines.append(f"{len(chain) + 1}. 原始问题：{initial['initial_query']} "
                     f"→ 原始答案：{initial['initial_answer']}")
        return "\n".join(lines)

    # ------------------------------------------------------------------ 输出选择
    def select(self) -> list[Candidate]:
        """按请求方向各取一题；同一方向内优先顺序为宽度、深度、原子。"""
        chosen: list[int] = []
        for direction in self.directions:
            for index, candidate in enumerate(self.pool):
                if index in chosen:
                    continue
                if candidate.question.task_type == direction.task_type:
                    chosen.append(index)
                    break
        return [self.pool[index] for index in chosen]
