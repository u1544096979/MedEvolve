"""Shared data contracts for MedEvolve modules."""

from typing import Literal

from pydantic import BaseModel, Field, JsonValue, model_validator


JsonObject = dict[str, JsonValue]
TaskType = Literal[
    "诊断与解释", "检查与风险分层", "治疗规划", "药物处方", "治疗重规划"
]


class EnvironmentConfig(BaseModel):
    name: str
    manager_url: str
    seed_data: str


class EnvironmentInstance(BaseModel):
    instance_id: str
    endpoint: str
    environment: str


class Evidence(BaseModel):
    source_id: str
    statement: str


class PatientTrajectory(BaseModel):
    patient_id: str
    environment: str
    text: str
    sources: list[Evidence]


class ExplorationGuidance(BaseModel):
    data_guidance: str
    environment_guidance: str


class GenerationDirection(BaseModel):
    task_type: TaskType
    difficulty: str
    weight: float = Field(ge=0, allow_inf_nan=False)


class Strategy(BaseModel):
    generation_guidance: str
    allocation: list[GenerationDirection]

    @model_validator(mode="after")
    def require_positive_weight(self) -> "Strategy":
        if not any(direction.weight > 0 for direction in self.allocation):
            raise ValueError("allocation must contain at least one positive weight")
        return self


class GeneratedQuestion(BaseModel):
    instruction: str
    task_type: str
    reference_answer: str
    evidence: list[Evidence]
    decision_point: str | None = None


class TaskBatch(BaseModel):
    questions: list[GeneratedQuestion]


class Task(GeneratedQuestion):
    id: str
    patient_id: str
    environment: str
    round_index: int
    source_method: str


class ModelConfig(BaseModel):
    model: str
    base_url: str | None = None
    api_key_env: str
    instructions_role: Literal["system", "user"] = "system"
    parallel_tool_calls: bool = True
    # 提供方声明的模型上限；只用于记录运行预算，未设置时不改动请求参数。
    context_window: int | None = Field(default=None, gt=0)
    max_output_tokens: int | None = Field(default=None, gt=0)


class Budget(BaseModel):
    max_model_turns: int = Field(ge=0)
    max_tool_calls: int = Field(ge=0)


class RunResult(BaseModel):
    output: JsonObject | None = None
    trace: list[JsonObject]
    usage: JsonObject
    status: Literal["completed", "budget_exhausted", "failed"]
    error: str | None = None


class Verification(BaseModel):
    valid: bool
    reason: str


class QualityReview(BaseModel):
    dimensions: JsonObject


class CompletionReview(BaseModel):
    completed: bool
    reason: str


class EvaluationResult(BaseModel):
    task_id: str
    verification: Verification | None = None
    quality: QualityReview | None = None
    completion: CompletionReview | None = None
    solver_run: RunResult | None = None
    observed_state: JsonObject
    status: Literal["completed", "invalid", "failed"]
    failure_category: Literal[
        "clinical", "insufficient_evidence", "ambiguous_task", "tool_error",
        "environment_error", "model_error"
    ] | None = None
    runs: list[RunResult]


class Feedback(BaseModel):
    quality_findings: list[str]
    difficulty_findings: list[str]
    failure_findings: list[str]
    suggestions: list[str]


class SolverInput(BaseModel):
    instruction: str
    patient_id: str
    decision_point: str | None = None


class SolverAnswer(BaseModel):
    answer: str


class DataExploration(BaseModel):
    guidance: str


class EnvironmentExploration(BaseModel):
    guidance: str


class RuntimeConfig(BaseModel):
    models: dict[str, ModelConfig]
    budgets: dict[str, Budget]


class EvaluationConfig(BaseModel):
    runtime: RuntimeConfig


class ExperimentConfig(BaseModel):
    run_id: str
    method: str
    environment: EnvironmentConfig
    runtime: RuntimeConfig
    evaluation: EvaluationConfig
    evolution_patients: list[str]
    evaluation_patients: list[str]
    round_patients: list[list[str]]
    initial_strategy: Strategy
    random_seed: int
    output_dir: str
