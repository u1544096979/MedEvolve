import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from contracts import ExperimentConfig, GeneratedQuestion, Verification
from environments.capabilities import capability_instructions, get_capabilities
from experiments.runner import DEFAULT_RUN_CONCURRENCY, resolve_concurrency


ENVIRONMENTS = ("PhysicianBench", "MedAgentBench", "FHIR-AgentBench")


@pytest.mark.parametrize("name", ENVIRONMENTS)
def test_capability_instructions_describe_shared_read_only_scope(name):
    capability = get_capabilities(name)
    instructions = capability_instructions(name)

    assert capability["experiment_read_only"] is True
    assert "只允许基于既有患者记录生成查询任务" in instructions
    assert "禁止任何持久化写入" in instructions
    assert "禁止模拟提交" in instructions
    assert "可以基于既有记录形成临床建议" in instructions
    assert capability["operation_semantics"]["reads"]
    assert capability["operation_semantics"]["writes"]


def test_capabilities_are_fixed_and_unknown_names_are_rejected():
    capabilities = get_capabilities("PhysicianBench")
    capabilities["operation_semantics"]["writes"] = "changed"
    assert get_capabilities("PhysicianBench")["operation_semantics"]["writes"] != "changed"
    with pytest.raises(ValueError, match="Unsupported environment"):
        get_capabilities("unknown")


def test_contracts_keep_legacy_task_compatible_and_add_optional_scope_fields():
    question = GeneratedQuestion(
        instruction="查看已有记录并解释诊断。", task_type="诊断与解释",
        reference_answer="基于检验结果考虑诊断。", evidence=[],
    )
    assert question.execution_requirement is None
    assert GeneratedQuestion(**{**question.model_dump(), "execution_requirement": "read_only"})
    with pytest.raises(ValidationError):
        GeneratedQuestion(**{**question.model_dump(), "execution_requirement": "write"})
    assert Verification(valid=True, reason="依据记录判断").read_only_task is None
    assert Verification(valid=True, reason="仅核对事实", read_only_task=True).read_only_task is True


def test_experiment_config_allows_older_config_without_concurrency_fields():
    config = json.loads(Path("configs/experiment.json").read_text(encoding="utf-8"))
    assert config["generation_concurrency"] == 6
    assert config["evaluation_concurrency"] == 6
    config.pop("generation_concurrency")
    config.pop("evaluation_concurrency")
    parsed = ExperimentConfig.model_validate(config)
    assert parsed.generation_concurrency is None
    assert parsed.evaluation_concurrency is None


@pytest.mark.parametrize("value", [0, -1])
@pytest.mark.parametrize("field", ["generation_concurrency", "evaluation_concurrency"])
def test_experiment_config_rejects_nonpositive_concurrency(field, value):
    config = json.loads(Path("configs/experiment.json").read_text(encoding="utf-8"))
    config[field] = value
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(config)


def test_concurrency_values_resolve_independently_with_precedence():
    assert resolve_concurrency(5, "2") == 5
    assert resolve_concurrency(None, "2") == 2
    assert resolve_concurrency(None, "invalid") == DEFAULT_RUN_CONCURRENCY
    assert resolve_concurrency(None, "0") == DEFAULT_RUN_CONCURRENCY
    assert resolve_concurrency(None, "-3") == DEFAULT_RUN_CONCURRENCY
