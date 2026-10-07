"""Fixed descriptions of environment semantics and the experiment read-only scope."""

from copy import deepcopy


_READ_ONLY_LIMIT = (
    "本轮实验只允许基于既有患者记录生成查询任务。禁止任何持久化写入，也禁止模拟提交、"
    "伪造写入成功或以提交操作作为任务步骤。可以基于既有记录形成临床建议。"
)

_CAPABILITIES = {
    "PhysicianBench": {
        "name": "PhysicianBench",
        "experiment_read_only": True,
        "query_tools": (
            "fhir_condition_search_problems", "fhir_observation_search_labs",
            "fhir_observation_search_vitals", "fhir_patient_search_demographics",
            "fhir_procedure_search_orders", "fhir_medication_request_search_orders",
            "fhir_document_reference_search_clinical_notes", "fhir_service_request_search",
            "fhir_observation_search_social_history",
        ),
        "operation_semantics": {
            "reads": "FHIR 查询读取既有患者记录",
            "writes": "FHIR 创建接口会持久化资源",
        },
        "experiment_limit": _READ_ONLY_LIMIT,
    },
    "MedAgentBench": {
        "name": "MedAgentBench",
        "experiment_read_only": True,
        "query_tools": ("GET", "FINISH"),
        "operation_semantics": {
            "reads": "FHIR 查询读取既有患者记录",
            "writes": "POST 可接受请求，但不会持久化数据库变更",
        },
        "experiment_limit": _READ_ONLY_LIMIT,
    },
    "FHIR-AgentBench": {
        "name": "FHIR-AgentBench",
        "experiment_read_only": True,
        "query_tools": ("all",),
        "operation_semantics": {
            "reads": "通过只读 FHIR 工具检索既有患者记录",
            "writes": "环境不提供写入操作",
        },
        "experiment_limit": _READ_ONLY_LIMIT,
    },
}


def get_capabilities(name: str) -> dict:
    """Return a copy of the fixed capability description for an environment."""
    try:
        return deepcopy(_CAPABILITIES[name])
    except KeyError as exc:
        raise ValueError(f"Unsupported environment: {name}") from exc


def read_only_scope_instructions() -> str:
    """Return the shared task-generation restriction independent of environment facts."""
    return _READ_ONLY_LIMIT


def capability_instructions(name: str) -> str:
    """Return environment facts and the common read-only constraint for prompts."""
    capability = get_capabilities(name)
    semantics = capability["operation_semantics"]
    return (
        f"环境：{name}。实际操作语义：读取：{semantics['reads']}；写入：{semantics['writes']}。"
        f"{read_only_scope_instructions()}"
    )
