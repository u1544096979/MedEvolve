"""生成阶段的环境协议分派。

PhysicianBench 暴露 SDK 函数工具，生成阶段绑定当前患者的只读子集；
MedAgentBench 暴露原生 GET/POST/FINISH 文本动作工具对象，整份交给运行时执行器。
"""

from environments.errors import EnvironmentUnavailable

__all__ = [
    "NATIVE_FINISH_ANSWERS", "NATIVE_FINISH_OBJECT", "NATIVE_RULES",
    "READ_ONLY_TOOLS", "bind_generation_tools", "is_native", "read_only_tools",
]

# 生成阶段的检索与核验只使用只读工具；写工具留给评测阶段的解题 Agent。
READ_ONLY_TOOLS = (
    "fhir_condition_search_problems",
    "fhir_observation_search_labs",
    "fhir_observation_search_vitals",
    "fhir_patient_search_demographics",
    "fhir_procedure_search_orders",
    "fhir_medication_request_search_orders",
    "fhir_document_reference_search_clinical_notes",
    "fhir_service_request_search",
    "fhir_observation_search_social_history",
)

# 原生协议的生成阶段约束：只检索、不写入，患者范围由环境限定。
NATIVE_RULES = (
    "本次只做检索：只发送 GET 动作，不要发送 POST；"
    "不要手工填写 patient 或 subject 参数，环境已限定当前患者；"
    "用 _count 控制返回条数，不要超过 20；"
    "检索完成后按原生协议用 FINISH 返回结果。"
)

NATIVE_FINISH_ANSWERS = (
    "最后用 FINISH([\"答案1\", \"答案2\"]) 返回答案数组，答案用字符串，不要输出其他内容。"
)

NATIVE_FINISH_OBJECT = (
    "最后用 FINISH 返回只含一个对象的列表，例如 "
    "FINISH([{\"identifier\": \"...\", \"relation\": \"...\", \"source_ids\": [\"ResourceType/id\"]}])，"
    "不要输出其他内容。"
)


def is_native(tools) -> bool:
    from environments.medagentbench import NativeTools

    return isinstance(tools, NativeTools)


def read_only_tools(tools: list) -> list:
    selected = [tool for tool in tools if tool.name in READ_ONLY_TOOLS]
    missing = sorted(set(READ_ONLY_TOOLS) - {tool.name for tool in selected})
    if missing:
        raise EnvironmentUnavailable(
            f"TaskCraft requires the read-only medical tools, missing: {missing}"
        )
    return selected


def bind_generation_tools(tools):
    """原生协议的工具对象原样透传，函数工具环境只绑定只读子集。"""
    if is_native(tools):
        return tools
    return read_only_tools(tools)
