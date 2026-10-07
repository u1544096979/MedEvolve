"""生成阶段的环境协议分派。

PhysicianBench 暴露 SDK 函数工具，生成阶段只绑定环境已声明的只读工具，不做名称推断；
MedAgentBench 暴露原生 GET/POST/FINISH 文本动作工具对象，生成阶段只放行 GET 与 FINISH。
两种协议都不修改环境给出的原对象与患者访问范围。
"""

from copy import deepcopy

from contracts import JsonObject
from environments.errors import EnvironmentUnavailable
from environments.medagentbench import NativeTools

__all__ = [
    "NATIVE_FINISH_ANSWERS", "NATIVE_FINISH_OBJECT", "NATIVE_RULES",
    "READ_ONLY_TOOLS", "ReadOnlyNativeTools", "bind_generation_tools", "is_native",
    "read_only_tools",
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
    "get_resources_by_resource_id",
    "get_resources_by_patient_fhir_id",
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
    return isinstance(tools, NativeTools)


class ReadOnlyNativeTools(NativeTools):
    """生成阶段的原生协议视图：保留患者客户端与动作语义，只放行检索与收尾。

    动作交给来源对象执行，患者范围、实例地址与分页行为都由来源对象承担；
    工具声明复制为来源对象的 GET 声明，来源对象本身不被修改。
    """

    def __init__(self, source: NativeTools):
        super().__init__(source.endpoint, source.patient_ids, [])
        self._source = source
        self.functions = [deepcopy(item) for item in source.functions
                          if item.get("name", "").startswith("GET ")]

    async def execute(self, action: str) -> JsonObject:
        normalized = action.replace("```tool_code", "").replace("```", "").strip()
        if not (normalized.startswith("GET ") or
                (normalized.startswith("FINISH(") and normalized.endswith(")"))):
            return {"kind": "error", "request": {"action": action},
                    "reply": "TaskCraft generation permits GET and FINISH only"}
        return await self._source.execute(action)


def read_only_tools(tools: list) -> list:
    """按环境显式声明保留只读工具；没有任何声明即视为环境不可用。"""
    selected = [tool for tool in tools if tool.name in READ_ONLY_TOOLS]
    if not selected:
        raise EnvironmentUnavailable(
            "TaskCraft requires at least one declared read-only medical tool"
        )
    return selected


def bind_generation_tools(tools):
    """原生协议绑定只读视图，函数工具环境只绑定已声明的只读子集。"""
    if is_native(tools):
        return ReadOnlyNativeTools(tools)
    return read_only_tools(tools)
