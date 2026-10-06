"""PhysicianBench 工具返回的长度受限 JSON 序列化。

交付文本使用紧凑 JSON（`separators=(",", ":")` 与 `ensure_ascii=False`），与早期
`json.dumps` 默认输出的文本不同；配置上限按交付 JSON 的字符数计算，包含结构、记录
与裁剪标记。超限时只选择完整记录或省略资源，不对序列化字符串硬切。
"""

import json

MIN_TOOL_RESULT_LIMIT = 256

_ERROR_HINT = "Tool call failed; error message exceeds the limit."


def _dump(value):
    return json.dumps(value, ensure_ascii=False, default=str,
                      separators=(",", ":"))


def validate_tool_result_limit(limit: int) -> None:
    if limit < MIN_TOOL_RESULT_LIMIT:
        raise ValueError("tool_result_limit must be at least 256 characters")


def serialize_tool_result(value: object, limit: int, *, write: bool = False,
                          error: bool = False) -> str:
    validate_tool_result_limit(limit)
    raw = _dump(value)
    if len(raw) <= limit:
        return raw
    if error:
        result = _dump({
            "error": _ERROR_HINT,
            "truncation": {"truncated": True, "message": _ERROR_HINT},
        })
    elif isinstance(value, dict) and isinstance(value.get("entries"), list):
        records = value["entries"]
        message = ("One record exceeds the limit; narrow the query."
                   if records and len(_dump(records[0])) > limit else
                   "Response incomplete; narrow the query.")
        base = {key: item for key, item in value.items() if key != "entries"}

        def render(count):
            return _dump({**base, "entries": records[:count],
                          "truncation": {"truncated": True,
                              "total_records": len(records),
                              "returned_records": count, "message": message}})

        if len(render(0)) > limit:
            base = {}
        low, high = 0, len(records)
        while low < high:
            middle = (low + high + 1) // 2
            if len(render(middle)) <= limit:
                low = middle
            else:
                high = middle - 1
        if records and low == 0:
            message = "One record exceeds the limit; narrow the query."
        result = render(low)
    else:
        result = _dump({"status": "succeeded" if write else "response_omitted",
                        "truncation": {"truncated": True,
                            "message": ("Write succeeded; response exceeds limit. Do not repeat."
                                        if write else
                                        "One record exceeds the limit; narrow the query.")}})
    if len(result) > limit:
        raise ValueError("tool_result_limit cannot contain the truncation metadata")
    return result
