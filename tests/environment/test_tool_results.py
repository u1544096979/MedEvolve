"""Length-limited serialization of PhysicianBench tool results."""

import json

import pytest

from environments.tool_results import serialize_tool_result, validate_tool_result_limit


@pytest.mark.parametrize("limit", [500, 5000, 10000])
def test_complete_records_within_configured_limit(limit):
    records = [{"id": str(i), "valueString": '化验\\n"结果"' * 40}
               for i in range(100)]
    original = {"entries": records, "total": 100, "pages": 1}
    raw = serialize_tool_result(original, limit)
    result = json.loads(raw)
    assert len(raw) <= limit
    assert result["entries"] == records[:len(result["entries"])]
    assert result["truncation"]["total_records"] == 100
    assert result["truncation"]["returned_records"] == len(result["entries"])
    assert result["truncation"]["truncated"] is True
    assert original["entries"] == records


def test_oversized_first_record_is_not_partially_returned():
    result = json.loads(serialize_tool_result(
        {"entries": [{"id": "large", "valueString": "x" * 10000}],
         "total": 1, "pages": 1}, 500))
    assert result["entries"] == []
    assert result["truncation"]["returned_records"] == 0
    assert result["truncation"]["total_records"] == 1


def test_small_result_preserves_data_and_exact_boundary():
    value = {"entries": [{"id": "a"}], "total": 1, "pages": 1}
    raw = serialize_tool_result(value, 500)
    assert json.loads(raw) == value
    large = {"entries": [{"id": "a", "valueString": "x" * 500}]}
    full = serialize_tool_result(large, 10000)
    assert serialize_tool_result(large, len(full)) == full


def test_oversized_write_preserves_success_without_resource_fragment():
    raw = serialize_tool_result(
        {"resourceType": "MedicationRequest", "id": "created",
         "note": [{"text": "x" * 10000}]}, 500, write=True)
    result = json.loads(raw)
    assert len(raw) <= 500
    assert result["status"] == "succeeded"
    assert "note" not in result
    assert "Do not repeat" in result["truncation"]["message"]


def test_limit_too_small_is_rejected():
    with pytest.raises(ValueError):
        validate_tool_result_limit(1)


def test_oversized_error_returns_compact_valid_json():
    raw = serialize_tool_result(
        {"error": "PatientAccessError: " + "x" * 10000}, 256, error=True)
    result = json.loads(raw)
    assert len(raw) <= 256
    assert result["truncation"]["truncated"] is True
    assert result["truncation"]["message"] == result["error"]
    assert "succeeded" not in raw
    assert "x" * 100 not in raw


def test_error_within_limit_keeps_original_message():
    value = {"error": "ValueError: count must be positive. Please retry."}
    assert serialize_tool_result(value, 256, error=True) == json.dumps(
        value, ensure_ascii=False, separators=(",", ":"))


def test_error_takes_precedence_over_write_and_entries():
    raw = serialize_tool_result(
        {"error": "x" * 5000, "entries": [{"id": str(i)} for i in range(50)]},
        256, write=True, error=True)
    result = json.loads(raw)
    assert len(raw) <= 256
    assert "succeeded" not in raw
    assert "entries" not in result
