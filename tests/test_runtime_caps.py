import json

import pytest

from ALGOS.runtime_caps import (
    MAX_JSON_CHARS,
    MAX_TEXT_CHARS,
    cap_text,
    bounded_payload,
    clamp_int,
    assert_array_budget,
)


def test_cap_text_reports_truncation_without_materializing_more_than_limit():
    text, truncated = cap_text("x" * (MAX_TEXT_CHARS + 7))
    assert len(text) == MAX_TEXT_CHARS
    assert truncated is True


def test_bounded_payload_returns_capped_json_and_flag():
    raw, truncated = bounded_payload({"blob": "y" * (MAX_JSON_CHARS * 2)})
    assert len(raw) == MAX_JSON_CHARS
    assert truncated is True
    assert raw.startswith('{"blob"')


def test_clamp_int_rejects_out_of_range_values():
    assert clamp_int("5", 1, 10, "limit") == 5
    with pytest.raises(ValueError, match="outside"):
        clamp_int(99, 1, 10, "limit")


def test_assert_array_budget_rejects_excessive_size_object():
    class BigThing:
        size = 2_000_001

    with pytest.raises(MemoryError, match="array budget"):
        assert_array_budget(BigThing(), max_elems=2_000_000)
