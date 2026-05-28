from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def test_build_generation_trace_records_routing_payload_latency_and_raw_output() -> None:
    from model_invocation_trace import build_generation_trace

    request = {"model": "fixture-model", "messages": [{"role": "user", "content": "ping"}], "max_tokens": 4}
    trace = build_generation_trace(
        target="groq",
        model_name="fixture-model",
        request_payload=request,
        latency_ms=12.34,
        raw_output="pong",
        raw_response={"id": "abc", "choices": []},
        execute_performed=True,
    )

    assert trace["schema"] == "lucidota.model_generation_trace.v1"
    assert trace["target"] == "groq"
    assert trace["model_name"] == "fixture-model"
    assert trace["payload_size_bytes"] == len(json.dumps(request, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    assert trace["payload_size_chars"] == len(json.dumps(request, sort_keys=True, separators=(",", ":")))
    assert trace["latency_ms"] == 12.34
    assert trace["raw_output"] == "pong"
    assert trace["raw_output_chars"] == 4
    assert trace["raw_response_present"] is True
    assert trace["execute_performed"] is True


def test_generation_trace_refuses_missing_target_model_and_negative_latency() -> None:
    from model_invocation_trace import build_generation_trace

    with pytest.raises(ValueError):
        build_generation_trace(target="", model_name="m", request_payload={}, latency_ms=0, raw_output="")
    with pytest.raises(ValueError):
        build_generation_trace(target="groq", model_name="", request_payload={}, latency_ms=0, raw_output="")
    with pytest.raises(ValueError):
        build_generation_trace(target="groq", model_name="m", request_payload={}, latency_ms=-1, raw_output="")
