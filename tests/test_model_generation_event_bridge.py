from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def write_receipt(root: Path, name: str = "groq_chat_execute_20260101T000000Z.json") -> Path:
    path = root / "05_OUTPUTS" / "model_invocations" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "schema": "lucidota.model_invocation.groq_chat.v1",
        "generated_at": "2026-01-01T00:00:00Z",
        "provider": "groq",
        "mode": "execute",
        "status": "PASS",
        "raw_response": {"id": "abc", "choices": []},
        "generation_trace": {
            "schema": "lucidota.model_generation_trace.v1",
            "target": "groq",
            "model_name": "openai/gpt-oss-120b",
            "payload_size_bytes": 123,
            "payload_size_chars": 123,
            "latency_ms": 45.6,
            "raw_output": "OK",
            "raw_output_chars": 2,
            "raw_response_present": True,
            "execute_performed": True,
        },
    }), encoding="utf-8")
    return path


def test_generation_event_bridge_builds_exact_event_from_one_receipt(tmp_path) -> None:
    from model_generation_event_bridge import load_generation_event

    receipt = write_receipt(tmp_path)
    event = load_generation_event(receipt, root=tmp_path)

    assert event["schema"] == "lucidota.model_generation_event.v1"
    assert event["receipt_path"] == "05_OUTPUTS/model_invocations/groq_chat_execute_20260101T000000Z.json"
    assert len(event["receipt_sha256"]) == 64
    assert event["target"] == "groq"
    assert event["model_name"] == "openai/gpt-oss-120b"
    assert event["payload_size_bytes"] == 123
    assert event["latency_ms"] == 45.6
    assert event["raw_output"] == "OK"
    assert event["raw_response"] == {"id": "abc", "choices": []}
    assert event["execute_performed"] is True


def test_generation_event_bridge_dry_run_is_targeted_to_named_receipts(tmp_path) -> None:
    from model_generation_event_bridge import stage_receipts

    wanted = write_receipt(tmp_path, "wanted.json")
    ignored = write_receipt(tmp_path, "ignored.json")

    report = stage_receipts([wanted], root=tmp_path, execute=False, database_url="postgresql:///unused")

    assert report["execute_performed"] is False
    assert report["receipts_seen"] == 1
    assert report["events_valid"] == 1
    assert report["events_staged"] == 0
    assert report["would_stage"] == ["05_OUTPUTS/model_invocations/wanted.json"]
    assert "ignored.json" not in json.dumps(report)
    assert ignored.exists()


def test_model_generation_event_schema_registers_queue_table_and_worker_contract() -> None:
    schema = (ROOT / "06_SCHEMA" / "111_model_generation_event_lane.sql").read_text(encoding="utf-8")
    assert "lucidota_control.model_generation_event" in schema
    assert "model_generation" in schema
    assert "model_generation_receipt_stage" in schema
    assert "model_generation_event_bridge" in schema
    assert "canonical_graph_write_allowed" in schema
    assert "false" in schema.lower()
