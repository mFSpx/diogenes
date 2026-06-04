from __future__ import annotations

import json
import urllib.request


LIVE_BASE_URL = "http://127.0.0.1:3000"


def test_chrono_current_reports_prompt_work_and_learning_alignment() -> None:
    with urllib.request.urlopen(f"{LIVE_BASE_URL}/chrono_current?limit=1", timeout=5) as resp:
        assert resp.status == 200
        payload = json.loads(resp.read().decode("utf-8"))

    assert isinstance(payload, list) and payload, payload
    row = payload[0]
    assert row["chrono_packet_id"] == "chrono_current"
    assert isinstance(row.get("prompt_ledger"), dict)
    assert isinstance(row.get("work_ledger"), dict)
    assert isinstance(row.get("execution_history"), dict)
    assert isinstance(row.get("learning_loop"), dict)
    assert isinstance(row.get("routing_registry"), dict)
    assert row["prompt_ledger"]["prompt_count"] >= row["prompt_ledger"]["filed_count"]
    assert row["work_ledger"]["work_order_count"] >= row["work_ledger"]["queued_work_orders"]
    assert "bytewax_window_count" in row["learning_loop"]
    assert "active_models" in row["routing_registry"]


def test_manual_current_mentions_chrono_packet() -> None:
    with urllib.request.urlopen(f"{LIVE_BASE_URL}/manual_current?limit=1", timeout=5) as resp:
        assert resp.status == 200
        payload = json.loads(resp.read().decode("utf-8"))

    row = payload[0]
    route_ids = {route["route_id"] for route in row["route_list"]}
    assert "chrono_current" in route_ids
    assert "chrono_current" in row["live_surface"]
    assert "prompt ledger" in row["auth_expectations"]["prompt_filing"]
