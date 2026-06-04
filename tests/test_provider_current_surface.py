from __future__ import annotations

import json
import urllib.request


LIVE_BASE_URL = "http://127.0.0.1:3000"


def test_provider_current_reports_provider_coverage_and_defaults() -> None:
    with urllib.request.urlopen(f"{LIVE_BASE_URL}/provider_current?limit=1", timeout=5) as resp:
        assert resp.status == 200
        payload = json.loads(resp.read().decode("utf-8"))

    assert isinstance(payload, list) and payload, payload
    row = payload[0]
    assert row["provider_packet_id"] == "provider_current"
    assert isinstance(row.get("provider_summary"), dict)
    assert isinstance(row.get("kind_breakdown"), dict)
    assert isinstance(row.get("active_providers"), list)
    assert isinstance(row.get("goal"), dict)
    assert isinstance(row.get("db_law"), dict)
    assert row["db_law"]["statement"].startswith("Postgres/PostgREST is truth")
    assert isinstance(row.get("next_commands"), list)
    assert any("luci provider current --json" in cmd for cmd in row["next_commands"])
    assert row["provider_summary"]["provider_count"] >= row["provider_summary"]["active_count"]
    assert "provider_kind_names" in row["provider_summary"]
    assert "routing_notes" in row


def test_manual_current_mentions_provider_packet() -> None:
    with urllib.request.urlopen(f"{LIVE_BASE_URL}/manual_current?limit=1", timeout=5) as resp:
        assert resp.status == 200
        payload = json.loads(resp.read().decode("utf-8"))

    row = payload[0]
    route_ids = {route["route_id"] for route in row["route_list"]}
    assert "provider_current" in route_ids
    assert "provider_current" in row["live_surface"]
    assert "provider registry" in row["auth_expectations"]["manual_source"]
