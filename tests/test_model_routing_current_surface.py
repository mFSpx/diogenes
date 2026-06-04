from __future__ import annotations

import json
import subprocess
import urllib.request
from pathlib import Path


LIVE_BASE_URL = "http://127.0.0.1:3000"
ROOT = Path(__file__).resolve().parents[1]


def test_model_routing_current_reports_live_role_coverage() -> None:
    with urllib.request.urlopen(f"{LIVE_BASE_URL}/model_routing_current?limit=1", timeout=5) as resp:
        assert resp.status == 200
        payload = json.loads(resp.read().decode("utf-8"))

    assert isinstance(payload, list) and payload, payload
    row = payload[0]
    assert row["routing_packet_id"] == "model_routing_current"
    assert isinstance(row.get("model_registry"), dict)
    assert isinstance(row.get("provider_registry"), dict)
    assert isinstance(row.get("local_model_roles"), dict)
    assert isinstance(row.get("missing_roles"), list)
    assert isinstance(row.get("goal"), dict)
    assert isinstance(row.get("db_law"), dict)
    assert row["db_law"]["statement"].startswith("Postgres/PostgREST is truth")
    assert isinstance(row.get("next_commands"), list)
    assert any("luci model-routing-current --json" in cmd for cmd in row["next_commands"])
    assert row["model_registry"]["active_models"] >= 3
    assert "router" in row["local_model_roles"]
    assert "classifier" in row["missing_roles"]
    assert "treelite_gate" in row["missing_roles"]


def test_manual_current_mentions_model_routing_packet() -> None:
    with urllib.request.urlopen(f"{LIVE_BASE_URL}/manual_current?limit=1", timeout=5) as resp:
        assert resp.status == 200
        payload = json.loads(resp.read().decode("utf-8"))

    row = payload[0]
    route_ids = {route["route_id"] for route in row["route_list"]}
    assert "model_routing_current" in route_ids
    assert "model_routing_current" in row["live_surface"]
    assert "model routing" in row["auth_expectations"]["manual_source"].lower()


def test_model_routing_current_spaced_shell_alias_is_live() -> None:
    proc = subprocess.run([str(ROOT / "luci"), "model", "routing", "current", "--json"], cwd=ROOT, text=True, capture_output=True, check=True)
    payload = json.loads(proc.stdout)
    assert payload["ok"] is True
    assert payload["source_url"].endswith("/model_routing_current?limit=1")
    assert payload["payload"]
