from __future__ import annotations

import json
import subprocess
import urllib.request
from pathlib import Path


LIVE_BASE_URL = "http://127.0.0.1:3000"
ROOT = Path(__file__).resolve().parents[1]


def test_model_registry_current_reports_model_coverage_and_roles() -> None:
    with urllib.request.urlopen(f"{LIVE_BASE_URL}/model_registry_current?limit=1", timeout=5) as resp:
        assert resp.status == 200
        payload = json.loads(resp.read().decode("utf-8"))

    assert isinstance(payload, list) and payload, payload
    row = payload[0]
    assert row["model_packet_id"] == "model_registry_current"
    assert isinstance(row.get("model_summary"), dict)
    assert isinstance(row.get("role_breakdown"), dict)
    assert isinstance(row.get("loadout_breakdown"), dict)
    assert isinstance(row.get("active_models"), list)
    assert isinstance(row.get("goal"), dict)
    assert isinstance(row.get("db_law"), dict)
    assert row["db_law"]["statement"].startswith("Postgres/PostgREST is truth")
    assert isinstance(row.get("next_commands"), list)
    assert any("luci model registry current --json" in cmd for cmd in row["next_commands"])
    assert row["model_summary"]["model_count"] >= row["model_summary"]["active_count"]
    assert "role_names" in row["model_summary"]
    assert "routing_notes" in row


def test_manual_current_mentions_model_registry_packet() -> None:
    with urllib.request.urlopen(f"{LIVE_BASE_URL}/manual_current?limit=1", timeout=5) as resp:
        assert resp.status == 200
        payload = json.loads(resp.read().decode("utf-8"))

    row = payload[0]
    route_ids = {route["route_id"] for route in row["route_list"]}
    assert "model_registry_current" in route_ids
    assert "model_registry_current" in row["live_surface"]
    assert "model registry" in row["auth_expectations"]["manual_source"]


def test_model_registry_raw_shell_alias_is_live() -> None:
    proc = subprocess.run([str(ROOT / "luci"), "model", "registry", "--json"], cwd=ROOT, text=True, capture_output=True, check=True)
    payload = json.loads(proc.stdout)
    assert payload["ok"] is True
    assert payload["source_url"].endswith("/model_registry?order=updated_at.desc&limit=50")
    assert payload["payload"]
