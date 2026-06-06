from __future__ import annotations

import json
import subprocess
import urllib.request
from pathlib import Path


LIVE_BASE_URL = "http://127.0.0.1:3000"
ROOT = Path(__file__).resolve().parents[1]


def test_model_routing_blockers_reports_missing_roles_as_live_packet() -> None:
    with urllib.request.urlopen(f"{LIVE_BASE_URL}/model_routing_blockers?limit=1", timeout=5) as resp:
        assert resp.status == 200
        payload = json.loads(resp.read().decode("utf-8"))

    assert isinstance(payload, list) and payload, payload
    row = payload[0]
    assert row["routing_packet_id"] == "model_routing_blockers"
    assert isinstance(row.get("missing_roles"), list)
    assert row["missing_role_count"] == len(row["missing_roles"])
    assert row["missing_roles"] == []
    assert isinstance(row.get("honestly_skipped_roles"), list)
    assert row["honestly_skipped_role_count"] == len(row["honestly_skipped_roles"])
    assert "classifier" in row["honestly_skipped_roles"]
    assert "treelite_gate" in row["honestly_skipped_roles"]
    assert row["role_admission_decisions"]["classifier"]["admission_class"] == "HONESTLY_SKIPPED"
    assert row["role_admission_decisions"]["treelite_gate"]["skip_reason"]
    assert isinstance(row.get("goal"), dict)
    assert isinstance(row.get("db_law"), dict)
    assert isinstance(row.get("next_commands"), list) and row["next_commands"]
    assert isinstance(row.get("next_command_refs"), list) and row["next_command_refs"]
    assert "manual_current" in row["next_command_refs"]
    assert "root_orchestrator_current" in row["next_command_refs"]
    assert "command_registry" in row["next_command_refs"]
    assert "schema_owner_manifest" in row["next_command_refs"]
    assert isinstance(row.get("orchestration"), dict)
    assert row["orchestration"]["mode"] == "sub_orchestrator"
    assert row["orchestration"]["sub_orchestrator_priority"][0] == "live_truth_surfaces"


def test_manual_current_mentions_model_routing_blockers_packet() -> None:
    with urllib.request.urlopen(f"{LIVE_BASE_URL}/manual_current?limit=1", timeout=15) as resp:
        assert resp.status == 200
        payload = json.loads(resp.read().decode("utf-8"))

    row = payload[0]
    route_ids = {route["route_id"] for route in row["route_list"]}
    assert "model_routing_blockers" in route_ids
    assert "model_routing_blockers" in row["live_surface"]
    assert "model_routing_blockers" in row["next_commands"]
    assert set(row["next_commands"]).issubset(set(row["next_command_refs"]))
    assert set(row["next_commands"]).issubset(set(row["next_command_refs"]))
    assert "model routing blockers" in row["auth_expectations"]["manual_source"].lower()
    assert isinstance(row["live_surface"]["model_routing_blockers"], list)
    assert row["live_surface"]["model_routing_blockers"]


def test_model_routing_blockers_spaced_shell_alias_is_live() -> None:
    proc = subprocess.run([str(ROOT / "luci"), "model", "routing", "blockers", "--json"], cwd=ROOT, text=True, capture_output=True, check=True)
    payload = json.loads(proc.stdout)
    assert payload["ok"] is True
    assert payload["source_url"].endswith("/model_routing_blockers?limit=1")
    assert payload["payload"]
