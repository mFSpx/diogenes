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
    assert isinstance(row.get("role_admission_decisions"), dict)
    assert isinstance(row.get("admitted_roles"), list)
    assert isinstance(row.get("honestly_skipped_roles"), list)
    assert isinstance(row.get("goal"), dict)
    assert isinstance(row.get("db_law"), dict)
    assert row["db_law"]["statement"].startswith("Postgres/PostgREST is truth")
    assert isinstance(row.get("next_commands"), list)
    assert "model_routing_current" in row["next_commands"]
    assert "model_routing_blockers" in row["next_commands"]
    assert "model_registry_current" in row["next_commands"]
    assert isinstance(row.get("next_command_refs"), list) and row["next_command_refs"]
    assert "manual_current" in row["next_command_refs"]
    assert "root_orchestrator_current" in row["next_command_refs"]
    assert "command_registry" in row["next_command_refs"]
    assert "schema_owner_manifest" in row["next_command_refs"]
    assert isinstance(row.get("orchestration"), dict)
    assert row["orchestration"]["mode"] == "sub_orchestrator"
    assert row["orchestration"]["sub_orchestrator_priority"][0] == "live_truth_surfaces"
    assert row["orchestration"]["strict_priority_stack"][0] == "live_truth_surfaces"
    assert isinstance(row.get("resident_loadout"), dict)
    assert row["resident_loadout"]["loadout_id"] == "gtx1650-special-forces-v0"
    assert isinstance(row.get("resident_loadout_status"), dict)
    assert row["resident_loadout_status"]["status"] == "partial"
    assert row["resident_loadout_status"]["decision"] == "defer"
    assert row["model_registry"]["active_models"] >= 3
    assert "router" in row["local_model_roles"]
    assert row["missing_roles"] == []
    assert "router" in row["admitted_roles"]
    assert "thinker" in row["admitted_roles"]
    assert "classifier" in row["honestly_skipped_roles"]
    assert "treelite_gate" in row["honestly_skipped_roles"]
    assert row["role_admission_decisions"]["classifier"]["admission_class"] == "HONESTLY_SKIPPED"
    assert row["role_admission_decisions"]["classifier"]["skip_reason"]
    assert row["role_admission_decisions"]["router"]["admission_class"] == "ADMITTED"
    assert row["controller_grant"]["grant_key"] == "default_local_operator"
    assert row["agent_thread_runtime"]["thread_key"] == "root_operator_thread"


def test_manual_current_mentions_model_routing_packet() -> None:
    with urllib.request.urlopen(f"{LIVE_BASE_URL}/manual_current?limit=1", timeout=15) as resp:
        assert resp.status == 200
        payload = json.loads(resp.read().decode("utf-8"))

    assert isinstance(payload, list) and payload, payload
    row = payload[0]
    assert "model_routing_current" in row["next_command_refs"]
    assert "model_routing_current" in row["next_command_refs"]
    assert "model_routing_current" in row["route_refs"]


def test_model_routing_current_spaced_shell_alias_is_live() -> None:
    proc = subprocess.run([str(ROOT / "luci"), "model", "routing", "current", "--json"], cwd=ROOT, text=True, capture_output=True, check=True)
    payload = json.loads(proc.stdout)
    assert payload["ok"] is True
    assert payload["source_url"].endswith("/model_routing_current?limit=1")
    assert payload["payload"]
