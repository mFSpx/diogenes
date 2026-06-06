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
    assert isinstance(row.get("role_admission_decisions"), dict)
    assert isinstance(row.get("admitted_roles"), list)
    assert isinstance(row.get("honestly_skipped_roles"), list)
    assert isinstance(row.get("goal"), dict)
    assert isinstance(row.get("db_law"), dict)
    assert row["db_law"]["statement"].startswith("Postgres/PostgREST is truth")
    assert isinstance(row.get("next_commands"), list)
    assert "model_registry_current" in row["next_commands"]
    assert "model_registry" in row["next_commands"]
    assert "model_routing_current" in row["next_commands"]
    assert isinstance(row.get("next_command_refs"), list) and row["next_command_refs"]
    assert "manual_current" in row["next_command_refs"]
    assert "root_orchestrator_current" in row["next_command_refs"]
    assert "command_registry" in row["next_command_refs"]
    assert "schema_owner_manifest" in row["next_command_refs"]
    assert isinstance(row.get("orchestration"), dict)
    assert row["orchestration"]["mode"] == "sub_orchestrator"
    assert row["orchestration"]["sub_orchestrator_priority"][0] == "live_truth_surfaces"
    assert isinstance(row.get("controller_grant"), dict)
    assert row["controller_grant"]["grant_key"] == "default_local_operator"
    assert row["controller_grant"]["effective_status"] == "active"
    assert isinstance(row.get("agent_thread_runtime"), dict)
    assert row["agent_thread_runtime"]["thread_key"] == "root_operator_thread"
    assert row["agent_thread_runtime"]["runtime_kind"] == "local"
    assert row["model_summary"]["model_count"] >= row["model_summary"]["active_count"]
    assert "role_names" in row["model_summary"]
    assert "routing_notes" in row
    assert isinstance(row.get("resident_loadout"), dict)
    assert row["resident_loadout"]["loadout_id"] == "gtx1650-special-forces-v0"
    assert isinstance(row.get("resident_loadout_status"), dict)
    assert row["resident_loadout_status"]["status"] == "partial"
    assert row["resident_loadout_status"]["decision"] == "defer"
    assert row["model_summary"]["active_loadout_id"] == "gtx1650-special-forces-v0"
    assert row["model_summary"]["active_loadout_slot_count"] >= 1
    assert row["role_admission_decisions"]["router"]["admission_class"] == "ADMITTED"
    assert "classifier" in row["honestly_skipped_roles"]
    assert row["missing_roles"] == []


def test_manual_current_mentions_model_registry_packet() -> None:
    with urllib.request.urlopen(f"{LIVE_BASE_URL}/manual_current?limit=1", timeout=15) as resp:
        assert resp.status == 200
        payload = json.loads(resp.read().decode("utf-8"))

    assert isinstance(payload, list) and payload, payload
    row = payload[0]
    assert "model_registry_current" in row["next_command_refs"]
    assert "model_registry_current" in row["next_commands"]
    assert not any("luci model registry current --json" in cmd for cmd in row["next_commands"])
    assert "model_registry_current" in row["route_refs"]


def test_model_registry_raw_shell_alias_is_live() -> None:
    proc = subprocess.run([str(ROOT / "luci"), "model", "registry", "--json"], cwd=ROOT, text=True, capture_output=True, check=True)
    payload = json.loads(proc.stdout)
    assert payload["ok"] is True
    assert payload["source_url"].endswith("/model_registry?order=updated_at.desc&limit=50")
    assert payload["payload"]
    assert "resident_loadout_status" in json.dumps(payload["payload"][0])
