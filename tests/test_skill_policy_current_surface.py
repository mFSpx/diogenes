from __future__ import annotations

import json
import urllib.request


LIVE_BASE_URL = "http://127.0.0.1:3000"


def test_skill_policy_current_reports_policy_text_and_status() -> None:
    with urllib.request.urlopen(f"{LIVE_BASE_URL}/skill_policy_current?limit=1", timeout=5) as resp:
        assert resp.status == 200
        payload = json.loads(resp.read().decode("utf-8"))

    assert isinstance(payload, list) and payload, payload
    row = payload[0]
    assert row["policy_id"] == "superpowers_alignment"
    assert row["policy_title"] == "LUCIDOTA Skill Policy"
    assert row["status"] == "current"
    assert "Repository-local truth sources win" in row["policy_text"]
    assert isinstance(row.get("next_command_refs"), list) and row["next_command_refs"]
    assert "manual_current" in row["next_command_refs"]
    assert "root_orchestrator_current" in row["next_command_refs"]
    assert "command_registry" in row["next_command_refs"]
    assert "schema_owner_manifest" in row["next_command_refs"]
    assert isinstance(row.get("orchestration"), dict)
    assert row["orchestration"]["mode"] == "sub_orchestrator"
    assert row["orchestration"]["sub_orchestrator_priority"][0] == "live_truth_surfaces"


def test_manual_current_mentions_skill_policy_route() -> None:
    with urllib.request.urlopen(f"{LIVE_BASE_URL}/manual_current?limit=1", timeout=15) as resp:
        assert resp.status == 200
        payload = json.loads(resp.read().decode("utf-8"))

    row = payload[0]
    route_ids = {route["route_id"] for route in row["route_list"]}
    assert "skill_policy_current" in route_ids
    assert "skill_policy_current" in row["live_surface"]
    assert "skill policy" in row["auth_expectations"]["manual_source"]
