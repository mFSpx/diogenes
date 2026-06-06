from __future__ import annotations

import json
import urllib.request


LIVE_BASE_URL = "http://127.0.0.1:3000"


def test_manual_current_mentions_workload_audit_refs() -> None:
    with urllib.request.urlopen(f"{LIVE_BASE_URL}/manual_current?limit=1", timeout=15) as resp:
        assert resp.status == 200
        payload = json.loads(resp.read().decode("utf-8"))

    assert isinstance(payload, list) and payload, payload
    row = payload[0]
    assert "workload_audit_current" in row["route_refs"]
    assert "workload_audit_ledger" in row["route_refs"]
    assert "workload_audit_current" in row["next_command_refs"]
