from __future__ import annotations

import json
import urllib.request


LIVE_BASE_URL = "http://127.0.0.1:3000"


def test_root_orchestrator_contains_workload_audit_summary() -> None:
    with urllib.request.urlopen(f"{LIVE_BASE_URL}/root_orchestrator_current?limit=1", timeout=15) as resp:
        assert resp.status == 200
        payload = json.loads(resp.read().decode("utf-8"))

    assert isinstance(payload, list) and payload, payload
    row = payload[0]
    summary = row["live_surface"]["workload_audit_current"]
    assert summary["audit_status"] in {"PROVEN", "PARTIAL", "UNKNOWN", "CONTRADICTED"}
    assert isinstance(summary.get("has_unacknowledged_unknown_rows"), bool)
    assert isinstance(summary.get("can_claim_duplex_race"), bool)
