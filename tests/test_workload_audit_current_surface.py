from __future__ import annotations

import json
import urllib.request


LIVE_BASE_URL = "http://127.0.0.1:3000"


def test_workload_audit_current_surface_reports_summary_and_race_state() -> None:
    with urllib.request.urlopen(f"{LIVE_BASE_URL}/workload_audit_current?limit=1", timeout=15) as resp:
        assert resp.status == 200
        payload = json.loads(resp.read().decode("utf-8"))

    assert isinstance(payload, list) and payload, payload
    row = payload[0]
    assert row["audit_status"] in {"PROVEN", "PARTIAL", "UNKNOWN", "CONTRADICTED"}
    assert isinstance(row.get("actor_summary"), list)
    assert isinstance(row.get("has_unacknowledged_unknown_rows"), bool)
    assert isinstance(row.get("unknown_row_count"), int)
    assert isinstance(row.get("proven_row_count"), int)
    assert isinstance(row.get("partial_row_count"), int)
    assert isinstance(row.get("contradicted_row_count"), int)
    assert isinstance(row.get("can_claim_duplex_race"), bool)

