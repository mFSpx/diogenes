from __future__ import annotations

import json
import urllib.request


LIVE_BASE_URL = "http://127.0.0.1:3000"


def test_no_claim_without_receipt() -> None:
    with urllib.request.urlopen(f"{LIVE_BASE_URL}/workload_audit_current?limit=1", timeout=15) as resp:
        assert resp.status == 200
        payload = json.loads(resp.read().decode("utf-8"))

    row = payload[0]
    if row["has_unacknowledged_unknown_rows"]:
        assert row["can_claim_duplex_race"] is False
