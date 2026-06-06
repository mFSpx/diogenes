from __future__ import annotations

import json
import urllib.request


LIVE_BASE_URL = "http://127.0.0.1:3000"


def test_indy_reads_requires_receipt_for_proven_claims() -> None:
    with urllib.request.urlopen(f"{LIVE_BASE_URL}/workload_audit_current?limit=1", timeout=15) as resp:
        assert resp.status == 200
        payload = json.loads(resp.read().decode("utf-8"))

    assert isinstance(payload, list) and payload, payload
    row = payload[0]
    actor_summary = row["actor_summary"]
    indy_rows = [
        item for item in actor_summary
        if item.get("actor_class") == "indy_reads" and item.get("caller") == "indy_reads"
    ]
    assert indy_rows, actor_summary
    for item in indy_rows:
        if item.get("proof_status") in {"PROVEN", "PARTIAL"}:
            assert item.get("receipt_uuid"), item
