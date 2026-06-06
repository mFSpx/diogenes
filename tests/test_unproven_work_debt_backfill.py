from __future__ import annotations

import json
import urllib.request


LIVE_BASE_URL = "http://127.0.0.1:3000"


def test_unproven_work_debt_rows_exist_for_unreceipted_claims() -> None:
    with urllib.request.urlopen(f"{LIVE_BASE_URL}/unproven_work_debt?limit=50", timeout=15) as resp:
        assert resp.status == 200
        payload = json.loads(resp.read().decode("utf-8"))

    assert isinstance(payload, list) and payload, payload
    debt = payload[0]
    assert debt["proof_status"] == "UNKNOWN"
    assert debt["debt_reason"] == "no receipt-backed workload/token evidence"
    assert debt["tokens_in"] is None
    assert debt["tokens_out"] is None

