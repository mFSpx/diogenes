from __future__ import annotations

import json
import urllib.request


LIVE_BASE_URL = "http://127.0.0.1:3000"


def test_indy_reads_metacognition_current_is_live() -> None:
    with urllib.request.urlopen(f"{LIVE_BASE_URL}/indy_reads_metacognition_current?limit=1", timeout=15) as resp:
        assert resp.status == 200
        payload = json.loads(resp.read().decode("utf-8"))

    assert isinstance(payload, list) and payload, payload
    row = payload[0]
    assert row["owner_role"] == "indy_reads_runtime"
    assert row["proof_status"] in {"PROVEN", "PARTIAL", "UNKNOWN", "CONTRADICTED"}
