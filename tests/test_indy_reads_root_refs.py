from __future__ import annotations

import json
import urllib.request


LIVE_BASE_URL = "http://127.0.0.1:3000"


def test_root_orchestrator_mentions_indy_reads_exocortex_refs() -> None:
    with urllib.request.urlopen(f"{LIVE_BASE_URL}/root_orchestrator_current?limit=1", timeout=15) as resp:
        assert resp.status == 200
        payload = json.loads(resp.read().decode("utf-8"))

    assert isinstance(payload, list) and payload, payload
    row = payload[0]
    live_surface = row["live_surface"]
    assert "indy_reads_self_model" in live_surface
    assert "indy_reads_metacognition_current" in live_surface
