from __future__ import annotations

import json
import urllib.request


LIVE_BASE_URL = "http://127.0.0.1:3000"


def test_active_goal_reports_strict_priority_stack() -> None:
    with urllib.request.urlopen(f"{LIVE_BASE_URL}/active_goal?limit=1", timeout=5) as resp:
        assert resp.status == 200
        payload = json.loads(resp.read().decode("utf-8"))

    assert isinstance(payload, list) and payload, payload
    row = payload[0]
    assert isinstance(row.get("orchestration"), dict)
    assert row["orchestration"]["mode"] == "sub_orchestrator"
    assert row["orchestration"]["strict_priority_stack"][0] == "live_truth_surfaces"
    assert row["title"]
    assert row["status"]
