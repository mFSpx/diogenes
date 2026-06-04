#!/usr/bin/env python3
from __future__ import annotations

import json
import urllib.request

LIVE_BASE_URL = "http://127.0.0.1:3000"


def test_api_workflow_registry_route_is_readable() -> None:
    with urllib.request.urlopen(f"{LIVE_BASE_URL}/api_workflow_registry?limit=1", timeout=5) as resp:
        assert resp.status == 200
        payload = json.loads(resp.read().decode("utf-8"))

    assert isinstance(payload, list)
    assert payload
    row = payload[0]
    assert "workflow_id" in row
    assert "workflow_name" in row
    assert "status" in row
    assert row["status"] in {"active", "deprecated", "planned"}
