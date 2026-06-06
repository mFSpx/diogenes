from __future__ import annotations

import json
import subprocess


def test_luci_api_indy_responses_surface_is_live() -> None:
    proc = subprocess.run(
        ["./luci", "api", "indy", "responses", "--json"],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(proc.stdout)
    assert isinstance(payload, list)
    assert payload, payload
    row = payload[0]
    assert row.get("response_id") or row.get("id") or row.get("event_id")
    assert row.get("response_delivery_status") or row.get("status") or row.get("processed_status")
