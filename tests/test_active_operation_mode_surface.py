from __future__ import annotations

import json
import urllib.request


LIVE_BASE_URL = "http://127.0.0.1:3000"


def test_active_operation_mode_surface_reports_build_swarm_mode() -> None:
    with urllib.request.urlopen(f"{LIVE_BASE_URL}/active_operation_mode?limit=1", timeout=15) as resp:
        assert resp.status == 200
        payload = json.loads(resp.read().decode("utf-8"))

    assert isinstance(payload, list) and payload, payload
    row = payload[0]
    assert row["current_mode"] == "BUILD_SWARM_MODE"
    assert row["cloud_policy"] == "OPERATOR_ALLOWED_RECEIPT_REQUIRED"
    assert row["swarm_policy"] == "OPERATOR_ALLOWED_RECEIPT_REQUIRED"
    assert row["indy_reads_policy"] == "EXOCORTEX_ALLOWED_RECEIPT_REQUIRED"
    assert row["receipt_policy"] == "DB_RECEIPT_OR_UNKNOWN_DEBT"
    assert row["runtime_default_policy"] == "RUNTIME_FAST_PATH_AFTER_BUILD"
    assert row["operator_override"] == "RAC_ON"
    assert isinstance(row.get("evidence_refs"), list) and row["evidence_refs"]
