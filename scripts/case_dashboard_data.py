#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from spine_common import rel, write_json

MAX_DASHBOARD_ROWS = 200


def _list(value: Any, *, limit: int = MAX_DASHBOARD_ROWS) -> list[Any]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("dashboard inputs must use lists for row collections")
    return value[:limit]


def _count_package(custody: dict[str, Any], key: str) -> int:
    value = (custody.get("package") or {}).get(key, 0)
    try:
        return int(value)
    except Exception:
        return 0


def compile_dashboard_data(
    *,
    case_id: str,
    custody: dict[str, Any],
    timeline: dict[str, Any],
    claims: list[dict[str, Any]],
    contradictions: list[dict[str, Any]],
    next_actions: list[dict[str, Any]],
    receipts: list[str],
    output_path: str | Path,
) -> dict[str, Any]:
    if not str(case_id).strip():
        raise ValueError("case_id_required")
    claims_l = _list(claims)
    contradictions_l = _list(contradictions)
    next_actions_l = _list(next_actions)
    receipts_l = [str(r) for r in _list(receipts)]
    timeline_rows = _list(timeline.get("timeline", []), limit=50)
    data = {
        "schema": "lucidota.case_dashboard.v1",
        "case_id": case_id,
        "bounded": True,
        "row_limits": {"claims": MAX_DASHBOARD_ROWS, "contradictions": MAX_DASHBOARD_ROWS, "next_actions": MAX_DASHBOARD_ROWS, "timeline_summary": 50},
        "counts": {
            "normal_files": _count_package(custody, "normal_count"),
            "quarantined_files": _count_package(custody, "quarantine_count"),
            "timeline_claims": int(timeline.get("claim_count") or 0),
            "claim_count": len(claims),
            "claim_count_included": len(claims_l),
            "contradiction_count": len(contradictions),
            "contradiction_count_included": len(contradictions_l),
            "next_action_count": len(next_actions),
            "next_action_count_included": len(next_actions_l),
            "receipt_count": len(receipts),
            "receipt_count_included": len(receipts_l),
        },
        "blocked_items": {
            "quarantine": _list(custody.get("quarantine", [])),
            "contradictions": [c for c in contradictions_l if isinstance(c, dict) and c.get("resolution_status") == "OPEN"],
        },
        "timeline_summary": timeline_rows,
        "claim_clusters": sorted({str(c.get("cluster_id")) for c in claims_l if isinstance(c, dict) and c.get("cluster_id")}),
        "contradictions": contradictions_l,
        "next_actions": next_actions_l,
        "receipts": receipts_l,
    }
    write_json(output_path, data)
    data["dashboard_data_path"] = rel(output_path)
    return data
