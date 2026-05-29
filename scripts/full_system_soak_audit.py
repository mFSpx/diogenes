#!/usr/bin/env python3
"""Receipt-backed full-system soak auditor.

This does not run a daemon and does not mutate the graph. It answers one
narrow question: do the required runtime lanes have PASS receipts spanning at
least N hours, while canonical graph writes stay absent?
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import median
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "05_OUTPUTS" / "soak"
TIME_KEYS = ("generated_at", "generated_at_utc", "started_at_utc", "finished_at_utc")
LANES = {
    "bytewax_stream": "05_OUTPUTS/project2501_board_stream/project2501_bytewax_board_stream_once_execute_*.json",
    "dbos_gate": "05_OUTPUTS/abductive_db_os/abductive_db_os_gate_fast_*.json",
    "absurd_gate": "05_OUTPUTS/absurd_abductive/absurd_gate_fast_*.json",
    "swarm_usage": "05_OUTPUTS/goals/swarm_usage_ledger_*.json",
    "slop_audit": "05_OUTPUTS/slop_audit/slop_audit_law_*.json",
}
POINT_INVARIANTS = {
    "operator_graph_policy_staged": "05_OUTPUTS/graph/graph_promotion_gate_execute_*.json",
}
BAD_STATUSES = {"FAIL", "BLOCKED"}
GOOD_STATUSES = {"PASS", "VERIFIED", "CHECK_OK"}


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def rel(root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except Exception:
        return str(path)


def parse_time(value: Any) -> datetime | None:
    if not value:
        return None
    text = str(value)
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


def receipt_time(path: Path, payload: dict[str, Any]) -> datetime | None:
    for key in TIME_KEYS:
        parsed = parse_time(payload.get(key))
        if parsed:
            return parsed
    match = re.search(r"(\d{8}T\d{6})", path.name)
    if match:
        return datetime.strptime(match.group(1), "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
    return None


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def canonical_write_present(value: Any) -> bool:
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {"canonical_graph_writes", "canonical_graph_writes_performed", "canonical_graph_materialization"} and item is True:
                return True
            if canonical_write_present(item):
                return True
    elif isinstance(value, list):
        return any(canonical_write_present(item) for item in value)
    return False


def status_of(payload: dict[str, Any]) -> str:
    status = payload.get("status") or payload.get("verdict")
    if isinstance(status, str):
        return status.upper()
    if payload.get("schema") == "lucidota.goals.swarm_usage_ledger.v1" and int(payload.get("totals", {}).get("all_accounted_tokens") or 0) > 0:
        return "PASS"
    if payload.get("blockers") == [] and (payload.get("db_writes_performed") is True or payload.get("execute_performed") is True):
        return "PASS"
    return "UNKNOWN"


def receipt_rows(root: Path, pattern: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob(pattern)):
        payload = load_json(path)
        if not payload:
            continue
        ts = receipt_time(path, payload)
        if not ts:
            continue
        rows.append({
            "path": rel(root, path),
            "time": ts,
            "status": status_of(payload),
            "canonical_write_present": canonical_write_present(payload),
        })
    rows.sort(key=lambda row: row["time"])
    return rows


def summarize_lane(root: Path, name: str, pattern: str, min_hours: float) -> dict[str, Any]:
    rows = receipt_rows(root, pattern)
    good = [row for row in rows if row["status"] in GOOD_STATUSES and not row["canonical_write_present"]]
    bad = [row for row in rows if row["status"] in BAD_STATUSES or row["canonical_write_present"]]
    window_start = None
    blockers: list[str] = []
    span_hours = 0.0
    cadence_seconds_p50 = None
    if len(good) >= 2:
        span_hours = (good[-1]["time"] - good[0]["time"]).total_seconds() / 3600.0
        window_start = good[-1]["time"] - timedelta(hours=min_hours)
        gaps = [(b["time"] - a["time"]).total_seconds() for a, b in zip(good, good[1:])]
        cadence_seconds_p50 = median(gaps) if gaps else None
    recent_bad = [row for row in bad if window_start is None or row["time"] >= window_start]
    if not good:
        blockers.append(f"{name}:no_pass_receipts")
    elif len(good) < 2 or span_hours < min_hours:
        blockers.append(f"{name}:span_below_min_hours")
    if recent_bad:
        blockers.append(f"{name}:hard_bad_receipts_present")
    return {
        "pattern": pattern,
        "receipt_count": len(rows),
        "pass_receipt_count": len(good),
        "bad_receipt_count": len(bad),
        "recent_bad_receipt_count": len(recent_bad),
        "current_window_start": window_start.isoformat() if window_start else None,
        "first_pass": good[0]["time"].isoformat() if good else None,
        "latest_pass": good[-1]["time"].isoformat() if good else None,
        "latest_pass_path": good[-1]["path"] if good else None,
        "span_hours": round(span_hours, 3),
        "cadence_seconds_p50": round(cadence_seconds_p50, 3) if cadence_seconds_p50 is not None else None,
        "passed": not blockers,
        "blockers": blockers,
    }


def latest_point(root: Path, pattern: str) -> dict[str, Any]:
    rows = receipt_rows(root, pattern)
    if not rows:
        return {"pattern": pattern, "passed": False, "blockers": ["point_receipt_missing"]}
    latest = rows[-1]
    blockers = []
    if latest["status"] not in GOOD_STATUSES:
        blockers.append("latest_point_status_not_pass")
    if latest["canonical_write_present"]:
        blockers.append("latest_point_has_canonical_write")
    return {
        "pattern": pattern,
        "receipt_count": len(rows),
        "latest_time": latest["time"].isoformat(),
        "latest_path": latest["path"],
        "latest_status": latest["status"],
        "passed": not blockers,
        "blockers": blockers,
    }


def build_audit(*, root: Path = ROOT, min_hours: float = 4.0) -> dict[str, Any]:
    root = Path(root)
    lanes = {name: summarize_lane(root, name, pattern, min_hours) for name, pattern in LANES.items()}
    points = {name: latest_point(root, pattern) for name, pattern in POINT_INVARIANTS.items()}
    all_patterns = list(LANES.values()) + list(POINT_INVARIANTS.values())
    canonical_bad = []
    for pattern in all_patterns:
        canonical_bad.extend(row for row in receipt_rows(root, pattern) if row["canonical_write_present"])
    blockers = [b for lane in lanes.values() for b in lane["blockers"]]
    blockers += [f"{name}:{b}" for name, point in points.items() for b in point.get("blockers", [])]
    invariants = {
        "canonical_graph_writes_absent": {
            "passed": not canonical_bad,
            "bad_receipt_count": len(canonical_bad),
            "sample_paths": [row["path"] for row in canonical_bad[:10]],
        }
    }
    if canonical_bad:
        status = "FAIL"
    elif blockers:
        status = "DEGRADED"
    else:
        status = "PASS"
    return {
        "schema": "lucidota.full_system_soak_audit.v1",
        "generated_at": now(),
        "status": status,
        "full_system_soak_passed": status == "PASS",
        "min_hours": float(min_hours),
        "definition": "Required lanes must have PASS receipts spanning min_hours; point invariants must be PASS; canonical graph writes/materialization must remain absent in audited receipts.",
        "lanes": lanes,
        "point_invariants": points,
        "invariants": invariants,
        "blockers": blockers,
        "canonical_graph_writes_performed": bool(canonical_bad),
    }


def write_report(audit: dict[str, Any]) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"full_system_soak_audit_{stamp()}.json"
    audit["report_path"] = rel(ROOT, path)
    path.write_text(json.dumps(audit, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description="Audit receipt-backed full-system soak evidence.")
    ap.add_argument("--min-hours", type=float, default=4.0)
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    audit = build_audit(root=Path(args.root), min_hours=args.min_hours)
    path = write_report(audit)
    if args.json:
        print(json.dumps(audit, sort_keys=True, default=str))
    print("REPORT_PATH=" + rel(ROOT, path))
    print("FULL_SYSTEM_SOAK_AUDIT=" + audit["status"])
    return 0 if audit["status"] == "PASS" else 2 if audit["status"] == "DEGRADED" else 4


if __name__ == "__main__":
    raise SystemExit(main())
