#!/usr/bin/env python3
"""LUCIDOTA terminal Big Board v0.

Read-only dashboard: docs are truth for bars; Postgres supplies live counters.
No Drive access, no writes, no secrets.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import psycopg

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "00_PROJECT_BRAIN" / "BUILD_PLAN_AUDIT.md"
STATE_DB = os.environ.get("DBOS_SYSTEM_DATABASE_URL", "postgresql://mfspx@/lucidota_state")
GRAPH_DB = os.environ.get("LUCIDOTA_GRAPH_DATABASE_URL", "postgresql://mfspx@/lucidota_graph")


def bars() -> dict:
    text = PLAN.read_text(encoding="utf-8")
    overall = re.search(r"## Overall Build Bar: (.+)", text)
    phases: list[str] = []
    in_bars = False
    for line in text.splitlines():
        if line == "## Phase Bars":
            in_bars = True
            continue
        if in_bars and line.startswith("## "):
            break
        if in_bars and line.startswith("- **"):
            phases.append(re.sub(r"^- \*\*(.*?)\*\* `(.*?)`(.*)$", r"\1 \2\3", line))
    return {"overall": overall.group(1) if overall else "unknown", "phases": phases}


def scalar(db_url: str, sql: str):
    try:
        with psycopg.connect(db_url, connect_timeout=3) as conn:
            return conn.execute(sql).fetchone()[0]
    except Exception as exc:
        return f"error:{str(exc)[:80]}"


def drive_manifest_count() -> int:
    try:
        from lucidota_drive_manifest import scan
        return len(scan())
    except Exception:
        return -1


def counters() -> dict:
    return {
        "workflow_events": scalar(STATE_DB, "SELECT count(*) FROM lucidota_control.workflow_event"),
        "wake_pending": scalar(STATE_DB, "SELECT count(*) FROM lucidota_control.event_outbox WHERE status='pending'"),
        "wake_delivered": scalar(STATE_DB, "SELECT count(*) FROM lucidota_control.event_outbox WHERE status='delivered'"),
        "cas_artifacts": scalar(GRAPH_DB, "SELECT count(*) FROM lucidota_vault.cas_manifest"),
        "body_capture_captures": scalar(GRAPH_DB, "SELECT count(*) FROM lucidota_body_capture.capture"),
        "body_capture_bundles": scalar(GRAPH_DB, "SELECT count(*) FROM lucidota_body_capture.evidence_bundle"),
        "river_scores": scalar(STATE_DB, "SELECT count(*) FROM lucidota_learning.river_score"),
        "bytewax_hints": scalar(STATE_DB, "SELECT count(*) FROM lucidota_learning.bytewax_hint"),
        "treelite_runs": scalar(STATE_DB, "SELECT count(*) FROM lucidota_learning.treelite_router_run"),
        "indy_queue": scalar(STATE_DB, "SELECT count(*) FROM lucidota_indy.side_queue WHERE status='queued'"),
        "auth_records": scalar(STATE_DB, "SELECT count(*) FROM lucidota_indy.auth_inventory"),
        "operator_corrections": scalar(STATE_DB, "SELECT count(*) FROM lucidota_indy.task_memory WHERE kind='correction'"),
        "drive_manifest_targets": drive_manifest_count(),
        "model_governor_decisions": scalar(STATE_DB, "SELECT count(*) FROM lucidota_runtime.load_governor_decision"),
    }


def gpu() -> dict:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return {"status": "missing"}
    try:
        out = subprocess.check_output([exe, "--query-gpu=name,memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"], text=True, timeout=3).strip()
        return {"status": "ok", "summary": out}
    except Exception as exc:
        return {"status": "error", "error": str(exc)[:120]}


def render(report: dict) -> str:
    lines = ["LUCIDOTA BIG BOARD v0", "=" * 22, f"Overall {report['bars']['overall']}", "", "Build phases:"]
    lines.extend(f"  {p}" for p in report["bars"]["phases"][:16])
    lines += ["", "Live counters:"]
    lines.extend(f"  {k}: {v}" for k, v in report["counters"].items())
    lines += ["", "GPU:", f"  {report['gpu']}"]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(prog="lucidota-big-board")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    report = {"ok": True, "bars": bars(), "counters": counters(), "gpu": gpu()}
    print(json.dumps(report, sort_keys=True) if args.json else render(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
