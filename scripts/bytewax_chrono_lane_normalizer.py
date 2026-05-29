#!/usr/bin/env python3
"""Chrono lane normalization stream runner.

RFC-CHRONO-001 requires Bytewax-owned normalization semantics, no Docker, and no
legacy queue-runtime dependency. Bytewax is optional at runtime on this host; this runner keeps the
same append -> normalize -> projection -> promotion-gate topology and executes it
as an Absurd-compatible durable foreground loop. If Bytewax is installed later,
the import probe records that fact without changing the persistence contract.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psycopg
from psycopg.rows import dict_row

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = ROOT / "06_SCHEMA" / "109_chrono_lane_split_projection_gate.sql"
OUT = ROOT / "05_OUTPUTS" / "chrono_ledger"
REFRESH = ROOT / "scripts" / "chrono_lane_split_projection_gate.py"
MAX_LOOP_CYCLES = int(os.environ.get("LUCIDOTA_CHRONO_NORMALIZER_MAX_LOOP_CYCLES", "100"))
MAX_REFRESH_TIMEOUT_SECONDS = int(os.environ.get("LUCIDOTA_CHRONO_NORMALIZER_MAX_REFRESH_TIMEOUT_SECONDS", "600"))


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def rel(path: str | Path) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def db_url(args: argparse.Namespace) -> str:
    return args.database_url or os.environ.get("KORPUS_DATABASE_URL") or os.environ.get("DATABASE_URL") or "postgresql:///lucidota_storage"


def bytewax_probe() -> dict[str, Any]:
    try:
        import bytewax  # type: ignore
        return {"available": True, "module": "bytewax", "version": getattr(bytewax, "__version__", None)}
    except Exception as exc:
        return {"available": False, "module": "bytewax", "reason": f"{type(exc).__name__}: {exc}", "fallback": "absurd_polling_stream"}


def apply_schema(database_url: str) -> None:
    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(SCHEMA.read_text(encoding="utf-8"))
        conn.commit()


def state_signature(database_url: str) -> dict[str, Any]:
    with psycopg.connect(database_url, row_factory=dict_row) as conn:
        row = conn.execute(
            """
            SELECT count(*)::bigint AS claim_count,
                   max(created_at) AS max_created_at,
                   count(*) FILTER (WHERE coalesce(invalid,false))::bigint AS invalid_count
            FROM lucidota_korpus.temporal_claim
            """
        ).fetchone()
        return {"claim_count": int(row["claim_count"]), "max_created_at": str(row["max_created_at"]), "invalid_count": int(row["invalid_count"])}


def run_refresh(database_url: str, timeout: int) -> dict[str, Any]:
    cmd = [sys.executable, str(REFRESH), "--database-url", database_url, "--execute"]
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, timeout=timeout)
    report_path = None
    for line in proc.stdout.splitlines():
        if line.startswith("REPORT_PATH="):
            report_path = line.split("=", 1)[1]
    return {
        "command": [str(x) for x in cmd],
        "returncode": proc.returncode,
        "report_path": report_path,
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
    }


def write_report(name: str, payload: dict[str, Any]) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"bytewax_chrono_lane_normalizer_{name}_{stamp()}.json"
    payload.setdefault("generated_at", now_iso())
    payload["report_path"] = rel(path)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"REPORT_PATH={rel(path)}")
    return path


def once(args: argparse.Namespace) -> int:
    database_url = db_url(args)
    args.refresh_timeout_seconds = max(1, min(args.refresh_timeout_seconds, MAX_REFRESH_TIMEOUT_SECONDS))
    apply_schema(database_url)
    before = state_signature(database_url)
    refresh = run_refresh(database_url, args.refresh_timeout_seconds) if args.execute else {"returncode": 0, "dry_run": True}
    after = state_signature(database_url)
    report = {
        "schema": "lucidota.bytewax_chrono_lane_normalizer.once.v1",
        "action": "once",
        "execute_performed": bool(args.execute),
        "queue_spine_used": False,
        "docker_used": False,
        "bytewax_probe": bytewax_probe(),
        "topology": [
            "temporal_claim append",
            "chrono_claim_normalized",
            "chrono_batch_cluster",
            "chrono_file_projection",
            "graph_promotion_candidate",
            "promotion gate",
        ],
        "before": before,
        "refresh": refresh,
        "after": after,
        "status": "PASS" if refresh.get("returncode") == 0 else "FAIL",
    }
    write_report("once_execute" if args.execute else "once_dry_run", report)
    print("BYTEWAX_CHRONO_NORMALIZER=" + report["status"])
    print("CLAIM_COUNT=" + str(after["claim_count"]))
    return 0 if report["status"] == "PASS" else 4


def loop(args: argparse.Namespace) -> int:
    database_url = db_url(args)
    args.max_cycles = max(1, min(args.max_cycles, MAX_LOOP_CYCLES))
    args.refresh_timeout_seconds = max(1, min(args.refresh_timeout_seconds, MAX_REFRESH_TIMEOUT_SECONDS))
    apply_schema(database_url)
    probe = bytewax_probe()
    last_sig: dict[str, Any] | None = None
    cycles = 0
    failures = 0
    print(json.dumps({"event": "chrono_lane_stream_start", "time": now_iso(), "bytewax_probe": probe, "queue_spine_used": False, "docker_used": False, "max_cycles": args.max_cycles}, sort_keys=True))
    while cycles < args.max_cycles:
        cycles += 1
        sig = state_signature(database_url)
        changed = sig != last_sig
        if changed or args.force_every_cycle:
            try:
                refresh = run_refresh(database_url, args.refresh_timeout_seconds) if args.execute else {"returncode": 0, "dry_run": True}
                ok = refresh.get("returncode") == 0
                failures = 0 if ok else failures + 1
                print(json.dumps({"event": "chrono_lane_refresh", "time": now_iso(), "cycle": cycles, "changed": changed, "status": "PASS" if ok else "FAIL", "signature": sig, "report_path": refresh.get("report_path")}, sort_keys=True))
                if not ok and failures >= args.max_failures:
                    write_report("loop_failed", {"cycles": cycles, "failures": failures, "last_refresh": refresh, "last_signature": sig, "bytewax_probe": probe, "status": "FAIL"})
                    return 4
            except Exception as exc:
                failures += 1
                print(json.dumps({"event": "chrono_lane_refresh_exception", "time": now_iso(), "cycle": cycles, "error": f"{type(exc).__name__}: {exc}"}, sort_keys=True))
                if failures >= args.max_failures:
                    write_report("loop_exception", {"cycles": cycles, "failures": failures, "error": repr(exc), "last_signature": sig, "bytewax_probe": probe, "status": "FAIL"})
                    return 4
        last_sig = sig
        if cycles < args.max_cycles:
            time.sleep(args.idle_sleep)
    write_report("loop_complete", {"cycles": cycles, "failures": failures, "last_signature": last_sig, "bytewax_probe": probe, "status": "PASS"})
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Bytewax-compatible Chrono lane normalization runner")
    parser.add_argument("--database-url")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--refresh-timeout-seconds", type=int, default=180, help="bounded refresh subprocess timeout")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("once")
    p.set_defaults(func=once)
    p = sub.add_parser("loop")
    p.add_argument("--idle-sleep", type=float, default=5.0)
    p.add_argument("--max-cycles", type=int, default=1, help="cycle cap for ephemeral execution; values below 1 are clamped to 1")
    p.add_argument("--max-failures", type=int, default=3)
    p.add_argument("--force-every-cycle", action="store_true")
    p.set_defaults(func=loop)
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
