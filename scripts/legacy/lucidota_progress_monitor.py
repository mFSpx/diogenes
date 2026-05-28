#!/usr/bin/env python3
"""LUCIDOTA active-loop progress monitor.

Reads process state, ABSURD queue rows, Bytewax/ABCD ledgers, and GO ontology
fidelity without mutating worker queues or canonical graph tables.  Default mode
prints one terminal frame; `loop` repeats forever.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psycopg
from psycopg.rows import dict_row

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

STATE_DSN = os.environ.get("LUCIDOTA_ABSURD_STATE_DSN") or os.environ.get("LUCIDOTA_GO_STATE_DSN") or os.environ.get("DATABASE_URL") or os.environ.get("DBOS_SYSTEM_DATABASE_URL") or "postgresql:///lucidota_state"
STORAGE_DSN = os.environ.get("LUCIDOTA_GO_STORAGE_DSN") or "postgresql:///lucidota_storage"
OUT = ROOT / "05_OUTPUTS" / "progress_monitor"
ABCD_LEDGER = ROOT / "05_OUTPUTS" / "work_loops" / "updated_abcd_execution_ledger.jsonl"
ABCD_DEAD = ROOT / "05_OUTPUTS" / "work_loops" / "updated_abcd_dead_letter.jsonl"
ABCD_SUMMARY_DIR = ROOT / "05_OUTPUTS" / "work_loops" / "updated_abcd_reports"
DUCKDB_PATH = ROOT / "04_RUNTIME" / "bytewax_abductive_blender" / "abductive_blender.duckdb"

GO25 = [
    "ENTITY", "ATTRIBUTE", "RELATIONSHIP", "FRICTION", "LEVERAGE", "VISIBILITY",
    "ACTION", "EVENT", "TIME", "PATTERN", "HYPOTHESIS", "CLAIM", "EVIDENCE",
    "ATOMIC_ID", "SIGNAL", "GLOW", "TERM", "TOOL", "ALGORITHM", "NAUGHTY",
    "NICE", "GROUP", "OPERATOR", "MODE", "COMMENT",
]
PLURAL_OPERATOR_ALIASES = {"ACTIONS": "ACTION", "EVENTS": "EVENT", "HYPOTHESES": "HYPOTHESIS"}


def now_z() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def rel(path: str | Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def rows(dsn: str, sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
    try:
        with psycopg.connect(dsn, row_factory=dict_row) as conn:
            return [dict(r) for r in conn.execute(sql, params).fetchall()]
    except Exception as exc:
        return [{"error": f"{type(exc).__name__}: {exc}"}]


def scalar(dsn: str, sql: str, params: tuple[Any, ...] = ()) -> Any:
    try:
        with psycopg.connect(dsn) as conn:
            row = conn.execute(sql, params).fetchone()
            return row[0] if row else None
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"


def process_snapshot() -> dict[str, Any]:
    cp = subprocess.run(["ps", "-eo", "pid,ppid,user,stat,etime,rss,cmd"], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    needles = {
        "absurd_workers": "unified_absurd_ingest_worker.py work",
        "bytewax_blenders": "bytewax_abductive_blender.py loop",
        "abcd_runners": "updated_abcd_sequence_runner.py",
        "krampus_rechronologizers": "krampus_rechronologize.py",
        "krampus_time_machines": "krampus_time_machine.py",
        "river_reflexes": "lucidota_river_reflex.py",
        "treelite_routers": "lucidota_treelite_router.py",
    }
    found = {k: [] for k in needles}
    for line in cp.stdout.splitlines()[1:]:
        for key, needle in needles.items():
            if needle in line and "rg " not in line:
                parts = line.split(None, 6)
                if len(parts) == 7:
                    if any(skip in parts[6] for skip in ("bash -c", "grep ", "awk ", "sed ")):
                        continue
                    found[key].append({"pid": int(parts[0]), "ppid": int(parts[1]), "user": parts[2], "stat": parts[3], "etime": parts[4], "rss_kb": int(parts[5]), "cmd": parts[6]})
                else:
                    found[key].append({"raw": line})
    return found


def absurd_snapshot() -> dict[str, Any]:
    return {
        "counts": rows(STATE_DSN, """
            SELECT workflow_name,status,count(*)::int AS n,min(created_at)::text AS oldest,max(updated_at)::text AS newest
            FROM lucidota_control.absurd_workflow GROUP BY workflow_name,status ORDER BY workflow_name,status
        """),
        "running": rows(STATE_DSN, """
            SELECT workflow_id::text, workflow_name, status, locked_by, locked_at::text, lease_expires_at::text, state->>'step' AS step
            FROM lucidota_control.absurd_workflow WHERE status='running' ORDER BY updated_at DESC LIMIT 20
        """),
        "skip_locked_present": "FOR UPDATE SKIP LOCKED" in (ROOT / "scripts" / "unified_absurd_ingest_worker.py").read_text(encoding="utf-8", errors="replace"),
    }


def blender_snapshot() -> dict[str, Any]:
    return {
        "hint_counts": rows(STATE_DSN, """
            SELECT epistemic_flag,count(*)::int AS n,max(created_at)::text AS newest
            FROM lucidota_learning.bytewax_abductive_hint GROUP BY epistemic_flag ORDER BY epistemic_flag
        """),
        "stream_runs": rows(STATE_DSN, """
            SELECT status,count(*)::int AS n,max(created_at)::text AS newest FROM lucidota_learning.bytewax_stream_run GROUP BY status ORDER BY status
        """),
        "cursors": rows(STATE_DSN, "SELECT cursor_name,last_seen_at::text,last_seen_ref,updated_at::text FROM lucidota_learning.bytewax_abductive_cursor ORDER BY cursor_name"),
        "duckdb": {"path": rel(DUCKDB_PATH), "exists": DUCKDB_PATH.exists(), "bytes": DUCKDB_PATH.stat().st_size if DUCKDB_PATH.exists() else 0, "memory_limit_env": os.environ.get("LUCIDOTA_DUCKDB_MEMORY_LIMIT", "1536MB")},
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except Exception as exc:
            out.append({"json_error": str(exc), "raw": line[:500]})
    return out


def latest_summary() -> dict[str, Any] | None:
    xs = sorted(ABCD_SUMMARY_DIR.glob("updated_abcd_sequence_summary_*.json"), key=lambda p: p.stat().st_mtime)
    if not xs:
        return None
    try:
        data = json.loads(xs[-1].read_text(encoding="utf-8"))
        data["path"] = rel(xs[-1])
        return data
    except Exception as exc:
        return {"path": rel(xs[-1]), "error": str(exc)}


def abcd_snapshot() -> dict[str, Any]:
    ledger = read_jsonl(ABCD_LEDGER)
    dead = read_jsonl(ABCD_DEAD)
    cycles: dict[str, Counter[str]] = defaultdict(Counter)
    for row in ledger:
        cyc = str(row.get("cycle", "unknown"))
        cycles[cyc]["rows"] += 1
        cycles[cyc]["counted" if row.get("counted") else "dead_lettered"] += 1
    tail = ledger[-10:]
    return {
        "ledger_path": rel(ABCD_LEDGER),
        "dead_letter_path": rel(ABCD_DEAD),
        "ledger_rows": len(ledger),
        "counted_total": sum(1 for r in ledger if r.get("counted")),
        "dead_letter_total": sum(1 for r in ledger if not r.get("counted")),
        "dead_letter_file_rows": len(dead),
        "cycles": {k: dict(v) for k, v in sorted(cycles.items(), key=lambda kv: kv[0])},
        "last_sequence_index": tail[-1].get("sequence_index") if tail else None,
        "last_cycle": tail[-1].get("cycle") if tail else None,
        "last_target": tail[-1].get("target_key") if tail else None,
        "last_epistemic_flag": tail[-1].get("epistemic_flag") if tail else None,
        "tail": [{"sequence_index": r.get("sequence_index"), "cycle": r.get("cycle"), "target_key": r.get("target_key"), "counted": r.get("counted"), "epistemic_flag": r.get("epistemic_flag"), "blocked_by": r.get("blocked_by")} for r in tail],
        "latest_summary": latest_summary(),
    }


def ontology_snapshot() -> dict[str, Any]:
    official_ok = False
    official_terms: list[str] = []
    try:
        official = json.loads((ROOT / "OFFICIAL_ONTOLOGY.json").read_text(encoding="utf-8"))
        official_terms = list(official.get("active_terms") or official.get("terms") or [])[:25]
        official_ok = official_terms == GO25
    except Exception:
        pass
    term_rows = rows(STORAGE_DSN, "SELECT term FROM lucidota_go.term_registry ORDER BY term")
    db_terms = [r.get("term") for r in term_rows if "term" in r]
    db_unknown = sorted(set(t for t in db_terms if t not in GO25))
    staging_unknown = rows(STORAGE_DSN, """
        SELECT proposed_term, count(*)::int AS n
        FROM lucidota_go.staging_packet
        WHERE proposed_term IS NOT NULL AND proposed_term <> ALL(%s)
        GROUP BY proposed_term ORDER BY proposed_term
    """, (GO25,))
    staging_counts = rows(STORAGE_DSN, """
        SELECT proposed_term,count(*)::int AS n,max(created_at)::text AS newest
        FROM lucidota_go.staging_packet GROUP BY proposed_term ORDER BY proposed_term
    """)
    return {
        "canonical_terms": GO25,
        "plural_operator_aliases_observed_as_noncanonical_input_only": PLURAL_OPERATOR_ALIASES,
        "official_ontology_exact_go25": official_ok,
        "official_terms_sample": official_terms,
        "term_registry_unknown_terms": db_unknown,
        "staging_packet_unknown_terms": staging_unknown,
        "staging_packet_term_counts": staging_counts,
        "raw_visibility_policy": "internal local tables preserve literal values; outbound redaction only at release edges",
    }


def snapshot() -> dict[str, Any]:
    return {
        "schema": "lucidota.progress_monitor.v1",
        "generated_at": now_z(),
        "processes": process_snapshot(),
        "absurd": absurd_snapshot(),
        "bytewax_blender": blender_snapshot(),
        "abcd": abcd_snapshot(),
        "ontology": ontology_snapshot(),
    }


def print_frame(payload: dict[str, Any], json_mode: bool = False) -> None:
    if json_mode:
        print(json.dumps(payload, indent=2, ensure_ascii=False, default=str), flush=True)
        return
    abcd = payload["abcd"]
    proc = payload["processes"]
    print(f"===== LUCIDOTA_PROGRESS_MONITOR :: {payload['generated_at']} =====", flush=True)
    print(
        " ".join([
            f"ABSURD_WORKERS={len(proc.get('absurd_workers', []))}",
            f"BYTEWAX_BLENDERS={len(proc.get('bytewax_blenders', []))}",
            f"ABCD_RUNNERS={len(proc.get('abcd_runners', []))}",
            f"KRAMPUS_RECHRONOLOGIZERS={len(proc.get('krampus_rechronologizers', []))}",
            f"KRAMPUS_TIME_MACHINES={len(proc.get('krampus_time_machines', []))}",
            f"RIVER_REFLEXES={len(proc.get('river_reflexes', []))}",
            f"TREELITE_ROUTERS={len(proc.get('treelite_routers', []))}",
        ]),
        flush=True,
    )
    print(f"ABCD_LEDGER_ROWS={abcd['ledger_rows']} ABCD_COUNTED_TOTAL={abcd['counted_total']} ABCD_DEAD_LETTER_TOTAL={abcd['dead_letter_total']} ABCD_LAST_CYCLE={abcd['last_cycle']} ABCD_LAST_SEQUENCE_INDEX={abcd['last_sequence_index']} ABCD_LAST_TARGET={abcd['last_target']}", flush=True)
    print(f"ABSURD_SKIP_LOCKED={payload['absurd']['skip_locked_present']} ONTOLOGY_GO25_EXACT={payload['ontology']['official_ontology_exact_go25']} UNKNOWN_DB_TERMS={len(payload['ontology']['term_registry_unknown_terms'])} UNKNOWN_STAGING_TERMS={len(payload['ontology']['staging_packet_unknown_terms'])}", flush=True)
    print("PROCESS_FRAME=" + json.dumps(proc, ensure_ascii=False, default=str), flush=True)
    print("ABSURD_FRAME=" + json.dumps(payload["absurd"], ensure_ascii=False, default=str), flush=True)
    print("BYTEWAX_FRAME=" + json.dumps(payload["bytewax_blender"], ensure_ascii=False, default=str), flush=True)
    print("ABCD_FRAME=" + json.dumps(abcd, ensure_ascii=False, default=str), flush=True)
    print("ONTOLOGY_FRAME=" + json.dumps(payload["ontology"], ensure_ascii=False, default=str), flush=True)
    print("===== /LUCIDOTA_PROGRESS_MONITOR =====", flush=True)


def write_report(payload: dict[str, Any]) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"lucidota_progress_monitor_{stamp()}.json"
    payload["report_path"] = rel(path)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description="LUCIDOTA active progress monitor")
    ap.add_argument("action", nargs="?", default="once", choices=["once", "loop"])
    ap.add_argument("--interval", type=float, default=10.0)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--write-report", action="store_true")
    args = ap.parse_args()
    while True:
        payload = snapshot()
        if args.write_report:
            write_report(payload)
        print_frame(payload, json_mode=args.json)
        if args.action == "once":
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
