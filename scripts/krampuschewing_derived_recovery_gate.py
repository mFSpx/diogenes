#!/usr/bin/env python3
"""Derived KRAMPUSCHEWING recovery gate from existing index only."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "05_OUTPUTS" / "krampuschewing"


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def rel(path: Path | str) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def run(cmd: list[str]) -> dict[str, Any]:
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
    return {"command": cmd, "rc": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr}


def kv(stdout: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in stdout.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def count_lines(path: Path) -> int:
    with path.open("rb") as fh:
        return sum(1 for _ in fh)


def chroma_hits(path: Path) -> int:
    hits = 0
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if "CHROMADB" in line or "chroma.sqlite3" in line:
                hits += 1
    return hits


def main() -> int:
    ap = argparse.ArgumentParser(description="Run derived KRAMPUSCHEWING recovery from existing index, no filesystem re-index")
    ap.add_argument("--index", required=True)
    ap.add_argument("--summary", required=True)
    ap.add_argument("--large-file-validation", required=True)
    args = ap.parse_args()

    source_index = resolve(args.index)
    source_summary = resolve(args.summary)
    large_validation = resolve(args.large_file_validation)
    source_summary_data = json.loads(source_summary.read_text(encoding="utf-8"))
    blockers: list[str] = []
    commands_run: list[dict[str, Any]] = []
    receipts_written: list[str] = []

    reclass_cmd = [sys.executable, "scripts/krampuschewing_reclassify_index.py", "--index", rel(source_index), "--large-file-validation", rel(large_validation)]
    r1 = run(reclass_cmd)
    commands_run.append(r1)
    if r1["rc"] != 0:
        blockers.append(f"reclassify_failed_rc:{r1['rc']}")
    r1kv = kv(r1["stdout"])
    normalized_index = resolve(r1kv.get("NORMALIZED_INDEX_PATH", "")) if r1kv.get("NORMALIZED_INDEX_PATH") else None
    normalized_summary = resolve(r1kv.get("NORMALIZED_SUMMARY_PATH", "")) if r1kv.get("NORMALIZED_SUMMARY_PATH") else None
    if normalized_index and normalized_index.exists():
        receipts_written.append(rel(normalized_index))
    else:
        blockers.append("normalized_index_missing")
    if normalized_summary and normalized_summary.exists():
        receipts_written.append(rel(normalized_summary))
    else:
        blockers.append("normalized_summary_missing")

    graph_path = graph_summary_path = river_path = river_summary_path = None
    if normalized_index and normalized_index.exists():
        graph_cmd = [sys.executable, "scripts/krampuschewing_graph_stage.py", "--index", rel(normalized_index)]
        r2 = run(graph_cmd)
        commands_run.append(r2)
        if r2["rc"] != 0:
            blockers.append(f"graph_stage_failed_rc:{r2['rc']}")
        r2kv = kv(r2["stdout"])
        graph_path = resolve(r2kv.get("GRAPH_CANDIDATES_PATH", "")) if r2kv.get("GRAPH_CANDIDATES_PATH") else None
        graph_summary_path = resolve(r2kv.get("GRAPH_CANDIDATES_SUMMARY_PATH", "")) if r2kv.get("GRAPH_CANDIDATES_SUMMARY_PATH") else None
        for p, name in [(graph_path, "graph_candidates"), (graph_summary_path, "graph_summary")]:
            if p and p.exists():
                receipts_written.append(rel(p))
            else:
                blockers.append(f"{name}_missing")

        river_cmd = [sys.executable, "scripts/krampuschewing_river_rows.py", "--index", rel(normalized_index)]
        r3 = run(river_cmd)
        commands_run.append(r3)
        if r3["rc"] != 0:
            blockers.append(f"river_rows_failed_rc:{r3['rc']}")
        r3kv = kv(r3["stdout"])
        river_path = resolve(r3kv.get("RIVER_TRAINING_CANDIDATES_PATH", "")) if r3kv.get("RIVER_TRAINING_CANDIDATES_PATH") else None
        river_summary_path = resolve(r3kv.get("RIVER_TRAINING_SUMMARY_PATH", "")) if r3kv.get("RIVER_TRAINING_SUMMARY_PATH") else None
        for p, name in [(river_path, "river_candidates"), (river_summary_path, "river_summary")]:
            if p and p.exists():
                receipts_written.append(rel(p))
            else:
                blockers.append(f"{name}_missing")

    normalized_summary_data = json.loads(normalized_summary.read_text(encoding="utf-8")) if normalized_summary and normalized_summary.exists() else {}
    graph_summary_data = json.loads(graph_summary_path.read_text(encoding="utf-8")) if graph_summary_path and graph_summary_path.exists() else {}
    river_summary_data = json.loads(river_summary_path.read_text(encoding="utf-8")) if river_summary_path and river_summary_path.exists() else {}

    rows_read = int(normalized_summary_data.get("rows_read") or 0)
    rows_normalized = int(normalized_summary_data.get("rows_normalized") or 0)
    graph_count = count_lines(graph_path) if graph_path and graph_path.exists() else 0
    river_count = count_lines(river_path) if river_path and river_path.exists() else 0
    active_db_count = int(normalized_summary_data.get("active_runtime_db_risk_count") or 0)
    graph_chroma_hits = chroma_hits(graph_path) if graph_path and graph_path.exists() else -1
    river_chroma_hits = chroma_hits(river_path) if river_path and river_path.exists() else -1
    graph_db_excluded = active_db_count == 1 and graph_chroma_hits == 0 and bool(graph_summary_data.get("active_runtime_db_excluded"))
    river_db_excluded = active_db_count == 1 and river_chroma_hits == 0 and bool(river_summary_data.get("active_runtime_db_excluded"))

    if rows_read != count_lines(source_index):
        blockers.append(f"rows_read_mismatch:source={count_lines(source_index)}:normalized={rows_read}")
    if rows_read != rows_normalized:
        blockers.append(f"rows_normalized_mismatch:{rows_read}!={rows_normalized}")
    if graph_count <= 0:
        blockers.append("graph_candidates_empty")
    if river_count <= 0:
        blockers.append("river_candidates_empty")
    if not graph_db_excluded:
        blockers.append(f"active_db_not_excluded_from_graph:hits={graph_chroma_hits}:active_db_count={active_db_count}")
    if not river_db_excluded:
        blockers.append(f"active_db_not_excluded_from_river:hits={river_chroma_hits}:active_db_count={active_db_count}")

    source_partial = source_summary_data.get("verdict") != "PASS"
    known_gaps = []
    if source_partial:
        known_gaps.append(f"source_master_summary_verdict:{source_summary_data.get('verdict')}")
        known_gaps.append("source_index_partial_due_large_hash_policy")

    hard_blockers = [b for b in blockers if not b.startswith("source_master_summary")]
    verdict = "FAIL" if hard_blockers else "PARTIAL_FAIL" if source_partial else "PASS"

    receipt = {
        "schema": "lucidota.krampuschewing.derived_recovery_receipt.v1",
        "generated_at_utc": now(),
        "verdict": verdict,
        "source_index": rel(source_index),
        "source_summary": rel(source_summary),
        "source_large_file_validation": rel(large_validation),
        "normalized_index": rel(normalized_index) if normalized_index else None,
        "normalized_summary": rel(normalized_summary) if normalized_summary else None,
        "graph_candidates": rel(graph_path) if graph_path else None,
        "graph_candidates_summary": rel(graph_summary_path) if graph_summary_path else None,
        "river_training_candidates": rel(river_path) if river_path else None,
        "river_training_summary": rel(river_summary_path) if river_summary_path else None,
        "rows_read": rows_read,
        "rows_normalized": rows_normalized,
        "graph_candidates_count": graph_count,
        "river_training_candidates_count": river_count,
        "active_runtime_db_risk_count": active_db_count,
        "active_runtime_db_excluded_from_graph": graph_db_excluded,
        "active_runtime_db_excluded_from_river": river_db_excluded,
        "graph_chromadb_hits": graph_chroma_hits,
        "river_chromadb_hits": river_chroma_hits,
        "files_moved": [],
        "files_deleted": [],
        "full_hashing_performed": False,
        "archive_unpacking_performed": False,
        "river_training_performed": False,
        "canonical_graph_materialization": False,
        "canonical_graph_writes": False,
        "commands_run": commands_run,
        "receipts_written": receipts_written,
        "blockers": hard_blockers,
        "known_gaps": known_gaps,
        "next_smallest_safe_work": "Chunk-hash large archives with an allowlist that excludes active runtime DBs, then optionally promote safe proof-kernel custody candidates.",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    receipt_path = OUT / f"krampuschewing_derived_recovery_receipt_{stamp()}.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=False, ensure_ascii=False), encoding="utf-8")
    print("REPORT_PATH=" + rel(receipt_path))
    print("KRAMPUSCHEWING_DERIVED_RECOVERY_GATE=" + verdict)
    print("NORMALIZED_INDEX=" + str(receipt["normalized_index"]))
    print("NORMALIZED_SUMMARY=" + str(receipt["normalized_summary"]))
    print("GRAPH_CANDIDATES=" + str(receipt["graph_candidates"]))
    print("GRAPH_CANDIDATES_SUMMARY=" + str(receipt["graph_candidates_summary"]))
    print("RIVER_TRAINING_CANDIDATES=" + str(receipt["river_training_candidates"]))
    print("RIVER_TRAINING_SUMMARY=" + str(receipt["river_training_summary"]))
    print("ROWS_READ=" + str(rows_read))
    print("ROWS_NORMALIZED=" + str(rows_normalized))
    print("GRAPH_CANDIDATES_COUNT=" + str(graph_count))
    print("RIVER_TRAINING_CANDIDATES_COUNT=" + str(river_count))
    print("ACTIVE_RUNTIME_DB_EXCLUDED_FROM_GRAPH=" + str(graph_db_excluded).lower())
    print("ACTIVE_RUNTIME_DB_EXCLUDED_FROM_RIVER=" + str(river_db_excluded).lower())
    if hard_blockers:
        print("BLOCKERS=" + json.dumps(hard_blockers, sort_keys=True))
    if known_gaps:
        print("KNOWN_GAPS=" + json.dumps(known_gaps, sort_keys=True))
    return 0 if verdict == "PASS" else 2 if verdict == "PARTIAL_FAIL" else 1


if __name__ == "__main__":
    raise SystemExit(main())
