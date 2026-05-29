#!/usr/bin/env python3
"""File-backed ABSURD abductive operating ledger."""
from __future__ import annotations
import argparse, json, hashlib, subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "05_OUTPUTS" / "absurd_abductive"
RUNTIME = ROOT / "04_RUNTIME" / "absurd_abductive"
RECEIPT_GLOBS = [
    "05_OUTPUTS/model_invocation_audits/model_invocation_audit_*.json",
    "05_OUTPUTS/goals/swarm_usage_ledger_*.json",
    "05_OUTPUTS/hunch_hypertimeline/*.json",
    "05_OUTPUTS/indy_reads/**/*.json",
    "05_OUTPUTS/krampuschewing/*.json",
    "05_OUTPUTS/project2501_board_stream/*.json",
    "05_OUTPUTS/slop_audit/*.json",
]

def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")

def rel(path: Path | str) -> str:
    try: return str(Path(path).resolve().relative_to(ROOT))
    except Exception: return str(path)

def load_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        return data if isinstance(data, dict) else {}
    except Exception as exc:
        return {"schema":"lucidota.unreadable_receipt.v1", "status":"FAIL", "error":f"{type(exc).__name__}:{exc}"}

def latest_receipts(limit: int = 80) -> list[Path]:
    paths: list[Path] = []
    for pat in RECEIPT_GLOBS:
        paths.extend(ROOT.glob(pat))
    uniq = sorted(set(paths), key=lambda p: p.stat().st_mtime, reverse=True)
    return uniq[:limit]

def normalize_receipt(path: Path) -> dict[str, Any]:
    data = load_json(path)
    schema = str(data.get("schema") or "unknown")
    status = str(data.get("verdict") or data.get("status") or data.get("audit_status") or "UNKNOWN")
    row = {
        "schema":"lucidota.absurd_abductive.ledger_row.v1",
        "row_id":"receipt:" + hashlib.sha256(str(path).encode()).hexdigest()[:16],
        "object_type":"Receipt",
        "source_path": rel(path),
        "source_schema": schema,
        "status": status,
        "verdict": status,
        "canonical_graph_writes": bool(data.get("canonical_graph_writes_performed") or data.get("canonical_graph_writes")),
        "external_effects": bool(data.get("external_effects", False)),
        "summary": schema,
        "counts": {},
        "raw_keys": sorted(data.keys())[:40],
    }
    if "model_invocation_audit" in schema:
        missing = int(data.get("missing_dedicated_model_audit_blocks") or 0)
        row.update({"object_type":"ModelAuditBlock", "summary":f"model audit {status}; missing={missing}", "counts":{"missing_model_audit_blocks":missing, "blocks":len(data.get("five_task_audit_blocks", []))}})
    elif "hunch_postgres_ingest" in schema:
        row.update({"object_type":"GraphCandidate", "summary":"hunch ingest graph candidates staged", "counts":{"records_upserted":int(data.get("records_upserted") or 0), "graph_candidates":int(data.get("graph_candidates_written") or 0)}})
    elif "indy_reads" in schema:
        row.update({"object_type":"Evidence", "summary":"Indy corpus/evidence receipt", "counts":{"chunks":int(data.get("chunks_written") or 0)}})
    elif "swarm_usage_ledger" in schema:
        row.update({"object_type":"SwarmUsageLedger", "summary":"receipt-backed model usage ledger"})
    elif "bytewax" in schema or "board_stream" in schema:
        row.update({"object_type":"StreamEvent", "summary":"Bytewax/project2501 stream receipt"})
    return row

def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    blockers: list[dict[str, Any]] = []
    completed: list[dict[str, Any]] = []
    graph_candidates = 0
    for row in rows:
        counts[row["object_type"]] = counts.get(row["object_type"], 0) + 1
        graph_candidates += int(row.get("counts", {}).get("graph_candidates", 0) or 0)
        if row.get("canonical_graph_writes"):
            blockers.append({"class":"graph_safety", "severity":100, "summary":"canonical graph write observed", "source":row["source_path"]})
        if row["object_type"] == "ModelAuditBlock" and int(row.get("counts", {}).get("missing_model_audit_blocks", 0) or 0) > 0:
            blockers.append({"class":"model_audit", "severity":95, "summary":row["summary"], "source":row["source_path"]})
        if str(row.get("status")) in {"PASS", "verified", "executed", "MODEL_AUDIT_RECEIPT_PRESENT"}:
            completed.append({"summary":row["summary"], "source":row["source_path"]})
    if graph_candidates:
        counts["GraphCandidate"] = max(counts.get("GraphCandidate", 0), graph_candidates)
    return {
        "schema":"lucidota.absurd_abductive.board.v1",
        "generated_at_utc": now(),
        "object_counts": {"Receipt": len(rows), **counts},
        "open_blockers": sorted(blockers, key=lambda b: b.get("severity",0), reverse=True),
        "completed_work": completed[:20],
        "pending_hypothesis_work_queues": [],
        "canonical_graph_materialization": False,
        "canonical_graph_writes": any(r.get("canonical_graph_writes") for r in rows),
        "external_effects": any(r.get("external_effects") for r in rows),
        "ledger_rows": rows,
    }

def build_board(limit: int = 80) -> dict[str, Any]:
    return summarize_rows([normalize_receipt(p) for p in latest_receipts(limit)])

def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str)+"\n", encoding="utf-8")

def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows: f.write(json.dumps(row, sort_keys=True, default=str)+"\n")

def write_board(board: dict[str, Any]) -> Path:
    path = OUT / f"board_{stamp()}.json"
    board["receipt_path"] = rel(path)
    write_json(path, board)
    write_json(OUT / "board_latest.json", board)
    return path

def init(dry_run: bool) -> dict[str, Any]:
    payload={"schema":"lucidota.absurd_abductive.init.v1","generated_at_utc":now(),"dry_run":dry_run,"backend":"jsonl","runtime_dir":rel(RUNTIME),"output_dir":rel(OUT),"status":"PASS"}
    if not dry_run: RUNTIME.mkdir(parents=True, exist_ok=True); OUT.mkdir(parents=True, exist_ok=True)
    path=OUT/f"ledger_init_{stamp()}.json"; payload["receipt_path"]=rel(path); write_json(path,payload); return payload

def ingest_receipts(since: str, dry_run: bool) -> dict[str, Any]:
    rows=[normalize_receipt(p) for p in latest_receipts()]
    if not dry_run: write_jsonl(RUNTIME/"ledger.jsonl", rows)
    payload={"schema":"lucidota.absurd_abductive.ingest_receipts.v1","generated_at_utc":now(),"since":since,"dry_run":dry_run,"rows":len(rows),"status":"PASS","canonical_graph_writes":False}
    path=OUT/f"ingest_receipts_{stamp()}.json"; payload["receipt_path"]=rel(path); write_json(path,payload); return payload

def main() -> int:
    ap=argparse.ArgumentParser(description="File-backed ABSURD abductive ledger")
    sub=ap.add_subparsers(dest="cmd", required=True)
    p=sub.add_parser("init"); p.add_argument("--dry-run", action="store_true")
    p=sub.add_parser("ingest-receipts"); p.add_argument("--since", default="latest"); p.add_argument("--dry-run", action="store_true")
    p=sub.add_parser("board"); p.add_argument("--json", action="store_true")
    p=sub.add_parser("next-moves"); p.add_argument("--json", action="store_true")
    args=ap.parse_args()
    if args.cmd=="init": payload=init(args.dry_run)
    elif args.cmd=="ingest-receipts": payload=ingest_receipts(args.since, args.dry_run)
    elif args.cmd=="board": payload=build_board(); path=write_board(payload); print("BOARD_PATH="+rel(path))
    else:
        from absurd_next_move_engine import generate_next_moves
        payload={"schema":"lucidota.absurd_abductive.next_moves_receipt.v1","generated_at_utc":now(),"next_moves":generate_next_moves(build_board()),"status":"PASS"}
        path=OUT/f"next_moves_{stamp()}.json"; payload["receipt_path"]=rel(path); write_json(path,payload)
    if getattr(args,"json",False): print(json.dumps(payload, sort_keys=True, default=str))
    if payload.get("receipt_path"): print("REPORT_PATH="+payload["receipt_path"])
    print("ABSURD_ABDUCTIVE_LEDGER="+str(payload.get("status","PASS")))
    return 0
if __name__ == "__main__": raise SystemExit(main())
