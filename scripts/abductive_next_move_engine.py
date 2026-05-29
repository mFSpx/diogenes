#!/usr/bin/env python3
"""Rank next safe moves from abductive DB OS board state."""
from __future__ import annotations
import argparse, json, hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"05_OUTPUTS"/"abductive_db_os"

def now(): return datetime.now(timezone.utc).isoformat().replace("+00:00","Z")
def stamp(): return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
def rel(p):
    try: return str(Path(p).resolve().relative_to(ROOT))
    except Exception: return str(p)
def move_id(text: str) -> str: return "NM-"+hashlib.sha256(text.encode()).hexdigest()[:12]

def next_move(cls: str, target: str, why: str, gain: float, risk: str="low", commands: list[str]|None=None) -> dict[str, Any]:
    return {"schema":"lucidota.next_move.v1","move_id":move_id(cls+target+why),"source":"abductive_next_move_engine","class":cls,"target":target,"why_now":why,"expected_gain":gain,"risk":risk,"requires_operator_approval":False,"forbidden_effects":["canonical_graph_write","external_send","evidence_delete"],"commands":commands or [],"receipt_expected":f"05_OUTPUTS/abductive_db_os/{cls}_*.json","status":"candidate"}

def generate_next_moves(board: dict[str, Any]) -> list[dict[str, Any]]:
    moves=[]
    blockers=board.get("open_blockers") or []
    if any(b.get("class")=="model_audit" for b in blockers):
        moves.append(next_move("model_audit","scripts/model_invocation_audit.py","complete 5-task audit block has missing/invalid dedicated audit",0.99,"low",["python3 scripts/model_invocation_audit.py --json || true"]))
    if board.get("canonical_graph_writes"):
        moves.append(next_move("audit","canonical_graph_writes","canonical graph write observed in receipt state",1.0,"high",[]))
    if int((board.get("object_counts") or {}).get("GraphCandidate",0))>0:
        moves.append(next_move("graph_stage","graph_candidates","staged candidates exist and need review before promotion",0.72,"medium",["python3 scripts/graph_promotion_gate.py --help"]))
    moves.append(next_move("stream_check","project2501-bytewax-board-stream.service","keep live DB event organ typed and bounded",0.55,"low",["python3 scripts/bytewax_db_os_stream_audit.py --json"]))
    moves.append(next_move("status_update","00_PROJECT_BRAIN/STATUS_LEDGER.md","preserve operator-visible board truth from receipts",0.45,"low",["python3 scripts/lucidota_status_ledger.py --check"]))
    return sorted(moves, key=lambda m: float(m["expected_gain"]), reverse=True)[:5]

def load_board(arg: str) -> dict[str, Any]:
    if arg == "latest":
        p=OUT/"board_latest.json"
        if p.exists(): return json.loads(p.read_text())
    p=Path(arg)
    if not p.is_absolute(): p=ROOT/p
    return json.loads(p.read_text()) if p.exists() else {"open_blockers":[],"object_counts":{}}

def write_json(path: Path, payload: dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(payload,indent=2,sort_keys=True,default=str)+"\n",encoding="utf-8")

def main() -> int:
    ap=argparse.ArgumentParser(); ap.add_argument("--board", default="latest"); ap.add_argument("--emit-work-orders", action="store_true"); ap.add_argument("--dry-run", action="store_true"); ap.add_argument("--json", action="store_true")
    args=ap.parse_args(); board=load_board(args.board); moves=generate_next_moves(board)
    payload={"schema":"lucidota.abductive_db_os.next_moves.v1","generated_at_utc":now(),"board":args.board,"dry_run":args.dry_run,"emit_work_orders":args.emit_work_orders,"next_moves":moves,"status":"PASS"}
    path=OUT/f"next_moves_{stamp()}.json"; payload["receipt_path"]=rel(path); write_json(path,payload)
    if args.json: print(json.dumps(payload, sort_keys=True, default=str))
    print("REPORT_PATH="+payload["receipt_path"]); print("ABDUCTIVE_NEXT_MOVE_ENGINE=PASS"); return 0
if __name__=="__main__": raise SystemExit(main())
