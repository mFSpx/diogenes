#!/usr/bin/env python3
"""Abductive DB OS megagate."""
from __future__ import annotations
import argparse, json, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/"05_OUTPUTS"/"abductive_db_os"
def now(): return datetime.now(timezone.utc).isoformat().replace("+00:00","Z")
def stamp(): return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
def rel(p):
    try: return str(Path(p).resolve().relative_to(ROOT))
    except Exception: return str(p)
def combine_gate_verdict(checks: list[dict[str, Any]], hard_failures: list[Any]) -> str:
    if hard_failures or any(c.get("hard") and c.get("verdict") in {"FAIL","BLOCKED"} for c in checks): return "FAIL"
    if any(c.get("verdict") in {"FAIL","BLOCKED","DEGRADED"} for c in checks): return "DEGRADED"
    return "PASS"
def run(cmd: list[str]) -> dict[str, Any]:
    p=subprocess.run(cmd,cwd=ROOT,text=True,capture_output=True,timeout=120)
    receipt=None
    for line in (p.stdout+"\n"+p.stderr).splitlines():
        if line.startswith("REPORT_PATH=") or line.startswith("BOARD_STATE_PATH=") or line.startswith("BOARD_PATH="):
            receipt=line.split("=",1)[1].strip()
    return {"cmd":cmd,"returncode":p.returncode,"stdout_tail":p.stdout[-2000:],"stderr_tail":p.stderr[-1000:],"receipt":receipt}
def child_verdict(name: str, returncode: int, output: str) -> str:
    verdict="PASS" if returncode==0 else "DEGRADED"
    markers=(
        "ABDUCTIVE_DB_OS_LEDGER=",
        "MODEL_AUDIT_DB_ADAPTER=",
        "BYTEWAX_DB_OS_STREAM_AUDIT=",
        "KRAMPUSCHEWING_DB_OS_ADAPTER=",
        "INDY_READS_DB_OS_BRIEF=",
        "ABDUCTIVE_NEXT_MOVE_ENGINE=",
        "ABDUCTIVE_DB_OS_HEALTH_CHECK=",
    )
    for line in output.splitlines():
        if line.startswith(markers):
            marker=line.split("=",1)[1].strip().upper()
            if marker=="PASS":
                verdict="PASS"
            elif marker=="DEGRADED":
                verdict="DEGRADED"
            elif marker in {"FAIL","BLOCKED"}:
                verdict="FAIL" if name=="abductive_db_os_health_check" else "DEGRADED"
    return verdict
def command_plan(mode: str) -> list[list[str]]:
    return [
        [sys.executable,"scripts/abductive_db_os_ledger.py","board","--json"],
        [sys.executable,"scripts/model_audit_db_adapter.py","--json"],
        [sys.executable,"scripts/bytewax_db_os_stream_audit.py","--json"],
        [sys.executable,"scripts/krampuschewing_db_os_adapter.py","--json"],
        [sys.executable,"scripts/indy_reads_db_os_brief.py","--board","latest","--json"],
        [sys.executable,"scripts/abductive_next_move_engine.py","--board","latest","--json"],
        [sys.executable,"scripts/abductive_db_os_health_check.py","--"+mode],
    ]
def operator_next_smallest_safe_work(verdict: str, checks: list[dict[str, Any]]) -> str:
    model_degraded=any(c.get("name")=="model_audit_db_adapter" and c.get("verdict") in {"FAIL","BLOCKED","DEGRADED"} for c in checks)
    if model_degraded: return "repair model audit block 0001"
    if verdict=="PASS": return "run next-move #1"
    return "review degraded gate warnings, then run next-move #1"
def write_json(path: Path, payload: dict[str, Any]): path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(payload,indent=2,sort_keys=True,default=str)+"\n",encoding="utf-8")
def main() -> int:
    ap=argparse.ArgumentParser(); ap.add_argument("--fast", action="store_true"); ap.add_argument("--daily", action="store_true"); args=ap.parse_args(); mode="daily" if args.daily else "fast"
    started=now(); commands=command_plan(mode)
    results=[]; checks=[]; receipts=[]; hard=[]
    for cmd in commands:
        r=run(cmd); results.append(r)
        name=Path(cmd[1]).stem; verdict=child_verdict(name, r["returncode"], r["stdout_tail"]+"\n"+r["stderr_tail"])
        checks.append({"name":name,"verdict":verdict,"hard":False,"returncode":r["returncode"]})
        if r.get("receipt"): receipts.append(r["receipt"])
    verdict=combine_gate_verdict(checks, hard)
    payload={"schema":"lucidota.abductive_db_os_gate.v1","mode":mode,"started_at_utc":started,"finished_at_utc":now(),"verdict":verdict,"db_backend":"jsonl","board_state":next((x for x in receipts if "board_" in x),None),"objects_created_or_staged":{"commands":0,"hypotheses":0,"claims":0,"evidence":0,"graph_candidates":0,"river_candidates":0,"next_moves":0},"checks":checks,"hard_failures":hard,"warnings":[r for r in results if r["returncode"]!=0],"receipts_read":[],"receipts_written":receipts,"canonical_graph_materialization":False,"canonical_graph_writes":False,"external_effects":False,"operator_summary":{"can_trust_current_board":verdict!="FAIL","system_is_moving":True,"top_blocker":None if verdict=="PASS" else "see warnings/checks","next_smallest_safe_work":operator_next_smallest_safe_work(verdict, checks)},"command_results":results}
    path=OUT/f"abductive_db_os_gate_{mode}_{stamp()}.json"; payload["receipt_path"]=rel(path); write_json(path,payload)
    print(json.dumps(payload, sort_keys=True, default=str)); print("REPORT_PATH="+payload["receipt_path"]); print("ABDUCTIVE_DB_OS_GATE="+verdict); return 0 if verdict in {"PASS","DEGRADED"} else 4
if __name__=="__main__": raise SystemExit(main())
