#!/usr/bin/env python3
"""Convert model invocation audit receipts into Abductive DB OS typed rows."""
from __future__ import annotations
import argparse, json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from model_audit_absurd_adapter import build_rows, latest_audit, load

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'05_OUTPUTS'/'abductive_db_os'
def now(): return datetime.now(timezone.utc).isoformat().replace('+00:00','Z')
def stamp(): return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
def rel(p):
    try: return str(Path(p).resolve().relative_to(ROOT))
    except Exception: return str(p)
def run() -> dict[str, Any]:
    audit_path=latest_audit()
    if not audit_path:
        payload={'schema':'lucidota.abductive_db_os.model_audit_adapter.v1','generated_at_utc':now(),'verdict':'BLOCKED','blockers':['no_model_invocation_audit_found'],'canonical_graph_writes':False,'canonical_graph_materialization':False,'external_effects':False}
    else:
        data=load(audit_path); rows=build_rows(data,audit_path); missing=int(data.get('missing_dedicated_model_audit_blocks') or 0)
        payload={'schema':'lucidota.abductive_db_os.model_audit_adapter.v1','generated_at_utc':now(),'verdict':'PASS' if missing==0 else 'FAIL','latest_model_invocation_audit':rel(audit_path),'latest_audit_verdict':data.get('verdict') or data.get('status') or ('PASS' if missing==0 else 'FAIL'),'missing_dedicated_model_audit_blocks':missing,'typed_rows':rows,'row_counts':{k:len(v) for k,v in rows.items()},'rules':{'receipt_backed_percentages_only':True,'main_window_codex_unreceipted_without_explicit_receipt':True,'complete_5_task_blocks_require_valid_assigned_json_audit':True},'canonical_graph_writes':False,'canonical_graph_materialization':False,'external_effects':False}
    path=OUT/f'model_audit_db_adapter_{stamp()}.json'; payload['receipt_path']=rel(path); path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(payload,indent=2,sort_keys=True,default=str)+'\n')
    return payload
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--json',action='store_true'); args=ap.parse_args(); payload=run()
    if args.json: print(json.dumps(payload,sort_keys=True,default=str))
    print('REPORT_PATH='+payload.get('receipt_path','')); print('MODEL_AUDIT_DB_ADAPTER='+payload['verdict']); return 0 if payload['verdict']=='PASS' else 4
if __name__=='__main__': raise SystemExit(main())
