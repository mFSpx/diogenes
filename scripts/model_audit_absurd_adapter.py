#!/usr/bin/env python3
"""Convert model invocation audit receipts into ABSURD typed rows."""
from __future__ import annotations
import argparse, json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'05_OUTPUTS'/'absurd_abductive'
def now(): return datetime.now(timezone.utc).isoformat().replace('+00:00','Z')
def stamp(): return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
def rel(p):
    try: return str(Path(p).resolve().relative_to(ROOT))
    except Exception: return str(p)
def latest_audit() -> Path | None:
    paths=sorted((ROOT/'05_OUTPUTS/model_invocation_audits').glob('model_invocation_audit_*.json'), key=lambda p:p.stat().st_mtime, reverse=True)
    return paths[0] if paths else None
def load(path: Path) -> dict[str, Any]: return json.loads(path.read_text(encoding='utf-8'))
def build_rows(data: dict[str, Any], path: Path) -> dict[str, Any]:
    inv=data.get('invocations', []) if isinstance(data.get('invocations'), list) else []
    blocks=data.get('five_task_audit_blocks', []) if isinstance(data.get('five_task_audit_blocks'), list) else []
    by_provider=data.get('by_provider', {}) or {}
    deterministic=[i for i in inv if i.get('deterministic_audit_receipt')]
    model_inv=[i for i in inv if not i.get('deterministic_audit_receipt')]
    rows={
      'model_invocation':[{'provider':i.get('provider'),'model':i.get('model'),'status':i.get('status'),'receipt_ref':i.get('receipt_path'),'audit_block_id':i.get('audit_block_id')} for i in model_inv],
      'deterministic_audit_receipt':[{'provider':i.get('provider'),'model':i.get('model'),'status':i.get('status'),'receipt_ref':i.get('receipt_path'),'audit_block_id':i.get('audit_block_id'),'model_calls_performed':False} for i in deterministic],
      'model_audit_block':[{'block_id':b.get('block_id'),'block_signature':b.get('block_signature'),'assigned_auditor':b.get('auditor_provider'),'audit_status':b.get('audit_status'),'task_count':b.get('task_count'),'audit_receipts':b.get('audit_receipts',[])} for b in blocks],
      'auditor_assignment':[{'block_id':b.get('block_id'),'auditor_provider':b.get('auditor_provider'),'requires_different_model':b.get('requires_different_model', True)} for b in blocks],
      'audit_verdict':[{'block_id':b.get('block_id'),'verdict':'PASS' if b.get('audit_status')=='MODEL_AUDIT_RECEIPT_PRESENT' else ('PENDING' if b.get('audit_status')=='PENDING_UNTIL_5_TASKS' else 'FAIL'),'reason':b.get('audit_status')} for b in blocks],
      'provider_usage':[],
      'unreceipted_main_window_usage':[{'provider':'codex_main_window','status':'UNRECEIPTED','tokens':None,'reason':'main-window usage is external unless explicit receipt exists'}],
    }
    total=sum(int(v) for v in by_provider.values() if isinstance(v,int)) or 0
    for provider,count in by_provider.items(): rows['provider_usage'].append({'provider':provider,'receipt_count':count,'share_by_receipt_count':(count/total if total else 0.0),'source':'receipt_count_not_token_truth'})
    return rows
def run() -> dict[str, Any]:
    path=latest_audit();
    if not path:
        return {'schema':'lucidota.absurd.model_audit_adapter.v1','generated_at_utc':now(),'verdict':'BLOCKED','blockers':['no_model_invocation_audit_found'],'canonical_graph_writes':False}
    data=load(path); rows=build_rows(data,path); missing=int(data.get('missing_dedicated_model_audit_blocks') or 0)
    payload={'schema':'lucidota.absurd.model_audit_adapter.v1','generated_at_utc':now(),'verdict':'PASS' if missing==0 else 'FAIL','latest_model_invocation_audit':rel(path),'latest_audit_verdict':data.get('verdict') or data.get('status') or ('PASS' if missing==0 else 'FAIL'),'missing_dedicated_model_audit_blocks':missing,'typed_rows':rows,'row_counts':{k:len(v) for k,v in rows.items()},'rules':{'receipt_backed_percentages_only':True,'main_window_codex_unreceipted_without_explicit_receipt':True,'complete_5_task_blocks_require_valid_assigned_json_audit':True},'canonical_graph_writes':False,'canonical_graph_materialization':False,'external_effects':False}
    path_out=OUT/f'model_audit_absurd_adapter_{stamp()}.json'; payload['receipt_path']=rel(path_out); path_out.parent.mkdir(parents=True,exist_ok=True); path_out.write_text(json.dumps(payload,indent=2,sort_keys=True,default=str)+'\n')
    return payload
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--json',action='store_true'); args=ap.parse_args(); payload=run()
    if args.json: print(json.dumps(payload,sort_keys=True,default=str))
    print('REPORT_PATH='+payload.get('receipt_path','')); print('MODEL_AUDIT_ABSURD_ADAPTER='+payload['verdict']); return 0 if payload['verdict']=='PASS' else 4
if __name__=='__main__': raise SystemExit(main())
