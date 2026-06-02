#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, re
from pathlib import Path
from cep_builder import build_cep
from cep_to_kernel_route import route_cep_to_pipeline
from case_workspace import CaseWorkspace
from spine_job_adapter import ABSURDJobAdapter
from spine_common import receipt, rel
from work_order_importer import submit_pipeline_jobs

OUT = Path("05_OUTPUTS/operator_command_router")

def route_operator_command(raw_command: str, *, case_id: str, source_folder: str, base_dir: str|Path|None=None, ledger_path=None, event_log=None, receipt_dir: str|Path|None=None, emit_receipt: bool=True) -> dict:
    if not Path(source_folder).exists():
        payload={'schema':'lucidota.operator_command_router.v1','status':'DENIED','error':'source_folder_missing','case_id':case_id,'source_folder':str(source_folder),'raw_command':raw_command}
        if receipt_dir is not None:
            receipt('operator_command_router_denied', payload, root=receipt_dir, emit=emit_receipt)
        return payload
    cep=build_cep(raw_command=raw_command, normalized_intent='case.create_and_build_packet', target_refs=[source_folder], evidence_refs=[])
    route=route_cep_to_pipeline(cep, source_folder=source_folder, case_id=case_id, ledger_path=ledger_path, event_log=event_log)
    if route['status'] != 'ROUTED':
        payload={'schema':'lucidota.operator_command_router.v1','status':'DENIED','cep':cep,'route':route,'case_id':case_id,'source_folder':str(source_folder),'raw_command':raw_command}
        if receipt_dir is not None:
            receipt('operator_command_router_denied', payload, root=receipt_dir, emit=emit_receipt)
        return payload
    ws=CaseWorkspace.create(case_id, base_dir=base_dir)
    adapter=ABSURDJobAdapter(ws.root/'absurd')
    jobs=submit_pipeline_jobs(adapter=adapter, case_id=case_id, source_folder=source_folder)
    payload={'schema':'lucidota.operator_command_router.v1','status':'PASSED','cep':cep,'route':route,'case_id':case_id,'source_folder':str(source_folder),'workspace':str(ws.root),'job_count':len(jobs),'jobs':[j['job_id'] for j in jobs],'raw_command':raw_command}
    payload['visible_response']={'summary':f"Indy_READs: routed case {case_id} from {source_folder} into {len(jobs)} pipeline jobs.", 'workspace': rel(ws.root), 'route_status': route['status']}
    if receipt_dir is not None:
        receipt('operator_command_router_passed', payload, root=receipt_dir, emit=emit_receipt)
    return payload

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--raw-command',required=True); ap.add_argument('--case-id',required=True); ap.add_argument('--source-folder',required=True); ap.add_argument('--base-dir'); ap.add_argument('--receipt-dir',default=str(OUT)); ap.add_argument('--json',action='store_true'); a=ap.parse_args(); payload=route_operator_command(a.raw_command,case_id=a.case_id,source_folder=a.source_folder,base_dir=a.base_dir,receipt_dir=a.receipt_dir,emit_receipt=not a.json); print(json.dumps(payload,sort_keys=True)); 
    if not a.json:
        print(f"OPERATOR_ROUTE={payload['status']}");
    return 0 if payload['status']=='PASSED' else 3
if __name__=='__main__': raise SystemExit(main())
