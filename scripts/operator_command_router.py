#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, re
from pathlib import Path
from cep_builder import build_cep
from cep_to_kernel_route import route_cep_to_pipeline
from case_workspace import CaseWorkspace
from spine_job_adapter import ABSURDJobAdapter
from work_order_importer import submit_pipeline_jobs

def route_operator_command(raw_command: str, *, case_id: str, source_folder: str, base_dir: str|Path|None=None, ledger_path=None, event_log=None) -> dict:
    if not Path(source_folder).exists():
        return {'status':'DENIED','error':'source_folder_missing'}
    cep=build_cep(raw_command=raw_command, normalized_intent='case.create_and_build_packet', target_refs=[source_folder], evidence_refs=[])
    route=route_cep_to_pipeline(cep, source_folder=source_folder, case_id=case_id, ledger_path=ledger_path, event_log=event_log)
    if route['status'] != 'ROUTED': return {'status':'DENIED','cep':cep,'route':route}
    ws=CaseWorkspace.create(case_id, base_dir=base_dir)
    adapter=ABSURDJobAdapter(ws.root/'absurd')
    jobs=submit_pipeline_jobs(adapter=adapter, case_id=case_id, source_folder=source_folder)
    return {'status':'PASSED','cep':cep,'route':route,'case_id':case_id,'workspace':str(ws.root),'job_count':len(jobs),'jobs':[j['job_id'] for j in jobs]}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--raw-command',required=True); ap.add_argument('--case-id',required=True); ap.add_argument('--source-folder',required=True); ap.add_argument('--base-dir'); a=ap.parse_args(); print(json.dumps(route_operator_command(a.raw_command,case_id=a.case_id,source_folder=a.source_folder,base_dir=a.base_dir),sort_keys=True)); return 0
if __name__=='__main__': raise SystemExit(main())
