#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Any
from spine_job_adapter import ABSURDJobAdapter
from spine_common import receipt, rel, sha256_json
STAGES=['intake','parse','timeline','staging','graph_candidate','case_packet']

def submit_pipeline_jobs(*, adapter: ABSURDJobAdapter, case_id: str, source_folder: str) -> list[dict[str,Any]]:
    jobs=[]; prev=None
    for stage in STAGES:
        payload={'case_id':case_id,'stage':stage,'source_folder':source_folder}
        job=adapter.create_job(lane=f'pipeline.{stage}', payload=payload, idempotency_key=sha256_json(payload), depends_on=[prev] if prev else [])
        if job['state']=='CREATED':
            job=adapter.transition(job['job_id'],'QUEUED')
        jobs.append(job); prev=job['job_id']
    return jobs


def work_orders_from_next_actions(actions: list[dict]) -> list[dict]:
    return [{'work_order_id':'wo:'+a['action_id'].split(':',1)[1], 'kind':'missing_evidence', 'target_ref':a['target_ref'], 'instruction':a['plain_language_instruction'], 'closure_gate':a['closure_gate'], 'status':'CREATED'} for a in actions]


def planned_pipeline_jobs(*, case_id: str, source_folder: str) -> list[dict[str,Any]]:
    jobs=[]; prev=None
    for stage in STAGES:
        payload={'case_id':case_id,'stage':stage,'source_folder':source_folder}
        job_id='job:'+sha256_json({'lane':f'pipeline.{stage}','idempotency_key':sha256_json(payload)})[:24]
        jobs.append({'job_id':job_id,'lane':f'pipeline.{stage}','payload':payload,'idempotency_key':sha256_json(payload),'depends_on':[prev] if prev else [],'state':'PLANNED'})
        prev=job_id
    return jobs


def load_actions(args: argparse.Namespace) -> list[dict[str,Any]]:
    if args.actions_json:
        data=json.loads(args.actions_json)
    else:
        data=json.loads(Path(args.actions_file).read_text(encoding='utf-8'))
    if not isinstance(data,list):
        raise SystemExit('actions input must be a JSON list')
    return data


def cmd_pipeline(args: argparse.Namespace) -> int:
    source=Path(args.source_folder)
    blockers=[]
    if not source.exists():
        blockers.append('source_folder_missing')
    if blockers:
        payload={'schema':'lucidota.work_order_importer.pipeline.v1','status':'BLOCKED','execute_performed':False,'case_id':args.case_id,'source_folder':str(source),'blockers':blockers}
        receipt('work_order_importer_pipeline_blocked',payload,root='05_OUTPUTS/work_orders')
        if args.json: print(json.dumps(payload,sort_keys=True))
        print('WORK_ORDER_IMPORTER=BLOCKED')
        return 3
    if args.execute:
        adapter=ABSURDJobAdapter(args.absurd_dir)
        jobs=submit_pipeline_jobs(adapter=adapter,case_id=args.case_id,source_folder=str(source))
        status='PASSED'
        state_path=rel(adapter.state_path)
    else:
        jobs=planned_pipeline_jobs(case_id=args.case_id,source_folder=str(source))
        status='DRY_RUN'
        state_path=None
    payload={'schema':'lucidota.work_order_importer.pipeline.v1','status':status,'execute_performed':bool(args.execute),'case_id':args.case_id,'source_folder':str(source),'job_count':len(jobs),'jobs':jobs,'absurd_state_path':state_path,'canonical_graph_writes_performed':False,'db_writes_performed':False}
    receipt('work_order_importer_pipeline_execute' if args.execute else 'work_order_importer_pipeline_dry_run',payload,root='05_OUTPUTS/work_orders')
    if args.json: print(json.dumps(payload,sort_keys=True))
    print('WORK_ORDER_IMPORTER='+status)
    return 0


def cmd_next_actions(args: argparse.Namespace) -> int:
    actions=load_actions(args)
    work_orders=work_orders_from_next_actions(actions)
    jsonl_path=None
    if args.execute:
        out=Path(args.jsonl_out)
        rows=''.join(json.dumps(row,sort_keys=True)+'\n' for row in work_orders)
        out.parent.mkdir(parents=True,exist_ok=True); out.write_text(rows,encoding='utf-8')
        jsonl_path=rel(out)
    payload={'schema':'lucidota.work_order_importer.next_actions.v1','status':'PASSED' if args.execute else 'DRY_RUN','execute_performed':bool(args.execute),'action_count':len(actions),'work_order_count':len(work_orders),'work_orders':work_orders,'jsonl_path':jsonl_path,'canonical_graph_writes_performed':False,'db_writes_performed':False}
    receipt('work_order_importer_next_actions_execute' if args.execute else 'work_order_importer_next_actions_dry_run',payload,root='05_OUTPUTS/work_orders')
    if args.json: print(json.dumps(payload,sort_keys=True))
    print('WORK_ORDER_IMPORTER='+payload['status'])
    return 0


def build_parser() -> argparse.ArgumentParser:
    p=argparse.ArgumentParser(description='Import work orders into local ABSURD-compatible queues with receipts.')
    sub=p.add_subparsers(dest='cmd',required=True)
    pp=sub.add_parser('pipeline',help='Plan or enqueue the standard case pipeline jobs.')
    pp.add_argument('--case-id',required=True); pp.add_argument('--source-folder',required=True); pp.add_argument('--absurd-dir',required=True); pp.add_argument('--execute',action='store_true'); pp.add_argument('--json',action='store_true'); pp.set_defaults(fn=cmd_pipeline)
    na=sub.add_parser('next-actions',help='Convert missing-evidence next actions into work orders.')
    src=na.add_mutually_exclusive_group(required=True); src.add_argument('--actions-json'); src.add_argument('--actions-file')
    na.add_argument('--jsonl-out',default='05_OUTPUTS/work_orders/imported_next_actions.jsonl'); na.add_argument('--execute',action='store_true'); na.add_argument('--json',action='store_true'); na.set_defaults(fn=cmd_next_actions)
    return p


def main() -> int:
    args=build_parser().parse_args()
    return args.fn(args)


if __name__=='__main__':
    raise SystemExit(main())
