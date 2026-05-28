#!/usr/bin/env python3
"""ABSURD wrapper for Surface/CEP fan-in.
Writes queue/workflow receipts and optionally stages conversation_command rows.
No canonical graph mutation.
"""
from __future__ import annotations
import argparse, hashlib, json, os, socket, sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import psycopg
from absurd_worker_contracts import gate_worker_payload_hygiene
from absurd_worker_contracts import record_worker_contract_rejection, validate_worker_contract
from spine_kernel_authorization import validate_job_kernel_authorization

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'05_OUTPUTS/absurd'
SCHEMA=ROOT/'06_SCHEMA/043_absurd_remaining_worker_contracts.sql'
QUEUE='surface_cep'
WORKFLOW='absurd-surface-cep-fan-in'
KIND='surface_cep_health_check'
KINDS={KIND,'surface_instruction_fan_in','conversation_command_work_order','surface_instruction_compiled_command'}
WORKER_KEY='surface_cep_worker'

def now(): return datetime.now(timezone.utc).isoformat().replace('+00:00','Z')
def stamp(): return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
def sha(x:Any)->str: return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),default=str).encode()).hexdigest()
def state_url(a): return a.state_database_url or os.environ.get('ABSURD_SYSTEM_DATABASE_URL') or 'postgresql:///lucidota_state'
def write(name,d):
    OUT.mkdir(parents=True,exist_ok=True); p=OUT/f'absurd_surface_cep_{name}_{stamp()}.json'; d.setdefault('generated_at',now()); d['report_path']=str(p.relative_to(ROOT)); p.write_text(json.dumps(d,indent=2,sort_keys=False,default=str)); print(f'REPORT_PATH={p.relative_to(ROOT)}'); return p

def init_schema(a):
    r={'action':'init_schema','execute_performed':False}
    if not a.execute: write('init_schema_dry_run',r); return 0
    with psycopg.connect(state_url(a)) as c:
      with c.cursor() as cur: cur.execute(SCHEMA.read_text())
      c.commit()
    r['execute_performed']=True; write('init_schema_execute',r); return 0

def enqueue(a):
    payload=json.loads(a.payload_json or '{}')
    if not payload: payload={'check':'surface_cep_health'}
    idem=a.idempotency_key or sha({'queue':QUEUE,'payload':payload})
    r={'action':'enqueue','idempotency_key':idem,'execute_performed':False,'inserted_new':False}
    if not a.execute: write('enqueue_dry_run',r); return 0
    with psycopg.connect(state_url(a)) as c:
      with c.cursor() as cur:
        cur.execute("""INSERT INTO lucidota_control.absurd_queue_job(queue_name,workflow_name,job_kind,idempotency_key,payload)
          VALUES (%s,%s,%s,%s,%s::jsonb) ON CONFLICT(queue_name,idempotency_key) DO UPDATE SET updated_at=lucidota_control.absurd_queue_job.updated_at RETURNING job_uuid::text,(xmax=0)""",(QUEUE,WORKFLOW,a.job_kind,idem,json.dumps(payload)))
        job,inserted=cur.fetchone(); r.update({'job_uuid':job,'inserted_new':bool(inserted)})
        if inserted: cur.execute("INSERT INTO lucidota_control.absurd_queue_event(job_uuid,queue_name,event_kind,event_source,detail) VALUES (%s,%s,'enqueued','spine_surface_cep_worker',%s::jsonb)",(job,QUEUE,json.dumps({'payload_sha256':sha(payload)})))
      c.commit()
    r['execute_performed']=True; write('enqueue_execute',r); print(f'JOB_UUID={job}'); return 0

def worker_once(a):
    wid=a.worker_id or f'{socket.gethostname()}:{os.getpid()}'
    r={'action':'worker_once','worker_id':wid,'target_job_uuid':a.job_uuid,'execute_performed':False,'db_writes_performed':False,'job_processed':False,'canonical_graph_writes_performed':False,'graph_writes_performed':False}
    with psycopg.connect(state_url(a)) as c:
      with c.cursor() as cur:
        if not a.execute:
          if a.job_uuid:
            cur.execute("SELECT job_uuid::text,payload,job_kind FROM lucidota_control.absurd_queue_job WHERE queue_name=%s AND status='queued' AND job_uuid=%s::uuid ORDER BY created_at LIMIT 1",(QUEUE,a.job_uuid))
          else:
            cur.execute("SELECT job_uuid::text,payload,job_kind FROM lucidota_control.absurd_queue_job WHERE queue_name=%s AND status='queued' ORDER BY created_at LIMIT 1",(QUEUE,))
          row=cur.fetchone()
          r['would_process']=row[0] if row else None
          if row:
            r['worker_contract']=validate_worker_contract(cur, queue_name=QUEUE, job_kind=str(row[2]), worker_key=WORKER_KEY).as_result()
            verdict=validate_job_kernel_authorization(queue_name=QUEUE, job_kind=row[2], payload=row[1] or {})
            r['kernel_authorization']=verdict.as_result()
          write('worker_dry_run',r); return 0
        cur.execute("SET LOCAL lucidota.actor_role='worker'")
        if a.job_uuid:
          cur.execute("SELECT job_uuid::text,payload,idempotency_key,job_kind FROM lucidota_control.absurd_queue_job WHERE queue_name=%s AND status='queued' AND job_uuid=%s::uuid ORDER BY created_at FOR UPDATE SKIP LOCKED LIMIT 1",(QUEUE,a.job_uuid))
        else:
          cur.execute("SELECT job_uuid::text,payload,idempotency_key,job_kind FROM lucidota_control.absurd_queue_job WHERE queue_name=%s AND status='queued' ORDER BY created_at FOR UPDATE SKIP LOCKED LIMIT 1",(QUEUE,))
        row=cur.fetchone()
        if not row: r['no_job_available']=True; write('worker_execute',r); return 0
        job,payload,idem,job_kind=row
        contract=validate_worker_contract(cur, queue_name=QUEUE, job_kind=str(job_kind), worker_key=WORKER_KEY)
        r['worker_contract']=contract.as_result()
        if not contract.ok:
          gate_result,error_kind=record_worker_contract_rejection(cur, job_uuid=job, queue_name=QUEUE, payload=payload, contract=contract, event_source='spine_surface_cep_worker')
          c.commit()
          r.update({'execute_performed':True,'db_writes_performed':True,'job_processed':False,'job_uuid':job,'status':'FAIL','result':gate_result,'blockers':[error_kind]})
          write('worker_execute',r); print(f'JOB_UUID={job}'); return 4
        if str(job_kind) not in KINDS:
          r.update({'execute_performed':True,'job_processed':False,'job_uuid':job,'status':'FAIL','blockers':['unexpected_job_kind']})
          write('worker_execute',r); print(f'JOB_UUID={job}'); return 4
        auth_verdict=validate_job_kernel_authorization(queue_name=QUEUE, job_kind=job_kind, payload=payload or {})
        r['kernel_authorization']=auth_verdict.as_result()
        if not auth_verdict.ok:
          cur.execute("UPDATE lucidota_control.absurd_queue_job SET status='failed',result=%s::jsonb,completed_at=now() WHERE job_uuid=%s",(json.dumps({'ok':False,'error':'kernel_authorization_failed','kernel_authorization':auth_verdict.as_result()}),job))
          c.commit()
          r.update({'execute_performed':True,'db_writes_performed':True,'job_processed':False,'job_uuid':job,'status':'FAIL','blockers':['kernel_authorization_failed']})
          write('worker_execute',r); print(f'JOB_UUID={job}'); return 4
        cur.execute("UPDATE lucidota_control.absurd_queue_job SET status='running',locked_by=%s,locked_at=now() WHERE job_uuid=%s",(wid,job))
        lineage_id=(payload.get('command_envelope') or {}).get('current_object_state',{}).get('lineage_id') if isinstance(payload,dict) else None
        execution_uuid=f'surface_cep:{job}'
        command_uuid=None
        if payload.get('plain_language_instruction') and payload.get('command_envelope'):
          cur.execute("""INSERT INTO lucidota_control.conversation_command(plain_language_instruction,command_envelope,authority_class,idempotency_key,allowed_effect,detail)
            VALUES (%s,%s::jsonb,'operator_authored_assertion',%s,%s,%s::jsonb) ON CONFLICT(cep_dedupe_key) DO UPDATE SET updated_at=lucidota_control.conversation_command.updated_at RETURNING command_uuid::text""",(payload['plain_language_instruction'],json.dumps(payload['command_envelope']),idem,payload.get('allowed_effect','stage_only'),json.dumps({'source':'spine_surface_cep_worker'})))
          command_uuid=cur.fetchone()[0]
        result_payload = {"command_uuid": command_uuid, "ok": True, "outcome": "succeeded"}
        payload_ok, hygiene = gate_worker_payload_hygiene(
            result_payload,
            queue_name=QUEUE,
            worker_key=WORKER_KEY,
            job_kind=str(job_kind),
            required_keys=("command_uuid", "outcome"),
            min_score=200,
        )
        if not payload_ok:
            cur.execute("UPDATE lucidota_control.absurd_queue_job SET status='failed',result=%s::jsonb,completed_at=now() WHERE job_uuid=%s",(json.dumps({"result":result_payload, "hygiene": hygiene, "outcome": "failed"}),job))
            cur.execute("INSERT INTO lucidota_control.workflow_event(workflow_id,run_id,phase,status,source,detail) VALUES (%s,%s,'surface_cep','failed','spine_surface_cep_worker',%s::jsonb)",(WORKFLOW,job,json.dumps({'hygiene': hygiene})))
            r.update({"status":"FAIL","execute_performed":True,'db_writes_performed':True,'job_processed':False,'blockers':['decision_hygiene_failed']}); write('worker_execute',r); print(f'JOB_UUID={job}'); return 4
        cur.execute("UPDATE lucidota_control.absurd_queue_job SET status='succeeded',result=%s::jsonb,completed_at=now() WHERE job_uuid=%s",(json.dumps(result_payload),job))
        cur.execute("INSERT INTO lucidota_control.workflow_event(workflow_id,run_id,phase,status,source,detail) VALUES (%s,%s,'surface_cep','succeeded','spine_surface_cep_worker',%s::jsonb)",(WORKFLOW,job,json.dumps(result_payload)))
      c.commit()
    r.update({'execute_performed':True,'db_writes_performed':True,'job_processed':True,'job_uuid':job,'execution_uuid':execution_uuid,'lineage_id':lineage_id,'command_uuid':command_uuid,'status':'PASS'}); write('worker_execute',r); print(f'JOB_UUID={job}'); print(f'EXECUTION_UUID={execution_uuid}'); return 0

def main():
    p=argparse.ArgumentParser(); p.add_argument('--state-database-url'); sub=p.add_subparsers(dest='cmd',required=True)
    for name,func in [('init-schema',init_schema),('enqueue',enqueue),('worker-once',worker_once)]:
      sp=sub.add_parser(name); sp.add_argument('--execute',action='store_true'); sp.set_defaults(func=func)
      if name=='enqueue': sp.add_argument('--payload-json'); sp.add_argument('--idempotency-key'); sp.add_argument('--job-kind',default=KIND)
      if name=='worker-once': sp.add_argument('--worker-id'); sp.add_argument('--job-uuid')
    a=p.parse_args(); return a.func(a)
if __name__=='__main__': raise SystemExit(main())
