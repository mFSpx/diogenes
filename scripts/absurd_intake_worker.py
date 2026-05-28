#!/usr/bin/env python3
"""ABSURD custody wrapper for the supervised Rust lucidota-intake service.
Observes service/drop-dir state and writes ABSURD workflow/custody receipts. It does not move drops.
"""
from __future__ import annotations
import argparse, hashlib, json, os, socket, subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import psycopg
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'05_OUTPUTS/absurd'; SCHEMAS=[ROOT/'06_SCHEMA/035_absurd_queue_spine.sql',ROOT/'06_SCHEMA/039_absurd_real_work_loop.sql',ROOT/'06_SCHEMA/043_absurd_remaining_worker_contracts.sql',ROOT/'06_SCHEMA/049_absurd_intake_wrapper.sql']; QUEUE='intake'; WORKFLOW='absurd-intake-health-check'
SERVICE_CHECK_TIMEOUT_SEC=10
MAX_PENDING_DROP_SCAN=100000
def now(): return datetime.now(timezone.utc).isoformat().replace('+00:00','Z')
def stamp(): return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
def sha(x:Any)->str: return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),default=str).encode()).hexdigest()
def db(a): return a.state_database_url or os.environ.get('ABSURD_SYSTEM_DATABASE_URL') or 'postgresql:///lucidota_state'
def rel(p):
 try: return str(Path(p).resolve().relative_to(ROOT))
 except Exception: return str(p)
def write(name,d): OUT.mkdir(parents=True,exist_ok=True); p=OUT/f'absurd_intake_{name}_{stamp()}.json'; d.setdefault('generated_at',now()); d['report_path']=rel(p); p.write_text(json.dumps(d,indent=2,sort_keys=False,default=str)); print(f'REPORT_PATH={rel(p)}'); return p
def service_active()->bool:
 try:
  return subprocess.run(['systemctl','--user','is-active','--quiet','lucidota-intake.service'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=SERVICE_CHECK_TIMEOUT_SEC).returncode==0
 except Exception:
  return False
def pending_count(drop_dir:Path, max_entries:int=MAX_PENDING_DROP_SCAN)->int:
 if not drop_dir.exists(): return 0
 count=0
 for _ in drop_dir.iterdir():
  count+=1
  if count>=max_entries: return max_entries
 return count
def init(a):
 if a.execute:
  with psycopg.connect(db(a)) as c:
   with c.cursor() as cur:
    for schema in SCHEMAS: cur.execute(schema.read_text())
   c.commit()
 write('init_schema_execute' if a.execute else 'init_schema_dry_run',{'execute_performed':bool(a.execute),'schemas':[rel(s) for s in SCHEMAS]}); return 0
def enqueue(a):
 payload={'drop_dir':a.drop_dir,'digested_dir':a.digested_dir,'quarantine_dir':a.quarantine_dir}; idem=a.idempotency_key or sha({'queue':QUEUE,'payload':payload,'bucket':datetime.now(timezone.utc).strftime('%Y%m%dT%H%M')})
 if not a.execute: write('enqueue_dry_run',{'execute_performed':False,'idempotency_key':idem}); return 0
 with psycopg.connect(db(a)) as c:
  with c.cursor() as cur:
   cur.execute("""INSERT INTO lucidota_control.absurd_queue_job(queue_name,workflow_name,job_kind,idempotency_key,payload)
     VALUES (%s,%s,'intake_health_check',%s,%s::jsonb) ON CONFLICT(queue_name,idempotency_key) DO UPDATE SET updated_at=lucidota_control.absurd_queue_job.updated_at RETURNING job_uuid::text,(xmax=0)""",(QUEUE,WORKFLOW,idem,json.dumps(payload)))
   job,inserted=cur.fetchone()
  c.commit()
 write('enqueue_execute',{'execute_performed':True,'job_uuid':job,'inserted_new':bool(inserted),'idempotency_key':idem}); print(f'JOB_UUID={job}'); return 0
def worker_once(a):
 wid=a.worker_id or f'{socket.gethostname()}:{os.getpid()}'; report={'execute_performed':False,'job_processed':False,'canonical_graph_writes_performed':False}
 with psycopg.connect(db(a)) as c:
  with c.cursor() as cur:
   if not a.execute:
    cur.execute("SELECT job_uuid::text FROM lucidota_control.absurd_queue_job WHERE queue_name=%s AND status='queued' ORDER BY created_at LIMIT 1",(QUEUE,)); row=cur.fetchone(); report['would_process']=row[0] if row else None; write('worker_dry_run',report); return 0
   cur.execute("SET LOCAL lucidota.actor_role='worker'")
   cur.execute("SELECT job_uuid::text,payload FROM lucidota_control.absurd_queue_job WHERE queue_name=%s AND status='queued' ORDER BY created_at FOR UPDATE SKIP LOCKED LIMIT 1",(QUEUE,)); row=cur.fetchone()
   if not row: report['no_job_available']=True; write('worker_execute',report); return 0
   job,payload=row; active=service_active(); drop=Path(payload.get('drop_dir') or ROOT/'03_VAULT/korpus_krampii/DROP'); pend=pending_count(drop)
   cur.execute("UPDATE lucidota_control.absurd_queue_job SET status='running',locked_by=%s,locked_at=now() WHERE job_uuid=%s",(wid,job))
   cur.execute("""INSERT INTO lucidota_control.workflow_event(workflow_id,run_id,phase,status,source,detail)
     VALUES (%s,%s,'intake_health','succeeded','absurd_intake_worker',%s::jsonb) RETURNING event_id::text""",(WORKFLOW,job,json.dumps({'active':active,'pending_drop_count':pend})))
   ev=cur.fetchone()[0]
   cur.execute("""INSERT INTO lucidota_control.intake_service_custody_event(event_kind,active,drop_dir,digested_dir,quarantine_dir,pending_drop_count,workflow_event_id,evidence,detail)
     VALUES (%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s::jsonb) RETURNING custody_uuid::text""",('service_active' if active else 'service_inactive',active,str(drop),payload.get('digested_dir',''),payload.get('quarantine_dir',''),pend,ev,json.dumps([f'workflow_event:{ev}']),json.dumps({'worker_id':wid})))
   custody=cur.fetchone()[0]
   status='succeeded' if active else 'failed'
   cur.execute("UPDATE lucidota_control.absurd_queue_job SET status=%s,result=%s::jsonb,completed_at=CASE WHEN %s='succeeded' THEN now() ELSE completed_at END,last_error=%s WHERE job_uuid=%s",(status,json.dumps({'active':active,'pending_drop_count':pend,'custody_uuid':custody}),status,'' if active else 'intake service inactive',job))
  c.commit()
 report.update({'execute_performed':True,'job_processed':True,'job_uuid':job,'active':active,'pending_drop_count':pend,'custody_uuid':custody,'workflow_event_id':ev}); write('worker_execute',report); print(f'JOB_UUID={job}'); return 0 if active else 3
def main():
 p=argparse.ArgumentParser(); p.add_argument('--state-database-url'); sub=p.add_subparsers(dest='cmd',required=True)
 defaults={'drop_dir':str(ROOT/'03_VAULT/korpus_krampii/DROP'),'digested_dir':str(ROOT/'03_VAULT/korpus_krampii/DIGESTED'),'quarantine_dir':str(ROOT/'03_VAULT/korpus_krampii/QUARANTINE')}
 for name,func in [('init-schema',init),('enqueue',enqueue),('worker-once',worker_once)]:
  sp=sub.add_parser(name); sp.add_argument('--execute',action='store_true'); sp.set_defaults(func=func)
  if name=='enqueue':
   sp.add_argument('--drop-dir',default=defaults['drop_dir']); sp.add_argument('--digested-dir',default=defaults['digested_dir']); sp.add_argument('--quarantine-dir',default=defaults['quarantine_dir']); sp.add_argument('--idempotency-key')
  if name=='worker-once': sp.add_argument('--worker-id')
 a=p.parse_args(); return a.func(a)
if __name__=='__main__': raise SystemExit(main())
