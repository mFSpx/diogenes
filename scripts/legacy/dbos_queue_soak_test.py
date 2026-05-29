#!/usr/bin/env python3
"""DBOS queue spine soak: many jobs, retries, duplicate suppression."""
from __future__ import annotations
import argparse,json,os,hashlib
from datetime import datetime, timezone
from pathlib import Path
import psycopg
from psycopg.rows import dict_row
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'05_OUTPUTS/dbos'
def now(): return datetime.now(timezone.utc).isoformat().replace('+00:00','Z')
def stamp(): return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
def rel(p):
 try: return str(Path(p).resolve().relative_to(ROOT))
 except Exception: return str(p)
def db(a): return a.database_url or os.environ.get('DBOS_SYSTEM_DATABASE_URL') or 'postgresql:///lucidota_state'
def write(payload):
 OUT.mkdir(parents=True,exist_ok=True); p=OUT/f'dbos_queue_soak_test_{stamp()}.json'; payload.setdefault('generated_at',now()); payload['report_path']=rel(p); p.write_text(json.dumps(payload,indent=2,default=str),encoding='utf-8'); print('REPORT_PATH='+rel(p))
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--database-url'); ap.add_argument('--execute',action='store_true'); ap.add_argument('--jobs',type=int,default=30); a=ap.parse_args(); blockers=[]
 if not a.execute: blockers.append('execute_required')
 inserted=dup_reused=claimed=failed=dead=success=0; sample=[]; run_id=f'round91-soak-{stamp()}'.lower()
 if a.execute:
  with psycopg.connect(db(a), row_factory=dict_row) as conn:
   with conn.cursor() as cur:
    cur.execute("SET LOCAL lucidota.actor_role='system'")
    cur.execute("INSERT INTO lucidota_control.dbos_queue(queue_name,owner_subsystem,notes) VALUES ('soak_probe','round91','DBOS soak probe') ON CONFLICT(queue_name) DO UPDATE SET updated_at=now()")
    for i in range(a.jobs):
     idem=f"{run_id}-{i//2 if i<10 else i}"  # first ten deliberately duplicate pairs
     kind='fail_once' if i%7==0 else 'noop'
     cur.execute('''INSERT INTO lucidota_control.dbos_queue_job(queue_name,workflow_name,job_kind,idempotency_key,payload,priority,max_attempts)
                    VALUES ('soak_probe','round91-soak',%s,%s,%s::jsonb,%s,2)
                    ON CONFLICT(queue_name,idempotency_key) DO UPDATE SET updated_at=lucidota_control.dbos_queue_job.updated_at
                    RETURNING job_uuid::text,(xmax=0) inserted_new,status,attempt_count''',(kind,idem,json.dumps({'i':i,'kind':kind}),i))
     row=cur.fetchone(); inserted += int(row['inserted_new']); dup_reused += int(not row['inserted_new'])
    for _ in range(a.jobs+5):
     cur.execute("""SELECT job_uuid::text,job_kind,attempt_count,max_attempts,payload FROM lucidota_control.dbos_queue_job
                    WHERE queue_name='soak_probe' AND status IN ('queued','failed') AND attempt_count < max_attempts
                    ORDER BY priority ASC,created_at ASC FOR UPDATE SKIP LOCKED LIMIT 1""")
     row=cur.fetchone()
     if not row: break
     claimed += 1; job=row['job_uuid']; kind=row['job_kind']; attempt=int(row['attempt_count'])+1
     
     if row['attempt_count'] and row['job_kind']=='fail_once':
      cur.execute("UPDATE lucidota_control.dbos_queue_job SET status='queued',updated_at=now() WHERE job_uuid=%s::uuid",(job,))
     cur.execute("UPDATE lucidota_control.dbos_queue_job SET status='running',attempt_count=attempt_count+1,locked_by='round91_soak',locked_at=now(),last_heartbeat_at=now() WHERE job_uuid=%s::uuid",(job,))
     if kind=='fail_once' and attempt==1:
      failed += 1
      cur.execute("UPDATE lucidota_control.dbos_queue_job SET status='failed',error_kind='round91_forced_retry',error_message='forced first failure',updated_at=now() WHERE job_uuid=%s::uuid",(job,))
      cur.execute("INSERT INTO lucidota_control.dbos_queue_event(job_uuid,queue_name,event_kind,event_source,detail) VALUES (%s::uuid,'soak_probe','failed','dbos_queue_soak_test',%s::jsonb)",(job,json.dumps({'attempt':attempt})))
     else:
      success += 1
      cur.execute("UPDATE lucidota_control.dbos_queue_job SET status='succeeded',result=%s::jsonb,completed_at=now(),updated_at=now() WHERE job_uuid=%s::uuid",(json.dumps({'handled':True,'attempt':attempt}),job))
      cur.execute("INSERT INTO lucidota_control.dbos_queue_event(job_uuid,queue_name,event_kind,event_source,detail) VALUES (%s::uuid,'soak_probe','succeeded','dbos_queue_soak_test',%s::jsonb)",(job,json.dumps({'attempt':attempt})))
     sample.append({'job_uuid':job,'kind':kind,'attempt':attempt})
    cur.execute("UPDATE lucidota_control.dbos_queue_job SET status='dead_lettered',updated_at=now() WHERE queue_name='soak_probe' AND status='failed' AND attempt_count>=max_attempts RETURNING job_uuid::text")
    dead=len(cur.fetchall())
   conn.commit()
 if inserted < 1: blockers.append('no_jobs_inserted')
 if dup_reused < 1: blockers.append('duplicate_suppression_not_observed')
 if failed < 1: blockers.append('retry_failure_not_observed')
 if success < 1: blockers.append('success_not_observed')
 payload={'action':'dbos_queue_soak_test','execute_performed':bool(a.execute),'db_writes_performed':bool(a.execute),'graph_writes_performed':False,'jobs_requested':a.jobs,'inserted_new':inserted,'duplicates_reused':dup_reused,'claimed':claimed,'forced_failures':failed,'succeeded':success,'dead_lettered':dead,'sample':sample[:20],'blockers':blockers,'status':'PASS' if not blockers else 'FAIL'}
 write(payload); print('DBOS_QUEUE_SOAK='+payload['status']); return 0 if not blockers else 4
if __name__=='__main__': raise SystemExit(main())
