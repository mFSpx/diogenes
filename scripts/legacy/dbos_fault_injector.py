#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,os
from datetime import datetime,timezone
from pathlib import Path
import psycopg
from psycopg.rows import dict_row
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'05_OUTPUTS/chaos'
def ts(): return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
def rel(p):
 try:return str(Path(p).resolve().relative_to(ROOT))
 except Exception:return str(p)
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--execute',action='store_true'); a=ap.parse_args(); blockers=[]; facts={}
 if a.execute:
  with psycopg.connect(os.environ.get('DBOS_SYSTEM_DATABASE_URL') or 'postgresql:///lucidota_state', row_factory=dict_row) as c:
   with c.cursor() as cur:
    cur.execute("SET LOCAL lucidota.actor_role='system'")
    cur.execute("INSERT INTO lucidota_control.dbos_queue(queue_name,owner_subsystem) VALUES ('fault_probe','chaos') ON CONFLICT(queue_name) DO NOTHING")
    cur.execute("INSERT INTO lucidota_control.dbos_queue_job(queue_name,workflow_name,job_kind,idempotency_key,payload,max_attempts) VALUES ('fault_probe','fault-inject','forced_failure',%s,'{}'::jsonb,1) ON CONFLICT(queue_name,idempotency_key) DO UPDATE SET status='queued',attempt_count=0 RETURNING job_uuid::text",('fault-'+ts(),)); job=cur.fetchone()['job_uuid']
    cur.execute("UPDATE lucidota_control.dbos_queue_job SET status='running',attempt_count=1 WHERE job_uuid=%s::uuid",(job,)); cur.execute("UPDATE lucidota_control.dbos_queue_job SET status='dead_lettered',error_kind='forced_failure',error_message='chaos forced failure' WHERE job_uuid=%s::uuid",(job,))
    facts={'job_uuid':job}
   c.commit()
 else: blockers.append('EXECUTE_REQUIRED_FOR_FAULT_FIXTURE')
 payload={'action':'dbos_fault_injector','execute_performed':a.execute,'facts':facts,'blockers':[] if a.execute else blockers,'status':'PASS' if a.execute else 'FAIL'}
 OUT.mkdir(parents=True,exist_ok=True); p=OUT/f'dbos_fault_injector_{ts()}.json'; payload['report_path']=rel(p); p.write_text(json.dumps(payload,indent=2),encoding='utf-8'); print('REPORT_PATH='+rel(p)); print('DBOS_FAULT_INJECTOR='+payload['status']); return 0 if a.execute else 4
if __name__=='__main__': raise SystemExit(main())
