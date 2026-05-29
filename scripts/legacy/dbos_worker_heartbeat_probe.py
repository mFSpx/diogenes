#!/usr/bin/env python3
"""Probe worker heartbeat writes and stale recovery by last_heartbeat_at."""
from __future__ import annotations
import argparse,json,os,subprocess,sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
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
 OUT.mkdir(parents=True,exist_ok=True); p=OUT/f'dbos_worker_heartbeat_probe_{stamp()}.json'; payload.setdefault('generated_at',now()); payload['report_path']=rel(p); p.write_text(json.dumps(payload,indent=2,default=str),encoding='utf-8'); print(f'REPORT_PATH={rel(p)}'); return p
def run(cmd):
 proc=subprocess.run(cmd,cwd=ROOT,text=True,capture_output=True,timeout=180); return {'command':' '.join(cmd),'returncode':proc.returncode,'stdout_tail':proc.stdout[-2000:],'stderr_tail':proc.stderr[-2000:]}
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--database-url'); ap.add_argument('--execute',action='store_true'); a=ap.parse_args()
 if not a.execute: write({'action':'heartbeat_probe','execute_performed':False,'blockers':['execute_required']}); return 2
 idem=f'heartbeat-probe-{stamp()}'.lower(); stale_idem=f'heartbeat-stale-{stamp()}'.lower()
 payload={'target_number':10,'target_name':'Worker telemetry heartbeat','idempotency_key':idem,'handler':'noop','message':'heartbeat probe','files_changed':['scripts/boring_beast.py','scripts/dbos_consume_one.py','scripts/dbos_worker_heartbeat_probe.py'],'validation_commands':['python3 scripts/dbos_worker_heartbeat_probe.py --execute']}
 commands=[run([sys.executable,'scripts/boring_beast.py','init-schema','--execute']), run([sys.executable,'scripts/boring_beast.py','enqueue','--execute','--priority','-8000','--payload-json',json.dumps(payload)]), run([sys.executable,'scripts/boring_beast.py','worker-once','--execute','--worker-id','heartbeat-probe-worker','--idempotency-key',idem])]
 blockers=[]; rows={}
 with psycopg.connect(db(a), row_factory=dict_row) as conn:
  with conn.cursor() as cur:
   cur.execute('SELECT job_uuid::text,status::text,last_heartbeat_at::text,locked_at::text FROM lucidota_control.dbos_queue_job WHERE queue_name=\'boring_beast\' AND idempotency_key=%s',(idem,)); rows['processed']=dict(cur.fetchone())
   if not rows['processed']['last_heartbeat_at']: blockers.append('processed_job_missing_heartbeat')
   # create stale heartbeat job: locked_at recent, heartbeat old; recovery must use heartbeat.
   stale_payload={'target_number':10,'target_name':'heartbeat stale recovery','idempotency_key':stale_idem,'handler':'noop'}
   cur.execute('''INSERT INTO lucidota_control.dbos_queue_job(queue_name,workflow_name,job_kind,idempotency_key,payload,status,priority,attempt_count,max_attempts,locked_by,locked_at,last_heartbeat_at)
                  VALUES ('boring_beast','boring-beast-work-loop','boring_beast_work_item',%s,%s::jsonb,'running',-7999,1,2,'heartbeat-stale-probe',now(),now()-interval '2 hours')
                  ON CONFLICT(queue_name,idempotency_key) DO UPDATE SET status='running',locked_at=now(),last_heartbeat_at=now()-interval '2 hours',locked_by='heartbeat-stale-probe'
                  RETURNING job_uuid::text''',(stale_idem,json.dumps(stale_payload)))
   stale_job=cur.fetchone()['job_uuid']
  conn.commit()
 recover=run([sys.executable,'scripts/boring_beast.py','recover-stale','--execute','--timeout-seconds','60'])
 with psycopg.connect(db(a), row_factory=dict_row) as conn:
  with conn.cursor() as cur:
   cur.execute('SELECT job_uuid::text,status::text,last_heartbeat_at::text,locked_at::text,locked_by FROM lucidota_control.dbos_queue_job WHERE job_uuid=%s::uuid',(stale_job,)); rows['stale_after_recovery']=dict(cur.fetchone())
 if rows['stale_after_recovery']['status']!='queued': blockers.append('stale_heartbeat_job_not_recovered')
 report={'action':'heartbeat_probe','execute_performed':True,'db_writes_performed':True,'graph_writes_performed':False,'commands':commands,'recover_command':recover,'rows':rows,'blockers':blockers,'status':'PASS' if not blockers else 'FAIL'}
 write(report); print('WORKER_HEARTBEAT='+report['status']); return 0 if not blockers else 4
if __name__=='__main__': raise SystemExit(main())
