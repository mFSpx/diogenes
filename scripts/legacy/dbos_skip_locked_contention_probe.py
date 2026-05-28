#!/usr/bin/env python3
"""Prove DBOS queue multi-worker contention uses SKIP LOCKED."""
from __future__ import annotations
import argparse,json,os
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
 OUT.mkdir(parents=True,exist_ok=True); p=OUT/f'dbos_skip_locked_contention_probe_{payload["status"].lower()}_{stamp()}.json'; payload.setdefault('generated_at',now()); payload['report_path']=rel(p); p.write_text(json.dumps(payload,indent=2,default=str),encoding='utf-8'); print(f'REPORT_PATH={rel(p)}'); return p
def seed(cur,queue,tag,priority):
 cur.execute('''INSERT INTO lucidota_control.dbos_queue(queue_name,owner_subsystem,notes) VALUES (%s,'SKIP LOCKED probe','contention test') ON CONFLICT(queue_name) DO UPDATE SET updated_at=now()''',(queue,))
 cur.execute('''INSERT INTO lucidota_control.dbos_queue_job(queue_name,workflow_name,job_kind,idempotency_key,payload,priority,max_attempts,status)
                VALUES (%s,'skip-locked-probe','noop',%s,%s::jsonb,%s,1,'queued')
                ON CONFLICT(queue_name,idempotency_key) DO UPDATE SET status='queued',priority=EXCLUDED.priority,updated_at=now()
                RETURNING job_uuid::text''',(queue,f'skip_locked:{tag}',json.dumps({'tag':tag}),priority)); return cur.fetchone()['job_uuid']
def claim(cur,queue):
 cur.execute('''SELECT job_uuid::text,idempotency_key FROM lucidota_control.dbos_queue_job WHERE queue_name=%s AND status='queued' AND run_after<=now() ORDER BY priority,created_at FOR UPDATE SKIP LOCKED LIMIT 1''',(queue,)); r=cur.fetchone(); return dict(r) if r else None
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--database-url'); ap.add_argument('--execute',action='store_true'); a=ap.parse_args(); blockers=[]; queue='skip_locked_probe'; result={}
 if not a.execute: write({'action':'probe','execute_performed':False,'blockers':['execute_required'],'status':'FAIL'}); return 2
 with psycopg.connect(db(a), row_factory=dict_row) as setup:
  with setup.cursor() as cur:
   seed(cur,queue,'a',1); seed(cur,queue,'b',2)
  setup.commit()
 c1=psycopg.connect(db(a), row_factory=dict_row); c2=psycopg.connect(db(a), row_factory=dict_row)
 try:
  cur1=c1.cursor(); cur2=c2.cursor(); j1=claim(cur1,queue); j2=claim(cur2,queue)
  result={'worker1_claim':j1,'worker2_claim':j2}
  if not j1 or not j2: blockers.append('both_workers_did_not_claim')
  elif j1['job_uuid']==j2['job_uuid']: blockers.append('skip_locked_failed_same_job_claimed')
 finally:
  c1.rollback(); c2.rollback(); c1.close(); c2.close()
 report={'action':'skip_locked_contention_probe','execute_performed':True,'db_writes_performed':True,'graph_writes_performed':False,'queue_name':queue,**result,'blockers':blockers,'status':'PASS' if not blockers else 'FAIL'}
 write(report); print('SKIP_LOCKED_CONTENTION='+report['status']); return 0 if not blockers else 4
if __name__=='__main__': raise SystemExit(main())
