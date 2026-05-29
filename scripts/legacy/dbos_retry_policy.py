#!/usr/bin/env python3
"""Configure and validate per-queue DBOS retry policy."""
from __future__ import annotations
import argparse,json,os
from datetime import datetime, timezone
from pathlib import Path
import psycopg
from psycopg.rows import dict_row
ROOT=Path(__file__).resolve().parents[1]; SCHEMA=ROOT/'06_SCHEMA/098_dbos_retry_policy_registry.sql'; OUT=ROOT/'05_OUTPUTS/dbos'
def now(): return datetime.now(timezone.utc).isoformat().replace('+00:00','Z')
def stamp(): return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
def rel(p):
 try: return str(Path(p).resolve().relative_to(ROOT))
 except Exception: return str(p)
def db(a): return a.database_url or os.environ.get('DBOS_SYSTEM_DATABASE_URL') or 'postgresql:///lucidota_state'
def write(name,payload):
 OUT.mkdir(parents=True,exist_ok=True); p=OUT/f'dbos_retry_policy_{name}_{stamp()}.json'; payload.setdefault('generated_at',now()); payload['report_path']=rel(p); p.write_text(json.dumps(payload,indent=2,default=str),encoding='utf-8'); print(f'REPORT_PATH={rel(p)}'); return p
def init(a):
 if a.execute:
  with psycopg.connect(db(a)) as conn:
   with conn.cursor() as cur: cur.execute(SCHEMA.read_text())
   conn.commit()
 write('init_schema_execute' if a.execute else 'init_schema_dry_run',{'action':'init_schema','execute_performed':bool(a.execute),'schema':rel(SCHEMA)}); return 0
def set_policy(a):
 if a.dead_letter_after_attempts>a.max_attempts:
  write('set_fail',{'blockers':['dead_letter_after_attempts_gt_max_attempts'],'status':'FAIL'}); return 4
 policy=None
 if a.execute:
  with psycopg.connect(db(a), row_factory=dict_row) as conn:
   with conn.cursor() as cur:
    cur.execute('INSERT INTO lucidota_control.dbos_queue(queue_name,owner_subsystem,notes,max_attempts) VALUES (%s,%s,%s,%s) ON CONFLICT(queue_name) DO UPDATE SET max_attempts=EXCLUDED.max_attempts,updated_at=now()',(a.queue_name,'retry_policy_registry','registered by retry policy',a.max_attempts))
    cur.execute('''INSERT INTO lucidota_control.dbos_retry_policy_registry(queue_name,max_attempts,backoff_seconds,dead_letter_after_attempts,detail)
                   VALUES (%s,%s,%s,%s,%s::jsonb) ON CONFLICT(queue_name) DO UPDATE SET max_attempts=EXCLUDED.max_attempts,backoff_seconds=EXCLUDED.backoff_seconds,dead_letter_after_attempts=EXCLUDED.dead_letter_after_attempts,updated_at=now() RETURNING *''',(a.queue_name,a.max_attempts,a.backoff_seconds,a.dead_letter_after_attempts,json.dumps({'script':'scripts/dbos_retry_policy.py'})))
    policy=dict(cur.fetchone())
   conn.commit()
 report={'action':'set','execute_performed':bool(a.execute),'db_writes_performed':bool(a.execute),'graph_writes_performed':False,'policy':policy,'status':'PASS'}; write('set_execute' if a.execute else 'set_dry_run',report); print('DBOS_RETRY_POLICY=PASS'); return 0
def validate(a):
 blockers=[]
 with psycopg.connect(db(a), row_factory=dict_row) as conn:
  with conn.cursor() as cur:
   cur.execute('SELECT q.queue_name,q.max_attempts AS queue_max,p.max_attempts AS policy_max,p.dead_letter_after_attempts FROM lucidota_control.dbos_queue q LEFT JOIN lucidota_control.dbos_retry_policy_registry p ON p.queue_name=q.queue_name WHERE q.queue_name=%s',(a.queue_name,)); row=cur.fetchone()
 if not row: blockers.append('queue_missing')
 elif row['policy_max'] is None: blockers.append('retry_policy_missing')
 elif int(row['policy_max'])!=int(row['queue_max']): blockers.append('queue_policy_max_attempts_mismatch')
 report={'action':'validate','queue_name':a.queue_name,'row':dict(row) if row else None,'blockers':blockers,'status':'PASS' if not blockers else 'FAIL'}; write('validate_pass' if not blockers else 'validate_fail',report); print('DBOS_RETRY_POLICY_VALIDATE='+report['status']); return 0 if not blockers else 4
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--database-url'); sub=ap.add_subparsers(dest='cmd',required=True)
 s=sub.add_parser('init-schema'); s.add_argument('--execute',action='store_true')
 sp=sub.add_parser('set'); sp.add_argument('--queue-name',required=True); sp.add_argument('--max-attempts',type=int,required=True); sp.add_argument('--backoff-seconds',type=int,default=60); sp.add_argument('--dead-letter-after-attempts',type=int,required=True); sp.add_argument('--execute',action='store_true')
 v=sub.add_parser('validate'); v.add_argument('--queue-name',required=True)
 a=ap.parse_args(); return init(a) if a.cmd=='init-schema' else set_policy(a) if a.cmd=='set' else validate(a)
if __name__=='__main__': raise SystemExit(main())
