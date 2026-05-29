#!/usr/bin/env python3
"""Probe DBOS queue scheduler run_after/priority selection across queues."""
from __future__ import annotations

import argparse,json,os
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
def db(a): return a.database_url or os.environ.get('DBOS_SYSTEM_DATABASE_URL') or os.environ.get('DATABASE_URL') or 'postgresql:///lucidota_state'
def write(payload):
 OUT.mkdir(parents=True,exist_ok=True); p=OUT/f'dbos_queue_scheduler_probe_{stamp()}.json'; payload.setdefault('generated_at',now()); payload['report_path']=rel(p); p.write_text(json.dumps(payload,indent=2,default=str),encoding='utf-8'); print(f'REPORT_PATH={rel(p)}'); return p

def enqueue(cur,queue,tag,priority,run_after):
 payload={'probe':'dbos_queue_scheduler_probe','tag':tag}
 cur.execute('''INSERT INTO lucidota_control.dbos_queue_job(queue_name,workflow_name,job_kind,idempotency_key,payload,priority,run_after,max_attempts,detail)
                VALUES (%s,'scheduler-probe','noop',%s,%s::jsonb,%s,%s::timestamptz,1,%s::jsonb)
                ON CONFLICT (queue_name,idempotency_key) DO UPDATE SET status='queued', priority=EXCLUDED.priority, run_after=EXCLUDED.run_after, updated_at=now()
                RETURNING job_uuid::text, priority, run_after''',(queue,f'scheduler_probe:{queue}:{tag}',json.dumps(payload),priority,run_after,json.dumps({'round':41})))
 return dict(cur.fetchone())

def selected(cur,queue):
 cur.execute('''SELECT job_uuid::text,idempotency_key,priority,run_after FROM lucidota_control.dbos_queue_job
                WHERE queue_name=%s AND status='queued' AND run_after<=now()
                ORDER BY priority ASC, created_at ASC FOR UPDATE SKIP LOCKED LIMIT 1''',(queue,))
 r=cur.fetchone(); return dict(r) if r else None

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--database-url'); ap.add_argument('--execute',action='store_true'); a=ap.parse_args()
 if not a.execute:
  write({'action':'probe','execute_performed':False,'blockers':['execute_required_for_scheduler_probe']}); return 2
 queues=['scheduler_probe_a','scheduler_probe_b']; result={'action':'probe','execute_performed':True,'db_writes_performed':True,'graph_writes_performed':False,'queues':{},'blockers':[]}
 with psycopg.connect(db(a), row_factory=dict_row) as conn:
  with conn.cursor() as cur:
   for q in queues:
    cur.execute("INSERT INTO lucidota_control.dbos_queue(queue_name,owner_subsystem,notes) VALUES (%s,'DBOS scheduler probe','temporary scheduler probe queue') ON CONFLICT (queue_name) DO UPDATE SET updated_at=now()",(q,))
    rows=[enqueue(cur,q,'future_high',-999,'2999-01-01T00:00:00Z'), enqueue(cur,q,'ready_low',50,'2000-01-01T00:00:00Z'), enqueue(cur,q,'ready_high',-50,'2000-01-01T00:00:00Z')]
    pick=selected(cur,q)
    ok=bool(pick and pick['idempotency_key'].endswith(':ready_high') and int(pick['priority'])==-50)
    result['queues'][q]={'inserted_or_reset':rows,'selected':pick,'passed':ok}
    if not ok: result['blockers'].append(f'scheduler_bad_selection:{q}')
  conn.commit()
 write(result); print('DBOS_QUEUE_SCHEDULER=' + ('PASS' if not result['blockers'] else 'FAIL'))
 return 0 if not result['blockers'] else 4
if __name__=='__main__': raise SystemExit(main())
