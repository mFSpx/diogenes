#!/usr/bin/env python3
"""Bridge a Marrow command receipt into a DBOS work-order job."""
from __future__ import annotations
import argparse,hashlib,json,os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import psycopg
from psycopg.rows import dict_row
ROOT=Path(__file__).resolve().parents[1]; SCHEMA=ROOT/'06_SCHEMA/089_marrow_dbos_work_order_bridge.sql'; OUT=ROOT/'05_OUTPUTS/marrow_loop'
def now(): return datetime.now(timezone.utc).isoformat().replace('+00:00','Z')
def stamp(): return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
def rel(p):
 try: return str(Path(p).resolve().relative_to(ROOT))
 except Exception: return str(p)
def sha_file(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def db(a): return a.database_url or os.environ.get('DBOS_SYSTEM_DATABASE_URL') or os.environ.get('DATABASE_URL') or 'postgresql:///lucidota_state'
def write(name,payload):
 OUT.mkdir(parents=True,exist_ok=True); p=OUT/f'marrow_dbos_bridge_{name}_{stamp()}.json'; payload.setdefault('generated_at',now()); payload['report_path']=rel(p); p.write_text(json.dumps(payload,indent=2,default=str),encoding='utf-8'); print(f'REPORT_PATH={rel(p)}'); return p

def init(a):
 if a.execute:
  with psycopg.connect(db(a)) as conn:
   with conn.cursor() as cur: cur.execute(SCHEMA.read_text())
   conn.commit()
 write('init_schema_execute' if a.execute else 'init_schema_dry_run',{'action':'init_schema','execute_performed':bool(a.execute),'schema':rel(SCHEMA)}); return 0

def bridge(a):
 p=Path(a.receipt); p=p if p.is_absolute() else ROOT/p
 blockers=[]; job_uuid=None; bridge_uuid=None; inserted_new=False
 if not p.exists(): blockers.append('receipt_missing')
 data={}
 if not blockers:
  data=json.loads(p.read_text())
  if not data.get('command_uuid'): blockers.append('receipt_missing_command_uuid')
  if data.get('db_writes_performed') is True: blockers.append('receipt_already_db_written_not_accepted_for_local_bridge')
 h=sha_file(p) if p.exists() else None
 idk=f"marrow_dbos:{data.get('command_uuid', 'missing')}:{h[:12] if h else 'missing'}"
 payload={'receipt_path':rel(p),'receipt_sha256':h,'command_uuid':data.get('command_uuid'),'raw_command':data.get('raw_command'),'normalized_intent':data.get('normalized_intent'),'authority_class':data.get('authority_class'),'source':data.get('source'),'db_writes_performed':False,'graph_writes_performed':False}
 if a.execute and not blockers:
  with psycopg.connect(db(a), row_factory=dict_row) as conn:
   with conn.cursor() as cur:
    cur.execute('''INSERT INTO lucidota_control.dbos_queue_job(queue_name,workflow_name,job_kind,idempotency_key,payload,priority,max_attempts,detail)
                   VALUES ('marrow_loop','marrow-command-receipt','marrow_command_work_order',%s,%s::jsonb,%s,3,%s::jsonb)
                   ON CONFLICT(queue_name,idempotency_key) DO UPDATE SET updated_at=lucidota_control.dbos_queue_job.updated_at
                   RETURNING job_uuid::text,(xmax=0) AS inserted_new''',(idk,json.dumps(payload),a.priority,json.dumps({'script':'scripts/marrow_dbos_bridge.py'})))
    row=cur.fetchone(); job_uuid=row['job_uuid']; inserted_new=bool(row['inserted_new'])
    cur.execute('''INSERT INTO lucidota_control.marrow_dbos_bridge_receipt(command_uuid,receipt_path,receipt_sha256,job_uuid,idempotency_key,detail)
                   VALUES (%s::uuid,%s,%s,%s::uuid,%s,%s::jsonb)
                   ON CONFLICT(idempotency_key) DO UPDATE SET detail=lucidota_control.marrow_dbos_bridge_receipt.detail || EXCLUDED.detail
                   RETURNING bridge_uuid::text''',(data['command_uuid'],rel(p),h,job_uuid,idk,json.dumps({'job_inserted_new':inserted_new})))
    bridge_uuid=cur.fetchone()['bridge_uuid']
    if inserted_new:
     cur.execute("INSERT INTO lucidota_control.dbos_queue_event(job_uuid,queue_name,event_kind,event_source,detail) VALUES (%s::uuid,'marrow_loop','enqueued','marrow_dbos_bridge',%s::jsonb)",(job_uuid,json.dumps({'bridge_uuid':bridge_uuid,'receipt_path':rel(p)})))
   conn.commit()
 report={'action':'bridge','execute_performed':bool(a.execute),'db_writes_performed':bool(a.execute and job_uuid),'graph_writes_performed':False,'receipt_path':rel(p),'receipt_sha256':h,'job_uuid':job_uuid,'bridge_uuid':bridge_uuid,'job_inserted_new':inserted_new,'idempotency_key':idk,'blockers':blockers,'status':'PASS' if not blockers else 'FAIL'}
 write('bridge_execute' if a.execute else 'bridge_dry_run',report); print('MARROW_DBOS_BRIDGE='+report['status']);
 if job_uuid: print('DBOS_JOB_UUID='+job_uuid)
 return 0 if not blockers else 4

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--database-url'); sub=ap.add_subparsers(dest='cmd',required=True)
 s=sub.add_parser('init-schema'); s.add_argument('--execute',action='store_true')
 b=sub.add_parser('bridge'); b.add_argument('--receipt',required=True); b.add_argument('--priority',type=int,default=25); b.add_argument('--execute',action='store_true')
 a=ap.parse_args(); return init(a) if a.cmd=='init-schema' else bridge(a)
if __name__=='__main__': raise SystemExit(main())
