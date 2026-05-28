#!/usr/bin/env python3
"""Enforce conversation_command legal status transitions."""
from __future__ import annotations
import argparse,json,os
from datetime import datetime, timezone
from pathlib import Path
import psycopg
from psycopg.rows import dict_row
ROOT=Path(__file__).resolve().parents[1]; SCHEMA=ROOT/'06_SCHEMA/097_conversation_command_status_transition.sql'; OUT=ROOT/'05_OUTPUTS/dbos'
LEGAL={('staged','queued'),('queued','accepted'),('accepted','executed'),('queued','rejected'),('accepted','rejected'),('staged','rejected'),('staged','superseded'),('queued','superseded')}
def now(): return datetime.now(timezone.utc).isoformat().replace('+00:00','Z')
def stamp(): return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
def rel(p):
 try: return str(Path(p).resolve().relative_to(ROOT))
 except Exception: return str(p)
def db(a): return a.database_url or os.environ.get('DBOS_SYSTEM_DATABASE_URL') or 'postgresql:///lucidota_state'
def write(name,payload):
 OUT.mkdir(parents=True,exist_ok=True); p=OUT/f'conversation_command_status_worker_{name}_{stamp()}.json'; payload.setdefault('generated_at',now()); payload['report_path']=rel(p); p.write_text(json.dumps(payload,indent=2,default=str),encoding='utf-8'); print(f'REPORT_PATH={rel(p)}'); return p
def init(a):
 if a.execute:
  with psycopg.connect(db(a)) as conn:
   with conn.cursor() as cur: cur.execute(SCHEMA.read_text())
   conn.commit()
 write('init_schema_execute' if a.execute else 'init_schema_dry_run',{'action':'init_schema','execute_performed':bool(a.execute),'schema':rel(SCHEMA)}); return 0
def transition(a):
 blockers=[]; before=None; event_uuid=None
 with psycopg.connect(db(a), row_factory=dict_row) as conn:
  with conn.cursor() as cur:
   cur.execute('SELECT command_uuid::text,status FROM lucidota_control.conversation_command WHERE command_uuid=%s::uuid FOR UPDATE',(a.command_uuid,)); row=cur.fetchone()
   if not row: blockers.append('command_not_found')
   else:
    before=row['status']
    if (before,a.to_status) not in LEGAL: blockers.append(f'illegal_transition:{before}->{a.to_status}')
    if a.execute and not blockers:
     cur.execute('UPDATE lucidota_control.conversation_command SET status=%s, executed_at=CASE WHEN %s=\'executed\' THEN now() ELSE executed_at END, updated_at=now(), detail=detail || %s::jsonb WHERE command_uuid=%s::uuid',(a.to_status,a.to_status,json.dumps({'status_worker':'scripts/conversation_command_status_worker.py'}),a.command_uuid))
     cur.execute('INSERT INTO lucidota_control.conversation_command_status_event(command_uuid,from_status,to_status,actor,detail) VALUES (%s::uuid,%s,%s,%s,%s::jsonb) RETURNING event_uuid::text',(a.command_uuid,before,a.to_status,a.actor,json.dumps({'script':'scripts/conversation_command_status_worker.py'})))
     event_uuid=cur.fetchone()['event_uuid']
  conn.commit()
 report={'action':'transition','execute_performed':bool(a.execute),'db_writes_performed':bool(a.execute and event_uuid),'graph_writes_performed':False,'command_uuid':a.command_uuid,'from_status':before,'to_status':a.to_status,'event_uuid':event_uuid,'blockers':blockers,'status':'PASS' if not blockers else 'FAIL'}
 write('transition_execute' if a.execute else 'transition_dry_run',report); print('CONVERSATION_STATUS_TRANSITION='+report['status']); return 0 if not blockers else 4
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--database-url'); sub=ap.add_subparsers(dest='cmd',required=True)
 s=sub.add_parser('init-schema'); s.add_argument('--execute',action='store_true')
 t=sub.add_parser('transition'); t.add_argument('--command-uuid',required=True); t.add_argument('--to-status',required=True,choices=['queued','accepted','executed','rejected','superseded']); t.add_argument('--actor',default='conversation_command_status_worker'); t.add_argument('--execute',action='store_true')
 a=ap.parse_args(); return init(a) if a.cmd=='init-schema' else transition(a)
if __name__=='__main__': raise SystemExit(main())
