#!/usr/bin/env python3
"""Probe graph_item approved guard: operator UUID, scope, time required."""
from __future__ import annotations
import argparse,json,os,uuid
from datetime import datetime, timezone
from pathlib import Path
import psycopg
from psycopg.rows import dict_row
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'05_OUTPUTS/graph'; SCHEMA=ROOT/'06_SCHEMA/016_go_graph_core.sql'; BARRIER=ROOT/'06_SCHEMA/040_graph_write_barrier_enforcement.sql'
def now(): return datetime.now(timezone.utc).isoformat().replace('+00:00','Z')
def stamp(): return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
def rel(p):
 try: return str(Path(p).resolve().relative_to(ROOT))
 except Exception: return str(p)
def db(a): return a.database_url or os.environ.get('KORPUS_DATABASE_URL') or os.environ.get('DATABASE_URL') or 'postgresql:///lucidota_storage'
def write(payload):
 OUT.mkdir(parents=True,exist_ok=True); p=OUT/f'graph_item_approval_guard_probe_{payload["status"].lower()}_{stamp()}.json'; payload.setdefault('generated_at',now()); payload['report_path']=rel(p); p.write_text(json.dumps(payload,indent=2,default=str),encoding='utf-8'); print(f'REPORT_PATH={rel(p)}'); return p
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--database-url'); ap.add_argument('--execute',action='store_true'); a=ap.parse_args(); blockers=[]; bad_blocked=False; good_uuid=None
 if not a.execute: write({'action':'probe','execute_performed':False,'blockers':['execute_required'],'status':'FAIL'}); return 2
 with psycopg.connect(db(a), row_factory=dict_row) as conn:
  with conn.cursor() as cur:
   cur.execute(SCHEMA.read_text()); cur.execute(BARRIER.read_text())
   cur.execute("SET LOCAL lucidota.graph_promotion_path='on'")
   try:
    cur.execute("INSERT INTO lucidota_go.graph_item(term,label,status,location_at_on_graph,location_real_at_added,payload) VALUES ('CLAIM','bad approved fixture','approved','approval_guard_probe',%s::jsonb,%s::jsonb)",(json.dumps({'probe':'bad'}),json.dumps({'evidence_note':'bad'})))
   except Exception:
    conn.rollback(); bad_blocked=True
   else:
    conn.rollback()
 if not bad_blocked: blockers.append('approved_without_operator_scope_time_not_blocked')
 with psycopg.connect(db(a), row_factory=dict_row) as conn:
  with conn.cursor() as cur:
   cur.execute(SCHEMA.read_text()); cur.execute(BARRIER.read_text()); cur.execute("SET LOCAL lucidota.graph_promotion_path='on'")
   cur.execute("INSERT INTO lucidota_go.graph_item(term,label,status,location_at_on_graph,location_real_at_added,time_approved,approval_scope,operator_uuid,payload) VALUES ('CLAIM','good approved fixture','approved','approval_guard_probe',%s::jsonb,now(),'current',%s::uuid,%s::jsonb) RETURNING uuid::text",(json.dumps({'probe':'good'}),str(uuid.UUID(int=67)),json.dumps({'evidence_note':'approval guard proof'})))
   good_uuid=cur.fetchone()['uuid']
  conn.commit()
 report={'action':'approval_guard_probe','execute_performed':True,'db_writes_performed':True,'canonical_graph_writes_performed':True,'bad_insert_blocked':bad_blocked,'good_graph_item_uuid':good_uuid,'blockers':blockers,'status':'PASS' if not blockers else 'FAIL'}
 write(report); print('GRAPH_ITEM_APPROVAL_GUARD='+report['status']); return 0 if not blockers else 4
if __name__=='__main__': raise SystemExit(main())
