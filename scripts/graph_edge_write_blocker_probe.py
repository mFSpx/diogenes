#!/usr/bin/env python3
"""Regression probe: direct graph_edge writes are blocked outside promotion path."""
from __future__ import annotations
import argparse,json,os
from datetime import datetime, timezone
from pathlib import Path
import psycopg
from psycopg.rows import dict_row
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'05_OUTPUTS/graph'; SCHEMA=ROOT/'06_SCHEMA/040_graph_write_barrier_enforcement.sql'
def now(): return datetime.now(timezone.utc).isoformat().replace('+00:00','Z')
def stamp(): return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
def rel(p):
 try: return str(Path(p).resolve().relative_to(ROOT))
 except Exception: return str(p)
def db(a): return a.database_url or os.environ.get('KORPUS_DATABASE_URL') or os.environ.get('DATABASE_URL') or 'postgresql:///lucidota_storage'
def write(payload):
 OUT.mkdir(parents=True,exist_ok=True); p=OUT/f'graph_edge_write_blocker_probe_{payload["status"].lower()}_{stamp()}.json'; payload.setdefault('generated_at',now()); payload['report_path']=rel(p); p.write_text(json.dumps(payload,indent=2,default=str),encoding='utf-8'); print(f'REPORT_PATH={rel(p)}'); return p
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--database-url'); ap.add_argument('--execute',action='store_true'); a=ap.parse_args(); blockers=[]; results=[]
 if not a.execute: write({'execute_performed':False,'blockers':['execute_required'],'status':'FAIL'}); return 2
 with psycopg.connect(db(a), row_factory=dict_row) as conn:
  with conn.cursor() as cur:
   cur.execute(SCHEMA.read_text())
   cur.execute('SELECT uuid::text FROM lucidota_go.graph_item ORDER BY created_at DESC LIMIT 2'); nodes=[r['uuid'] for r in cur.fetchall()]
  conn.commit()
 if len(nodes)<2: blockers.append('need_two_graph_items')
 else:
  with psycopg.connect(db(a)) as conn:
   with conn.cursor() as cur:
    try: cur.execute("INSERT INTO lucidota_go.graph_edge(source_uuid,target_uuid,edge_type) VALUES (%s::uuid,%s::uuid,'DIRECT_BLOCK_TEST')",(nodes[0],nodes[1]))
    except Exception as exc: results.append({'probe':'insert_without_promotion_path','blocked':True,'error':str(exc)[:300]}); conn.rollback()
    else: results.append({'probe':'insert_without_promotion_path','blocked':False}); conn.rollback(); blockers.append('insert_not_blocked')
 report={'action':'edge_direct_write_blocker','execute_performed':True,'db_writes_performed':False,'graph_writes_performed':False,'results':results,'blockers':blockers,'status':'PASS' if not blockers else 'FAIL'}
 write(report); print('GRAPH_EDGE_WRITE_BLOCKER='+report['status']); return 0 if not blockers else 4
if __name__=='__main__': raise SystemExit(main())
