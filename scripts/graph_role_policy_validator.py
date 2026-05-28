#!/usr/bin/env python3
"""Validate graph role promotion policy matrix in executable code."""
from __future__ import annotations

import argparse,json,os,uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import psycopg
from psycopg.rows import dict_row

ROOT=Path(__file__).resolve().parents[1]; SCHEMA=ROOT/'06_SCHEMA/044_graph_promotion_policy_roles.sql'; OUT=ROOT/'05_OUTPUTS/graph'
def now(): return datetime.now(timezone.utc).isoformat().replace('+00:00','Z')
def stamp(): return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
def rel(p):
 try: return str(Path(p).resolve().relative_to(ROOT))
 except Exception: return str(p)
def db(a): return a.database_url or os.environ.get('KORPUS_DATABASE_URL') or os.environ.get('DATABASE_URL') or 'postgresql:///lucidota_storage'
def write(payload):
 OUT.mkdir(parents=True,exist_ok=True); p=OUT/f'graph_role_policy_validator_{payload["status"].lower()}_{stamp()}.json'
 payload.setdefault('generated_at',now()); payload['report_path']=rel(p); p.write_text(json.dumps(payload,indent=2,default=str),encoding='utf-8'); print(f'REPORT_PATH={rel(p)}'); return p

CASES=[
 {'name':'reader_can_defer_with_evidence','role':'graph_reader','decision':'defer','materialize':False,'operator_confirmed':False,'evidence':['tests/evidence'],'command':None,'expect':True},
 {'name':'reader_cannot_materialize','role':'graph_reader','decision':'promote','materialize':True,'operator_confirmed':True,'evidence':['tests/evidence'],'command':str(uuid.UUID(int=1)),'expect':False},
 {'name':'promoter_can_defer','role':'graph_promoter','decision':'defer','materialize':False,'operator_confirmed':False,'evidence':['tests/evidence'],'command':None,'expect':True},
 {'name':'promoter_materialize_needs_command','role':'graph_promoter','decision':'operator_confirmed','materialize':True,'operator_confirmed':True,'evidence':['tests/evidence'],'command':None,'expect':False},
 {'name':'promoter_materialize_with_command','role':'graph_promoter','decision':'operator_confirmed','materialize':True,'operator_confirmed':True,'evidence':['tests/evidence'],'command':str(uuid.UUID(int=2)),'expect':True},
 {'name':'operator_promote_with_evidence','role':'operator','decision':'promote','materialize':True,'operator_confirmed':True,'evidence':['tests/evidence'],'command':None,'expect':True},
 {'name':'missing_evidence_denied','role':'operator','decision':'promote','materialize':True,'operator_confirmed':True,'evidence':[],'command':None,'expect':False},
]

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--database-url'); ap.add_argument('--execute',action='store_true'); a=ap.parse_args()
 results=[]; blockers=[]
 with psycopg.connect(db(a), row_factory=dict_row) as conn:
  with conn.cursor() as cur:
   if a.execute: cur.execute(SCHEMA.read_text())
   for case in CASES:
    cur.execute('SELECT lucidota_go.graph_promotion_preflight(%s,%s,%s,%s,%s::jsonb,%s::uuid) AS preflight',(case['role'],case['decision'],case['materialize'],case['operator_confirmed'],json.dumps(case['evidence']),case['command']))
    row=cur.fetchone(); pre=row['preflight']; ok=bool(pre.get('allowed')) is case['expect']
    results.append({**case,'preflight':pre,'passed':ok})
    if not ok: blockers.append(case['name'])
  conn.commit()
 payload={'action':'validate_graph_role_policies','execute_performed':bool(a.execute),'db_writes_performed':bool(a.execute),'graph_writes_performed':False,'cases':results,'blockers':blockers,'status':'PASS' if not blockers else 'FAIL'}
 write(payload); print('GRAPH_ROLE_POLICY=' + payload['status']); return 0 if not blockers else 4
if __name__=='__main__': raise SystemExit(main())
