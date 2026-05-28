#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os
from datetime import datetime, timezone
from pathlib import Path
import psycopg
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'05_OUTPUTS/graph_promotion'; SCHEMA=ROOT/'06_SCHEMA/044_graph_promotion_policy_roles.sql'
def stamp(): return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
def db(a): return a.database_url or os.environ.get('KORPUS_DATABASE_URL') or os.environ.get('DATABASE_URL') or 'postgresql:///lucidota_storage'
def write(name,d): OUT.mkdir(parents=True,exist_ok=True); p=OUT/f'graph_promotion_policy_{name}_{stamp()}.json'; d['report_path']=str(p.relative_to(ROOT)); p.write_text(json.dumps(d,indent=2,sort_keys=False,default=str)); print(f'REPORT_PATH={p.relative_to(ROOT)}'); return p
def main():
 p=argparse.ArgumentParser(); p.add_argument('--database-url'); p.add_argument('--execute',action='store_true'); p.add_argument('--role-name',default='graph_promoter'); p.add_argument('--decision',default='promote'); p.add_argument('--materialize',action='store_true'); p.add_argument('--operator-confirmed',action='store_true'); p.add_argument('--evidence-ref',action='append',default=[]); p.add_argument('--command-envelope-uuid'); a=p.parse_args()
 r={'execute_performed':False,'preflight':None,'blockers':[]}
 with psycopg.connect(db(a)) as c:
  with c.cursor() as cur:
   if a.execute: cur.execute(SCHEMA.read_text()); r['execute_performed']=True
   cur.execute('SELECT lucidota_go.graph_promotion_preflight(%s,%s,%s,%s,%s::jsonb,%s)',(a.role_name,a.decision,bool(a.materialize),bool(a.operator_confirmed),json.dumps(a.evidence_ref),a.command_envelope_uuid))
   r['preflight']=cur.fetchone()[0]
  c.commit()
 write('check',r)
 return 0 if r['preflight'].get('allowed') else 2
if __name__=='__main__': raise SystemExit(main())
