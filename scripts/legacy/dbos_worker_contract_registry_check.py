#!/usr/bin/env python3
"""Validate required DBOS worker contracts for core LUCIDOTA workers."""
from __future__ import annotations
import argparse, json, os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import psycopg
from psycopg.rows import dict_row
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'05_OUTPUTS/dbos'
SCHEMA=ROOT/'06_SCHEMA/082_dbos_worker_contract_registry_enforcement.sql'
REQUIRED=['chrono_worker','krampus_worker','river_worker','surface_cep_worker','graph_promotion_worker']
def now(): return datetime.now(timezone.utc).isoformat().replace('+00:00','Z')
def stamp(): return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
def rel(p):
    try: return str(Path(p).resolve().relative_to(ROOT))
    except Exception: return str(p)
def db(a): return a.database_url or os.environ.get('DBOS_SYSTEM_DATABASE_URL') or 'postgresql:///lucidota_state'
def write(n,d):
    OUT.mkdir(parents=True,exist_ok=True); p=OUT/f'dbos_worker_contract_registry_{n}_{stamp()}.json'; d.setdefault('generated_at',now()); d['report_path']=rel(p); p.write_text(json.dumps(d,indent=2,sort_keys=False,default=str)); print(f'REPORT_PATH={rel(p)}'); return p
def init_schema(a):
    if a.execute:
      with psycopg.connect(db(a)) as c:
        with c.cursor() as cur: cur.execute(SCHEMA.read_text())
        c.commit()
    write('init_schema_execute' if a.execute else 'init_schema_dry_run', {'action':'init_schema','execute_performed':bool(a.execute),'schema':rel(SCHEMA)}); return 0
def check(a):
    blockers=[]; rows=[]
    with psycopg.connect(db(a), row_factory=dict_row) as c:
      with c.cursor() as cur:
        cur.execute('''SELECT c.worker_key,c.queue_name,c.script_path,c.status,c.canonical_graph_write_allowed,q.queue_name IS NOT NULL AS queue_exists
          FROM lucidota_control.dbos_worker_contract c LEFT JOIN lucidota_control.dbos_queue q ON q.queue_name=c.queue_name
          WHERE c.worker_key = ANY(%s) ORDER BY c.worker_key''',(REQUIRED,))
        rows=[dict(r) for r in cur.fetchall()]
    got={r['worker_key'] for r in rows}
    for key in REQUIRED:
      if key not in got: blockers.append(f'missing_contract:{key}')
    for r in rows:
      if not r['queue_exists']: blockers.append(f'missing_queue:{r["worker_key"]}:{r["queue_name"]}')
      if not (ROOT/r['script_path']).exists(): blockers.append(f'missing_script:{r["worker_key"]}:{r["script_path"]}')
      if r['status'] not in {'implemented','verified'}: blockers.append(f'bad_status:{r["worker_key"]}:{r["status"]}')
      if r['canonical_graph_write_allowed']: blockers.append(f'canonical_graph_write_allowed:{r["worker_key"]}')
    report={'action':'check','status':'PASS' if not blockers else 'FAIL','required_workers':REQUIRED,'contracts':rows,'blockers':blockers,'execute_performed':False,'db_writes_performed':False,'graph_writes_performed':False}
    write('pass' if not blockers else 'fail', report); print('DBOS_WORKER_CONTRACT_REGISTRY='+report['status']); return 0 if not blockers else 4
def main():
    p=argparse.ArgumentParser(); p.add_argument('--database-url'); sub=p.add_subparsers(dest='cmd',required=True)
    sp=sub.add_parser('init-schema'); sp.add_argument('--execute', action='store_true'); sp.set_defaults(func=init_schema)
    sp=sub.add_parser('check'); sp.set_defaults(func=check)
    a=p.parse_args(); return a.func(a)
if __name__=='__main__': raise SystemExit(main())
