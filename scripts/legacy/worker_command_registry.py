#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,os
from datetime import datetime, timezone
from pathlib import Path
import psycopg
from psycopg.rows import dict_row
ROOT=Path(__file__).resolve().parents[1]; SCHEMA=ROOT/'06_SCHEMA/096_worker_command_registry.sql'; OUT=ROOT/'05_OUTPUTS/dbos'
def now(): return datetime.now(timezone.utc).isoformat().replace('+00:00','Z')
def stamp(): return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
def rel(p):
 try: return str(Path(p).resolve().relative_to(ROOT))
 except Exception: return str(p)
def db(a): return a.database_url or os.environ.get('DBOS_SYSTEM_DATABASE_URL') or 'postgresql:///lucidota_state'
def write(name,payload):
 OUT.mkdir(parents=True,exist_ok=True); p=OUT/f'worker_command_registry_{name}_{stamp()}.json'; payload.setdefault('generated_at',now()); payload['report_path']=rel(p); p.write_text(json.dumps(payload,indent=2,default=str),encoding='utf-8'); print(f'REPORT_PATH={rel(p)}'); return p
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--database-url'); sub=ap.add_subparsers(dest='cmd',required=True)
 i=sub.add_parser('init-schema'); i.add_argument('--execute',action='store_true')
 v=sub.add_parser('validate'); v.add_argument('--queue-name',default='*'); v.add_argument('--job-kind',default='*'); v.add_argument('--handler',required=True)
 a=ap.parse_args()
 if a.cmd=='init-schema':
  if a.execute:
   with psycopg.connect(db(a)) as conn:
    with conn.cursor() as cur: cur.execute(SCHEMA.read_text())
    conn.commit()
  write('init_schema_execute' if a.execute else 'init_schema_dry_run',{'action':'init_schema','execute_performed':bool(a.execute),'schema':rel(SCHEMA)}); return 0
 with psycopg.connect(db(a), row_factory=dict_row) as conn:
  with conn.cursor() as cur:
   cur.execute("SELECT * FROM lucidota_control.worker_command_registry WHERE active AND handler=%s AND queue_name IN (%s,'*') AND job_kind IN (%s,'*')",(a.handler,a.queue_name,a.job_kind)); rows=[dict(r) for r in cur.fetchall()]
 passed=bool(rows); write('validate_pass' if passed else 'validate_fail',{'action':'validate','queue_name':a.queue_name,'job_kind':a.job_kind,'handler':a.handler,'matches':rows,'status':'PASS' if passed else 'FAIL'}); print('WORKER_COMMAND_REGISTRY='+('PASS' if passed else 'FAIL')); return 0 if passed else 4
if __name__=='__main__': raise SystemExit(main())
