#!/usr/bin/env python3
"""Preflight registered DBOS workers before supervisor launch."""
from __future__ import annotations
import argparse,json,os,subprocess,sys
from datetime import datetime, timezone
from pathlib import Path
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
 OUT.mkdir(parents=True,exist_ok=True); p=OUT/f'dbos_worker_supervisor_preflight_{stamp()}.json'; payload.setdefault('generated_at',now()); payload['report_path']=rel(p); p.write_text(json.dumps(payload,indent=2,default=str),encoding='utf-8'); print(f'REPORT_PATH={rel(p)}')
def pycheck(path):
 p=ROOT/path if not Path(path).is_absolute() else Path(path)
 if not p.exists(): return {'path':path,'exists':False,'py_compile':None,'blocker':'script_missing'}
 if p.suffix!='.py': return {'path':path,'exists':True,'py_compile':'skipped_non_python','blocker':None}
 proc=subprocess.run([sys.executable,'-m','py_compile',str(p)],cwd=ROOT,text=True,capture_output=True)
 return {'path':path,'exists':True,'py_compile':'PASS' if proc.returncode==0 else 'FAIL','blocker':None if proc.returncode==0 else proc.stderr[-1000:]}
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--database-url'); a=ap.parse_args(); rows=[]; blockers=[]
 with psycopg.connect(db(a), row_factory=dict_row) as conn:
  with conn.cursor() as cur:
   cur.execute("""SELECT queue_name, job_kind, handler, script_path, active FROM lucidota_control.worker_command_registry WHERE active=true ORDER BY queue_name, job_kind""")
   for r in cur.fetchall():
    script=r.get('script_path')
    item=dict(r); item['script_check']=pycheck(script) if script else {'blocker':'script_path_missing'}
    if item['script_check'].get('blocker'): blockers.append(f"{r['queue_name']}:{r['job_kind']}:{item['script_check']['blocker']}")
    rows.append(item)
 payload={'action':'supervisor_preflight','workers_checked':len(rows),'workers':rows,'blockers':blockers,'status':'PASS' if not blockers else 'FAIL'}
 write(payload); print('DBOS_SUPERVISOR_PREFLIGHT='+payload['status']); return 0 if not blockers else 4
if __name__=='__main__': raise SystemExit(main())
