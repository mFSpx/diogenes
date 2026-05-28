#!/usr/bin/env python3
"""Prove graph promotion execute path fails safely on invalid payloads."""
from __future__ import annotations
import argparse, json, os, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import psycopg
from psycopg.rows import dict_row
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'05_OUTPUTS/graph'
TABLES=['lucidota_go.graph_item','lucidota_go.graph_edge','lucidota_go.graph_journal','lucidota_go.graph_promotion_materialization','lucidota_go.graph_promotion_packet','lucidota_go.graph_promotion_decision']
def now(): return datetime.now(timezone.utc).isoformat().replace('+00:00','Z')
def stamp(): return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
def db(a): return a.database_url or os.environ.get('KORPUS_DATABASE_URL') or os.environ.get('DATABASE_URL') or 'postgresql:///lucidota_storage'
def rel(p):
    try: return str(Path(p).resolve().relative_to(ROOT))
    except Exception: return str(p)
def counts(conn):
    out={}
    with conn.cursor() as cur:
      for t in TABLES:
        cur.execute(f'SELECT count(*) FROM {t}')
        out[t]=int(cur.fetchone()[0])
    return out
def write(n,d):
    OUT.mkdir(parents=True,exist_ok=True); p=OUT/f'graph_materialization_rollback_probe_{n}_{stamp()}.json'; d.setdefault('generated_at',now()); d['report_path']=rel(p); p.write_text(json.dumps(d,indent=2,sort_keys=False,default=str)); print(f'REPORT_PATH={rel(p)}'); return p

def script_exists(path: str) -> bool:
    p = ROOT / path
    return p.exists() and p.is_file()

def probe(a):
    if not script_exists('scripts/graph_promotion_execute.py'):
        report={'action':'probe','status':'FAIL','execute_performed':False,'db_writes_performed':False,'graph_writes_performed':False,'blockers':['graph_promotion_execute_missing']}
        write('fail', report); print('GRAPH_MATERIALIZATION_ROLLBACK=FAIL'); return 4
    with psycopg.connect(db(a)) as c: before=counts(c)
    invalid_payload='{"term":'
    cmd=[sys.executable,'scripts/graph_promotion_execute.py','--execute','--source-system','rollback_probe','--candidate-kind','other','--candidate-payload-json',invalid_payload,'--evidence-ref','rollback_probe','--authority-class','operator_authored_assertion','--decision','operator_confirmed','--operator-confirmed','--rationale','rollback probe malformed json']
    try:
        proc=subprocess.run(cmd,cwd=ROOT,text=True,capture_output=True,timeout=max(10, min(int(a.timeout_seconds), 180)))
    except subprocess.TimeoutExpired as exc:
        proc = subprocess.CompletedProcess(cmd, 124, exc.stdout or "", exc.stderr or "")
    with psycopg.connect(db(a)) as c: after=counts(c)
    unchanged=before==after
    failed_as_expected=proc.returncode!=0
    report={'action':'probe','status':'PASS' if unchanged and failed_as_expected else 'FAIL','execute_performed':True,'db_writes_performed':False,'graph_writes_performed':False,'command':' '.join(cmd),'returncode':proc.returncode,'stdout_tail':(proc.stdout or '')[-2000:],'stderr_tail':(proc.stderr or '')[-2000:],'counts_before':before,'counts_after':after,'counts_unchanged':unchanged,'failed_as_expected':failed_as_expected,'blockers':[] if unchanged and failed_as_expected else ['rollback_probe_failed']}
    write('pass' if report['status']=='PASS' else 'fail', report); print('GRAPH_MATERIALIZATION_ROLLBACK='+report['status']); return 0 if report['status']=='PASS' else 4
def main():
    p=argparse.ArgumentParser(); p.add_argument('--database-url'); p.add_argument('--timeout-seconds', type=int, default=120); p.set_defaults(func=probe); a=p.parse_args(); return a.func(a)
if __name__=='__main__': raise SystemExit(main())
