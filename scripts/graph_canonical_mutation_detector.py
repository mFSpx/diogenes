#!/usr/bin/env python3
"""Detect canonical graph mutation around graph worker commands."""
from __future__ import annotations
import argparse,json,os,subprocess,sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import psycopg
from psycopg.rows import dict_row
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'05_OUTPUTS/graph'
TABLES=['lucidota_go.graph_item','lucidota_go.graph_edge','lucidota_go.graph_journal']
def now(): return datetime.now(timezone.utc).isoformat().replace('+00:00','Z')
def stamp(): return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
def rel(p):
 try: return str(Path(p).resolve().relative_to(ROOT))
 except Exception: return str(p)
def db(a): return a.database_url or os.environ.get('KORPUS_DATABASE_URL') or os.environ.get('DATABASE_URL') or 'postgresql:///lucidota_storage'
def counts(cur):
 out={}
 for t in TABLES:
  cur.execute('SELECT to_regclass(%s)',(t,)); exists=cur.fetchone()['to_regclass']
  if exists is None: out[t]=None
  else:
   cur.execute(f'SELECT count(*) AS n FROM {t}'); out[t]=int(cur.fetchone()['n'])
 return out
def run(cmd):
 proc=subprocess.run(cmd,cwd=ROOT,text=True,capture_output=True,timeout=180); return {'command':' '.join(cmd),'returncode':proc.returncode,'stdout_tail':proc.stdout[-2000:],'stderr_tail':proc.stderr[-2000:]}
def write(payload):
 OUT.mkdir(parents=True,exist_ok=True); p=OUT/f'graph_canonical_mutation_detector_{payload["status"].lower()}_{stamp()}.json'; payload.setdefault('generated_at',now()); payload['report_path']=rel(p); p.write_text(json.dumps(payload,indent=2,default=str),encoding='utf-8'); print(f'REPORT_PATH={rel(p)}'); return p
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--database-url'); ap.add_argument('--execute',action='store_true'); a=ap.parse_args()
 commands=[] if not a.execute else [
  [sys.executable,'scripts/graph_promotion_gate.py','--authority-class','deterministic_metric','--evidence-ref','06_SCHEMA/034_graph_promotion_pipeline.sql','--candidate-payload-json','{"kind":"mutation_detector"}'],
  [sys.executable,'scripts/graph_promotion_approval_worker.py','create-candidate','--source-system','mutation_detector','--candidate-kind','node','--candidate-payload-json','{"name":"mutation_detector"}','--authority-class','deterministic_metric','--evidence-ref','06_SCHEMA/034_graph_promotion_pipeline.sql'],
  [sys.executable,'scripts/graph_role_policy_validator.py'],
 ]
 blockers=[]; command_results=[]
 with psycopg.connect(db(a), row_factory=dict_row) as conn:
  with conn.cursor() as cur: before=counts(cur)
 for cmd in commands:
  command_results.append(run(cmd))
 with psycopg.connect(db(a), row_factory=dict_row) as conn:
  with conn.cursor() as cur: after=counts(cur)
 if before!=after: blockers.append('canonical_graph_counts_changed')
 for r in command_results:
  if r['returncode'] not in (0,2): blockers.append('unexpected_worker_returncode:'+r['command'])
 payload={'action':'detect','execute_performed':bool(a.execute),'db_writes_performed':False,'graph_writes_performed':False,'before_counts':before,'after_counts':after,'commands':command_results,'blockers':blockers,'status':'PASS' if not blockers else 'FAIL'}
 write(payload); print('GRAPH_CANONICAL_MUTATION_DETECTOR='+payload['status']); return 0 if not blockers else 4
if __name__=='__main__': raise SystemExit(main())
