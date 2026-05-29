#!/usr/bin/env python3
"""Graph promotion soak: invalid packets blocked, valid staged packets accepted without direct canonical mutation."""
from __future__ import annotations
import argparse,json,os,subprocess,sys
from datetime import datetime, timezone
from pathlib import Path
import psycopg
from psycopg.rows import dict_row
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'05_OUTPUTS/graph'
def now(): return datetime.now(timezone.utc).isoformat().replace('+00:00','Z')
def stamp(): return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
def rel(p):
 try: return str(Path(p).resolve().relative_to(ROOT))
 except Exception: return str(p)
def sdb(a): return a.storage_database_url or os.environ.get('KORPUS_DATABASE_URL') or os.environ.get('DATABASE_URL') or 'postgresql:///lucidota_storage'
def count_graph(url):
 with psycopg.connect(url, row_factory=dict_row) as conn:
  with conn.cursor() as cur:
   out={}
   for t in ('graph_item','graph_edge','graph_journal'):
    cur.execute(f'SELECT count(*) n FROM lucidota_go.{t}'); out[t]=int(cur.fetchone()['n'])
   return out
def run(cmd):
 p=subprocess.run(cmd,cwd=ROOT,text=True,capture_output=True); d={'cmd':' '.join(cmd),'rc':p.returncode,'stdout_tail':p.stdout[-1200:],'stderr_tail':p.stderr[-1200:]}
 for line in p.stdout.splitlines():
  if line.startswith('REPORT_PATH=') or line.startswith('PACKET_UUID='): d[line.split('=',1)[0]]=line.split('=',1)[1]
 return d
def write(payload):
 OUT.mkdir(parents=True,exist_ok=True); p=OUT/f'graph_promotion_soak_test_{stamp()}.json'; payload.setdefault('generated_at',now()); payload['report_path']=rel(p); p.write_text(json.dumps(payload,indent=2,default=str),encoding='utf-8'); print('REPORT_PATH='+rel(p))
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--storage-database-url'); ap.add_argument('--execute',action='store_true'); a=ap.parse_args(); blockers=[]; steps=[]
 if not a.execute: blockers.append('execute_required')
 before=count_graph(sdb(a))
 if a.execute:
  invalid=run([sys.executable,'scripts/graph_promotion_gate.py','gate','--execute','--candidate-payload-json','{"term":"ENTITY","label":"invalid no evidence"}','--authority-class','operator_authored_assertion'])
  steps.append(invalid)
  if invalid['rc']==0: blockers.append('invalid_no_evidence_not_blocked')
  for i in range(3):
   valid=run([sys.executable,'scripts/graph_promotion_gate.py','gate','--execute','--candidate-payload-json',json.dumps({'term':'ENTITY','label':f'round92 valid staged {i}'}),'--evidence-ref','05_OUTPUTS/work_loops/real_code_loop_ledger.jsonl','--authority-class','operator_authored_assertion','--decision','defer','--rationale','round92 soak valid staged packet'])
   steps.append(valid)
   if valid['rc']!=0 or not valid.get('PACKET_UUID'): blockers.append(f'valid_packet_failed_{i}')
 after=count_graph(sdb(a))
 if before != after: blockers.append('canonical_graph_counts_changed')
 payload={'action':'graph_promotion_soak','execute_performed':bool(a.execute),'db_writes_performed':bool(a.execute),'graph_writes_performed':False,'canonical_graph_writes_performed':False,'counts_before':before,'counts_after':after,'steps':steps,'valid_packets':sum(1 for s in steps if s.get('PACKET_UUID')),'blockers':blockers,'status':'PASS' if not blockers else 'FAIL'}
 write(payload); print('GRAPH_PROMOTION_SOAK='+payload['status']); return 0 if not blockers else 4
if __name__=='__main__': raise SystemExit(main())
