#!/usr/bin/env python3
"""Expand SimpleMem recall over claim packets/design atoms across pass kinds."""
from __future__ import annotations
import argparse,json,subprocess,sys
from datetime import datetime, timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'05_OUTPUTS/simplemem'
PASS_QUERIES={'broad':'lucidota workflow graph chrono','lexical':'ABSURD Chrono graph promotion','contradiction':'not truth candidate blocker violation','temporal':'timeline timestamp chrono ledger','variant':'workflow blueprint design atom','weird_neighbor':'proof hoard strange useful later'}
def now(): return datetime.now(timezone.utc).isoformat().replace('+00:00','Z')
def stamp(): return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
def rel(p):
 try: return str(Path(p).resolve().relative_to(ROOT))
 except Exception: return str(p)
def run(cmd):
 proc=subprocess.run(cmd,cwd=ROOT,text=True,capture_output=True,timeout=180); return {'command':' '.join(cmd),'returncode':proc.returncode,'stdout_tail':proc.stdout[-2000:],'stderr_tail':proc.stderr[-2000:]}
def write(payload):
 OUT.mkdir(parents=True,exist_ok=True); p=OUT/f'simplemem_recall_expansion_{stamp()}.json'; payload.setdefault('generated_at',now()); payload['report_path']=rel(p); p.write_text(json.dumps(payload,indent=2,default=str),encoding='utf-8'); print(f'REPORT_PATH={rel(p)}'); return p
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--execute',action='store_true'); ap.add_argument('--limit',type=int,default=15); a=ap.parse_args(); commands=[]; blockers=[]; total_seen=0; total_inserted=0
 commands.append(run([sys.executable,'scripts/simplemem_candidate_index.py','init-schema','--execute']))
 for kind,query in PASS_QUERIES.items():
  cmd=[sys.executable,'scripts/simplemem_candidate_index.py','index','--query',query,'--source','all','--limit',str(a.limit),'--include-zero-score']
  if a.execute: cmd.append('--execute')
  res=run(cmd); res['pass_kind']=kind; commands.append(res)
  for line in res['stdout_tail'].splitlines():
   if line.startswith('CANDIDATES_SEEN='): total_seen+=int(line.split('=',1)[1])
   if line.startswith('CANDIDATES_INSERTED='): total_inserted+=int(line.split('=',1)[1])
  if res['returncode']!=0: blockers.append(f'pass_failed:{kind}')
 report={'action':'recall_expansion','execute_performed':bool(a.execute),'db_writes_performed':bool(a.execute),'graph_writes_performed':False,'safe_to_answer_from_this_alone':False,'not_truth':True,'passes':list(PASS_QUERIES.keys()),'total_candidates_seen':total_seen,'total_candidates_inserted':total_inserted,'commands':commands,'blockers':blockers,'status':'PASS' if not blockers else 'FAIL'}
 write(report); print('SIMPLEMEM_RECALL_EXPANSION='+report['status']); return 0 if not blockers else 4
if __name__=='__main__': raise SystemExit(main())
