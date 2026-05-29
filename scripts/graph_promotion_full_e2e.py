#!/usr/bin/env python3
"""Full graph promotion E2E: surface instruction -> command -> promotion helper -> receipts."""
from __future__ import annotations
import argparse,json,subprocess,sys
from datetime import datetime, timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'05_OUTPUTS/graph'
def now(): return datetime.now(timezone.utc).isoformat().replace('+00:00','Z')
def stamp(): return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
def rel(p):
 try: return str(Path(p).resolve().relative_to(ROOT))
 except Exception: return str(p)
def runc(cmd):
 p=subprocess.run(cmd,cwd=ROOT,text=True,capture_output=True,timeout=180); out={'cmd':' '.join(cmd),'rc':p.returncode,'stdout_tail':p.stdout[-2000:],'stderr_tail':p.stderr[-2000:]}
 for line in p.stdout.splitlines():
  if '=' in line and line.split('=',1)[0] in {'REPORT_PATH','COMMAND_UUID','ABSURD_JOB_UUID','MATERIALIZATION_UUID','GRAPH_ITEM_UUID','JOURNAL_UUID'}:
   out[line.split('=',1)[0]]=line.split('=',1)[1]
 return out
def exists(path): return (ROOT/path).exists()
def write(payload):
 OUT.mkdir(parents=True,exist_ok=True); p=OUT/f'graph_promotion_full_e2e_{stamp()}.json'; payload.setdefault('generated_at',now()); payload['report_path']=rel(p); p.write_text(json.dumps(payload,indent=2),encoding='utf-8'); print('REPORT_PATH='+rel(p))
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--execute',action='store_true'); a=ap.parse_args(); blockers=[]; steps=[]
 if not a.execute:
  payload={'action':'graph_promotion_full_e2e','execute_performed':False,'blockers':['execute_required'],'status':'FAIL'}; write(payload); return 2
 if not exists('scripts/surface_instruction_compile_dry_run.py'): blockers.append('surface_instruction_compile_missing')
 if not exists('scripts/graph_materialization_helper.py'): blockers.append('graph_materialization_helper_missing')
 if not exists('scripts/graph_materialization_receipt_query.py'): blockers.append('graph_materialization_receipt_query_missing')
 if blockers:
  payload={'action':'graph_promotion_full_e2e','execute_performed':False,'blockers':blockers,'status':'FAIL'}; write(payload); return 2
 comp=runc([sys.executable,'scripts/surface_instruction_compile_dry_run.py','--surface-id','graph_promotion_full_e2e','--operator-action','selected','--target-ref','graph_item:new_round87','--evidence-refs','05_OUTPUTS/work_loops/real_code_loop_ledger.jsonl','--artifact-refs','scripts/graph_promotion_full_e2e.py','--current-object-state','{"round":87}','--allowed-effect','stage graph promotion candidate through governed path','--execute','--queue-absurd'])
 steps.append(comp); cmd_uuid=comp.get('COMMAND_UUID')
 if not cmd_uuid: blockers.append('command_uuid_missing')
 mat={}
 if cmd_uuid:
  payload=json.dumps({'term':'ENTITY','label':'Round 87 full graph promotion E2E','status':'staged','detail':{'source':'scripts/graph_promotion_full_e2e.py'}})
  mat=runc([sys.executable,'scripts/graph_materialization_helper.py','materialize','--execute','--operator-confirmed','--promotion-only','--command-envelope-uuid',cmd_uuid,'--candidate-payload-json',payload,'--evidence-ref','05_OUTPUTS/work_loops/real_code_loop_ledger.jsonl','--source-system','graph_promotion_full_e2e','--authority-class','operator_confirmed_finding','--rationale','Round 87 governed full E2E proof'])
  steps.append(mat)
  if mat['rc']!=0: blockers.append('materialize_failed')
 if mat.get('MATERIALIZATION_UUID'):
  q=runc([sys.executable,'scripts/graph_materialization_receipt_query.py','--materialization-uuid',mat['MATERIALIZATION_UUID']]); steps.append(q)
  if q['rc']!=0: blockers.append('receipt_query_failed')
 for s in steps:
  if s['rc']!=0: blockers.append(f"step_rc_{s['rc']}:{s['cmd']}")
 payload={'action':'graph_promotion_full_e2e','execute_performed':True,'db_writes_performed':True,'canonical_graph_writes_performed':bool(mat.get('GRAPH_ITEM_UUID')),'graph_writes_performed':bool(mat.get('GRAPH_ITEM_UUID')),'steps':steps,'command_uuid':cmd_uuid,'materialization_uuid':mat.get('MATERIALIZATION_UUID'),'journal_uuid':mat.get('JOURNAL_UUID'),'blockers':blockers,'status':'PASS' if not blockers else 'FAIL'}
 write(payload); print('GRAPH_PROMOTION_FULL_E2E='+payload['status']); return 0 if not blockers else 4
if __name__=='__main__': raise SystemExit(main())
