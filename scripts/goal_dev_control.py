#!/usr/bin/env python3
"""Tiny GOALS cadence/effective-LOC/supply router; no model calls."""
from __future__ import annotations
import argparse,json,re,sys
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'05_OUTPUTS/goals'
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from ALGOS.bandit_router import select_action  # noqa:E402
from ALGOS.decision_hygiene import counts, score_features  # noqa:E402
TS=re.compile(r"Generated: `([^`]+)`")
CODE={'.py','.sh','.rs','.js','.ts'}
DEFAULT_PATHS=['scripts/goal_handoff.py','scripts/goal_dev_control.py','GOALS/AGENT_ORCHESTRATION_POLICY.md','GOALS/EXTERNAL_PLUGIN_BUILD_MODE.md','GOALS/plugin_build_mode_bootstrap.json']
def now(): return datetime.now(timezone.utc).isoformat().replace('+00:00','Z')
def stamp(): return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
def rel(p):
 try: return str(Path(p).resolve().relative_to(ROOT))
 except Exception: return str(p)
def parse_time(s):
 return datetime.fromisoformat(s.replace('Z','+00:00'))
def read(p:Path)->str:
 return p.read_text(encoding='utf-8',errors='replace') if p.exists() else ''
def code_lines(text:str)->int:
 return sum(1 for l in text.splitlines() if l.strip() and not l.lstrip().startswith('#'))
def cadence(root:Path)->dict:
 text=read(root/'GOALS/GOAL_LOG.md')+'\n'+read(root/'GOALS/CURRENT_HANDOFF.md')
 times=sorted(parse_time(x) for x in TS.findall(text))
 gaps=[(b-a).total_seconds()/60 for a,b in zip(times,times[1:])]
 return {'handoff_count':len(times),'avg_minutes_between_handoffs':round(sum(gaps)/len(gaps),2) if gaps else None,'last_handoff':times[-1].isoformat().replace('+00:00','Z') if times else None}
def path_metrics(root:Path, paths:list[str])->dict:
 code=docs=doc_lines=0; scanned=[]
 for raw in paths:
  p=root/raw
  if not p.exists(): continue
  text=read(p); scanned.append(raw)
  if p.suffix in CODE: code+=code_lines(text)
  else: docs+=1; doc_lines+=sum(1 for l in text.splitlines() if l.strip())
 receipts=len(list((root/'05_OUTPUTS/goals').glob('*.json')))+len(list((root/'05_OUTPUTS/model_runtime').glob('strict_model_stack_admission_*.json')))
 return {'scanned_paths':scanned,'code_lines':code,'doc_files':docs,'doc_lines':doc_lines,'receipt_count':receipts}
def route(text:str, away:int, hygiene:int)->dict:
 low=text.lower(); complex_=bool(re.search(r'architecture|security|mamba|deepseek|bonsai|model|integration|database|graph',low))
 if away<10: actions=['tiny_local','operator_return_review']
 elif complex_: actions=['mid_agent','strong_orchestrator']
 else: actions=['tiny_local','mid_agent']
 act=select_action({'away_minutes':float(away),'hygiene':max(hygiene,0)/10000.0},actions,seed=away).action_id
 return {'selected_action':act,'candidate_actions':actions,'rule':'asymmetric_gate_then_bandit_prior','uses_existing_algo':'ALGOS.bandit_router.select_action'}
def build_report(root:Path=ROOT, away_minutes:int=0, text:str='', paths:list[str]|None=None)->dict:
 root=Path(root); paths=paths or DEFAULT_PATHS; cur=read(root/'GOALS/CURRENT_HANDOFF.md'); c=counts(text+'\n'+cur); hygiene,label=score_features(c)
 m=path_metrics(root,paths); elapsed=max(1,away_minutes); eff=max(0,m['code_lines']+3*m['receipt_count']+hygiene/1000-m['doc_lines']/20)
 return {'schema':'lucidota.goals.dev_supply_control.v1','generated_at':now(),'model_calls_performed':False,'canonical_graph_writes_performed':False,'away_minutes':away_minutes,'dev_mode_entrypoints':{'handoff':'GOALS/CURRENT_HANDOFF.md','subsystems':'GOALS/DEV_MODE_SUBSYSTEMS.json','feature_audit':'GOALS/DEV_MODE_FEATURE_AUDIT.json'},'cadence':cadence(root),'effective_loc':{**m,'formula':'code_lines + 3*receipt_count + hygiene/1000 - doc_lines/20','effective_loc':round(eff,2),'effective_loc_per_hour':round(eff/(elapsed/60),2)},'decision_hygiene':{'score':hygiene,'label':label,'counts':c},'route':route(text+'\n'+cur,away_minutes,hygiene)}
def main()->int:
 p=argparse.ArgumentParser(); p.add_argument('--away-minutes',type=int,default=0); p.add_argument('--text',default=''); p.add_argument('--path',action='append'); p.add_argument('--json',action='store_true'); a=p.parse_args()
 r=build_report(away_minutes=a.away_minutes,text=a.text,paths=a.path); OUT.mkdir(parents=True,exist_ok=True); out=OUT/f'goal_dev_control_{stamp()}.json'; r['report_path']=rel(out); out.write_text(json.dumps(r,indent=2,sort_keys=True)+'\n')
 print('REPORT_PATH='+rel(out)); print('GOAL_DEV_CONTROL=PASS'); print('ROUTE='+r['route']['selected_action']); print('ELOC_PER_HOUR='+str(r['effective_loc']['effective_loc_per_hour']))
 if a.json: print(json.dumps(r,sort_keys=True))
 return 0
if __name__=='__main__': raise SystemExit(main())
