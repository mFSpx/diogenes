#!/usr/bin/env python3
"""Bound one GOALS slice for Groq; writes wrapper receipt and subreceipt."""
from __future__ import annotations
import argparse,json,os,subprocess,sys,hashlib
from datetime import datetime,timezone
from pathlib import Path
from groq_env import load_groq_env
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'05_OUTPUTS/goals'; RUN=ROOT/'04_RUNTIME/goals'
def now(): return datetime.now(timezone.utc).isoformat().replace('+00:00','Z')
def stamp(): return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
def rel(p):
 try:return str(Path(p).resolve().relative_to(ROOT))
 except Exception:return str(p)
def read_text(v):
 if v and v.startswith('@'):
  p=Path(v[1:]); p=p if p.is_absolute() else ROOT/p; return p.read_text(encoding='utf-8')
 return v or ''
def sha_text(v): return hashlib.sha256(v.encode()).hexdigest()
def ontology():
 try:
  o=json.loads((ROOT/'OFFICIAL_ONTOLOGY.json').read_text())
  return {'official_ontology':o.get('official_ontology'),'core_sentence':o.get('core_sentence'),'active_terms':o.get('active_terms',[])[:25]}
 except Exception:return {'official_ontology':'missing','core_sentence':'','active_terms':[]}
def prompt(task,files,kind):
 onto=json.dumps(ontology(),separators=(',',':'))
 return f"""You are a bounded LUCIDOTA Groq worker. Do exactly this {kind} slice, no repo mutation.
ONTOLOGY: {onto}
TASK: {task}
FILES: {', '.join(files) if files else 'none specified'}
RULES: local-first; no VRAM; no secrets; no completion claims without proof; keep answer terse.
Return JSON only: {{"summary":"","findings":[],"commands_to_run":[],"safe_patch_plan":[],"blockers":[],"next_small_step":""}}"""
def write(r):
 OUT.mkdir(parents=True,exist_ok=True); p=OUT/f'groq_goal_delegate_{stamp()}.json'; r['report_path']=rel(p); p.write_text(json.dumps(r,indent=2,sort_keys=True)+'\n'); return p
def main()->int:
 ap=argparse.ArgumentParser(); ap.add_argument('--task',required=True); ap.add_argument('--file',action='append',default=[]); ap.add_argument('--kind',choices=['audit','plan','review','code-slice'],default='audit'); ap.add_argument('--model',default=os.environ.get('GROQ_GOAL_MODEL','llama-3.1-8b-instant')); ap.add_argument('--max-tokens',type=int,default=512); ap.add_argument('--execute',action='store_true'); ap.add_argument('--json',action='store_true'); a=ap.parse_args()
 body=prompt(read_text(a.task),a.file,a.kind); blockers=[]
 if a.execute:
  load_groq_env()
  if not os.environ.get('GROQ_API_KEY'):
   blockers.append('missing_api_key_env:GROQ_API_KEY')
 sub=None; usage=None; text=''; prompt_path=None
 if a.execute and not blockers:
  RUN.mkdir(parents=True,exist_ok=True); pp=RUN/f'groq_delegate_{stamp()}.prompt'; pp.write_text(body,encoding='utf-8')
  prompt_path=rel(pp)
  cmd=[sys.executable,'scripts/model_runner_cli.py','groq-chat','--prompt','@'+rel(pp),'--system','Return valid compact JSON only.','--model',a.model,'--max-tokens',str(a.max_tokens),'--temperature','0','--json','--execute']
  pr=subprocess.run(cmd,cwd=ROOT,text=True,capture_output=True,timeout=120)
  sub=next((l.split('=',1)[1] for l in pr.stdout.splitlines() if l.startswith('RECEIPT_PATH=')),None)
  if pr.returncode: blockers.append('groq_delegate_call_failed')
  if sub and (ROOT/sub).exists():
   sd=json.loads((ROOT/sub).read_text()); usage=sd.get('usage'); text=sd.get('text','')
   for b in sd.get('blockers',[]) or []:
    if b not in blockers: blockers.append(b)
 r={'schema':'lucidota.goals.groq_delegate.v1','generated_at':now(),'provider':'groq','model':a.model,'kind':a.kind,'task_chars':len(read_text(a.task)),'files':a.file,'prompt_path':prompt_path,'prompt_sha256':sha_text(body),'api_key_env_used':'GROQ_API_KEY' if os.environ.get('GROQ_API_KEY') else None,'api_key_redacted':bool(os.environ.get('GROQ_API_KEY')),'execute_performed':bool(a.execute and not blockers),'model_calls_performed':bool(a.execute and not blockers),'subreceipt_path':sub,'usage':usage,'text':text,'blockers':blockers}
 p=write(r); print('REPORT_PATH='+rel(p)); print('GROQ_GOAL_DELEGATE='+('PASS' if not blockers else 'BLOCKED'))
 if a.json: print(json.dumps(r,sort_keys=True))
 return 0 if not blockers else 4
if __name__=='__main__': raise SystemExit(main())
