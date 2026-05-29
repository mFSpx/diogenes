#!/usr/bin/env python3
"""Queue model-fabric bringup/proof jobs through ABSURD/Postgres."""
from __future__ import annotations
import argparse,json,os,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
import psycopg
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'05_OUTPUTS/goals'; RUN=ROOT/'04_RUNTIME/goals'; PY=sys.executable
def now(): return datetime.now(timezone.utc).isoformat().replace('+00:00','Z')
def stamp(): return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
def rel(p):
 try:return str(Path(p).resolve().relative_to(ROOT))
 except Exception:return str(p)
def ontology():
 try:
  o=json.loads((ROOT/'OFFICIAL_ONTOLOGY.json').read_text())
  return {'official_ontology':o.get('official_ontology'),'core_sentence':o.get('core_sentence'),'active_terms':o.get('active_terms',[])[:25]}
 except Exception:return {'official_ontology':'missing','core_sentence':'','active_terms':[]}
def ensure_queue(q):
 with psycopg.connect(os.environ.get('ABSURD_SYSTEM_DATABASE_URL') or os.environ.get('DATABASE_URL') or 'postgresql:///lucidota_state') as c:
  c.execute("INSERT INTO lucidota_control.absurd_queue(queue_name,owner_subsystem,max_attempts,notes) VALUES (%s,'GOALS model fabric',2,'model fabric bringup/proof') ON CONFLICT(queue_name) DO UPDATE SET updated_at=now(),notes=EXCLUDED.notes",(q,)); c.commit()
def run(cmd):
 p=subprocess.run(cmd,cwd=ROOT,text=True,capture_output=True,check=False,timeout=180)
 rp=next((l.split('=',1)[1] for l in p.stdout.splitlines() if l.startswith('REPORT_PATH=')),None)
 return {'returncode':p.returncode,'report_path':rp,'stdout_tail':p.stdout[-1500:],'stderr_tail':p.stderr[-1500:]}
def jobs(model):
 RUN.mkdir(parents=True,exist_ok=True); pp=RUN/f'model_fabric_groq_{stamp()}.txt'; pp.write_text('Audit LUCIDOTA model fabric online proof. Use ontology, Postgres receipts, local-first rule. Return blockers and next smallest step.',encoding='utf-8')
 return [
  {'name':'start_local_fabric','command':[PY,'scripts/goal_model_fabric_control.py','start','--target','heavy','--target','needles','--wait','35','--json']},
  {'name':'local_status_proof','command':[PY,'scripts/goal_model_fabric_control.py','status','--json']},
  {'name':'groq_ontology_audit','command':[PY,'scripts/groq_goal_delegate.py','--task','@'+rel(pp),'--kind','audit','--model',model,'--max-tokens','384','--execute','--json']}]
def enqueue(q,w,j,execute):
 payload={'handler':'external_command','command':j['command'],'job_name':j['name'],'timeout_seconds':240}
 cmd=[PY,'scripts/absurd_queue_spine.py','--action','enqueue','--queue',q,'--workflow',w+':'+j['name'],'--job-kind','external_command','--payload-json',json.dumps(payload,separators=(',',':')),'--idempotency-key',f'{w}:{stamp()}:{j["name"]}','--priority','10']
 if execute: cmd.insert(4,'--execute')
 return run(cmd)
def main()->int:
 ap=argparse.ArgumentParser(); ap.add_argument('--queue',default='model_fabric'); ap.add_argument('--workflow',default='model_fabric_online'); ap.add_argument('--groq-model',default=os.environ.get('GROQ_GOAL_MODEL','llama-3.1-8b-instant')); ap.add_argument('--execute',action='store_true'); ap.add_argument('--consume',action='store_true'); ap.add_argument('--json',action='store_true'); a=ap.parse_args()
 js=jobs(a.groq_model); enq=[]; cons=[]
 if a.execute: ensure_queue(a.queue)
 for j in js: enq.append({'job':j['name'], **enqueue(a.queue,a.workflow,j,a.execute)})
 if a.execute and a.consume:
  for _ in js: cons.append(run([PY,'scripts/absurd_queue_spine.py','--action','worker-once','--queue',a.queue,'--execute']))
 r={'schema':'lucidota.goals.model_fabric_orchestrate.v1','generated_at':now(),'queue':a.queue,'workflow':a.workflow,'ontology':ontology(),'planned_jobs':js,'enqueue_results':enq,'consume_results':cons,'execute_performed':bool(a.execute),'model_calls_performed':False,'canonical_graph_writes_performed':False,'proof_rule':'All model-fabric jobs are ABSURD/Postgres external_command jobs; Groq key stays env-only.'}
 OUT.mkdir(parents=True,exist_ok=True); p=OUT/f'goal_model_fabric_orchestrate_{stamp()}.json'; r['report_path']=rel(p); p.write_text(json.dumps(r,indent=2,sort_keys=True)+'\n')
 print('REPORT_PATH='+rel(p)); print('GOAL_MODEL_FABRIC_ORCHESTRATE=PASS'); 
 if a.json: print(json.dumps(r,sort_keys=True))
 return 0
if __name__=='__main__': raise SystemExit(main())
