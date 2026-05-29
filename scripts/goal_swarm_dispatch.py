#!/usr/bin/env python3
"""Bridge GOALS packets into durable ABSURD queue jobs with receipts."""
from __future__ import annotations
import argparse,json,os,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
import psycopg
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'05_OUTPUTS/goals'; PY=sys.executable
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
def now(): return datetime.now(timezone.utc).isoformat().replace('+00:00','Z')
def stamp(): return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
def rel(p):
 try:return str(Path(p).resolve().relative_to(ROOT))
 except Exception:return str(p)
def run(cmd:list[str])->dict:
 p=subprocess.run(cmd,cwd=ROOT,text=True,capture_output=True,check=False)
 rp=next((l.split('=',1)[1].strip() for l in p.stdout.splitlines() if l.startswith('REPORT_PATH=')), '')
 return {'returncode':p.returncode,'stdout_tail':p.stdout[-3000:],'stderr_tail':p.stderr[-3000:],'report_path':rp,'report':json.loads(Path(rp).read_text()) if rp else None}
def default_command(target:str, task:str, complexity:str)->list[str]:
 return [PY,'scripts/goal_agent_packet.py','--target',target,'--task',task,'--complexity',complexity,'--json']
def ensure_queue(name:str)->None:
 with psycopg.connect(os.environ.get('ABSURD_SYSTEM_DATABASE_URL') or os.environ.get('DATABASE_URL') or 'postgresql:///lucidota_state') as conn:
  conn.execute("INSERT INTO lucidota_control.absurd_queue(queue_name,owner_subsystem,max_attempts,notes) VALUES (%s,'GOALS swarm dispatcher',3,'GOALS async command lane') ON CONFLICT(queue_name) DO UPDATE SET owner_subsystem=EXCLUDED.owner_subsystem,max_attempts=EXCLUDED.max_attempts,notes=EXCLUDED.notes,updated_at=now()", (name,))
  conn.commit()
def main()->int:
 a=argparse.ArgumentParser(); a.add_argument('--target',default='generic'); a.add_argument('--task',required=True); a.add_argument('--file',action='append',default=[]); a.add_argument('--complexity',default='standard'); a.add_argument('--queue',default='goal_swarm'); a.add_argument('--workflow',default='goal_swarm'); a.add_argument('--jobs',type=int,default=3); a.add_argument('--json',action='store_true'); a.add_argument('--command',nargs=argparse.REMAINDER); x=a.parse_args()
 from scripts.goal_agent_packet import build_packet
 pkt=build_packet(target=x.target,task=x.task,files=x.file,complexity=x.complexity)
 cmd=x.command or default_command(x.target,x.task,x.complexity)
 selected='groq' if any(Path(part).name=='groq_goal_delegate.py' for part in cmd) else pkt['adapter']['selected']
 ensure_queue(x.queue)
 jobs=[]
 for i in range(max(1,x.jobs)):
  payload={'handler':'external_command','command':cmd,'goal_packet':{'target':x.target,'task':x.task,'files':x.file,'complexity':x.complexity,'adapter':selected,'schema':pkt['schema']},'job_index':i,'selected_adapter':selected}
  idem=f"{x.workflow}:{stamp()}:{i}"
  r=run([PY,'scripts/absurd_queue_spine.py','--action','enqueue','--execute','--queue',x.queue,'--workflow',f'{x.workflow}:{selected}','--job-kind','external_command','--payload-json',json.dumps(payload,separators=(',',':')),'--idempotency-key',idem,'--priority',str(i+1)])
  jobs.append({'enqueue_receipt':r['report_path'],'job_uuid':((r['report'] or {}).get('action_result') or {}).get('job_uuid'),'blockers':(r['report'] or {}).get('blockers',[])})
 report={'schema':'lucidota.goals.swarm_dispatch.v1','generated_at':now(),'model_calls_performed':False,'canonical_graph_writes_performed':False,'queue':x.queue,'workflow':x.workflow,'jobs':jobs,'goal_packet':{'target':x.target,'task':x.task,'complexity':x.complexity,'adapter':selected,'report_path':pkt.get('report_path')},'command':cmd,'adapter_registry_source':'GOALS/plugin_build_mode_bootstrap.json','proof_rule':'No swarm claim without Postgres queue receipts and workflow_event rows.','report_path':''}
 OUT.mkdir(parents=True,exist_ok=True); out=OUT/f'goal_swarm_dispatch_{stamp()}.json'; report['report_path']=rel(out); out.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
 print('REPORT_PATH='+rel(out)); print('GOAL_SWARM_DISPATCH=PASS'); print(json.dumps(report,sort_keys=True) if x.json else rel(out)); return 0
if __name__=='__main__': raise SystemExit(main())
