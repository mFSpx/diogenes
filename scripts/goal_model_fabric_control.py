#!/usr/bin/env python3
"""Tiny GOAL 3 local model fabric status/start/stop helper; no daemon."""
from __future__ import annotations
import argparse,json,os,signal,subprocess,time,urllib.request
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'05_OUTPUTS/goals'
LANES={
 'deepseek':('04_RUNTIME/inference_os/deepseek_q4.pid',8080),
 'mamba_cpu':('04_RUNTIME/inference_os/mamba7b_ternary.pid',8081),
 'bonsai':('04_RUNTIME/inference_os/bonsai4b_ternary.pid',8082),
 'mamba_gpu':('04_RUNTIME/inference_os/mamba7b_gpu.pid',8083),
 'needle_0':('04_RUNTIME/needle_swarm/needle-0.pid',8090),
 'needle_1':('04_RUNTIME/needle_swarm/needle-1.pid',8091),
 'needle_2':('04_RUNTIME/needle_swarm/needle-2.pid',8092),
 'needle_3':('04_RUNTIME/needle_swarm/needle-3.pid',8093),
 'needle_4':('04_RUNTIME/needle_swarm/needle-4.pid',8094),
 'needle_5':('04_RUNTIME/needle_swarm/needle-5.pid',8095)}
GROUPS={'all':list(LANES),'heavy':['deepseek','mamba_cpu','bonsai'],'needles':[f'needle_{i}' for i in range(6)],'optional':['mamba_gpu']}
START={'deepseek':('scripts/lucidota_start_deepseek_llama.sh','04_RUNTIME/inference_os/deepseek_q4_llama_server.log'),
 'mamba_cpu':('scripts/lucidota_start_mamba_llama.sh','04_RUNTIME/inference_os/mamba7b_ternary_cpu_llama_server.log'),
 'bonsai':('scripts/lucidota_start_bonsai_ternary_llama.sh','04_RUNTIME/inference_os/bonsai4b_ternary_cpu_llama_server.log'),
 'mamba_gpu':('scripts/lucidota_start_mamba_gpu_partial.sh','04_RUNTIME/inference_os/mamba7b_gpu_llama_server.log')}
def now(): return datetime.now(timezone.utc).isoformat().replace('+00:00','Z')
def stamp(): return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
def rel(p):
 try: return str(Path(p).resolve().relative_to(ROOT))
 except Exception: return str(p)
def alive(pid:int)->bool:
 try: os.kill(pid,0); return True
 except OSError: return False
def read_pid(root:Path, lane:str)->int|None:
 p=root/LANES[lane][0]
 try: return int(p.read_text().strip())
 except Exception: return None
def health(port:int)->dict:
 try:
  with urllib.request.urlopen(f'http://127.0.0.1:{port}/health',timeout=.8) as r: raw=r.read(500).decode('utf-8','replace')
  return {'ok':True,'raw':raw}
 except Exception as e: return {'ok':False,'error':type(e).__name__}
def gpu_free()->int:
 try:
  return int(subprocess.run(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True,capture_output=True,timeout=3).stdout.splitlines()[0].split(',')[0].strip())
 except Exception: return 0
def expand(names:list[str])->list[str]:
 out=[]
 for n in names:
  for lane in GROUPS.get(n,[n]):
   if lane=='mamba_gpu' and os.environ.get('LUCIDOTA_ENABLE_MAMBA_GPU_PARTIAL','0')!='1': continue
   if lane in LANES and lane not in out: out.append(lane)
 return out
def build_status(root:Path=ROOT)->dict:
 lanes={}
 for lane,(_,port) in LANES.items():
  pid=read_pid(root,lane); lanes[lane]={'pid':pid,'pid_alive':bool(pid and alive(pid)),'port':port,'health':health(port)}
 return {'schema':'lucidota.goals.model_fabric_control.v1','generated_at':now(),'action':'status','lanes':lanes,'model_calls_performed':False,'canonical_graph_writes_performed':False}
def stop_lanes(root:Path=ROOT, names:list[str]|None=None)->dict:
 stopped=[]; names=expand(names or ['optional'])
 for lane in names:
  pid=read_pid(root,lane)
  item={'lane':lane,'pid':pid,'signal_sent':False}
  if pid and alive(pid):
   os.kill(pid,signal.SIGTERM); item['signal_sent']=True; time.sleep(.05)
   if alive(pid):
    try: os.kill(pid,signal.SIGKILL); item['sigkill_sent']=True
    except OSError: pass
  stopped.append(item)
 return {'schema':'lucidota.goals.model_fabric_control.v1','generated_at':now(),'action':'stop','targets':names,'stopped':stopped,'status_after':build_status(root)}
def start_lanes(root:Path=ROOT,names:list[str]|None=None,wait:int=45)->dict:
 started=[]; names=expand(names or ['needles'])
 if any(n.startswith('needle_') for n in names):
  subprocess.run([str(root/'scripts/lucidota_start_needle_swarm.sh')],cwd=root,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
 for lane in names:
  if lane.startswith('needle_'): continue
  pid=read_pid(root,lane); port=LANES[lane][1]; rec={'lane':lane,'port':port,'pid_before':pid}
  if health(port).get('ok'): rec['already_online']=True; started.append(rec); continue
  if pid and alive(pid): rec['already_running']=True; started.append(rec); continue
  if lane not in START: rec['skipped']='no_start_command'; started.append(rec); continue
  script,log=START[lane]; env=os.environ.copy()
  if lane in {'deepseek','mamba_gpu'} and gpu_free() < int(env.get('LUCIDOTA_VRAM_RESERVE_MB','768')) + {'deepseek':1100,'mamba_gpu':1100}[lane]:
   rec['skipped']='vram_guard'; rec['gpu_free_mb']=gpu_free(); started.append(rec); continue
  (root/log).parent.mkdir(parents=True,exist_ok=True)
  env.update({'DRI_PRIME':'0','__NV_PRIME_RENDER_OFFLOAD':'0','OMP_NUM_THREADS':env.get('OMP_NUM_THREADS','2'),'OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1'})
  if lane == 'mamba_cpu': env['CUDA_VISIBLE_DEVICES']=''
  if lane == 'bonsai': env['CUDA_VISIBLE_DEVICES']=''
  with (root/log).open('ab') as f: proc=subprocess.Popen([str(root/script)],cwd=root,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,env=env)
  (root/LANES[lane][0]).parent.mkdir(parents=True,exist_ok=True); (root/LANES[lane][0]).write_text(str(proc.pid)); rec['pid_started']=proc.pid; rec['log']=rel(root/log)
  for _ in range(max(0,wait)):
   if health(port).get('ok'): rec['health_ok_after_start']=True; break
   time.sleep(1)
  started.append(rec)
 return {'schema':'lucidota.goals.model_fabric_control.v1','generated_at':now(),'action':'start','targets':names,'started':started,'status_after':build_status(root),'model_calls_performed':False,'canonical_graph_writes_performed':False}
def write(payload:dict)->Path:
 OUT.mkdir(parents=True,exist_ok=True); p=OUT/f'goal_model_fabric_control_{stamp()}.json'; payload['report_path']=rel(p); p.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n'); return p
def main()->int:
 ap=argparse.ArgumentParser(); ap.add_argument('cmd',choices=['status','start','stop']); ap.add_argument('--target',action='append',default=[]); ap.add_argument('--wait',type=int,default=45); ap.add_argument('--json',action='store_true'); a=ap.parse_args()
 r=build_status() if a.cmd=='status' else start_lanes(names=a.target or ['needles'],wait=a.wait) if a.cmd=='start' else stop_lanes(names=a.target or ['optional']); p=write(r)
 print('REPORT_PATH='+rel(p)); print('GOAL_MODEL_FABRIC_CONTROL=PASS'); print('ACTION='+r['action'])
 if a.json: print(json.dumps(r,sort_keys=True))
 return 0
if __name__=='__main__': raise SystemExit(main())
