#!/usr/bin/env python3
"""Tiny CPU/RAM/VRAM/PCI/temp telemetry sampler for GOALS."""
from __future__ import annotations
import argparse,json,shutil,subprocess,time
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'05_OUTPUTS/goals'
def now(): return datetime.now(timezone.utc).isoformat().replace('+00:00','Z')
def stamp(): return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
def sh(cmd:list[str])->str:
 try: return subprocess.run(cmd,text=True,capture_output=True,timeout=3,check=False).stdout.strip()
 except Exception as e: return f'ERROR:{type(e).__name__}:{e}'
def mem()->dict:
 vals={}
 for line in Path('/proc/meminfo').read_text().splitlines():
  k,v=line.split(':',1); vals[k]=int(v.split()[0])
 return {k:vals.get(k) for k in ['MemTotal','MemAvailable','MemFree','SwapTotal','SwapFree']}
def cpu()->dict:
 parts=Path('/proc/stat').read_text().splitlines()[0].split()[1:]; return {'stat':[int(x) for x in parts[:8]],'loadavg':Path('/proc/loadavg').read_text().split()[:3]}
def gpu()->dict:
 if not shutil.which('nvidia-smi'): return {'available':False}
 q='name,temperature.gpu,power.draw,memory.total,memory.used,memory.free,utilization.gpu,pcie.link.gen.current,pcie.link.width.current'
 out=sh(['nvidia-smi',f'--query-gpu={q}','--format=csv,noheader,nounits'])
 keys=['name','temp_c','power_w','mem_total_mb','mem_used_mb','mem_free_mb','util_pct','pcie_gen','pcie_width']
 return {'available':bool(out and not out.startswith('ERROR')),'raw':out,'parsed':dict(zip(keys,[x.strip() for x in out.split(',')])) if ',' in out else {}}
def temps()->dict:
 out={}
 for p in Path('/sys/class/thermal').glob('thermal_zone*/temp'):
  try: out[p.parent.name]=round(int(p.read_text().strip())/1000,1)
  except Exception: pass
 return out
def snapshot()->dict:
 c=cpu(); return {'schema':'lucidota.goals.telemetry.sample.v1','at':now(),'cpu':c,'loadavg':c.get('loadavg',[]),'mem':mem(),'gpu':gpu(),'temps':temps()}
def summarize(rows:list[dict])->dict:
 def nums(path):
  out=[]
  for r in rows:
   v=r
   for k in path:
    if isinstance(v,dict): v=v.get(k)
    elif isinstance(v,(list,tuple)) and isinstance(k,int) and k<len(v): v=v[k]
    else: v=None
   try: out.append(float(v))
   except Exception: pass
  return out
 return {'samples':len(rows),'max_load1':max(nums(['loadavg',0]) or [0]),'min_mem_available_kb':min(nums(['mem','MemAvailable']) or [0]),'max_gpu_used_mb':max(nums(['gpu','parsed','mem_used_mb']) or [0]),'max_gpu_temp_c':max(nums(['gpu','parsed','temp_c']) or [0])}
def monitor(duration:int,interval:int)->dict:
 OUT.mkdir(parents=True,exist_ok=True); start=time.time(); rows=[]; jl=OUT/f'telemetry_{stamp()}.jsonl'
 with jl.open('w') as f:
  while time.time()-start<=duration:
   r=snapshot(); rows.append(r); f.write(json.dumps(r,sort_keys=True)+'\n'); f.flush(); time.sleep(max(1,interval))
 report={'schema':'lucidota.goals.telemetry.report.v1','generated_at':now(),'jsonl':str(jl.relative_to(ROOT)),'duration_s':duration,'interval_s':interval,'summary':summarize(rows),'passed':bool(rows)}
 rp=OUT/f'telemetry_report_{stamp()}.json'; rp.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n'); return report|{'report_path':str(rp.relative_to(ROOT))}
def main()->int:
 ap=argparse.ArgumentParser(); ap.add_argument('cmd',choices=['snapshot','monitor']); ap.add_argument('--duration',type=int,default=30); ap.add_argument('--interval',type=int,default=5); ap.add_argument('--json',action='store_true'); a=ap.parse_args()
 r=snapshot() if a.cmd=='snapshot' else monitor(a.duration,a.interval); print('GOAL_TELEMETRY=PASS'); print(json.dumps(r,sort_keys=True) if a.json else r.get('report_path','snapshot'))
 return 0
if __name__=='__main__': raise SystemExit(main())
