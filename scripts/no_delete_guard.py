#!/usr/bin/env python3
"""Forward-only guard for tracked deletes: preserve or receipt before erasing."""
from __future__ import annotations
import argparse,json,subprocess
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; BASE=ROOT/'GOALS/NO_DELETE_BASELINE.json'; OUT=ROOT/'05_OUTPUTS/goals'
def now(): return datetime.now(timezone.utc).isoformat().replace('+00:00','Z')
def stamp(): return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
def git_status()->list[str]:
 r=subprocess.run(['git','status','--porcelain'],cwd=ROOT,text=True,capture_output=True,check=False); return r.stdout.splitlines()
def parse_deleted(lines:list[str])->set[str]:
 out=set()
 for l in lines:
  if len(l)>3 and 'D' in l[:2] and not l.startswith('??'): out.add(l[3:])
 return out
def load_baseline()->set[str]:
 return set(json.loads(BASE.read_text()).get('baseline_deletes',[])) if BASE.exists() else set()
def new_deletes(baseline:set[str], current:set[str])->list[str]: return sorted(current-baseline)
def write_receipt(payload:dict)->dict:
 OUT.mkdir(parents=True,exist_ok=True); p=OUT/f'no_delete_guard_{stamp()}.json'; payload['report_path']=str(p.relative_to(ROOT)); p.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n'); return payload
def snapshot()->dict:
 cur=sorted(parse_deleted(git_status())); BASE.parent.mkdir(exist_ok=True); BASE.write_text(json.dumps({'schema':'lucidota.no_delete_baseline.v1','created_at':now(),'baseline_deletes':cur},indent=2,sort_keys=True)+'\n'); return write_receipt({'schema':'lucidota.no_delete_guard.v1','generated_at':now(),'action':'snapshot','baseline_count':len(cur),'passed':True,'baseline':str(BASE.relative_to(ROOT))})
def check()->dict:
 base=load_baseline(); cur=parse_deleted(git_status()); nd=new_deletes(base,cur)
 return write_receipt({'schema':'lucidota.no_delete_guard.v1','generated_at':now(),'action':'check','baseline_count':len(base),'current_deletes':len(cur),'new_deletes':nd,'passed':not nd,'rule':'archive/quarantine to KRAMPUSCHEWING or justify fresh runaway cleanup before tracked delete'})
def main()->int:
 ap=argparse.ArgumentParser(); ap.add_argument('cmd',choices=['snapshot','check']); a=ap.parse_args(); r=snapshot() if a.cmd=='snapshot' else check(); print('REPORT_PATH='+r['report_path']); print('NO_DELETE_GUARD='+('PASS' if r['passed'] else 'FAIL')); return 0 if r['passed'] else 2
if __name__=='__main__': raise SystemExit(main())
