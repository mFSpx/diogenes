#!/usr/bin/env python3
"""Run Mega-Gate repeatedly and validate each report."""
from __future__ import annotations
import argparse,json,subprocess,sys
from datetime import datetime, timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'05_OUTPUTS/mega_gate'
def stamp(): return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
def rel(p):
 try: return str(Path(p).resolve().relative_to(ROOT))
 except Exception: return str(p)
def run(cmd):
 p=subprocess.run(cmd,cwd=ROOT,text=True,capture_output=True); rp=None
 for l in p.stdout.splitlines():
  if l.startswith('REPORT_PATH='): rp=l.split('=',1)[1]
 return {'command':' '.join(cmd),'rc':p.returncode,'report_path':rp,'stdout_tail':p.stdout[-1200:],'stderr_tail':p.stderr[-1200:]}
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--iterations',type=int,default=2); a=ap.parse_args(); steps=[]; blockers=[]
 for i in range(a.iterations):
  gate=run([sys.executable,'scripts/lucidota_mega_gate.py']); steps.append(gate)
  if gate['rc']!=0: blockers.append(f'gate_failed_{i}')
  if gate.get('report_path'):
   val=run([sys.executable,'scripts/mega_gate_metrics_validator.py','--report',gate['report_path']]); steps.append(val)
   if val['rc']!=0: blockers.append(f'validator_failed_{i}')
  else: blockers.append(f'gate_report_missing_{i}')
 payload={'action':'mega_gate_regression_loop','iterations':a.iterations,'steps':steps,'blockers':blockers,'status':'PASS' if not blockers else 'FAIL'}
 OUT.mkdir(parents=True,exist_ok=True); out=OUT/f'mega_gate_regression_loop_{stamp()}.json'; payload['generated_at']=datetime.now(timezone.utc).isoformat().replace('+00:00','Z'); payload['report_path']=rel(out); out.write_text(json.dumps(payload,indent=2),encoding='utf-8'); print('REPORT_PATH='+rel(out)); print('MEGA_GATE_REGRESSION_LOOP='+payload['status']); return 0 if not blockers else 4
if __name__=='__main__': raise SystemExit(main())
