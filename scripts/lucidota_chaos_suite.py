#!/usr/bin/env python3
from __future__ import annotations
import json,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'05_OUTPUTS/chaos'
def ts():return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
def rel(p):
 try:return str(Path(p).resolve().relative_to(ROOT))
 except Exception:return str(p)
def run(cmd):
 p=subprocess.run(cmd,cwd=ROOT,text=True,capture_output=True); rp=None
 for l in p.stdout.splitlines():
  if l.startswith('REPORT_PATH='): rp=l.split('=',1)[1]
 return {'cmd':' '.join(cmd),'rc':p.returncode,'report_path':rp,'stdout_tail':p.stdout[-1000:],'stderr_tail':p.stderr[-1000:]}
def main():
 steps=[run([sys.executable,'scripts/mega_gate_fault_injector.py']),run([sys.executable,'scripts/status_ledger_fault_injector.py']),run([sys.executable,'scripts/tickletrunk_fault_injector.py']),run([sys.executable,'scripts/recovery_matrix.py'])]
 blockers=[s['cmd'] for s in steps if s['rc']!=0]
 payload={'action':'lucidota_chaos_suite','steps':steps,'blockers':blockers,'status':'PASS' if not blockers else 'FAIL'}; OUT.mkdir(parents=True,exist_ok=True); p=OUT/f'chaos_suite_{ts()}.json'; payload['report_path']=rel(p); p.write_text(json.dumps(payload,indent=2),encoding='utf-8'); print('REPORT_PATH='+rel(p)); print('CHAOS_SUITE='+payload['status']); return 0 if not blockers else 4
if __name__=='__main__': raise SystemExit(main())
