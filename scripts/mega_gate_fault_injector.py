#!/usr/bin/env python3
from __future__ import annotations
import json,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'05_OUTPUTS/chaos'
def ts(): return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
def rel(p):
 try:return str(Path(p).resolve().relative_to(ROOT))
 except Exception:return str(p)
def main():
 OUT.mkdir(parents=True,exist_ok=True); bad=OUT/f'mega_gate_corrupt_child_{ts()}.json'; bad.write_text('{"schema":"lucidota.mega_gate.v2","status":"PASS"}',encoding='utf-8')
 proc=subprocess.run([sys.executable,'scripts/lucidota_mega_gate.py','--validate-report',str(bad)],cwd=ROOT,text=True,capture_output=True)
 ok=proc.returncode!=0
 payload={'action':'mega_gate_fault_injector','corrupt_report':rel(bad),'validator_rc':proc.returncode,'expected_rejection':ok,'blockers':[] if ok else ['CORRUPT_REPORT_NOT_REJECTED'],'status':'PASS' if ok else 'FAIL'}
 p=OUT/f'mega_gate_fault_injector_{ts()}.json'; payload['report_path']=rel(p); p.write_text(json.dumps(payload,indent=2),encoding='utf-8'); print('REPORT_PATH='+rel(p)); print('MEGA_GATE_FAULT='+payload['status']); return 0 if ok else 4
if __name__=='__main__': raise SystemExit(main())
