#!/usr/bin/env python3
from __future__ import annotations
import json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'05_OUTPUTS/tests'; OUT.mkdir(parents=True,exist_ok=True)
def run(cmd): return subprocess.run(cmd,cwd=ROOT,text=True,capture_output=True)
def main():
 bad=OUT/'mega_gate_bad_fixture.json'; bad.write_text(json.dumps({'schema':'lucidota.mega_gate.v2','status':'PASS','repairs_applied':[],'metrics':{'steps_total':1,'steps_passed':1,'chrono_ranking_violations':0,'chrono_service_active':True},'blockers':[]}),encoding='utf-8')
 good=run([sys.executable,'scripts/lucidota_mega_gate.py'])
 rp=None
 for l in good.stdout.splitlines():
  if l.startswith('REPORT_PATH='): rp=l.split('=',1)[1]
 good_val=run([sys.executable,'scripts/lucidota_mega_gate.py','--validate-report',rp or 'missing'])
 bad_val=run([sys.executable,'scripts/lucidota_mega_gate.py','--validate-report',str(bad)])
 metrics=run([sys.executable,'scripts/mega_gate_metrics_validator.py','--report',rp or 'missing'])
 ok=good.returncode==0 and good_val.returncode==0 and bad_val.returncode!=0 and metrics.returncode==0
 payload={'status':'PASS' if ok else 'FAIL','good_report':rp,'good_rc':good.returncode,'good_validate_rc':good_val.returncode,'bad_validate_rc':bad_val.returncode,'metrics_rc':metrics.returncode}
 out=OUT/'mega_gate_v2_repairs_result.json'; out.write_text(json.dumps(payload,indent=2),encoding='utf-8'); print('REPORT_PATH='+str(out.relative_to(ROOT))); print('MEGA_GATE_V2_REPAIRS_TEST='+payload['status']); return 0 if ok else 4
if __name__=='__main__': raise SystemExit(main())
