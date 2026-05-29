#!/usr/bin/env python3
from __future__ import annotations
import json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'05_OUTPUTS/tests'; OUT.mkdir(parents=True,exist_ok=True)
def run(cmd): return subprocess.run(cmd,cwd=ROOT,text=True,capture_output=True)
def rp(out):
 for l in out.splitlines():
  if l.startswith('REPORT_PATH='): return l.split('=',1)[1]
 return None
def main():
 steps=[run([sys.executable,'scripts/post_gate_target_selector.py','init-schema','--execute']),run([sys.executable,'scripts/post_gate_target_selector.py','list']),run([sys.executable,'scripts/post_gate_target_selector.py','select','--target-key','chrono_audit_expansion','--evidence-ref','05_OUTPUTS/mega_gate','--execute']),run([sys.executable,'scripts/post_gate_target_selector.py','select','--target-key','chrono_audit_expansion'])]
 ok=steps[0].returncode==0 and steps[1].returncode==0 and steps[2].returncode==0 and steps[3].returncode!=0
 payload={'status':'PASS' if ok else 'FAIL','reports':[rp(s.stdout) for s in steps],'rcs':[s.returncode for s in steps]}
 out=OUT/'post_gate_target_selector_result.json'; out.write_text(json.dumps(payload,indent=2),encoding='utf-8'); print('REPORT_PATH='+str(out.relative_to(ROOT))); print('POST_GATE_TARGET_TEST='+payload['status']); return 0 if ok else 4
if __name__=='__main__': raise SystemExit(main())
