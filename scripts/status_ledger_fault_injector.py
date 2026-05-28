#!/usr/bin/env python3
from __future__ import annotations
import json
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'05_OUTPUTS/chaos'
def ts():return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
def rel(p):
 try:return str(Path(p).resolve().relative_to(ROOT))
 except Exception:return str(p)
def main():
 d=json.load(open(ROOT/'05_OUTPUTS/status_ledger.json')); d['software'][0]['executed']='yes'; d['software'][0]['evidence']=''
 OUT.mkdir(parents=True,exist_ok=True); fixture=OUT/f'status_ledger_corrupt_copy_{ts()}.json'; fixture.write_text(json.dumps(d,indent=2),encoding='utf-8')
 ok=not d['software'][0]['evidence']
 payload={'action':'status_ledger_fault_injector','fixture':rel(fixture),'expected_fault':'executed_missing_evidence','blockers':[] if ok else ['FIXTURE_NOT_CORRUPT'],'status':'PASS' if ok else 'FAIL'}
 p=OUT/f'status_ledger_fault_injector_{ts()}.json'; payload['report_path']=rel(p); p.write_text(json.dumps(payload,indent=2),encoding='utf-8'); print('REPORT_PATH='+rel(p)); print('STATUS_LEDGER_FAULT=PASS'); return 0
if __name__=='__main__': raise SystemExit(main())
