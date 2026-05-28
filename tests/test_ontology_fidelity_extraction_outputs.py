#!/usr/bin/env python3
"""Regression suite: extraction outputs fail on softened ontology and pass exact labels."""
from __future__ import annotations
import json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'05_OUTPUTS/contracts'; OUT.mkdir(parents=True,exist_ok=True)
def rp(stdout):
 for line in stdout.splitlines():
  if line.startswith('REPORT_PATH='): return line.split('=',1)[1]
 return None
def run(cmd): return subprocess.run(cmd,cwd=ROOT,text=True,capture_output=True)
def main():
 good=OUT/'ontology_exact_extraction_fixture.json'; bad=OUT/'ontology_softened_extraction_fixture.json'
 labels=json.load(open(ROOT/'05_OUTPUTS/contracts/operator_ontology_labels.json'))['required_exact_labels']
 good.write_text(json.dumps({'extracted_labels':labels,'note':'Cruelty Protocols and Master’s Eye remain exact'},ensure_ascii=False),encoding='utf-8')
 bad.write_text(json.dumps({'extracted_labels':['protective helper mode','wellness protector'],'note':'softened substitute without canonical term'},ensure_ascii=False),encoding='utf-8')
 init=run([sys.executable,'scripts/operator_ontology_fidelity_guard.py','init-schema','--execute'])
 g=run([sys.executable,'scripts/operator_ontology_fidelity_guard.py','check','--input',str(good),'--execute'])
 b=run([sys.executable,'scripts/operator_ontology_fidelity_guard.py','check','--input',str(bad),'--execute'])
 ok=init.returncode==0 and g.returncode==0 and b.returncode!=0
 payload={'status':'PASS' if ok else 'FAIL','good_report':rp(g.stdout),'bad_report':rp(b.stdout),'init_rc':init.returncode,'good_rc':g.returncode,'bad_rc':b.returncode,'bad_expected_failure':True}
 out=OUT/'ontology_fidelity_extraction_regression_result.json'; out.write_text(json.dumps(payload,indent=2),encoding='utf-8')
 print('REPORT_PATH='+str(out.relative_to(ROOT))); print('ONTOLOGY_EXTRACTION_REGRESSION='+payload['status']); return 0 if ok else 4
if __name__=='__main__': raise SystemExit(main())
