from __future__ import annotations
import json
import subprocess
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def _report(stdout:str)->Path:
 for line in stdout.splitlines():
  if line.startswith('REPORT_PATH='): return ROOT/line.split('=',1)[1]
 raise AssertionError(stdout)
def test_chrono_conservation_verify_passes():
 proc=subprocess.run([sys.executable,'scripts/chrono_conservation_verify.py','verify'],cwd=ROOT,text=True,capture_output=True,check=True)
 data=json.loads(_report(proc.stdout).read_text())
 assert data['status']=='PASS'
 assert next(c for c in data['checks'] if c['check']=='ranking_violations_zero')['value']==0
 assert all(c['passed'] for c in data['checks'])
def test_chrono_db_fact_audit_passes():
 proc=subprocess.run([sys.executable,'scripts/chrono_audit_db_report.py'],cwd=ROOT,text=True,capture_output=True,check=True)
 data=json.loads(_report(proc.stdout).read_text())
 assert data['status']=='PASS'
 assert int(data['ranking_violations'])==0
 assert data['source']=='database_facts_only'
if __name__=='__main__':
 test_chrono_conservation_verify_passes(); test_chrono_db_fact_audit_passes(); print('CHRONO_CONSERVATION_TESTS=PASS')
