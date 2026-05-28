#!/usr/bin/env python3
"""Full Chrono conservation gate after expanded ingest."""
from __future__ import annotations
import json,subprocess,sys
from datetime import datetime, timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'05_OUTPUTS/chrono_ledger'
def stamp(): return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
def rel(p):
 try: return str(Path(p).resolve().relative_to(ROOT))
 except Exception: return str(p)
def runc(cmd):
 p=subprocess.run(cmd,cwd=ROOT,text=True,capture_output=True); rp=None
 for line in p.stdout.splitlines():
  if line.startswith('REPORT_PATH='): rp=line.split('=',1)[1]
 return {'command':' '.join(cmd),'rc':p.returncode,'report_path':rp,'stdout_tail':p.stdout[-1500:],'stderr_tail':p.stderr[-1500:]}
def main():
 steps=[runc([sys.executable,'scripts/chrono_conservation_verify.py','verify']), runc([sys.executable,'scripts/chrono_audit_db_report.py']), runc([sys.executable,'scripts/chrono_source_trust_validator.py']), runc([sys.executable,'scripts/chrono_temporal_conflict_detector.py','--execute','--limit','2000'])]
 blockers=[f"step_failed:{s['command']}" for s in steps if s['rc']!=0]
 payload={'action':'chrono_full_conservation_gate','steps':steps,'blockers':blockers,'status':'PASS' if not blockers else 'FAIL'}
 OUT.mkdir(parents=True,exist_ok=True); p=OUT/f'chrono_full_conservation_gate_{stamp()}.json'; payload['report_path']=rel(p); payload['generated_at']=datetime.now(timezone.utc).isoformat().replace('+00:00','Z'); p.write_text(json.dumps(payload,indent=2),encoding='utf-8'); print('REPORT_PATH='+rel(p)); print('CHRONO_FULL_CONSERVATION='+payload['status']); return 0 if not blockers else 4
if __name__=='__main__': raise SystemExit(main())
