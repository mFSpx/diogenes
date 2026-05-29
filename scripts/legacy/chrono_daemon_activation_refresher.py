#!/usr/bin/env python3
"""Refresh Chrono daemon activation report and write a runtime fact."""
from __future__ import annotations
import argparse,json,os,subprocess,sys
from datetime import datetime, timezone
from pathlib import Path
import psycopg
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'05_OUTPUTS/chrono_ledger'
def now(): return datetime.now(timezone.utc).isoformat().replace('+00:00','Z')
def rel(p):
 try: return str(Path(p).resolve().relative_to(ROOT))
 except Exception: return str(p)
def state_db(a): return a.state_database_url or os.environ.get('DBOS_SYSTEM_DATABASE_URL') or 'postgresql:///lucidota_state'
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--state-database-url'); ap.add_argument('--execute',action='store_true'); a=ap.parse_args()
 proc=subprocess.run([sys.executable,'scripts/chrono_service_activation_report.py'],cwd=ROOT,text=True,capture_output=True)
 path=None; active=None; ranking=None
 try:
  data=json.loads(proc.stdout[proc.stdout.find('{'):])
  path=data.get('path'); active=data.get('active'); ranking=data.get('ranking_violations')
 except Exception: pass
 blockers=[]
 if proc.returncode!=0: blockers.append(f'activation_report_rc_{proc.returncode}')
 if active is not True: blockers.append('chrono_service_not_active')
 if ranking not in (0,'0'): blockers.append('ranking_violations_nonzero')
 if a.execute:
  with psycopg.connect(state_db(a)) as conn:
   with conn.cursor() as cur:
    cur.execute('''INSERT INTO lucidota_control.runtime_status_fact(subsystem,fact_key,fact_value,evidence_refs) VALUES ('Chrono-Ledger','chrono.daemon.activation_refreshed',%s::jsonb,%s::jsonb)
                   ON CONFLICT(subsystem,fact_key) DO UPDATE SET fact_value=EXCLUDED.fact_value,evidence_refs=EXCLUDED.evidence_refs,derived_at=now()''',(json.dumps({'active':active,'ranking_violations':ranking}),json.dumps({'activation_report':path,'stdout_tail':proc.stdout[-1000:],'stderr_tail':proc.stderr[-1000:]})))
   conn.commit()
 payload={'action':'chrono_daemon_activation_refresher','execute_performed':bool(a.execute),'db_writes_performed':bool(a.execute),'graph_writes_performed':False,'activation_report_path':path,'active':active,'ranking_violations':ranking,'blockers':blockers,'status':'PASS' if not blockers else 'FAIL'}
 OUT.mkdir(parents=True,exist_ok=True); p=OUT/f"chrono_daemon_activation_refresher_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}.json"; payload['report_path']=rel(p); payload['generated_at']=now(); p.write_text(json.dumps(payload,indent=2),encoding='utf-8'); print('REPORT_PATH='+rel(p)); print('CHRONO_DAEMON_REFRESH='+payload['status']); return 0 if not blockers else 4
if __name__=='__main__': raise SystemExit(main())
