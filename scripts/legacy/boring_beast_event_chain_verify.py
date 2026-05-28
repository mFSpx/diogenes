#!/usr/bin/env python3
from __future__ import annotations
import json,re,subprocess,sys,os
from datetime import datetime,timezone
from pathlib import Path
import psycopg
from psycopg.rows import dict_row
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'05_OUTPUTS/boring_beast'
def ts(): return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
def rel(p):
 try: return str(Path(p).resolve().relative_to(ROOT))
 except Exception: return str(p)
def main():
 proc=subprocess.run([sys.executable,'scripts/boring_beast.py','e2e','--execute'],cwd=ROOT,text=True,capture_output=True)
 job_ids=re.findall(r'JOB_UUID=([0-9a-f-]+)',proc.stdout); job=job_ids[0] if job_ids else None; blockers=[]
 if proc.returncode!=0: blockers.append('BORING_BEAST_E2E_FAILED')
 if not job: blockers.append('JOB_UUID_NOT_EMITTED')
 facts={}
 if job:
  with psycopg.connect(os.environ.get('DBOS_SYSTEM_DATABASE_URL') or 'postgresql:///lucidota_state', row_factory=dict_row) as c:
   with c.cursor() as cur:
    cur.execute('SELECT count(*) events FROM lucidota_control.dbos_queue_event WHERE job_uuid=%s::uuid',(job,)); facts['queue_events']=int(cur.fetchone()['events'])
    cur.execute('SELECT count(*) records FROM lucidota_control.boring_execution_record WHERE task_id=%s OR detail::text LIKE %s',(job,'%'+job+'%')); facts['execution_records']=int(cur.fetchone()['records'])
  with psycopg.connect(os.environ.get('KORPUS_DATABASE_URL') or os.environ.get('DATABASE_URL') or 'postgresql:///lucidota_storage', row_factory=dict_row) as c:
   with c.cursor() as cur:
    cur.execute("SELECT count(*) claims FROM lucidota_korpus.temporal_claim WHERE detail::text LIKE %s",('%'+job+'%',)); facts['chrono_claims']=int(cur.fetchone()['claims'])
  if facts.get('queue_events',0)<2: blockers.append('QUEUE_EVENT_CHAIN_INCOMPLETE')
  if facts.get('chrono_claims',0)<1: blockers.append('CHRONO_EVENT_MISSING')
 payload={'action':'boring_beast_event_chain_verify','job_uuid':job,'facts':facts,'stdout_tail':proc.stdout[-2000:],'stderr_tail':proc.stderr[-1000:],'blockers':blockers,'status':'PASS' if not blockers else 'FAIL'}
 OUT.mkdir(parents=True,exist_ok=True); p=OUT/f'boring_beast_event_chain_verify_{ts()}.json'; payload['generated_at']=datetime.now(timezone.utc).isoformat().replace('+00:00','Z'); payload['report_path']=rel(p); p.write_text(json.dumps(payload,indent=2),encoding='utf-8'); print('REPORT_PATH='+rel(p)); print('BORING_BEAST_EVENT_CHAIN='+payload['status']); return 0 if not blockers else 4
if __name__=='__main__': raise SystemExit(main())
