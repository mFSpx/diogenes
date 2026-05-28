#!/usr/bin/env python3
"""Simulate a missed Chrono LISTEN/NOTIFY event and prove replay/backfill recovery."""
from __future__ import annotations
import argparse,hashlib,json,os,subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import psycopg
from psycopg.rows import dict_row
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'05_OUTPUTS/chrono_ledger'; BIN=ROOT/'01_REPOS/lucidota_etl/target/release/lucidota-chrono-ledger'
def nowdt(): return datetime.now(timezone.utc)
def now(): return nowdt().isoformat().replace('+00:00','Z')
def stamp(): return nowdt().strftime('%Y%m%dT%H%M%S%fZ')
def rel(p):
 try: return str(Path(p).resolve().relative_to(ROOT))
 except Exception: return str(p)
def db(a): return a.database_url or os.environ.get('KORPUS_DATABASE_URL') or os.environ.get('DATABASE_URL') or 'postgresql:///lucidota_storage'
def write(payload):
 OUT.mkdir(parents=True,exist_ok=True); p=OUT/f'chrono_missed_event_simulation_{payload["status"].lower()}_{stamp()}.json'; payload.setdefault('generated_at',now()); payload['report_path']=rel(p); p.write_text(json.dumps(payload,indent=2,default=str),encoding='utf-8'); print(f'REPORT_PATH={rel(p)}'); return p
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--database-url'); ap.add_argument('--execute',action='store_true'); a=ap.parse_args(); blockers=[]; file_uuid=None; before=0; after=0; proc=None
 if not a.execute: write({'action':'simulate','execute_performed':False,'blockers':['execute_required']}); return 2
 ts=nowdt().strftime('%Y%m%dT%H%M%S'); source=f'round54-missed-event-{ts}-{os.getpid()}'; sha=hashlib.sha256(source.encode()).hexdigest(); path=f'KRAMPUSCHEWING/MISSED_EVENT_SIM/{ts}_round54.txt'
 with psycopg.connect(db(a), row_factory=dict_row) as conn:
  with conn.cursor() as cur:
   cur.execute('''INSERT INTO lucidota_korpus.file_object(sha256,size_bytes,mime,file_kind,suffix,cas_uri,locked_relative_path,first_seen_path,detail,status)
                  VALUES (%s,%s,'text/plain','missed_event_simulation','.txt',%s,%s,%s,%s::jsonb,'indexed')
                  ON CONFLICT(sha256) DO UPDATE SET last_seen_at=now(), seen_count=lucidota_korpus.file_object.seen_count+1
                  RETURNING file_uuid::text''',(sha,len(source),f'cas://sha256/{sha}',path,path,json.dumps({'script':'scripts/chrono_missed_event_simulation.py','simulates':'missed_notify_replay'})))
   file_uuid=cur.fetchone()['file_uuid']
   cur.execute('SELECT count(*) AS n FROM lucidota_korpus.temporal_claim WHERE file_uuid=%s::uuid',(file_uuid,)); before=int(cur.fetchone()['n'])
  conn.commit()
 if not BIN.exists(): blockers.append('chrono_binary_missing')
 else:
  proc=subprocess.run([str(BIN),'backfill','--database-url',db(a),'--limit','100'],cwd=ROOT,text=True,capture_output=True,timeout=180)
  if proc.returncode!=0: blockers.append('chrono_backfill_failed')
 with psycopg.connect(db(a), row_factory=dict_row) as conn:
  with conn.cursor() as cur:
   cur.execute('SELECT count(*) AS n FROM lucidota_korpus.temporal_claim WHERE file_uuid=%s::uuid',(file_uuid,)); after=int(cur.fetchone()['n'])
 if after<1: blockers.append('missed_event_target_has_no_temporal_claim')
 report={'action':'simulate_missed_event','execute_performed':True,'db_writes_performed':True,'graph_writes_performed':False,'file_uuid':file_uuid,'source_sha256':sha,'first_seen_path':path,'claims_before_backfill':before,'claims_after_backfill':after,'backfill_returncode':proc.returncode if proc else None,'backfill_stdout_tail':proc.stdout[-2000:] if proc else None,'backfill_stderr_tail':proc.stderr[-2000:] if proc else None,'blockers':blockers,'status':'PASS' if not blockers else 'FAIL'}
 write(report); print('CHRONO_MISSED_EVENT_SIM='+report['status']); return 0 if not blockers else 4
if __name__=='__main__': raise SystemExit(main())
