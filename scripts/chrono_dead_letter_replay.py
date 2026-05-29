#!/usr/bin/env python3
"""Chrono dead-letter replay safety CLI.

Dry-run by default. Execute runs the Chrono backfill safely and marks a dead-letter
resolved only if the target file gains a temporal_claim during the replay. It never
deletes dead-letter rows and never mutates file_object custody rows.
"""
from __future__ import annotations
import argparse,json,os,subprocess,sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import psycopg
from psycopg.rows import dict_row
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'05_OUTPUTS/chrono_ledger'; BIN=ROOT/'01_REPOS/lucidota_etl/target/release/lucidota-chrono-ledger'
def now(): return datetime.now(timezone.utc).isoformat().replace('+00:00','Z')
def stamp(): return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
def rel(p):
 try: return str(Path(p).resolve().relative_to(ROOT))
 except Exception: return str(p)
def db(a): return a.database_url or os.environ.get('KORPUS_DATABASE_URL') or os.environ.get('DATABASE_URL') or 'postgresql:///lucidota_storage'
def write(name,payload):
 OUT.mkdir(parents=True,exist_ok=True); p=OUT/f'chrono_dead_letter_replay_{name}_{stamp()}.json'; payload.setdefault('generated_at',now()); payload['report_path']=rel(p); p.write_text(json.dumps(payload,indent=2,default=str),encoding='utf-8'); print(f'REPORT_PATH={rel(p)}'); return p

def unresolved(a,limit=10):
 with psycopg.connect(db(a), row_factory=dict_row) as conn:
  with conn.cursor() as cur:
   cur.execute('SELECT dead_letter_uuid::text,file_uuid::text,extractor_name,error_kind,error_message,attempt_count,resolved FROM lucidota_korpus.chrono_dead_letter WHERE resolved=false ORDER BY last_seen_at DESC LIMIT %s',(limit,))
   return [dict(r) for r in cur.fetchall()]

def claim_count(cur,file_uuid):
 cur.execute('SELECT count(*) AS n FROM lucidota_korpus.temporal_claim WHERE file_uuid=%s::uuid',(file_uuid,)); return int(cur.fetchone()['n'])

def list_cmd(a):
 rows=unresolved(a,a.limit); write('list',{'action':'list','execute_performed':False,'rows':rows,'unresolved_count':len(rows),'db_writes_performed':False,'graph_writes_performed':False}); print(f'CHRONO_DLQ_ROWS={len(rows)}'); return 0

def replay(a):
 blockers=[]; target=None; before=None; after=None; proc=None; resolved=False
 with psycopg.connect(db(a), row_factory=dict_row) as conn:
  with conn.cursor() as cur:
   if a.dead_letter_uuid:
    cur.execute('SELECT dead_letter_uuid::text,file_uuid::text,extractor_name,error_kind,resolved FROM lucidota_korpus.chrono_dead_letter WHERE dead_letter_uuid=%s::uuid',(a.dead_letter_uuid,))
   else:
    cur.execute('SELECT dead_letter_uuid::text,file_uuid::text,extractor_name,error_kind,resolved FROM lucidota_korpus.chrono_dead_letter WHERE resolved=false ORDER BY last_seen_at DESC LIMIT 1')
   target=cur.fetchone()
   if not target: blockers.append('dead_letter_not_found')
   elif target['resolved']: blockers.append('dead_letter_already_resolved')
   else: before=claim_count(cur,target['file_uuid'])
 if a.execute and not blockers:
  if not BIN.exists(): blockers.append('chrono_binary_missing')
  else:
   cmd=[str(BIN),'backfill','--database-url',db(a),'--limit',str(a.limit)]
   proc=subprocess.run(cmd,cwd=ROOT,text=True,capture_output=True,timeout=180)
   if proc.returncode!=0: blockers.append('chrono_backfill_failed')
   with psycopg.connect(db(a), row_factory=dict_row) as conn:
    with conn.cursor() as cur:
     after=claim_count(cur,target['file_uuid']) if target else None
     if proc.returncode==0 and after is not None and before is not None and after>before:
      cur.execute("UPDATE lucidota_korpus.chrono_dead_letter SET resolved=true,resolved_at=now(),detail=detail || %s::jsonb WHERE dead_letter_uuid=%s::uuid",(json.dumps({'resolved_by':'scripts/chrono_dead_letter_replay.py','claims_before':before,'claims_after':after}),target['dead_letter_uuid']))
      resolved=True
    conn.commit()
 else:
  after=before
 report={'action':'replay','execute_performed':bool(a.execute),'db_writes_performed':bool(a.execute),'graph_writes_performed':False,'dead_letter':dict(target) if target else None,'claims_before':before,'claims_after':after,'resolved_mutated':resolved,'deleted_rows':0,'backfill_command':proc.args if proc else None,'backfill_returncode':proc.returncode if proc else None,'backfill_stdout_tail':proc.stdout[-2000:] if proc else None,'backfill_stderr_tail':proc.stderr[-2000:] if proc else None,'blockers':blockers,'status':'PASS' if not blockers else 'FAIL'}
 write('replay_execute' if a.execute else 'replay_dry_run',report); print('CHRONO_DLQ_REPLAY='+report['status']); return 0 if not blockers else 4

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--database-url'); sub=ap.add_subparsers(dest='cmd',required=True)
 l=sub.add_parser('list'); l.add_argument('--limit',type=int,default=10)
 r=sub.add_parser('replay'); r.add_argument('--dead-letter-uuid'); r.add_argument('--limit',type=int,default=100); r.add_argument('--execute',action='store_true')
 a=ap.parse_args(); return list_cmd(a) if a.cmd=='list' else replay(a)
if __name__=='__main__': raise SystemExit(main())
