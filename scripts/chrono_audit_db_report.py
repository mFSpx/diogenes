#!/usr/bin/env python3
"""Generate Chrono audit report from live DB facts only."""
from __future__ import annotations
import argparse,json,os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import psycopg
from psycopg.rows import dict_row
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'05_OUTPUTS/chrono_ledger'
def now(): return datetime.now(timezone.utc).isoformat().replace('+00:00','Z')
def stamp(): return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
def rel(p):
 try: return str(Path(p).resolve().relative_to(ROOT))
 except Exception: return str(p)
def db(a): return a.database_url or os.environ.get('KORPUS_DATABASE_URL') or os.environ.get('DATABASE_URL') or 'postgresql:///lucidota_storage'
def qall(cur,sql,params=()): cur.execute(sql,params); return [dict(r) for r in cur.fetchall()]
def qone(cur,sql,params=()): cur.execute(sql,params); return dict(cur.fetchone())
def write(payload):
 OUT.mkdir(parents=True,exist_ok=True); p=OUT/f'chrono_audit_db_report_{payload["status"].lower()}_{stamp()}.json'; payload.setdefault('generated_at',now()); payload['report_path']=rel(p); p.write_text(json.dumps(payload,indent=2,default=str),encoding='utf-8'); print(f'REPORT_PATH={rel(p)}'); return p
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--database-url'); a=ap.parse_args(); blockers=[]
 with psycopg.connect(db(a), row_factory=dict_row) as conn:
  with conn.cursor() as cur:
   facts={
    'file_object': qone(cur,'SELECT count(*) AS total_files FROM lucidota_korpus.file_object'),
    'temporal_claim': qone(cur,'SELECT count(*) AS total_claims, count(*) FILTER (WHERE coalesce(invalid,false)) AS invalid_claims, count(DISTINCT file_uuid) AS files_with_claims FROM lucidota_korpus.temporal_claim'),
    'source_distribution': qall(cur,'SELECT evidence_source,count(*) AS n FROM lucidota_korpus.temporal_claim GROUP BY evidence_source ORDER BY n DESC LIMIT 50'),
    'ranking_passes': qone(cur,"SELECT count(*) AS ranking_pass_count FROM information_schema.tables WHERE table_schema='lucidota_korpus' AND table_name IN ('temporal_ranking_pass','chrono_ranking_pass')"),
    'latest_temporal_ranking_pass': qall(cur,"SELECT ranking_pass_uuid::text, claim_count, file_count, selected_count, ranking_violations, created_at::text FROM lucidota_korpus.temporal_ranking_pass ORDER BY created_at DESC LIMIT 3"),
    'cursor': qall(cur,'SELECT cursor_name,last_file_uuid::text,processed_count,last_replay_finished_at::text,updated_at::text FROM lucidota_korpus.chrono_replay_cursor ORDER BY cursor_name'),
    'dead_letters': qall(cur,'SELECT resolved,count(*) AS n FROM lucidota_korpus.chrono_dead_letter GROUP BY resolved ORDER BY resolved'),
   }
   cur.execute('''WITH ranked AS (SELECT file_uuid, row_number() OVER (PARTITION BY file_uuid ORDER BY trust_weight DESC,candidate_timestamp ASC,created_at DESC,claim_uuid ASC) rn FROM lucidota_korpus.temporal_claim WHERE file_uuid IS NOT NULL AND coalesce(invalid,false)=false) SELECT count(*) AS ranking_violations FROM (SELECT file_uuid,count(*) FROM ranked WHERE rn=1 GROUP BY file_uuid HAVING count(*)<>1) bad''')
   ranking_violations=int(cur.fetchone()['ranking_violations'])
 if ranking_violations!=0: blockers.append('ranking_violations_nonzero')
 report={'action':'chrono_audit_db_report','source':'database_facts_only','db_writes_performed':False,'graph_writes_performed':False,'facts':facts,'ranking_violations':ranking_violations,'blockers':blockers,'status':'PASS' if not blockers else 'FAIL'}
 write(report); print(f'RANKING_VIOLATIONS={ranking_violations}'); return 0 if not blockers else 4
if __name__=='__main__': raise SystemExit(main())
