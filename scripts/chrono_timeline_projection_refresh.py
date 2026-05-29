#!/usr/bin/env python3
"""Refresh derived current Chrono timeline projection with selected claim UUIDs."""
from __future__ import annotations
import argparse,json,os
from datetime import datetime, timezone
from pathlib import Path
import psycopg
from psycopg.rows import dict_row
ROOT=Path(__file__).resolve().parents[1]; SCHEMA=ROOT/'06_SCHEMA/100_chrono_timeline_projection_refresh.sql'; OUT=ROOT/'05_OUTPUTS/chrono_ledger'
def now(): return datetime.now(timezone.utc).isoformat().replace('+00:00','Z')
def stamp(): return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
def rel(p):
 try: return str(Path(p).resolve().relative_to(ROOT))
 except Exception: return str(p)
def db(a): return a.database_url or os.environ.get('KORPUS_DATABASE_URL') or os.environ.get('DATABASE_URL') or 'postgresql:///lucidota_storage'
def write(name,payload):
 OUT.mkdir(parents=True,exist_ok=True); p=OUT/f'chrono_timeline_projection_refresh_{name}_{stamp()}.json'; payload.setdefault('generated_at',now()); payload['report_path']=rel(p); p.write_text(json.dumps(payload,indent=2,default=str),encoding='utf-8'); print(f'REPORT_PATH={rel(p)}'); return p
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--database-url'); ap.add_argument('--execute',action='store_true'); a=ap.parse_args()
 payload={'action':'refresh_projection','execute_performed':bool(a.execute),'db_writes_performed':False,'graph_writes_performed':False,'blockers':[]}
 sql_best="""
 WITH ranked AS (
   SELECT tc.claim_uuid, tc.file_uuid, tc.candidate_timestamp, tc.evidence_source, tc.trust_weight, tc.source_sha256,
          row_number() OVER (PARTITION BY tc.file_uuid ORDER BY tc.trust_weight DESC, tc.candidate_timestamp ASC, tc.created_at DESC) AS rn
   FROM lucidota_korpus.temporal_claim tc
   WHERE tc.file_uuid IS NOT NULL AND coalesce(tc.invalid,false)=false
 )
 SELECT * FROM ranked WHERE rn=1
 """
 with psycopg.connect(db(a), row_factory=dict_row) as conn:
  with conn.cursor() as cur:
   cur.execute(SCHEMA.read_text())
   cur.execute('SELECT count(*) AS n FROM ('+sql_best+') b'); selected=int(cur.fetchone()['n'])
   cur.execute('''WITH ranked AS (SELECT tc.file_uuid, tc.claim_uuid, tc.trust_weight, tc.candidate_timestamp, row_number() OVER (PARTITION BY tc.file_uuid ORDER BY tc.trust_weight DESC, tc.candidate_timestamp ASC, tc.created_at DESC) rn FROM lucidota_korpus.temporal_claim tc WHERE tc.file_uuid IS NOT NULL AND coalesce(tc.invalid,false)=false), best AS (SELECT * FROM ranked WHERE rn=1)
                 SELECT count(*) AS n FROM best b JOIN lucidota_korpus.temporal_claim c ON c.file_uuid=b.file_uuid AND coalesce(c.invalid,false)=false WHERE c.trust_weight>b.trust_weight''')
   violations=int(cur.fetchone()['n'])
   if a.execute:
    cur.execute('''WITH ranked AS (
        SELECT tc.claim_uuid, tc.file_uuid, tc.candidate_timestamp, tc.evidence_source, tc.trust_weight, tc.source_sha256,
               row_number() OVER (PARTITION BY tc.file_uuid ORDER BY tc.trust_weight DESC, tc.candidate_timestamp ASC, tc.created_at DESC) AS rn
        FROM lucidota_korpus.temporal_claim tc WHERE tc.file_uuid IS NOT NULL AND coalesce(tc.invalid,false)=false)
        INSERT INTO lucidota_korpus.current_chrono_timeline_projection(file_uuid,selected_claim_uuid,resolved_timestamp,evidence_source,trust_weight,source_sha256,projection_refreshed_at,detail)
        SELECT file_uuid, claim_uuid, candidate_timestamp, evidence_source, trust_weight, source_sha256, now(), jsonb_build_object('script','scripts/chrono_timeline_projection_refresh.py') FROM ranked WHERE rn=1
        ON CONFLICT(file_uuid) DO UPDATE SET selected_claim_uuid=EXCLUDED.selected_claim_uuid,resolved_timestamp=EXCLUDED.resolved_timestamp,evidence_source=EXCLUDED.evidence_source,trust_weight=EXCLUDED.trust_weight,source_sha256=EXCLUDED.source_sha256,projection_refreshed_at=now(),detail=EXCLUDED.detail''')
    refreshed=cur.rowcount; payload.update({'db_writes_performed':True,'refreshed_rows':refreshed})
   cur.execute('SELECT count(*) AS n FROM lucidota_korpus.current_chrono_timeline_projection p LEFT JOIN lucidota_korpus.temporal_claim tc ON tc.claim_uuid=p.selected_claim_uuid WHERE tc.claim_uuid IS NULL')
   broken_links=int(cur.fetchone()['n'])
  conn.commit()
 payload.update({'selected_files':selected,'ranking_violations':violations,'broken_claim_links':broken_links})
 if violations: payload['blockers'].append('ranking_violations_nonzero')
 if broken_links: payload['blockers'].append('projection_claim_link_missing')
 payload['status']='PASS' if not payload['blockers'] else 'FAIL'; write('execute' if a.execute else 'dry_run',payload); print('CHRONO_PROJECTION_REFRESH='+payload['status']); return 0 if payload['status']=='PASS' else 4
if __name__=='__main__': raise SystemExit(main())
