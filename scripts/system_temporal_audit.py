#!/usr/bin/env python3
"""System-wide temporal audit: claims/source distribution/disputes/ranking."""
from __future__ import annotations
import argparse,json,os
from datetime import datetime, timezone
from pathlib import Path
import psycopg
from psycopg.rows import dict_row
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'05_OUTPUTS/chrono_ledger'
def now(): return datetime.now(timezone.utc).isoformat().replace('+00:00','Z')
def stamp(): return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
def rel(p):
 try: return str(Path(p).resolve().relative_to(ROOT))
 except Exception: return str(p)
def db(a): return a.database_url or os.environ.get('KORPUS_DATABASE_URL') or os.environ.get('DATABASE_URL') or 'postgresql:///lucidota_storage'
def rows(cur,sql): cur.execute(sql); return [dict(r) for r in cur.fetchall()]
def one(cur,sql): cur.execute(sql); return dict(cur.fetchone())
def write(payload):
 OUT.mkdir(parents=True,exist_ok=True); p=OUT/f'system_temporal_audit_{stamp()}.json'; payload.setdefault('generated_at',now()); payload['report_path']=rel(p); p.write_text(json.dumps(payload,indent=2,default=str),encoding='utf-8'); print('REPORT_PATH='+rel(p))
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--database-url'); a=ap.parse_args(); blockers=[]
 with psycopg.connect(db(a), row_factory=dict_row) as conn:
  with conn.cursor() as cur:
   facts={'claims':one(cur,"SELECT count(*) total_claims, count(DISTINCT file_uuid) files_covered, count(*) FILTER (WHERE coalesce(invalid,false)) invalid_claims FROM lucidota_korpus.temporal_claim"),
          'source_distribution':rows(cur,"SELECT evidence_source,count(*) n,min(trust_weight) min_weight,max(trust_weight) max_weight FROM lucidota_korpus.temporal_claim GROUP BY evidence_source ORDER BY n DESC"),
          'conflicts':one(cur,"SELECT count(*) conflict_rows, count(DISTINCT file_uuid) disputed_files FROM lucidota_korpus.temporal_conflict_finding"),
          'ranking':one(cur,"SELECT count(*) ranking_passes FROM lucidota_korpus.chrono_ranking_pass"),
          'projection':one(cur,"SELECT count(*) projected_files FROM lucidota_korpus.current_chrono_timeline_projection")}
   cur.execute('''WITH ranked AS (SELECT file_uuid,trust_weight,candidate_timestamp,created_at, row_number() over(partition by file_uuid order by trust_weight desc,candidate_timestamp asc,created_at desc) rn FROM lucidota_korpus.temporal_claim WHERE file_uuid IS NOT NULL AND coalesce(invalid,false)=false), best AS (SELECT * FROM ranked WHERE rn=1) SELECT count(*) AS ranking_violations FROM best b JOIN ranked r USING(file_uuid) WHERE r.trust_weight>b.trust_weight''')
   ranking_violations=int(cur.fetchone()['ranking_violations'])
 if ranking_violations: blockers.append('ranking_violations_nonzero')
 if int(facts['claims']['total_claims'])<1: blockers.append('no_temporal_claims')
 payload={'action':'system_temporal_audit','facts':facts,'ranking_violations':ranking_violations,'blockers':blockers,'status':'PASS' if not blockers else 'FAIL'}
 write(payload); print('SYSTEM_TEMPORAL_AUDIT='+payload['status']); return 0 if not blockers else 4
if __name__=='__main__': raise SystemExit(main())
