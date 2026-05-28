#!/usr/bin/env python3
"""Detect disputed Chrono files from preserved temporal claims."""
from __future__ import annotations
import argparse,hashlib,json,os
from datetime import datetime, timezone
from pathlib import Path
import psycopg
from psycopg.rows import dict_row
ROOT=Path(__file__).resolve().parents[1]; SCHEMA=ROOT/'06_SCHEMA/102_chrono_temporal_conflict_detector.sql'; OUT=ROOT/'05_OUTPUTS/chrono_ledger'
def now(): return datetime.now(timezone.utc).isoformat().replace('+00:00','Z')
def stamp(): return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
def rel(p):
 try: return str(Path(p).resolve().relative_to(ROOT))
 except Exception: return str(p)
def db(a): return a.database_url or os.environ.get('KORPUS_DATABASE_URL') or os.environ.get('DATABASE_URL') or 'postgresql:///lucidota_storage'
def write(payload):
 OUT.mkdir(parents=True,exist_ok=True); p=OUT/f'chrono_temporal_conflict_detector_{stamp()}.json'; payload.setdefault('generated_at',now()); payload['report_path']=rel(p); p.write_text(json.dumps(payload,indent=2,default=str),encoding='utf-8'); print('REPORT_PATH='+rel(p))
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--database-url'); ap.add_argument('--execute',action='store_true'); ap.add_argument('--limit',type=int,default=1000); a=ap.parse_args()
 rows=[]; inserted=0
 with psycopg.connect(db(a), row_factory=dict_row) as conn:
  with conn.cursor() as cur:
   cur.execute(SCHEMA.read_text())
   cur.execute('''WITH agg AS (
      SELECT file_uuid, count(*) claim_count, count(DISTINCT candidate_timestamp) distinct_timestamp_count, count(DISTINCT evidence_source) source_count, max(trust_weight) maxw, min(trust_weight) minw, jsonb_agg(DISTINCT evidence_source) sources
      FROM lucidota_korpus.temporal_claim WHERE file_uuid IS NOT NULL AND coalesce(invalid,false)=false GROUP BY file_uuid)
      SELECT a.*, p.selected_claim_uuid::text FROM agg a LEFT JOIN lucidota_korpus.current_chrono_timeline_projection p USING(file_uuid)
      WHERE a.distinct_timestamp_count>1 OR a.source_count>1 ORDER BY a.claim_count DESC LIMIT %s''',(a.limit,))
   for r in cur.fetchall(): rows.append(dict(r))
   if a.execute:
    for r in rows:
     key=hashlib.sha256(f"{r['file_uuid']}|{r['claim_count']}|{r['distinct_timestamp_count']}|{r['source_count']}".encode()).hexdigest()
     cur.execute('''INSERT INTO lucidota_korpus.temporal_conflict_finding(file_uuid,claim_count,distinct_timestamp_count,source_count,highest_trust_weight,lowest_trust_weight,evidence_sources,selected_claim_uuid,idempotency_key,detail)
                    VALUES (%s,%s,%s,%s,%s,%s,%s::jsonb,%s::uuid,%s,%s::jsonb)
                    ON CONFLICT(idempotency_key) DO NOTHING RETURNING conflict_uuid''',(r['file_uuid'],r['claim_count'],r['distinct_timestamp_count'],r['source_count'],r['maxw'],r['minw'],json.dumps(r['sources']),r.get('selected_claim_uuid'),key,json.dumps({'script':'scripts/chrono_temporal_conflict_detector.py'})))
     if cur.fetchone(): inserted += 1
  conn.commit()
 payload={'action':'temporal_conflict_detect','execute_performed':bool(a.execute),'db_writes_performed':bool(a.execute),'graph_writes_performed':False,'disputed_files_count':len(rows),'inserted':inserted,'sample':rows[:10],'blockers':[],'status':'PASS'}
 write(payload); print('CHRONO_CONFLICT_DETECTOR=PASS'); return 0
if __name__=='__main__': raise SystemExit(main())
