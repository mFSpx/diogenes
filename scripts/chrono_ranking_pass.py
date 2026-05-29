#!/usr/bin/env python3
"""Create immutable Chrono ranking pass selections."""
from __future__ import annotations

import argparse,json,os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import psycopg
from psycopg.rows import dict_row

ROOT=Path(__file__).resolve().parents[1]; SCHEMA=ROOT/'06_SCHEMA/087_chrono_ranking_pass.sql'; OUT=ROOT/'05_OUTPUTS/chrono_ledger'
def now(): return datetime.now(timezone.utc).isoformat().replace('+00:00','Z')
def stamp(): return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
def rel(p):
 try: return str(Path(p).resolve().relative_to(ROOT))
 except Exception: return str(p)
def db(a): return a.database_url or os.environ.get('KORPUS_DATABASE_URL') or os.environ.get('DATABASE_URL') or 'postgresql:///lucidota_storage'
def write(name,payload):
 OUT.mkdir(parents=True,exist_ok=True); p=OUT/f'chrono_ranking_pass_{name}_{stamp()}.json'
 payload.setdefault('generated_at',now()); payload['report_path']=rel(p); p.write_text(json.dumps(payload,indent=2,default=str),encoding='utf-8'); print(f'REPORT_PATH={rel(p)}'); return p

def init(a):
 if a.execute:
  with psycopg.connect(db(a)) as conn:
   with conn.cursor() as cur: cur.execute(SCHEMA.read_text())
   conn.commit()
 write('init_schema_execute' if a.execute else 'init_schema_dry_run',{'action':'init_schema','execute_performed':bool(a.execute),'schema':rel(SCHEMA)})
 return 0

def run(a):
 with psycopg.connect(db(a), row_factory=dict_row) as conn:
  with conn.cursor() as cur:
   cur.execute('SELECT count(*) AS n FROM lucidota_korpus.temporal_claim'); claim_count=int(cur.fetchone()['n'])
   cur.execute('SELECT count(*) AS n FROM lucidota_korpus.file_object'); file_count=int(cur.fetchone()['n'])
   ranking_pass_uuid=None; selected=0; violations=0; source_distribution=[]
   if a.execute:
    cur.execute('INSERT INTO lucidota_korpus.temporal_ranking_pass(ranking_algorithm,claim_count,file_count,detail) VALUES (%s,%s,%s,%s::jsonb) RETURNING ranking_pass_uuid::text',('trust_weight_desc_timestamp_asc_created_desc_v1',claim_count,file_count,json.dumps({'script':'scripts/chrono_ranking_pass.py'})))
    ranking_pass_uuid=cur.fetchone()['ranking_pass_uuid']
    cur.execute('''WITH ranked AS (
      SELECT claim_uuid,file_uuid,candidate_timestamp,evidence_source,trust_weight,
             row_number() OVER (PARTITION BY file_uuid ORDER BY trust_weight DESC, candidate_timestamp ASC, created_at DESC, claim_uuid ASC) AS rn
      FROM lucidota_korpus.temporal_claim WHERE file_uuid IS NOT NULL
    )
    INSERT INTO lucidota_korpus.temporal_ranking_selection(ranking_pass_uuid,file_uuid,claim_uuid,candidate_timestamp,evidence_source,trust_weight,rank_priority)
    SELECT %s::uuid,file_uuid,claim_uuid,candidate_timestamp,evidence_source,trust_weight,1 FROM ranked WHERE rn=1
    RETURNING evidence_source''',(ranking_pass_uuid,))
    rows=cur.fetchall(); selected=len(rows)
    cur.execute('''SELECT count(*) AS n FROM lucidota_korpus.temporal_ranking_selection s
                   LEFT JOIN lucidota_korpus.temporal_claim c ON c.claim_uuid=s.claim_uuid AND c.file_uuid=s.file_uuid
                   WHERE s.ranking_pass_uuid=%s::uuid AND c.claim_uuid IS NULL''',(ranking_pass_uuid,)); violations=int(cur.fetchone()['n'])
    cur.execute('UPDATE lucidota_korpus.temporal_ranking_pass SET selected_count=%s, ranking_violations=%s WHERE ranking_pass_uuid=%s::uuid',(selected,violations,ranking_pass_uuid))
    cur.execute('SELECT evidence_source, count(*) AS n FROM lucidota_korpus.temporal_ranking_selection WHERE ranking_pass_uuid=%s::uuid GROUP BY evidence_source ORDER BY n DESC',(ranking_pass_uuid,)); source_distribution=[dict(r) for r in cur.fetchall()]
    conn.commit()
   else:
    cur.execute('''WITH ranked AS (SELECT file_uuid, row_number() OVER (PARTITION BY file_uuid ORDER BY trust_weight DESC, candidate_timestamp ASC, created_at DESC, claim_uuid ASC) rn FROM lucidota_korpus.temporal_claim WHERE file_uuid IS NOT NULL) SELECT count(*) AS n FROM ranked WHERE rn=1'''); selected=int(cur.fetchone()['n'])
 report={'action':'rank','execute_performed':bool(a.execute),'db_writes_performed':bool(a.execute),'graph_writes_performed':False,'ranking_pass_uuid':ranking_pass_uuid,'claim_count':claim_count,'file_count':file_count,'selected_count':selected,'ranking_violations':violations,'selected_source_distribution':source_distribution,'status':'PASS' if violations==0 else 'FAIL'}
 write('rank_pass' if violations==0 else 'rank_fail',report); print(f'RANKING_PASS_UUID={ranking_pass_uuid or "DRY_RUN"}'); print(f'RANKING_VIOLATIONS={violations}')
 return 0 if violations==0 else 4

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--database-url'); sub=ap.add_subparsers(dest='cmd',required=True)
 s=sub.add_parser('init-schema'); s.add_argument('--execute',action='store_true')
 r=sub.add_parser('rank'); r.add_argument('--execute',action='store_true')
 a=ap.parse_args(); return init(a) if a.cmd=='init-schema' else run(a)
if __name__=='__main__': raise SystemExit(main())
