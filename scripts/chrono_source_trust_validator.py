#!/usr/bin/env python3
"""Validate Chrono evidence_source trust weights."""
from __future__ import annotations
import argparse,json,os
from datetime import datetime, timezone
from pathlib import Path
import psycopg
from psycopg.rows import dict_row
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'05_OUTPUTS/chrono_ledger'
EXPECTED={'filename_strict_iso':1.00,'embedded_json_log_time':1.00,'comm_ingress_time':0.99,'git_containment_time':0.95,'filename_date':0.90,'markdown_header_date':0.80,'absurd_queue_event_bridge':0.55,'dbos_queue_event_bridge':0.55,'boring_beast_runtime_event':0.50,'real_work_loop_item':0.50,'execution_record_writer_event':0.50,'filesystem_mtime_db_occurrence':0.10,'mtime_snapshot_v1':0.10,'round49_invalidation_fixture':0.00}
def now(): return datetime.now(timezone.utc).isoformat().replace('+00:00','Z')
def stamp(): return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
def rel(p):
 try: return str(Path(p).resolve().relative_to(ROOT))
 except Exception: return str(p)
def db(a): return a.database_url or os.environ.get('KORPUS_DATABASE_URL') or os.environ.get('DATABASE_URL') or 'postgresql:///lucidota_storage'
def write(payload):
 OUT.mkdir(parents=True,exist_ok=True); p=OUT/f'chrono_source_trust_validator_{payload["status"].lower()}_{stamp()}.json'; payload.setdefault('generated_at',now()); payload['report_path']=rel(p); p.write_text(json.dumps(payload,indent=2,default=str),encoding='utf-8'); print(f'REPORT_PATH={rel(p)}'); return p
def validate_rows(rows):
 blockers=[]
 for r in rows:
  exp=EXPECTED.get(r['evidence_source'])
  r['expected_weight']=exp
  if exp is None: blockers.append(f"unknown_evidence_source:{r['evidence_source']}")
  elif round(float(r['min_weight']),2)!=round(exp,2) or round(float(r['max_weight']),2)!=round(exp,2): blockers.append(f"weight_mismatch:{r['evidence_source']}")
 return rows,blockers
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--database-url'); a=ap.parse_args(); blockers=[]
 with psycopg.connect(db(a), row_factory=dict_row) as conn:
  with conn.cursor() as cur:
   cur.execute('SELECT evidence_source, min(trust_weight)::float AS min_weight, max(trust_weight)::float AS max_weight, count(*) AS n FROM lucidota_korpus.temporal_claim GROUP BY evidence_source ORDER BY n DESC')
   rows=[dict(r) for r in cur.fetchall()]
 rows,blockers=validate_rows(rows)
 report={'action':'validate_source_trust','db_writes_performed':False,'graph_writes_performed':False,'sources':rows,'blockers':blockers,'status':'PASS' if not blockers else 'FAIL'}
 write(report); print('CHRONO_SOURCE_TRUST='+report['status']); return 0 if not blockers else 4
if __name__=='__main__': raise SystemExit(main())
