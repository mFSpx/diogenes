#!/usr/bin/env python3
"""System archaeology evidence audit: custody -> claim -> atom -> review coverage."""
from __future__ import annotations
import argparse,json,os
from datetime import datetime, timezone
from pathlib import Path
import psycopg
from psycopg.rows import dict_row
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'05_OUTPUTS/phase05'
def now(): return datetime.now(timezone.utc).isoformat().replace('+00:00','Z')
def stamp(): return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
def rel(p):
 try: return str(Path(p).resolve().relative_to(ROOT))
 except Exception: return str(p)
def db(a): return a.database_url or os.environ.get('KORPUS_DATABASE_URL') or os.environ.get('DATABASE_URL') or 'postgresql:///lucidota_storage'
def one(cur,sql): cur.execute(sql); return dict(cur.fetchone())
def write(payload):
 OUT.mkdir(parents=True,exist_ok=True); p=OUT/f'system_archaeology_evidence_audit_{stamp()}.json'; payload.setdefault('generated_at',now()); payload['report_path']=rel(p); p.write_text(json.dumps(payload,indent=2,default=str),encoding='utf-8'); print('REPORT_PATH='+rel(p))
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--database-url'); a=ap.parse_args(); blockers=[]
 with psycopg.connect(db(a), row_factory=dict_row) as conn:
  with conn.cursor() as cur:
   facts={
    'allowlisted_ingest': one(cur,"SELECT count(*) total, count(*) FILTER (WHERE ingest_status='custody_recorded') custody_recorded FROM lucidota_archaeology.allowlisted_ingest_artifact"),
    'design_atom': one(cur,"SELECT count(*) total, count(*) FILTER (WHERE jsonb_array_length(evidence)>0) with_evidence FROM lucidota_archaeology.design_atom"),
    'workflow_blueprint': one(cur,"SELECT count(*) total, count(*) FILTER (WHERE array_length(source_atom_uuids,1)>0) with_source_atoms FROM lucidota_archaeology.workflow_blueprint"),
    'master_eye_review': one(cur,"SELECT count(*) total FROM lucidota_archaeology.master_eye_review"),
    'telemetry_finding': one(cur,"SELECT count(*) total, count(*) FILTER (WHERE jsonb_array_length(evidence)>0) with_evidence FROM lucidota_archaeology.telemetry_finding"),
    'contradiction_ledger': one(cur,"SELECT count(*) total, count(*) FILTER (WHERE jsonb_array_length(evidence_refs)>0) with_evidence FROM lucidota_archaeology.contradiction_ledger"),
    'longmem_eval_seed': one(cur,"SELECT count(*) total, count(*) FILTER (WHERE jsonb_array_length(expected_evidence_refs)>0) with_evidence FROM lucidota_archaeology.longmem_eval_seed"),
   }
 if int(facts['allowlisted_ingest']['custody_recorded'])<1: blockers.append('no_allowlisted_custody')
 if int(facts['design_atom']['total'])<1: blockers.append('no_design_atoms')
 if int(facts['master_eye_review']['total'])<1: blockers.append('no_master_eye_reviews')
 payload={'action':'system_archaeology_evidence_audit','facts':facts,'blockers':blockers,'status':'PASS' if not blockers else 'FAIL'}
 write(payload); print('ARCHAEOLOGY_EVIDENCE_AUDIT='+payload['status']); return 0 if not blockers else 4
if __name__=='__main__': raise SystemExit(main())
