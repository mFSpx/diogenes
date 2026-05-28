#!/usr/bin/env python3
"""Invalidate temporal claims with evidence without deleting them."""
from __future__ import annotations
import argparse,json,os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import psycopg
from psycopg.rows import dict_row
ROOT=Path(__file__).resolve().parents[1]; SCHEMA=ROOT/'06_SCHEMA/091_chrono_temporal_claim_invalidation.sql'; OUT=ROOT/'05_OUTPUTS/chrono_ledger'
def now(): return datetime.now(timezone.utc).isoformat().replace('+00:00','Z')
def stamp(): return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
def rel(p):
 try: return str(Path(p).resolve().relative_to(ROOT))
 except Exception: return str(p)
def db(a): return a.database_url or os.environ.get('KORPUS_DATABASE_URL') or os.environ.get('DATABASE_URL') or 'postgresql:///lucidota_storage'
def write(name,payload):
 OUT.mkdir(parents=True,exist_ok=True); p=OUT/f'chrono_temporal_claim_invalidate_{name}_{stamp()}.json'; payload.setdefault('generated_at',now()); payload['report_path']=rel(p); p.write_text(json.dumps(payload,indent=2,default=str),encoding='utf-8'); print(f'REPORT_PATH={rel(p)}'); return p
def init(a):
 if a.execute:
  with psycopg.connect(db(a)) as conn:
   with conn.cursor() as cur: cur.execute(SCHEMA.read_text())
   conn.commit()
 write('init_schema_execute' if a.execute else 'init_schema_dry_run',{'action':'init_schema','execute_performed':bool(a.execute),'schema':rel(SCHEMA)}); return 0
def split(raw):
 out=[]
 for item in raw or []: out += [x.strip() for x in item.split(',') if x.strip()]
 return out
def invalidate(a):
 blockers=[]; claim_uuid=a.claim_uuid; event_uuid=None; before=None; after=None
 evidence=split(a.evidence_ref)
 if not evidence: blockers.append('evidence_ref_required')
 with psycopg.connect(db(a), row_factory=dict_row) as conn:
  with conn.cursor() as cur:
   if not claim_uuid and a.create_fixture:
    cur.execute('SELECT file_uuid::text, sha256 FROM lucidota_korpus.file_object ORDER BY first_seen_at NULLS LAST, file_uuid LIMIT 1')
    f=cur.fetchone()
    if not f: blockers.append('file_object_missing')
    elif a.execute:
     cur.execute('''INSERT INTO lucidota_korpus.temporal_claim(file_uuid,candidate_timestamp,evidence_source,trust_weight,raw_evidence,extractor,extractor_version,source_path,source_sha256,detail)
                    VALUES (%s::uuid,'1970-01-01T00:00:00Z','round49_invalidation_fixture',0.00,'round49 fixture','chrono_temporal_claim_invalidate.py','v1','round49',%s,%s::jsonb)
                    RETURNING claim_uuid::text''',(f['file_uuid'],f['sha256'],json.dumps({'fixture':True})))
     claim_uuid=cur.fetchone()['claim_uuid']
   if not claim_uuid: blockers.append('claim_uuid_required')
   if claim_uuid and not blockers:
    cur.execute('SELECT claim_uuid::text, invalid, invalidation_reason, invalidation_evidence FROM lucidota_korpus.temporal_claim WHERE claim_uuid=%s::uuid',(claim_uuid,)); row=cur.fetchone(); before=dict(row) if row else None
    if not row: blockers.append('claim_not_found')
   if a.execute and not blockers:
    cur.execute('''UPDATE lucidota_korpus.temporal_claim SET invalid=true, invalidated_at=coalesce(invalidated_at,now()), invalidation_reason=%s, invalidation_evidence=%s::jsonb, detail=detail || %s::jsonb WHERE claim_uuid=%s::uuid RETURNING claim_uuid::text, invalid, invalidation_reason, invalidation_evidence''',(a.reason,json.dumps(evidence),json.dumps({'invalidated_by':'scripts/chrono_temporal_claim_invalidate.py'}),claim_uuid))
    after=dict(cur.fetchone())
    cur.execute('INSERT INTO lucidota_korpus.temporal_claim_invalidation_event(claim_uuid,reason,evidence_refs,detail) VALUES (%s::uuid,%s,%s::jsonb,%s::jsonb) RETURNING invalidation_uuid::text',(claim_uuid,a.reason,json.dumps(evidence),json.dumps({'script':'scripts/chrono_temporal_claim_invalidate.py'})))
    event_uuid=cur.fetchone()['invalidation_uuid']
   conn.commit()
 report={'action':'invalidate','execute_performed':bool(a.execute),'db_writes_performed':bool(a.execute),'graph_writes_performed':False,'claim_uuid':claim_uuid,'before':before,'after':after,'invalidation_uuid':event_uuid,'deleted_rows':0,'evidence_refs':evidence,'blockers':blockers,'status':'PASS' if not blockers else 'FAIL'}
 write('invalidate_execute' if a.execute else 'invalidate_dry_run',report); print('TEMPORAL_CLAIM_INVALIDATION='+report['status']); return 0 if not blockers else 4
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--database-url'); sub=ap.add_subparsers(dest='cmd',required=True)
 s=sub.add_parser('init-schema'); s.add_argument('--execute',action='store_true')
 i=sub.add_parser('invalidate'); i.add_argument('--claim-uuid'); i.add_argument('--create-fixture',action='store_true'); i.add_argument('--reason',default='invalidated_with_evidence'); i.add_argument('--evidence-ref',action='append',required=True); i.add_argument('--execute',action='store_true')
 a=ap.parse_args(); return init(a) if a.cmd=='init-schema' else invalidate(a)
if __name__=='__main__': raise SystemExit(main())
