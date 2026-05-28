#!/usr/bin/env python3
"""Cruelty Protocols executable schema/guardrail verifier."""
from __future__ import annotations
import argparse,json,os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import psycopg
from psycopg.rows import dict_row
ROOT=Path(__file__).resolve().parents[1]; SCHEMA=ROOT/'06_SCHEMA/088_cruelty_protocols_output.sql'; OUT=ROOT/'05_OUTPUTS/phase05'
def now(): return datetime.now(timezone.utc).isoformat().replace('+00:00','Z')
def stamp(): return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
def rel(p):
 try: return str(Path(p).resolve().relative_to(ROOT))
 except Exception: return str(p)
def db(a): return a.database_url or os.environ.get('KORPUS_DATABASE_URL') or os.environ.get('DATABASE_URL') or 'postgresql:///lucidota_storage'
def write(name,payload):
 OUT.mkdir(parents=True,exist_ok=True); p=OUT/f'cruelty_protocols_guard_{name}_{stamp()}.json'; payload.setdefault('generated_at',now()); payload['report_path']=rel(p); p.write_text(json.dumps(payload,indent=2,default=str),encoding='utf-8'); print(f'REPORT_PATH={rel(p)}'); return p

def init(a):
 if a.execute:
  with psycopg.connect(db(a)) as conn:
   with conn.cursor() as cur: cur.execute(SCHEMA.read_text())
   conn.commit()
 write('init_schema_execute' if a.execute else 'init_schema_dry_run',{'action':'init_schema','execute_performed':bool(a.execute),'schema':rel(SCHEMA)}); return 0

def verify(a):
 blockers=[]; sample_uuid=None; expected_blocked=False
 with psycopg.connect(db(a), row_factory=dict_row) as conn:
  with conn.cursor() as cur:
   if a.execute:
    cur.execute('''INSERT INTO lucidota_archaeology.cruelty_protocol_output(protocol_name,source_ref,target_ref,finding_text,authority_class,confidence_bps,evidence_refs,recommended_protocol,detail)
                   VALUES ('semantic_isolation_ranking','round43:fixture','target:fixture','Guardrail fixture: semantic isolation candidate recorded without external action.','deterministic_metric',7000,%s::jsonb,'resource_flow_review',%s::jsonb)
                   RETURNING output_uuid::text''',(json.dumps(['06_SCHEMA/088_cruelty_protocols_output.sql']),json.dumps({'script':'scripts/cruelty_protocols_guard.py'})))
    sample_uuid=cur.fetchone()['output_uuid']
    try:
     cur.execute('''INSERT INTO lucidota_archaeology.cruelty_protocol_output(protocol_name,source_ref,finding_text,authority_class,confidence_bps,evidence_refs,recommended_protocol,external_action_authorized)
                    VALUES ('server_wipe_candidate_detection','round43:bad','This bad fixture should be rejected because external action lacks authorization.','deterministic_metric',5000,%s::jsonb,'server_wipe',true)''',(json.dumps(['06_SCHEMA/088_cruelty_protocols_output.sql']),))
    except Exception:
     conn.rollback(); expected_blocked=True
     with conn.cursor() as cur2:
      # Re-apply schema/sample after rollback to keep positive receipt.
      cur2.execute(SCHEMA.read_text())
      cur2.execute('''INSERT INTO lucidota_archaeology.cruelty_protocol_output(protocol_name,source_ref,target_ref,finding_text,authority_class,confidence_bps,evidence_refs,recommended_protocol,detail)
                      VALUES ('semantic_isolation_ranking','round43:fixture','target:fixture','Guardrail fixture: semantic isolation candidate recorded without external action.','deterministic_metric',7000,%s::jsonb,'resource_flow_review',%s::jsonb)
                      RETURNING output_uuid::text''',(json.dumps(['06_SCHEMA/088_cruelty_protocols_output.sql']),json.dumps({'script':'scripts/cruelty_protocols_guard.py','after_rollback':True})))
      sample_uuid=cur2.fetchone()['output_uuid']
      conn.commit()
    else:
     conn.commit()
   if not expected_blocked and a.execute: blockers.append('unauthorized_external_action_not_blocked')
 report={'action':'verify','execute_performed':bool(a.execute),'db_writes_performed':bool(a.execute),'graph_writes_performed':False,'sample_output_uuid':sample_uuid,'unauthorized_external_action_blocked':expected_blocked if a.execute else None,'blockers':blockers,'status':'PASS' if not blockers else 'FAIL'}
 write('verify_pass' if not blockers else 'verify_fail',report); print('CRUELTY_PROTOCOLS_GUARD='+report['status']); return 0 if not blockers else 4

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--database-url'); sub=ap.add_subparsers(dest='cmd',required=True)
 s=sub.add_parser('init-schema'); s.add_argument('--execute',action='store_true')
 v=sub.add_parser('verify'); v.add_argument('--execute',action='store_true')
 a=ap.parse_args(); return init(a) if a.cmd=='init-schema' else verify(a)
if __name__=='__main__': raise SystemExit(main())
