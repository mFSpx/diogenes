#!/usr/bin/env python3
"""Probe CEP conversation_command content idempotency."""
from __future__ import annotations
import argparse, json, os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import psycopg
from psycopg.rows import dict_row
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'05_OUTPUTS/dbos'
SCHEMAS=[ROOT/'06_SCHEMA/035_dbos_queue_spine.sql',ROOT/'06_SCHEMA/039_dbos_real_work_loop.sql',ROOT/'06_SCHEMA/081_cep_conversation_command_dedupe.sql']
def now(): return datetime.now(timezone.utc).isoformat().replace('+00:00','Z')
def stamp(): return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
def db(a): return a.database_url or os.environ.get('DBOS_SYSTEM_DATABASE_URL') or 'postgresql:///lucidota_state'
def rel(p):
    try: return str(Path(p).resolve().relative_to(ROOT))
    except Exception: return str(p)
def write(n,d):
    OUT.mkdir(parents=True,exist_ok=True); p=OUT/f'conversation_command_idempotency_{n}_{stamp()}.json'; d.setdefault('generated_at',now()); d['report_path']=rel(p); p.write_text(json.dumps(d,indent=2,sort_keys=False,default=str)); print(f'REPORT_PATH={rel(p)}'); return p
def init_schema(a):
    if a.execute:
      with psycopg.connect(db(a)) as c:
        with c.cursor() as cur:
          for s in SCHEMAS: cur.execute(s.read_text())
        c.commit()
    write('init_schema_execute' if a.execute else 'init_schema_dry_run', {'action':'init_schema','execute_performed':bool(a.execute),'schemas':[rel(s) for s in SCHEMAS]}); return 0
def probe(a):
    envelope={'protocol':'lucidota.surface_instruction_envelope.v1','surface_id':'round30_probe','canonical_mutation_allowed':False,'conversation_required':True,'target_refs':['round30:target'],'evidence_refs':['round30:evidence'],'allowed_effect':'stage_only'}
    instruction='Round 30 CEP idempotency probe instruction.'
    rows=[]
    with psycopg.connect(db(a), row_factory=dict_row) as c:
      with c.cursor() as cur:
        for idem in ['round30-cep-probe-a','round30-cep-probe-b']:
          cur.execute('''INSERT INTO lucidota_control.conversation_command(command_kind,plain_language_instruction,command_envelope,source_surface_id,source_artifact_refs,target_refs,evidence_refs,allowed_effect,authority_class,idempotency_key,detail)
            VALUES ('operator_instruction',%s,%s::jsonb,'round30_probe','[]'::jsonb,%s::jsonb,%s::jsonb,'stage_only','operator_authored_assertion',%s,%s::jsonb)
            ON CONFLICT(cep_dedupe_key) DO UPDATE SET updated_at=lucidota_control.conversation_command.updated_at
            RETURNING command_uuid::text, idempotency_key, cep_dedupe_key, (xmax=0) AS inserted_new''',(instruction,json.dumps(envelope),json.dumps(envelope['target_refs']),json.dumps(envelope['evidence_refs']),idem,json.dumps({'script':'scripts/conversation_command_idempotency_probe.py'})))
          rows.append(dict(cur.fetchone()))
      c.commit()
    same_uuid=len({r['command_uuid'] for r in rows})==1
    second_suppressed=rows[0]['inserted_new'] is True and rows[1]['inserted_new'] is False
    report={'action':'probe','status':'PASS' if same_uuid and second_suppressed else 'FAIL','execute_performed':True,'db_writes_performed':True,'graph_writes_performed':False,'rows':rows,'same_uuid':same_uuid,'second_suppressed':second_suppressed,'blockers':[] if same_uuid and second_suppressed else ['cep_dedupe_failed']}
    write('probe', report); print('CEP_IDEMPOTENCY='+report['status']); return 0 if report['status']=='PASS' else 4
def main():
    p=argparse.ArgumentParser(); p.add_argument('--database-url'); sub=p.add_subparsers(dest='cmd',required=True)
    sp=sub.add_parser('init-schema'); sp.add_argument('--execute', action='store_true'); sp.set_defaults(func=init_schema)
    sp=sub.add_parser('probe'); sp.set_defaults(func=probe)
    a=p.parse_args(); return a.func(a)
if __name__=='__main__': raise SystemExit(main())
