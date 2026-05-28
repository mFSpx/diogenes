#!/usr/bin/env python3
"""Stage graph promotion packets from CEP conversation_command rows."""
from __future__ import annotations
import argparse,hashlib,json,os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import psycopg
from psycopg.rows import dict_row
ROOT=Path(__file__).resolve().parents[1]; SCHEMAS=[ROOT/'06_SCHEMA/034_graph_promotion_pipeline.sql',ROOT/'06_SCHEMA/077_graph_promotion_packet_dedupe.sql',ROOT/'06_SCHEMA/095_cep_graph_packet_staging.sql']; OUT=ROOT/'05_OUTPUTS/graph'
def now(): return datetime.now(timezone.utc).isoformat().replace('+00:00','Z')
def stamp(): return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
def rel(p):
 try: return str(Path(p).resolve().relative_to(ROOT))
 except Exception: return str(p)
def state_db(a): return a.state_database_url or os.environ.get('DBOS_SYSTEM_DATABASE_URL') or 'postgresql:///lucidota_state'
def storage_db(a): return a.storage_database_url or os.environ.get('KORPUS_DATABASE_URL') or os.environ.get('DATABASE_URL') or 'postgresql:///lucidota_storage'
def sha(o): return hashlib.sha256(json.dumps(o,sort_keys=True,separators=(',',':'),default=str).encode()).hexdigest()
def write(name,payload):
 OUT.mkdir(parents=True,exist_ok=True); p=OUT/f'cep_graph_packet_stager_{name}_{stamp()}.json'; payload.setdefault('generated_at',now()); payload['report_path']=rel(p); p.write_text(json.dumps(payload,indent=2,default=str),encoding='utf-8'); print(f'REPORT_PATH={rel(p)}'); return p
def get_command(a):
 with psycopg.connect(state_db(a), row_factory=dict_row) as conn:
  with conn.cursor() as cur:
   cur.execute('''SELECT command_uuid::text, plain_language_instruction, command_envelope, target_refs, evidence_refs, source_artifact_refs, allowed_effect, authority_class, canonical_mutation_allowed, conversation_required
                  FROM lucidota_control.conversation_command WHERE command_uuid=%s::uuid''',(a.command_uuid,))
   r=cur.fetchone(); return dict(r) if r else None
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--state-database-url'); ap.add_argument('--storage-database-url'); ap.add_argument('--command-uuid',required=True); ap.add_argument('--execute',action='store_true'); a=ap.parse_args(); blockers=[]; packet_uuid=None; receipt_uuid=None; inserted_new=False
 cmd=get_command(a)
 if not cmd: blockers.append('conversation_command_not_found')
 elif cmd['canonical_mutation_allowed']: blockers.append('canonical_mutation_allowed_true')
 elif not cmd['conversation_required']: blockers.append('conversation_required_false')
 evidence=list(cmd.get('evidence_refs') or []) if cmd else []
 if cmd and not evidence: evidence=[f"conversation_command:{cmd['command_uuid']}"]
 payload={'command_uuid':cmd['command_uuid'] if cmd else None,'plain_language_instruction':cmd.get('plain_language_instruction') if cmd else None,'target_refs':cmd.get('target_refs') if cmd else [],'artifact_refs':cmd.get('source_artifact_refs') if cmd else [],'allowed_effect':cmd.get('allowed_effect') if cmd else None,'candidate_from':'conversation_command'} if cmd else {}
 if a.execute and not blockers:
  with psycopg.connect(storage_db(a), row_factory=dict_row) as conn:
   with conn.cursor() as cur:
    for schema in SCHEMAS: cur.execute(schema.read_text())
    cur.execute('''INSERT INTO lucidota_go.graph_promotion_packet(source_system,candidate_kind,candidate_payload,evidence_refs,authority_class,promotion_status,detail)
                   VALUES ('conversation_command','other',%s::jsonb,%s::jsonb,%s,'candidate',%s::jsonb)
                   ON CONFLICT(packet_dedupe_key) DO UPDATE SET detail=lucidota_go.graph_promotion_packet.detail || EXCLUDED.detail
                   RETURNING packet_uuid::text,(xmax=0) AS inserted_new''',(json.dumps(payload),json.dumps(evidence),cmd.get('authority_class') or 'operator_authored_assertion',json.dumps({'script':'scripts/cep_graph_packet_stager.py'})))
    row=cur.fetchone(); packet_uuid=row['packet_uuid']; inserted_new=bool(row['inserted_new'])
    cur.execute('''INSERT INTO lucidota_go.cep_graph_packet_stage_receipt(command_uuid,packet_uuid,command_payload_sha256,detail)
                   VALUES (%s::uuid,%s::uuid,%s,%s::jsonb)
                   ON CONFLICT(command_uuid,packet_uuid) DO UPDATE SET detail=lucidota_go.cep_graph_packet_stage_receipt.detail || EXCLUDED.detail
                   RETURNING receipt_uuid::text''',(cmd['command_uuid'],packet_uuid,sha(payload),json.dumps({'inserted_new':inserted_new})))
    receipt_uuid=cur.fetchone()['receipt_uuid']
   conn.commit()
 report={'action':'stage','execute_performed':bool(a.execute),'db_writes_performed':bool(a.execute and packet_uuid),'graph_writes_performed':False,'canonical_graph_writes_performed':False,'command_uuid':a.command_uuid,'packet_uuid':packet_uuid,'receipt_uuid':receipt_uuid,'inserted_new':inserted_new,'evidence_refs':evidence,'blockers':blockers,'status':'PASS' if not blockers else 'FAIL'}
 write('execute' if a.execute else 'dry_run',report); print('CEP_GRAPH_PACKET_STAGE='+report['status']);
 if packet_uuid: print('PACKET_UUID='+packet_uuid)
 return 0 if not blockers else 4
if __name__=='__main__': raise SystemExit(main())
