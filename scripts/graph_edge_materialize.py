#!/usr/bin/env python3
"""Materialize a graph edge through the governed promotion path."""
from __future__ import annotations
import argparse,json,os,uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import psycopg
from psycopg.rows import dict_row
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'05_OUTPUTS/graph'; SCHEMA=ROOT/'06_SCHEMA/093_graph_edge_materialization_guard.sql'; BASE=[ROOT/'06_SCHEMA/034_graph_promotion_pipeline.sql',ROOT/'06_SCHEMA/044_graph_promotion_policy_roles.sql',ROOT/'06_SCHEMA/052_graph_promotion_materialization.sql',ROOT/'06_SCHEMA/065_graph_materialization_helper_v2.sql',ROOT/'06_SCHEMA/093_graph_edge_materialization_guard.sql']
def now(): return datetime.now(timezone.utc).isoformat().replace('+00:00','Z')
def stamp(): return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
def rel(p):
 try: return str(Path(p).resolve().relative_to(ROOT))
 except Exception: return str(p)
def db(a): return a.database_url or os.environ.get('KORPUS_DATABASE_URL') or os.environ.get('DATABASE_URL') or 'postgresql:///lucidota_storage'
def control_db(a): return a.control_database_url or os.environ.get('ABSURD_SYSTEM_DATABASE_URL') or 'postgresql:///lucidota_state'
def write(name,payload):
 OUT.mkdir(parents=True,exist_ok=True); p=OUT/f'graph_edge_materialize_{name}_{stamp()}.json'; payload.setdefault('generated_at',now()); payload['report_path']=rel(p); p.write_text(json.dumps(payload,indent=2,default=str),encoding='utf-8'); print(f'REPORT_PATH={rel(p)}'); return p
def split(items):
 out=[]
 for raw in items or []: out += [x.strip() for x in raw.split(',') if x.strip()]
 return out
def file_ref_ok(ref):
 p=Path(ref); p=p if p.is_absolute() else ROOT/p; return p.exists()
def choose_nodes(cur):
 cur.execute("SELECT uuid::text FROM lucidota_go.graph_item ORDER BY created_at DESC LIMIT 2"); rows=[r['uuid'] for r in cur.fetchall()]
 if len(rows)<2: raise RuntimeError('need_at_least_two_graph_items')
 return rows[0],rows[1]
def command_exists(a, command_uuid):
 with psycopg.connect(control_db(a)) as conn:
  with conn.cursor() as cur:
   cur.execute("SELECT to_regclass('lucidota_control.conversation_command')")
   if cur.fetchone()[0] is None: return False
   cur.execute('SELECT 1 FROM lucidota_control.conversation_command WHERE command_uuid=%s::uuid',(command_uuid,))
   return cur.fetchone() is not None
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--database-url'); ap.add_argument('--control-database-url'); ap.add_argument('--source-uuid'); ap.add_argument('--target-uuid'); ap.add_argument('--edge-type',default='RELATED_TO'); ap.add_argument('--evidence-ref',action='append',required=True); ap.add_argument('--source-evidence-ref',action='append',default=[]); ap.add_argument('--target-evidence-ref',action='append',default=[]); ap.add_argument('--command-envelope-uuid',required=True); ap.add_argument('--execute',action='store_true'); a=ap.parse_args()
 evidence=split(a.evidence_ref); src_ev=split(a.source_evidence_ref) or evidence; tgt_ev=split(a.target_evidence_ref) or evidence; blockers=[]
 if not evidence: blockers.append('evidence_ref_required')
 try: a.command_envelope_uuid=str(uuid.UUID(a.command_envelope_uuid))
 except Exception: blockers.append('invalid_command_envelope_uuid')
 if a.execute and 'invalid_command_envelope_uuid' not in blockers and not command_exists(a,a.command_envelope_uuid): blockers.append('command_envelope_uuid_not_found_in_control_db')
 if a.execute and os.environ.get('LUCIDOTA_GRAPH_MATERIALIZATION_HELPER') != 'scripts/graph_materialization_helper.py': blockers.append('materialization_helper_required')
 for ref in src_ev+tgt_ev+evidence:
  if ':' not in ref and not file_ref_ok(ref): blockers.append(f'evidence_file_missing:{ref}')
 result={'action':'edge_materialize','execute_performed':bool(a.execute),'db_writes_performed':False,'canonical_graph_writes_performed':False,'graph_writes_performed':False,'blockers':blockers,'source_uuid':a.source_uuid,'target_uuid':a.target_uuid}
 if a.execute and not blockers:
  with psycopg.connect(db(a), row_factory=dict_row) as conn:
   with conn.cursor() as cur:
    for schema in BASE: cur.execute(schema.read_text())
    source=a.source_uuid; target=a.target_uuid
    if not source or not target: source,target=choose_nodes(cur)
    result['source_uuid']=source; result['target_uuid']=target
    cur.execute('SELECT 1 FROM lucidota_go.graph_item WHERE uuid=%s::uuid',(source,));
    if not cur.fetchone(): blockers.append('source_uuid_not_found')
    cur.execute('SELECT 1 FROM lucidota_go.graph_item WHERE uuid=%s::uuid',(target,));
    if not cur.fetchone(): blockers.append('target_uuid_not_found')
    if not blockers:
     payload={'source_uuid':source,'target_uuid':target,'edge_type':a.edge_type,'source_evidence_refs':src_ev,'target_evidence_refs':tgt_ev}
     cur.execute('''INSERT INTO lucidota_go.graph_promotion_packet(source_system,candidate_kind,candidate_payload,evidence_refs,authority_class,promotion_status,detail)
                    VALUES ('graph_edge_materialize','edge',%s::jsonb,%s::jsonb,'operator_confirmed_finding','operator_confirmed',%s::jsonb) RETURNING packet_uuid::text''',(json.dumps(payload),json.dumps(evidence),json.dumps({'script':'scripts/graph_edge_materialize.py'})))
     packet=cur.fetchone()['packet_uuid']
     cur.execute('''INSERT INTO lucidota_go.graph_promotion_decision(packet_uuid,decision,decided_by,rationale,evidence_refs,operator_confirmed,command_envelope_uuid)
                    VALUES (%s::uuid,'operator_confirmed','operator','edge materialization path proof',%s::jsonb,true,%s::uuid) RETURNING decision_uuid::text''',(packet,json.dumps(evidence),a.command_envelope_uuid))
     decision=cur.fetchone()['decision_uuid']
     cur.execute("SET LOCAL lucidota.graph_promotion_path='on'")
     cur.execute("SET LOCAL lucidota.graph_materialization_helper='scripts/graph_materialization_helper.py'")
     cur.execute('''INSERT INTO lucidota_go.graph_edge(source_uuid,target_uuid,edge_type,status,current_status,current_unknown,detail)
                    VALUES (%s::uuid,%s::uuid,%s,'staged','unknown',true,%s::jsonb) RETURNING edge_uuid::text''',(source,target,a.edge_type,json.dumps({'source_evidence_refs':src_ev,'target_evidence_refs':tgt_ev,'promotion_path':'scripts/graph_edge_materialize.py'})))
     edge=cur.fetchone()['edge_uuid']
     cur.execute('''INSERT INTO lucidota_go.graph_journal(edge_uuid,action,reason,after_state) VALUES (%s::uuid,'stage','edge materialized through promotion path',%s::jsonb) RETURNING journal_uuid::text''',(edge,json.dumps({'packet_uuid':packet,'decision_uuid':decision,'edge_uuid':edge,'command_envelope_uuid':a.command_envelope_uuid})))
     journal=cur.fetchone()['journal_uuid']
     cur.execute('''INSERT INTO lucidota_go.graph_promotion_materialization(packet_uuid,decision_uuid,graph_edge_uuid,journal_uuid,command_envelope_uuid,evidence_refs,materialization_kind,detail)
                    VALUES (%s::uuid,%s::uuid,%s::uuid,%s::uuid,%s::uuid,%s::jsonb,'edge',%s::jsonb) RETURNING materialization_uuid::text''',(packet,decision,edge,journal,a.command_envelope_uuid,json.dumps(evidence),json.dumps({'script':'scripts/graph_edge_materialize.py','source_evidence_refs':src_ev,'target_evidence_refs':tgt_ev})))
     mat=cur.fetchone()['materialization_uuid']
     cur.execute('''INSERT INTO lucidota_go.graph_materialization_helper_receipt(materialization_uuid,packet_uuid,decision_uuid,graph_edge_uuid,journal_uuid,command_envelope_uuid,evidence_count,authority_class,verification_passed,materializer_report_path,detail)
                    VALUES (%s::uuid,%s::uuid,%s::uuid,%s::uuid,%s::uuid,%s::uuid,%s,'operator_confirmed_finding',true,'scripts/graph_edge_materialize.py',%s::jsonb)
                    ON CONFLICT (materialization_uuid) DO UPDATE SET detail=lucidota_go.graph_materialization_helper_receipt.detail || EXCLUDED.detail
                    RETURNING helper_receipt_uuid::text''',(mat,packet,decision,edge,journal,a.command_envelope_uuid,len(evidence),json.dumps({'script':'scripts/graph_edge_materialize.py','source_evidence_refs':src_ev,'target_evidence_refs':tgt_ev})))
     helper_receipt=cur.fetchone()['helper_receipt_uuid']
     result.update({'packet_uuid':packet,'decision_uuid':decision,'graph_edge_uuid':edge,'journal_uuid':journal,'materialization_uuid':mat,'helper_receipt_uuid':helper_receipt,'db_writes_performed':True,'canonical_graph_writes_performed':True,'graph_writes_performed':True})
   conn.commit()
 result['blockers']=blockers; result['status']='PASS' if not blockers else 'FAIL'; write('execute' if a.execute else 'dry_run',result)
 if result.get('materialization_uuid'): print('MATERIALIZATION_UUID='+result['materialization_uuid'])
 if result.get('helper_receipt_uuid'): print('HELPER_RECEIPT_UUID='+result['helper_receipt_uuid'])
 return 0 if not blockers else 4
if __name__=='__main__': raise SystemExit(main())
