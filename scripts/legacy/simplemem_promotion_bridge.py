#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, os
from datetime import datetime, timezone
from pathlib import Path
import psycopg
ROOT=Path(__file__).resolve().parents[1]; SCHEMA=ROOT/'06_SCHEMA/047_simplemem_eval_promotion_bridge.sql'; OUT=ROOT/'05_OUTPUTS/simplemem'
def stamp(): return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
def state_db(a): return a.state_database_url or os.environ.get('DBOS_SYSTEM_DATABASE_URL') or 'postgresql:///lucidota_state'
def storage_db(a): return a.storage_database_url or os.environ.get('KORPUS_DATABASE_URL') or os.environ.get('DATABASE_URL') or 'postgresql:///lucidota_storage'
def sha(s): return hashlib.sha256(str(s).encode()).hexdigest()
def write(name,d): OUT.mkdir(parents=True,exist_ok=True); p=OUT/f'simplemem_bridge_{name}_{stamp()}.json'; d['report_path']=str(p.relative_to(ROOT)); p.write_text(json.dumps(d,indent=2,sort_keys=False,default=str)); print(f'REPORT_PATH={p.relative_to(ROOT)}'); return p
def init(a):
 if a.execute:
  with psycopg.connect(state_db(a)) as c:
   with c.cursor() as cur: cur.execute(SCHEMA.read_text())
   c.commit()
 write('init_schema_execute' if a.execute else 'init_schema_dry_run',{'execute_performed':bool(a.execute)}); return 0
def seed_candidate(a):
 q=sha(a.query); text=a.text; source=a.source_ref
 if not a.execute: write('seed_candidate_dry_run',{'execute_performed':False,'query_sha256':q}); return 0
 with psycopg.connect(state_db(a)) as c:
  with c.cursor() as cur:
   cur.execute('''INSERT INTO lucidota_control.simplemem_candidate(query_sha256,source_ref,candidate_text,recall_score,detail)
     VALUES (%s,%s,%s,%s,%s::jsonb) RETURNING candidate_uuid::text''',(q,source,text,a.score,json.dumps({'script':'simplemem_promotion_bridge.py'})))
   cand=cur.fetchone()[0]
  c.commit()
 write('seed_candidate_execute',{'execute_performed':True,'candidate_uuid':cand,'query_sha256':q}); print(f'CANDIDATE_UUID={cand}'); return 0
def evaluate(a):
 with psycopg.connect(state_db(a)) as c:
  with c.cursor() as cur:
   if a.candidate_uuid:
    cur.execute('SELECT candidate_uuid::text,query_sha256,source_ref,candidate_text,recall_score FROM lucidota_control.simplemem_candidate WHERE candidate_uuid=%s',(a.candidate_uuid,))
   else:
    cur.execute('SELECT candidate_uuid::text,query_sha256,source_ref,candidate_text,recall_score FROM lucidota_control.simplemem_candidate ORDER BY created_at DESC LIMIT 1')
   row=cur.fetchone()
 if not row:
  write('evaluate',{'blockers':['candidate_missing'],'execute_performed':False}); return 2
 cand,q,source,text,score=row; evidence=[f'simplemem_candidate:{cand}', source]
 packet=None; status='defer' if float(score)>=a.min_score else 'candidate_only'
 if a.execute and status=='defer' and a.create_promotion_packet:
  with psycopg.connect(storage_db(a)) as gc:
   with gc.cursor() as cur:
    before=None; after=None
    cur.execute('SELECT count(*) FROM lucidota_go.graph_item'); before=cur.fetchone()[0]
    payload={'term':'CLAIM','label':'SimpleMem candidate requires review','status':'staged','evidence_note':'SimpleMem candidate is NOT_TRUTH and requires promotion review','candidate_text_sha256':sha(text),'source_ref':source}
    cur.execute('''INSERT INTO lucidota_go.graph_promotion_packet(source_system,candidate_kind,candidate_payload,evidence_refs,authority_class,detail)
      VALUES ('simplemem_promotion_bridge','node',%s::jsonb,%s::jsonb,'model_computed_finding',%s::jsonb) RETURNING packet_uuid::text''',(json.dumps(payload),json.dumps(evidence),json.dumps({'not_truth':True})))
    packet=cur.fetchone()[0]
    cur.execute('SELECT count(*) FROM lucidota_go.graph_item'); after=cur.fetchone()[0]
   gc.commit()
  status='promotion_packet_created'
 with psycopg.connect(state_db(a)) as c:
  with c.cursor() as cur:
   cur.execute('''INSERT INTO lucidota_control.simplemem_candidate_eval(candidate_uuid,query_sha256,source_ref,candidate_text_sha256,eval_status,promotion_packet_uuid,evidence_refs,detail)
     VALUES (%s,%s,%s,%s,%s,%s,%s::jsonb,%s::jsonb) RETURNING eval_uuid::text''',(cand,q,source,sha(text),status,packet,json.dumps(evidence),json.dumps({'recall_score':float(score),'safe_to_answer_from_this_alone':False})))
   ev=cur.fetchone()[0]
  c.commit()
 report={'execute_performed':bool(a.execute),'candidate_uuid':cand,'eval_uuid':ev,'eval_status':status,'promotion_packet_uuid':packet,'safe_to_answer_from_this_alone':False,'not_truth':True,'canonical_graph_writes_performed':False,'evidence_refs':evidence}
 write('evaluate_execute' if a.execute else 'evaluate_dry_run',report); print(f'EVAL_UUID={ev}'); return 0
def main():
 p=argparse.ArgumentParser(); p.add_argument('--state-database-url'); p.add_argument('--storage-database-url'); sub=p.add_subparsers(dest='cmd',required=True)
 sp=sub.add_parser('init-schema'); sp.add_argument('--execute',action='store_true'); sp.set_defaults(func=init)
 sp=sub.add_parser('seed-candidate'); sp.add_argument('--execute',action='store_true'); sp.add_argument('--query',required=True); sp.add_argument('--text',required=True); sp.add_argument('--source-ref',default='operator_fixture'); sp.add_argument('--score',type=float,default=1.0); sp.set_defaults(func=seed_candidate)
 sp=sub.add_parser('evaluate'); sp.add_argument('--execute',action='store_true'); sp.add_argument('--candidate-uuid'); sp.add_argument('--min-score',type=float,default=1.0); sp.add_argument('--create-promotion-packet',action='store_true'); sp.set_defaults(func=evaluate)
 a=p.parse_args(); return a.func(a)
if __name__=='__main__': raise SystemExit(main())
