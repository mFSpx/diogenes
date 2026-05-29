#!/usr/bin/env python3
"""DBOS wrapper for graph promotion packets. Writes promotion packet/decision only; no direct canonical materialization."""
from __future__ import annotations
import argparse, hashlib, json, os, socket
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import psycopg
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'05_OUTPUTS/dbos'; SCHEMA=ROOT/'06_SCHEMA/043_dbos_remaining_worker_contracts.sql'; QUEUE='graph_promotion'; WORKFLOW='dbos-graph-promotion'
def now(): return datetime.now(timezone.utc).isoformat().replace('+00:00','Z')
def stamp(): return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
def sha(x:Any)->str: return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),default=str).encode()).hexdigest()
def state_url(a): return a.state_database_url or os.environ.get('DBOS_SYSTEM_DATABASE_URL') or 'postgresql:///lucidota_state'
def storage_url(a): return a.storage_database_url or os.environ.get('KORPUS_DATABASE_URL') or os.environ.get('DATABASE_URL') or 'postgresql:///lucidota_storage'
def write(name,d): OUT.mkdir(parents=True,exist_ok=True); p=OUT/f'dbos_graph_promotion_{name}_{stamp()}.json'; d.setdefault('generated_at',now()); d['report_path']=str(p.relative_to(ROOT)); p.write_text(json.dumps(d,indent=2,sort_keys=False,default=str)); print(f'REPORT_PATH={p.relative_to(ROOT)}'); return p
def init_schema(a):
    if a.execute:
      with psycopg.connect(state_url(a)) as c:
        with c.cursor() as cur: cur.execute(SCHEMA.read_text())
        c.commit()
    write('init_schema_execute' if a.execute else 'init_schema_dry_run',{'action':'init_schema','execute_performed':bool(a.execute)}); return 0
def enqueue(a):
    payload=json.loads(a.payload_json or '{"candidate_payload":{"term":"CLAIM","label":"graph promotion wrapper health","evidence_note":"wrapper health"},"evidence_refs":["dbos_graph_promotion_worker"],"authority_class":"operator_authored_assertion","decision":"defer"}')
    idem=a.idempotency_key or sha(payload); r={'action':'enqueue','idempotency_key':idem,'execute_performed':False,'inserted_new':False}
    if not a.execute: write('enqueue_dry_run',r); return 0
    with psycopg.connect(state_url(a)) as c:
      with c.cursor() as cur:
        cur.execute("INSERT INTO lucidota_control.dbos_queue_job(queue_name,workflow_name,job_kind,idempotency_key,payload) VALUES (%s,%s,'graph_promotion_packet_defer',%s,%s::jsonb) ON CONFLICT(queue_name,idempotency_key) DO UPDATE SET updated_at=lucidota_control.dbos_queue_job.updated_at RETURNING job_uuid::text,(xmax=0)",(QUEUE,WORKFLOW,idem,json.dumps(payload)))
        job,inserted=cur.fetchone(); r.update({'job_uuid':job,'inserted_new':bool(inserted)})
      c.commit()
    r['execute_performed']=True; write('enqueue_execute',r); print(f'JOB_UUID={job}'); return 0
def worker_once(a):
    wid=a.worker_id or f'{socket.gethostname()}:{os.getpid()}'; r={'action':'worker_once','worker_id':wid,'execute_performed':False,'job_processed':False,'canonical_graph_writes_performed':False}
    with psycopg.connect(state_url(a)) as sc:
      with sc.cursor() as cur:
        if not a.execute:
          cur.execute("SELECT job_uuid::text FROM lucidota_control.dbos_queue_job WHERE queue_name=%s AND status='queued' ORDER BY created_at LIMIT 1",(QUEUE,)); row=cur.fetchone(); r['would_process']=row[0] if row else None; write('worker_dry_run',r); return 0
        cur.execute("SET LOCAL lucidota.actor_role='worker'"); cur.execute("SELECT job_uuid::text,payload,idempotency_key FROM lucidota_control.dbos_queue_job WHERE queue_name=%s AND status='queued' ORDER BY created_at FOR UPDATE SKIP LOCKED LIMIT 1",(QUEUE,)); row=cur.fetchone()
        if not row: r['no_job_available']=True; write('worker_execute',r); return 0
        job,payload,idem=row; cur.execute("UPDATE lucidota_control.dbos_queue_job SET status='running',locked_by=%s,locked_at=now() WHERE job_uuid=%s",(wid,job))
      sc.commit()
    packet=decision=None; before=after=None
    with psycopg.connect(storage_url(a)) as gc:
      with gc.cursor() as cur:
        cur.execute("SELECT count(*) FROM lucidota_go.graph_item"); before=cur.fetchone()[0]
        ev=payload.get('evidence_refs') or ['dbos_graph_promotion_worker']; cp=payload.get('candidate_payload') or {'term':'CLAIM','label':'wrapper','evidence_note':'wrapper'}
        cur.execute("INSERT INTO lucidota_go.graph_promotion_packet(source_system,candidate_kind,candidate_payload,evidence_refs,authority_class,detail) VALUES ('dbos_graph_promotion_worker','node',%s::jsonb,%s::jsonb,%s,%s::jsonb) RETURNING packet_uuid::text",(json.dumps(cp),json.dumps(ev),payload.get('authority_class','operator_authored_assertion'),json.dumps({'job_uuid':job})))
        packet=cur.fetchone()[0]
        cur.execute("INSERT INTO lucidota_go.graph_promotion_decision(packet_uuid,decision,decided_by,rationale,evidence_refs,operator_confirmed) VALUES (%s,%s,'graph_promoter','DBOS wrapper defer/no materialization',%s::jsonb,false) RETURNING decision_uuid::text",(packet,payload.get('decision','defer'),json.dumps(ev)))
        decision=cur.fetchone()[0]
        cur.execute("SELECT count(*) FROM lucidota_go.graph_item"); after=cur.fetchone()[0]
      gc.commit()
    with psycopg.connect(state_url(a)) as sc:
      with sc.cursor() as cur:
        cur.execute("SET LOCAL lucidota.actor_role='worker'"); cur.execute("UPDATE lucidota_control.dbos_queue_job SET status='succeeded',result=%s::jsonb,completed_at=now() WHERE job_uuid=%s",(json.dumps({'packet_uuid':packet,'decision_uuid':decision}),job))
        cur.execute("INSERT INTO lucidota_control.workflow_event(workflow_id,run_id,phase,status,source,detail) VALUES (%s,%s,'graph_promotion','succeeded','dbos_graph_promotion_worker',%s::jsonb)",(WORKFLOW,job,json.dumps({'packet_uuid':packet,'decision_uuid':decision})))
      sc.commit()
    r.update({'execute_performed':True,'job_processed':True,'job_uuid':job,'packet_uuid':packet,'decision_uuid':decision,'canonical_graph_writes_performed':before!=after}); write('worker_execute',r); print(f'JOB_UUID={job}'); return 0 if before==after else 3
def main():
    p=argparse.ArgumentParser(); p.add_argument('--state-database-url'); p.add_argument('--storage-database-url'); sub=p.add_subparsers(dest='cmd',required=True)
    for name,func in [('init-schema',init_schema),('enqueue',enqueue),('worker-once',worker_once)]:
      sp=sub.add_parser(name); sp.add_argument('--execute',action='store_true'); sp.set_defaults(func=func)
      if name=='enqueue': sp.add_argument('--payload-json'); sp.add_argument('--idempotency-key')
      if name=='worker-once': sp.add_argument('--worker-id')
    a=p.parse_args(); return a.func(a)
if __name__=='__main__': raise SystemExit(main())
