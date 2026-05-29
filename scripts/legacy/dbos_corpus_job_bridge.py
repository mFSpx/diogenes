#!/usr/bin/env python3
"""DBOS-compatible corpus lane job bridge for KORPUS/KRAMPUS custody work."""
from __future__ import annotations
import argparse, hashlib, json, os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import psycopg
ROOT=Path(__file__).resolve().parents[1]
sys_path_added=False
import sys
if str(ROOT/"scripts") not in sys.path:
    sys.path.insert(0, str(ROOT/"scripts")); sys_path_added=True
from kernel_control_packet import dbos_enqueue_packet, require_control_packet
from dbos_kernel_authorization import validate_job_kernel_authorization
OUT=ROOT/'05_OUTPUTS/dbos'
def now(): return datetime.now(timezone.utc).isoformat().replace('+00:00','Z')
def stamp(): return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
def rel(p:Path|str)->str:
    try: return str(Path(p).resolve().relative_to(ROOT))
    except Exception: return str(p)
def sha(x:Any)->str: return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),default=str).encode()).hexdigest()
def dburl(a): return a.database_url or os.environ.get('DBOS_SYSTEM_DATABASE_URL') or 'postgresql:///lucidota_state'
def receipt(name,d):
    OUT.mkdir(parents=True,exist_ok=True); p=OUT/f'dbos_corpus_job_bridge_{name}_{stamp()}.json'; d['receipt_path']=rel(p); p.write_text(json.dumps(d,indent=2,default=str),encoding='utf-8'); print('RECEIPT_PATH='+rel(p)); return p
def load_control_packet(a, *, queue, lane, source_display, idem):
    if getattr(a, 'control_packet_json', None):
        return json.loads(a.control_packet_json)
    if getattr(a, 'control_packet', None):
        return json.loads(Path(a.control_packet).read_text(encoding='utf-8'))
    return dbos_enqueue_packet(queue_name=queue,lane=lane,source_path=source_display,idempotency_key=idem,authorized_by=a.authorized_by)

def bridge(a):
    source_arg = getattr(a, 'source_path', None) or getattr(a, 'source', None)
    source=Path(source_arg); source_display=rel(source) if source.exists() else source_arg
    idem=a.idempotency_key or sha({'queue':a.queue,'lane':a.lane,'source_path':source_display,'state':a.state,'max_files':a.max_files})
    generated_packet=dbos_enqueue_packet(queue_name=a.queue,lane=a.lane,source_path=source_display,idempotency_key=idem,authorized_by=a.authorized_by)
    kernel_packet=load_control_packet(a,queue=a.queue,lane=a.lane,source_display=source_display,idem=idem) if (a.control_packet or a.control_packet_json) else generated_packet
    payload={'source_path':source_display,'lane':a.lane,'requested_state':a.state,'result_receipt_path':a.result_receipt_path,'bridge_version':'v2','kernel_authorization':kernel_packet,'idempotency_key':idem,'max_files':a.max_files}
    verdict=validate_job_kernel_authorization(queue_name=a.queue,job_kind='korpus_lane_job',payload=payload)
    r={'schema':'diogenes.dbos_corpus_job_bridge.v2','generated_at':now(),'execute_performed':bool(a.execute),'queue':a.queue,'lane':a.lane,'state':a.state,'source_path':source_display,'max_files':a.max_files,'idempotency_key':idem,'result_receipt_path':a.result_receipt_path,'kernel_authorization':{'authorized_by':kernel_packet.get('authorized_by'),'packet_hash':kernel_packet.get('packet_hash'),'lane':kernel_packet.get('lane'),'valid':verdict.ok},'job_uuid':None,'inserted_new':False,'error':None,'status':'BLOCKED' if not source.exists() else 'DRY_RUN'}
    if not source.exists():
        r['error']='SOURCE_PATH_MISSING'; receipt('blocked',r); return 3
    if a.execute and not (a.control_packet or a.control_packet_json):
        r['status']='BLOCKED'; r['error']='MISSING_CONTROL_PACKET'; r['kernel_authorization']['valid']=False; receipt('blocked',r); return 2
    if not verdict.ok:
        r['status']='BLOCKED'; r['error']=verdict.error_kind; r['error_message']=verdict.error_message; receipt('blocked',r); return 2
    if not a.execute:
        receipt('dry_run',r); return 0
    try:
        with psycopg.connect(dburl(a)) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO lucidota_control.dbos_queue_job(queue_name,workflow_name,job_kind,idempotency_key,payload,status,detail)
                    VALUES (%s,'dbos-korpus-corpus-job-bridge','korpus_lane_job',%s,%s::jsonb,%s,%s::jsonb)
                    ON CONFLICT(queue_name,idempotency_key) DO UPDATE SET updated_at=lucidota_control.dbos_queue_job.updated_at
                    RETURNING job_uuid::text,(xmax=0) AS inserted_new
                """,(a.queue,idem,json.dumps(payload),a.state,json.dumps({'source':'scripts/dbos_corpus_job_bridge.py'})))
                job,inserted=cur.fetchone(); r['job_uuid']=job; r['inserted_new']=bool(inserted)
                if inserted:
                    cur.execute("INSERT INTO lucidota_control.dbos_queue_event(job_uuid,queue_name,event_kind,event_source,detail) VALUES (%s,%s,'enqueued','dbos_corpus_job_bridge',%s::jsonb)",(job,a.queue,json.dumps({'lane':a.lane,'source_path':source_display})))
            conn.commit()
        r['status']='PASSED'; receipt('execute',r); return 0
    except Exception as exc:
        r['status']='FAILED'; r['error']=repr(exc); receipt('failed',r); return 4
def main():
    p=argparse.ArgumentParser(); p.add_argument('--database-url'); p.add_argument('--source-path', '--source', dest='source_path', required=True); p.add_argument('--queue',default='korpus'); p.add_argument('--lane',default='manifest_inventory'); p.add_argument('--state',choices=['queued','deferred'],default='queued'); p.add_argument('--idempotency-key'); p.add_argument('--result-receipt-path'); p.add_argument('--authorized-by',default=os.environ.get('LUCIDOTA_AUTHORIZED_BY','operator_cli')); p.add_argument('--control-packet'); p.add_argument('--control-packet-json'); p.add_argument('--max-files',type=int,default=1); p.add_argument('--dry-run',action='store_true'); p.add_argument('--execute','--no-dry-run',dest='execute',action='store_true')
    return bridge(p.parse_args())
if __name__=='__main__': raise SystemExit(main())
