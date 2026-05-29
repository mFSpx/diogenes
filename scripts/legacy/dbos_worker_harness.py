#!/usr/bin/env python3
"""Generic DBOS worker harness: claim/execute/record/retry primitives."""
from __future__ import annotations
import argparse,json,os,subprocess,sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
import psycopg
from psycopg.rows import dict_row
from dbos_kernel_authorization import validate_job_kernel_authorization
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'05_OUTPUTS/dbos'
def now(): return datetime.now(timezone.utc).isoformat().replace('+00:00','Z')
def stamp(): return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
def rel(p):
 try: return str(Path(p).resolve().relative_to(ROOT))
 except Exception: return str(p)
def db_url(a=None): return (getattr(a,'database_url',None) if a else None) or os.environ.get('DBOS_SYSTEM_DATABASE_URL') or os.environ.get('DATABASE_URL') or 'postgresql:///lucidota_state'
def write(name,payload):
 OUT.mkdir(parents=True,exist_ok=True); p=OUT/f'dbos_worker_harness_{name}_{stamp()}.json'; payload.setdefault('generated_at',now()); payload['report_path']=rel(p); p.write_text(json.dumps(payload,indent=2,default=str),encoding='utf-8'); print(f'REPORT_PATH={rel(p)}'); return p
@dataclass
class ClaimedJob:
    job_uuid:str; queue_name:str; workflow_name:str; job_kind:str; idempotency_key:str; payload:dict[str,Any]; attempt_count:int; max_attempts:int
class DBOSWorkerHarness:
    def __init__(self, database_url:str, queue_name:str, worker_id:str): self.database_url=database_url; self.queue_name=queue_name; self.worker_id=worker_id
    def claim_one(self, job_kind:str|None=None)->ClaimedJob|None:
        with psycopg.connect(self.database_url, row_factory=dict_row) as conn:
            with conn.cursor() as cur:
                cur.execute('SET LOCAL lucidota.actor_role=\'worker\'')
                cur.execute('''SELECT job_uuid::text,queue_name,workflow_name,job_kind,idempotency_key,payload,attempt_count,max_attempts FROM lucidota_control.dbos_queue_job
                               WHERE queue_name=%s AND status='queued' AND run_after<=now() AND (%s::text IS NULL OR job_kind=%s)
                               ORDER BY priority ASC, created_at ASC FOR UPDATE SKIP LOCKED LIMIT 1''',(self.queue_name,job_kind,job_kind))
                row=cur.fetchone()
                if not row: return None
                cur.execute('''UPDATE lucidota_control.dbos_queue_job SET status='running',locked_by=%s,locked_at=now(),leased_by=%s,lease_expires_at=now()+interval '5 minutes',last_heartbeat_at=now(),attempt_count=attempt_count+1 WHERE job_uuid=%s::uuid''',(self.worker_id,self.worker_id,row['job_uuid']))
                cur.execute("INSERT INTO lucidota_control.dbos_queue_event(job_uuid,queue_name,event_kind,event_source,detail) VALUES (%s::uuid,%s,'started','dbos_worker_harness',%s::jsonb)",(row['job_uuid'],self.queue_name,json.dumps({'worker_id':self.worker_id})))
            conn.commit()
        return ClaimedJob(row['job_uuid'],row['queue_name'],row['workflow_name'],row['job_kind'],row['idempotency_key'],dict(row['payload']),int(row['attempt_count'])+1,int(row['max_attempts']))
    def finish(self, job:ClaimedJob, ok:bool, result:dict[str,Any], error_kind:str='', error_message:str='')->str:
        status='succeeded' if ok else ('dead_lettered' if job.attempt_count>=job.max_attempts else 'failed')
        with psycopg.connect(self.database_url) as conn:
            with conn.cursor() as cur:
                cur.execute('SET LOCAL lucidota.actor_role=\'worker\'')
                cur.execute('''UPDATE lucidota_control.dbos_queue_job SET status=%s,result=%s::jsonb,last_heartbeat_at=now(),error_kind=%s,error_message=%s,last_error=%s,completed_at=CASE WHEN %s='succeeded' THEN now() ELSE completed_at END WHERE job_uuid=%s::uuid''',(status,json.dumps(result),error_kind,error_message,error_message,status,job.job_uuid))
                cur.execute('INSERT INTO lucidota_control.dbos_queue_event(job_uuid,queue_name,event_kind,event_source,detail) VALUES (%s::uuid,%s,%s,\'dbos_worker_harness\',%s::jsonb)',(job.job_uuid,self.queue_name,status,json.dumps(result)))
                if status=='dead_lettered':
                    cur.execute('''INSERT INTO lucidota_control.dbos_queue_dead_letter(job_uuid,queue_name,workflow_name,job_kind,idempotency_key,error_kind,error_message,attempt_count,payload_sha256,context)
                                   VALUES (%s::uuid,%s,%s,%s,%s,%s,%s,%s,encode(digest(%s,'sha256'),'hex'),%s::jsonb)
                                   ON CONFLICT(job_uuid) WHERE resolved=false DO UPDATE SET last_seen_at=now(),error_message=EXCLUDED.error_message''',(job.job_uuid,self.queue_name,job.workflow_name,job.job_kind,job.idempotency_key,error_kind or 'handler_error',error_message,job.attempt_count,json.dumps(job.payload,sort_keys=True),json.dumps({'payload':job.payload})))
            conn.commit()
        return status

    def bridge_recent_events_to_chrono(self, limit:int=20)->dict[str,Any]:
        """Bridge recent DBOS queue events for this worker queue into Chrono via the canonical bridge script."""
        cmd=[sys.executable,str(ROOT/'scripts/chrono_queue_event_bridge.py'),'append','--execute','--limit',str(limit)]
        proc=subprocess.run(cmd,cwd=ROOT,text=True,capture_output=True)
        report_path=None
        for line in proc.stdout.splitlines():
            if line.startswith('REPORT_PATH='):
                report_path=line.split('=',1)[1]
        return {'command':' '.join(cmd),'returncode':proc.returncode,'report_path':report_path,'stdout_tail':proc.stdout[-2000:],'stderr_tail':proc.stderr[-2000:]}
    def consume_one(self, handler:Callable[[dict[str,Any]],tuple[bool,dict[str,Any]]], job_kind:str|None=None)->dict[str,Any]:
        job=self.claim_one(job_kind)
        if not job: return {'processed':False,'status':'no_job'}
        auth=validate_job_kernel_authorization(queue_name=job.queue_name, job_kind=job.job_kind, payload=job.payload)
        if not auth.ok:
            status=self.finish(job,False,{'error':'kernel_authorization_rejected','kernel_authorization':auth.as_result()},auth.error_kind or 'kernel_authorization_error',auth.error_message or 'kernel authorization rejected job')
            return {'processed':True,'job_uuid':job.job_uuid,'status':status,'kernel_authorization':auth.as_result()}
        try: ok,result=handler(job.payload); status=self.finish(job,ok,result,'' if ok else 'handler_error','' if ok else json.dumps(result))
        except Exception as exc: status=self.finish(job,False,{'error':str(exc)},'handler_exception',str(exc))
        return {'processed':True,'job_uuid':job.job_uuid,'status':status}
def self_test(a):
 if not a.execute: write('self_test_dry_run',{'action':'self_test','execute_performed':False,'blockers':['execute_required']}); return 2
 idem=f'harness-self-test-{stamp()}'.lower(); payload={'message':'harness self test'}
 with psycopg.connect(db_url(a)) as conn:
  with conn.cursor() as cur:
   cur.execute("INSERT INTO lucidota_control.dbos_queue(queue_name,owner_subsystem,notes) VALUES ('harness_probe','DBOS worker harness','generic harness probe') ON CONFLICT(queue_name) DO UPDATE SET updated_at=now()")
   cur.execute('''INSERT INTO lucidota_control.dbos_queue_job(queue_name,workflow_name,job_kind,idempotency_key,payload,priority,max_attempts)
                  VALUES ('harness_probe','harness-self-test','noop',%s,%s::jsonb,-9000,1)
                  ON CONFLICT(queue_name,idempotency_key) DO UPDATE SET status='queued',updated_at=now()''',(idem,json.dumps(payload)))
  conn.commit()
 h=DBOSWorkerHarness(db_url(a),'harness_probe','harness-self-test-worker')
 result=h.consume_one(lambda p:(True,{'handled':p.get('message')}),'noop')
 bridge = h.bridge_recent_events_to_chrono(20) if getattr(a,'bridge_chrono',False) else None
 blockers=[] if result.get('status')=='succeeded' else ['harness_consume_failed']
 if bridge and bridge.get('returncode')!=0: blockers.append('chrono_bridge_failed')
 write('self_test_execute',{'action':'self_test','execute_performed':True,'db_writes_performed':True,'graph_writes_performed':False,'result':result,'chrono_bridge':bridge,'blockers':blockers,'status':'PASS' if not blockers else 'FAIL'}); print('DBOS_WORKER_HARNESS=' + ('PASS' if not blockers else 'FAIL')); return 0 if not blockers else 4
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--database-url'); ap.add_argument('--self-test',action='store_true'); ap.add_argument('--execute',action='store_true'); ap.add_argument('--bridge-chrono', action='store_true'); a=ap.parse_args(); return self_test(a) if a.self_test else (print('use --self-test'),2)[1]
if __name__=='__main__': raise SystemExit(main())
