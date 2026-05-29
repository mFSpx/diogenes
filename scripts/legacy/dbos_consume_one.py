#!/usr/bin/env python3
"""Generic consume-one DBOS worker path for a named queue."""
from __future__ import annotations
import argparse,json,os,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
from typing import Any
import psycopg
from psycopg.rows import dict_row
from dbos_kernel_authorization import validate_job_kernel_authorization
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/"05_OUTPUTS/dbos"
ALLOWED_EXTERNAL_COMMANDS={
    "scripts/chrono_queue_event_bridge.py",
    "scripts/document_claim_packet_worker.py",
    "scripts/tracer_claim_packet_bridge.py",
    "scripts/phase05_design_atom_extractor.py",
    "scripts/simplemem_candidate_index.py",
    "scripts/graph_promotion_gate.py",
    "scripts/dbos_krampus_worker.py",
    "scripts/dbos_river_worker.py",
    "scripts/goal_agent_packet.py",
    "scripts/goal_dev_control.py",
    "scripts/goal_swarm_dispatch.py",
    "scripts/goal_model_fabric_control.py",
    "scripts/lucidota_model_turbine_overseer.py",
    "scripts/groq_goal_delegate.py",
    "scripts/goal_model_fabric_orchestrate.py",
    "scripts/model_runner_cli.py",
    "scripts/language_router.py",
    "scripts/lucidota_usecase_proof.py",
}
def stamp(): return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
def now(): return datetime.now(timezone.utc).isoformat().replace("+00:00","Z")
def db(a): return a.database_url or os.environ.get("DBOS_SYSTEM_DATABASE_URL") or "postgresql:///lucidota_state"
def rel(p):
    try: return str(Path(p).resolve().relative_to(ROOT))
    except Exception: return str(p)
def write(name,d): OUT.mkdir(parents=True,exist_ok=True); p=OUT/f"dbos_consume_one_{name}_{stamp()}.json"; d.setdefault("generated_at",now()); d["report_path"]=rel(p); p.write_text(json.dumps(d,indent=2,sort_keys=False,default=str)); print(f"REPORT_PATH={rel(p)}"); return p
def handle(payload):
    h=payload.get("handler","noop")
    if h=="noop": return True, {"message":payload.get("message","noop")}
    if h=="status_ledger_check":
        proc=subprocess.run([sys.executable,"scripts/lucidota_status_ledger.py","--check"],cwd=ROOT,text=True,capture_output=True,timeout=120)
        return proc.returncode==0, {"returncode":proc.returncode,"stdout_tail":proc.stdout[-1000:],"stderr_tail":proc.stderr[-1000:]}
    if h=="external_command":
        cmd=payload.get("command")
        if not isinstance(cmd,list) or len(cmd)<2:
            return False, {"error":"external_command_requires_command_list"}
        if cmd[0] not in {"python3","/usr/bin/python3",sys.executable} or str(cmd[1]) not in ALLOWED_EXTERNAL_COMMANDS:
            return False, {"error":"external_command_not_allowlisted","command":cmd[:2]}
        proc=subprocess.run(cmd,cwd=ROOT,text=True,capture_output=True,timeout=int(payload.get("timeout_seconds",180)))
        return proc.returncode==0, {"returncode":proc.returncode,"stdout_tail":proc.stdout[-3000:],"stderr_tail":proc.stderr[-3000:],"command":" ".join(str(x) for x in cmd)}
    return False, {"error":"unsupported_handler","handler":h}
def consume(a):
    report={"action":"consume_one","queue_name":a.queue_name,"execute_performed":bool(a.execute),"job_processed":False,"blockers":[]}
    with psycopg.connect(db(a), row_factory=dict_row) as c:
        with c.cursor() as cur:
            if not a.execute:
                cur.execute("SELECT job_uuid::text,idempotency_key,payload,status::text FROM lucidota_control.dbos_queue_job WHERE queue_name=%s AND status='queued' AND run_after<=now() ORDER BY priority,created_at LIMIT 1",(a.queue_name,))
                row=cur.fetchone(); report["would_process"]=dict(row) if row else None; write("dry_run",report); return 0
            cur.execute("SET LOCAL lucidota.actor_role='worker'")
            cur.execute("SELECT job_uuid::text,idempotency_key,payload,attempt_count,max_attempts,job_kind FROM lucidota_control.dbos_queue_job WHERE queue_name=%s AND status='queued' AND run_after<=now() ORDER BY priority,created_at FOR UPDATE SKIP LOCKED LIMIT 1",(a.queue_name,))
            row=cur.fetchone()
            if not row: report["blockers"].append("no_queued_job"); write("execute",report); return 0
            job=row["job_uuid"]; cur.execute("UPDATE lucidota_control.dbos_queue_job SET status='running',locked_by=%s,locked_at=now(),last_heartbeat_at=now(),attempt_count=attempt_count+1 WHERE job_uuid=%s",(a.worker_id,job))
            auth = validate_job_kernel_authorization(queue_name=a.queue_name, job_kind=str(row["job_kind"]), payload=dict(row["payload"]))
            if not auth.ok:
                ok = False
                result = {"error":"kernel_authorization_rejected","kernel_authorization":auth.as_result()}
                error_kind = auth.error_kind or "kernel_authorization_error"
                error_message = auth.error_message or "kernel authorization rejected job"
            else:
                handler_name=str(row["payload"].get("handler","noop"))
                cur.execute("SELECT to_regclass('lucidota_control.worker_command_registry') AS reg")
                if cur.fetchone()["reg"] is not None:
                    cur.execute("SELECT 1 FROM lucidota_control.worker_command_registry WHERE active AND handler=%s AND queue_name IN (%s,'*') AND job_kind IN (%s,'*') LIMIT 1", (handler_name, a.queue_name, row["job_kind"]))
                    if cur.fetchone() is None:
                        ok = False
                        result = {"error":"handler_not_registered","handler":handler_name}
                        error_kind = "handler_not_registered"
                        error_message = f"handler_not_registered:{handler_name}"
                    else:
                        ok,result=handle(row["payload"])
                        error_kind = "" if ok else "handler_error"
                        error_message = "" if ok else json.dumps(result)
                else:
                    ok,result=handle(row["payload"])
                    error_kind = "" if ok else "handler_error"
                    error_message = "" if ok else json.dumps(result)
            status="succeeded" if ok else ("dead_lettered" if int(row["attempt_count"])+1>=int(row["max_attempts"]) else "failed")
            cur.execute("UPDATE lucidota_control.dbos_queue_job SET status=%s,result=%s::jsonb,last_heartbeat_at=now(),error_kind=%s,error_message=%s,completed_at=CASE WHEN %s='succeeded' THEN now() ELSE completed_at END WHERE job_uuid=%s",(status,json.dumps(result),error_kind,error_message,status,job))
            cur.execute("INSERT INTO lucidota_control.dbos_queue_event(job_uuid,queue_name,event_kind,event_source,detail) VALUES (%s,%s,%s,'dbos_consume_one',%s::jsonb)",(job,a.queue_name,status if status in {"succeeded","failed","dead_lettered"} else "failed",json.dumps(result)))
        c.commit()
    report.update({"job_processed":True,"job_uuid":job,"status":status,"result":result})
    if status=="succeeded":
        record_cmd=[
            sys.executable,"scripts/execution_record_writer.py","--execute","--append-chrono",
            "--task-id",str(row["payload"].get("target_number","dbos_consume_one")),
            "--worker-name","dbos_consume_one",
            "--idempotency-key",f"dbos-consume-one:{row['idempotency_key']}",
            "--files-changed",json.dumps(row["payload"].get("files_changed",[])),
            "--validation-command",json.dumps(row["payload"].get("validation_commands",[])),
            "--result",json.dumps(result,sort_keys=True,default=str),
            "--evidence-ref",json.dumps([f"dbos_job:{job}","scripts/dbos_consume_one.py"]),
        ]
        rec=subprocess.run(record_cmd,cwd=ROOT,text=True,capture_output=True,timeout=180)
        report["execution_record_writer"]={"returncode":rec.returncode,"stdout_tail":rec.stdout[-2000:],"stderr_tail":rec.stderr[-2000:]}
    write("execute",report); print(f"JOB_UUID={job}"); print(f"STATUS={status}"); return 0 if status=="succeeded" else 2
def main():
    p=argparse.ArgumentParser(); p.add_argument("--database-url"); p.add_argument("--queue-name",default="boring_beast"); p.add_argument("--worker-id",default="dbos-consume-one"); p.add_argument("--execute",action="store_true"); a=p.parse_args(); return consume(a)
if __name__=="__main__": raise SystemExit(main())
