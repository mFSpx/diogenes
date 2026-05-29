#!/usr/bin/env python3
"""Strict work-order loader/validator for DBOS queue jobs."""
from __future__ import annotations
import argparse, hashlib, json, os, re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import psycopg
ROOT=Path(__file__).resolve().parents[1]; SCHEMAS=[ROOT/"06_SCHEMA/039_dbos_real_work_loop.sql", ROOT/"06_SCHEMA/062_work_order_loader_validator.sql"]; OUT=ROOT/"05_OUTPUTS/work_orders"
VALID_HANDLERS={"noop","status_ledger_check","fail_once","tracer_label","simplemem_index","external_command"}
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
def sha_obj(o): return hashlib.sha256(json.dumps(o,sort_keys=True,separators=(",",":"),default=str).encode()).hexdigest()
def norm(s): return "".join(ch for ch in "-".join(str(s).strip().lower().split()) if ch.isalnum() or ch in "-_.:")[:180]
def write(name,d): OUT.mkdir(parents=True,exist_ok=True); p=OUT/f"work_order_loader_{name}_{stamp()}.json"; d.setdefault("generated_at",now()); d["report_path"]=rel(p); p.write_text(json.dumps(d,indent=2,sort_keys=False,default=str)); print(f"REPORT_PATH={rel(p)}"); return p
def load(raw,path):
    if path: return json.loads(Path(path).read_text())
    return json.loads(raw)
def validate(p):
    e=[]
    if not isinstance(p,dict): return ["payload_must_be_object"]
    if str(p.get("target_number","")) not in {str(i) for i in range(1,21)}: e.append("target_number_must_be_1_to_20")
    if not str(p.get("target_name","")).strip(): e.append("target_name_required")
    if not str(p.get("idempotency_key","")).strip(): e.append("idempotency_key_required")
    if p.get("handler","noop") not in VALID_HANDLERS: e.append("unsupported_handler")
    if p.get("handler")=="external_command":
        cmd=p.get("command")
        if not isinstance(cmd,list) or not cmd: e.append("external_command_requires_command_list")
        elif len(cmd)<2 or cmd[0] not in {"python3","/usr/bin/python3",os.sys.executable}: e.append("external_command_requires_python3")
        elif str(cmd[1]) not in ALLOWED_EXTERNAL_COMMANDS: e.append("external_command_not_allowlisted")
    if "files_changed" in p and not isinstance(p["files_changed"],list): e.append("files_changed_must_be_list")
    if "validation_commands" in p and not isinstance(p["validation_commands"],list): e.append("validation_commands_must_be_list")
    return e
def init_schema(a):
    if a.execute:
        with psycopg.connect(db(a)) as c:
            with c.cursor() as cur:
                for s in SCHEMAS: cur.execute(s.read_text())
            c.commit()
    write("init_schema_execute" if a.execute else "init_schema_dry_run", {"execute_performed":bool(a.execute),"schemas":[rel(s) for s in SCHEMAS]}); return 0
def load_cmd(a):
    p=load(a.payload_json,a.payload_file); errors=validate(p); psha=sha_obj(p); job=None
    if not errors: p["idempotency_key"]=norm(p["idempotency_key"])
    if a.execute:
        with psycopg.connect(db(a)) as c:
            with c.cursor() as cur:
                if not errors:
                    cur.execute("SET LOCAL lucidota.actor_role='foreman'")
                    cur.execute("""INSERT INTO lucidota_control.dbos_queue_job(queue_name,workflow_name,job_kind,idempotency_key,payload,priority,max_attempts,detail)
                        VALUES ('boring_beast','boring-beast-work-loop','loaded_work_order',%s,%s::jsonb,%s,%s,%s::jsonb)
                        ON CONFLICT(queue_name,idempotency_key) DO UPDATE SET updated_at=lucidota_control.dbos_queue_job.updated_at
                        RETURNING job_uuid::text""",(p["idempotency_key"],json.dumps(p),a.priority,a.max_attempts,json.dumps({"script":"work_order_loader.py"})))
                    job=cur.fetchone()[0]
                cur.execute("INSERT INTO lucidota_control.work_order_load_event(source_ref,payload_sha256,valid,errors,job_uuid,detail) VALUES (%s,%s,%s,%s::jsonb,%s::uuid,%s::jsonb)",(a.source_ref,psha,not errors,json.dumps(errors),job,json.dumps({"script":"work_order_loader.py"})))
            c.commit()
    report={"action":"load","execute_performed":bool(a.execute),"db_writes_performed":bool(a.execute),"graph_writes_performed":False,"valid":not errors,"errors":errors,"payload_sha256":psha,"job_uuid":job}
    write("execute" if a.execute else "dry_run", report); print("WORK_ORDER_VALID="+("true" if not errors else "false")); return 0 if not errors else 2
def main():
    p=argparse.ArgumentParser(); p.add_argument("--database-url"); sub=p.add_subparsers(dest="cmd",required=True)
    sp=sub.add_parser("init-schema"); sp.add_argument("--execute",action="store_true"); sp.set_defaults(func=init_schema)
    sp=sub.add_parser("load"); sp.add_argument("--execute",action="store_true"); sp.add_argument("--payload-json"); sp.add_argument("--payload-file"); sp.add_argument("--source-ref",default="operator_cli"); sp.add_argument("--priority",type=int,default=100); sp.add_argument("--max-attempts",type=int,default=2); sp.set_defaults(func=load_cmd)
    a=p.parse_args(); return a.func(a)
if __name__=="__main__": raise SystemExit(main())
