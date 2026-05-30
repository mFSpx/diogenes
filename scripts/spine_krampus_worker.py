#!/usr/bin/env python3
"""ABSURD queue-spine wrapper for KRAMPUSCHEWING/KORPUS.

This is an observation/health wrapper. It writes ABSURD queue/job/workflow receipts
only. It never ingests drops, moves files, deletes files, mutates temporal claims,
or mutates KORPUS custody rows.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psycopg
from psycopg.rows import dict_row

from absurd_worker_contracts import gate_worker_payload_hygiene, record_worker_contract_rejection, validate_worker_contract

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "05_OUTPUTS" / "absurd"
STATE_SCHEMA = ROOT / "06_SCHEMA" / "035_absurd_queue_spine.sql"
KRAMPUS_SCHEMA = ROOT / "06_SCHEMA" / "037_absurd_krampus_wrapper.sql"
QUEUE_NAME = "korpus"
JOB_KIND = "krampus_health_check"
JOB_KINDS = {"krampus_health_check", "korpus_componentize"}
# korpus_lane_job is handled as an internal lane dispatch, not a registered contract job kind
_LANE_JOB_KIND = "korpus_lane_job"
WORKFLOW_NAME = "absurd-krampuschewing-health-check"
KORPUS_TABLES = [
    "lucidota_korpus.file_object",
    "lucidota_korpus.file_occurrence",
    "lucidota_korpus.component",
    "lucidota_korpus.temporal_claim",
]
CANONICAL_GRAPH_TABLES = ["lucidota_go.graph_item", "lucidota_go.graph_edge", "lucidota_go.graph_journal"]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def sha256_obj(obj: Any) -> str:
    return hashlib.sha256(dumps(obj).encode()).hexdigest()


def first_value(row: Any) -> Any:
    if row is None:
        return None
    if isinstance(row, dict):
        return next(iter(row.values()))
    return row[0]


def redact(url: str | None) -> str:
    if not url: return "unset"
    if url.startswith("postgresql:///"): return "postgresql:///<database>"
    if "@" in url: return "postgresql://<redacted>@" + url.split("@", 1)[1]
    return "set_redacted"


def state_url(args: argparse.Namespace) -> str:
    return args.state_database_url or os.environ.get("ABSURD_SYSTEM_DATABASE_URL") or "postgresql:///lucidota_state"


def storage_url(args: argparse.Namespace, payload: dict[str, Any] | None = None) -> str:
    if args.storage_database_url: return args.storage_database_url
    if payload and payload.get("storage_database_url"): return str(payload["storage_database_url"])
    return os.environ.get("KORPUS_DATABASE_URL") or "postgresql:///lucidota_storage"


def count_table(conn, table: str) -> int | None:
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT to_regclass(%s)", (table,))
            if first_value(cur.fetchone()) is None: return None
            cur.execute(f"SELECT count(*) FROM {table}")
            return int(first_value(cur.fetchone()))
    except Exception:
        return None


def count_tables(conn, tables: list[str]) -> dict[str, int | None]:
    return {t: count_table(conn, t) for t in tables}


def write_report(action: str, report: dict[str, Any]) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"absurd_krampus_wrapper_{action}_{stamp()}.json"
    report["report_path"] = str(out.relative_to(ROOT))
    out.write_text(json.dumps(report, indent=2, sort_keys=False, default=str), encoding="utf-8")
    print(f"REPORT_PATH={out.relative_to(ROOT)}")
    return out


def safe_dir_summary(path: Path, max_depth: int = 1) -> dict[str, Any]:
    result = {"path": str(path.relative_to(ROOT)) if str(path).startswith(str(ROOT)) else str(path), "exists": path.exists(), "files": 0, "dirs": 0, "sample_names": []}
    if not path.exists(): return result
    samples=[]
    for dirpath, dirnames, filenames in os.walk(path):
        p=Path(dirpath)
        try: depth=len(p.relative_to(path).parts)
        except Exception: depth=0
        if depth >= max_depth: dirnames[:] = []
        result["dirs"] += len(dirnames)
        result["files"] += len(filenames)
        for name in filenames[:10]:
            # names only, no contents
            samples.append(name)
        if len(samples) >= 12: break
    result["sample_names"] = samples[:12]
    return result


def init_schema(args: argparse.Namespace, execute: bool) -> tuple[dict[str, Any], list[str]]:
    url=state_url(args); blockers=[]; result={"state_database_url":redact(url),"execute_performed":False}
    if not execute:
        result["would_apply"]=[str(STATE_SCHEMA.relative_to(ROOT)), str(KRAMPUS_SCHEMA.relative_to(ROOT))]
        return result, blockers
    with psycopg.connect(url) as conn:
        with conn.cursor() as cur:
            cur.execute(STATE_SCHEMA.read_text(encoding="utf-8"))
            cur.execute(KRAMPUS_SCHEMA.read_text(encoding="utf-8"))
            cur.execute("SELECT count(*) FROM lucidota_control.absurd_queue WHERE queue_name='korpus'")
            result["korpus_queue_registered"] = int(first_value(cur.fetchone())) == 1
            cur.execute("SELECT count(*) FROM lucidota_control.workflow_registry WHERE workflow_name=%s", (WORKFLOW_NAME,))
            result["workflow_registered"] = int(first_value(cur.fetchone())) == 1
        conn.commit()
    result["execute_performed"]=True
    return result, blockers


def krampus_health(args: argparse.Namespace, payload: dict[str, Any] | None = None) -> tuple[dict[str, Any], list[str]]:
    blockers=[]
    db=storage_url(args, payload)
    watch_dir=Path(os.environ.get("KRAMPUScHEWING_DIR", str(ROOT/"KRAMPUSCHEWING")))
    digested=Path(os.environ.get("KRAMPUScHEWING_DIGESTED_DIR", str(ROOT/"03_VAULT/korpus_krampii/DIGESTED")))
    outputs=ROOT/"05_OUTPUTS/korpus_krampii"
    runtime_log=ROOT/"04_RUNTIME/krampuschewing_watcher.log"
    watcher=ROOT/"scripts/krampuschewing_watcher.sh"
    ingest_script=ROOT/"scripts/korpus_krampii.py"
    schema=ROOT/"06_SCHEMA/019_korpus_krampii.sql"
    if not watch_dir.exists(): blockers.append("krampus_watch_dir_missing")
    if not watcher.exists(): blockers.append("krampus_watcher_script_missing")
    if not ingest_script.exists(): blockers.append("korpus_krampii_script_missing")
    if not schema.exists(): blockers.append("korpus_schema_missing")
    db_counts={}
    graph_counts={}
    try:
        with psycopg.connect(db) as conn:
            db_counts=count_tables(conn, KORPUS_TABLES)
            graph_counts=count_tables(conn, CANONICAL_GRAPH_TABLES)
    except Exception as exc:
        blockers.append(f"storage_db_unreachable:{exc}")
    report_files=0
    if outputs.exists():
        report_files=sum(1 for p in outputs.iterdir() if p.is_file() and p.suffix in {".jsonl", ".json", ".txt", ".md"})
    health={
        "storage_database_url":redact(db),
        "watch_dir":safe_dir_summary(watch_dir, max_depth=1),
        "digested_dir":safe_dir_summary(digested, max_depth=2),
        "outputs_dir":safe_dir_summary(outputs, max_depth=1),
        "report_files_count":report_files,
        "runtime_log_exists":runtime_log.exists(),
        "runtime_log_size_bytes":runtime_log.stat().st_size if runtime_log.exists() else 0,
        "watcher_script_exists":watcher.exists(),
        "ingest_script_exists":ingest_script.exists(),
        "schema_exists":schema.exists(),
        "korpus_db_counts":db_counts,
        "canonical_graph_counts":graph_counts,
        "korpus_custody_mutated_by_wrapper":False,
        "temporal_claims_mutated_by_wrapper":False,
    }
    return health, blockers


def run_korpus_componentize(args: argparse.Namespace, payload: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    roots = payload.get("roots") or [str(ROOT / "03_VAULT/korpus_krampii/DROP")]
    if not isinstance(roots, list) or not roots:
        return {"error": "roots_must_be_nonempty_list"}, ["roots_must_be_nonempty_list"]
    limit = max(1, min(int(payload.get("limit", 10)), 100))
    command = [
        sys.executable,
        "scripts/korpus_krampii.py",
        "--json",
        "ingest",
        *[str(x) for x in roots],
        "--limit",
        str(limit),
        "--graph-mode",
        "storage-only",
        "--river-mode",
        str(payload.get("river_mode") or "bulk-light"),
        "--near-dup-mode",
        str(payload.get("near_dup_mode") or "bulk-light"),
    ]
    if payload.get("skip_existing_paths", True):
        command.append("--skip-existing-paths")
    proc = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=int(payload.get("timeout_seconds", 300)))
    result = {
        "returncode": proc.returncode,
        "command": command,
        "graph_mode": "storage-only",
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
    }
    return result, ([] if proc.returncode == 0 else ["korpus_componentize_failed"])


def enqueue(args: argparse.Namespace, execute: bool) -> tuple[dict[str, Any], list[str]]:
    url=state_url(args); blockers=[]
    payload={"check_kind":"krampus_health_check","storage_database_url":storage_url(args),"watch_dir":"KRAMPUSCHEWING","no_ingest":"true"}
    idempotency_key=args.idempotency_key or sha256_obj({"queue":QUEUE_NAME,"job_kind":JOB_KIND,"payload":{k:v for k,v in payload.items() if k!='storage_database_url'}})
    result={"state_database_url":redact(url),"queue":QUEUE_NAME,"workflow":WORKFLOW_NAME,"job_kind":JOB_KIND,"idempotency_key":idempotency_key,"payload_sha256":sha256_obj(payload),"execute_performed":False,"job_uuid":None,"inserted_new":False}
    if not execute: return result, blockers
    with psycopg.connect(url) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO lucidota_control.absurd_queue_job
                  (queue_name, workflow_name, job_kind, idempotency_key, payload, priority, max_attempts, detail)
                VALUES (%s,%s,%s,%s,%s::jsonb,%s,%s,%s::jsonb)
                ON CONFLICT (queue_name, idempotency_key) DO UPDATE SET updated_at=now()
                RETURNING job_uuid, (xmax = 0) AS inserted_new
            """, (QUEUE_NAME, WORKFLOW_NAME, JOB_KIND, idempotency_key, json.dumps(payload), args.priority, args.max_attempts, json.dumps({"source":"spine_krampus_worker"})))
            job_uuid, inserted_new=cur.fetchone()
            if inserted_new:
                cur.execute("INSERT INTO lucidota_control.absurd_queue_event(job_uuid, queue_name, event_kind, event_source, detail) VALUES (%s,%s,'enqueued','spine_krampus_worker',%s::jsonb)", (job_uuid, QUEUE_NAME, json.dumps({"job_kind":JOB_KIND,"idempotency_key":idempotency_key})))
            result.update({"execute_performed":True,"job_uuid":str(job_uuid),"inserted_new":bool(inserted_new)})
        conn.commit()
    return result, blockers


def worker_once(args: argparse.Namespace, execute: bool) -> tuple[dict[str, Any], list[str]]:
    url=state_url(args); blockers=[]; worker_id=args.worker_id or f"absurd_krampus:{socket.gethostname()}:{os.getpid()}"
    result={"state_database_url":redact(url),"queue":QUEUE_NAME,"worker_id":worker_id,"execute_performed":False,"job_processed":False}
    with psycopg.connect(url) as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute("""
                SELECT job_uuid::text, workflow_name, job_kind, idempotency_key, status::text, payload
                FROM lucidota_control.absurd_queue_job
                WHERE queue_name=%s AND status='queued' AND run_after <= now()
                ORDER BY priority ASC, created_at ASC
                LIMIT 1
            """, (QUEUE_NAME,))
            row=cur.fetchone()
            if not execute:
                result["would_process"]=dict(row) if row else None
                if row:
                    result["worker_contract"] = validate_worker_contract(cur, queue_name=QUEUE_NAME, job_kind=str(row["job_kind"]), worker_key="krampus_worker").as_result()
                return result, blockers
            if not row:
                result["no_job_available"]=True
                return result, blockers
            job_uuid=row["job_uuid"]; payload=row["payload"]
            contract = validate_worker_contract(cur, queue_name=QUEUE_NAME, job_kind=str(row["job_kind"]), worker_key="krampus_worker")
            result["worker_contract"] = contract.as_result()
            if not contract.ok:
                gate_result, error_kind = record_worker_contract_rejection(cur, job_uuid=job_uuid, queue_name=QUEUE_NAME, payload=payload, contract=contract, event_source="spine_krampus_worker")
                conn.commit()
                result.update({"execute_performed": True, "job_processed": False, "job_uuid": job_uuid, "status": "failed", "result": gate_result})
                blockers.append(error_kind)
                return result, blockers
            if row["job_kind"] not in JOB_KINDS and row["job_kind"] != _LANE_JOB_KIND:
                blockers.append("unexpected_job_kind")
                return result, blockers
            if row["job_kind"] == "korpus_componentize":
                health, hblockers = run_korpus_componentize(args, payload)
            elif row["job_kind"] == _LANE_JOB_KIND:
                source_path = payload.get("source_path", "")
                max_files = int(payload.get("max_files", 100))
                inv_out = str(ROOT / "05_OUTPUTS" / "krampus_inventory" / "krampus_queue_eligible.jsonl")
                import subprocess as _sp
                r = _sp.run(
                    [str(Path(sys.executable)), "scripts/krampus_bounded_inventory.py",
                     "--target", source_path, "--max-files", str(max_files),
                     "--jsonl-output", inv_out, "--execute"],
                    capture_output=True, text=True, cwd=str(ROOT), check=False,
                )
                if r.returncode == 0:
                    health = {"status": "PASS", "source_path": source_path, "max_files": max_files, "inv_out": inv_out}
                    hblockers = []
                else:
                    health = {"status": "FAIL", "error": r.stderr[-500:], "source_path": source_path}
                    hblockers = ["korpus_lane_job_inventory_failed"]
            else:
                health, hblockers=krampus_health(args, payload)
            ok=not hblockers
            if ok:
                result_payload = {"health": health, "outcome": "succeeded"}
                payload_ok, hygiene = gate_worker_payload_hygiene(
                    result_payload,
                    queue_name=QUEUE_NAME,
                    worker_key="krampus_worker",
                    job_kind=str(row["job_kind"]),
                    required_keys=(),
                    min_score=0,
                )
                if not payload_ok:
                    ok = False
                    hblockers.append(hygiene.get("error", "decision_hygiene_failed"))
                    health = {"health": result_payload, "hygiene": hygiene, "outcome": "failed"}
            status="succeeded" if ok else "failed"
            cur.execute("UPDATE lucidota_control.absurd_queue_job SET status='running', leased_by=%s, lease_expires_at=now()+interval '5 minutes', attempt_count=attempt_count+1, updated_at=now() WHERE job_uuid=%s::uuid", (worker_id, job_uuid))
            cur.execute("INSERT INTO lucidota_control.absurd_queue_event(job_uuid, queue_name, event_kind, event_source, detail) VALUES (%s::uuid,%s,'started','spine_krampus_worker',%s::jsonb)", (job_uuid, QUEUE_NAME, json.dumps({"worker_id":worker_id})))
            cur.execute("UPDATE lucidota_control.absurd_queue_job SET status=%s, result=%s::jsonb, completed_at=CASE WHEN %s THEN now() ELSE completed_at END, updated_at=now(), last_error=%s WHERE job_uuid=%s::uuid", (status, json.dumps(health, default=str), ok, ";".join(hblockers), job_uuid))
            cur.execute("INSERT INTO lucidota_control.absurd_queue_event(job_uuid, queue_name, event_kind, event_source, detail) VALUES (%s::uuid,%s,%s,'spine_krampus_worker',%s::jsonb)", (job_uuid, QUEUE_NAME, status, json.dumps({"health_blockers":hblockers,"file_objects":health.get("korpus_db_counts",{}).get("lucidota_korpus.file_object")})))
            cur.execute("INSERT INTO lucidota_control.workflow_event(workflow_id, run_id, phase, status, source, detail) VALUES (%s,%s,'absurd_krampus_wrapper',%s,'spine_krampus_worker',%s::jsonb) RETURNING event_id::text", (WORKFLOW_NAME, job_uuid, status, json.dumps({"queue":"korpus","job_uuid":job_uuid,"health":health}, default=str)))
            event_id=first_value(cur.fetchone())
            if not ok:
                cur.execute("""
                    INSERT INTO lucidota_control.absurd_queue_dead_letter
                      (job_uuid, queue_name, workflow_name, job_kind, idempotency_key, error_kind, error_message, attempt_count, payload_sha256, context)
                    SELECT job_uuid, queue_name, workflow_name, job_kind, idempotency_key, 'krampus_health_failed', %s, attempt_count, %s, %s::jsonb
                    FROM lucidota_control.absurd_queue_job WHERE job_uuid=%s::uuid
                    ON CONFLICT (job_uuid) WHERE resolved=false DO UPDATE SET error_message=EXCLUDED.error_message,last_seen_at=now(),context=EXCLUDED.context
                """, (";".join(hblockers), sha256_obj(payload), json.dumps(health, default=str), job_uuid))
            result.update({"execute_performed":True,"job_processed":True,"job_uuid":job_uuid,"workflow_event_id":event_id,"status":status,"health":health,"korpus_custody_mutated_by_wrapper":False,"temporal_claims_mutated_by_wrapper":False,"canonical_graph_writes_performed":False})
            blockers.extend(hblockers)
        conn.commit()
    return result, blockers


def main() -> int:
    ap=argparse.ArgumentParser()
    ap.add_argument("--action", choices=["audit","init-schema","enqueue-health-check","worker-once"], required=True)
    mode=ap.add_mutually_exclusive_group(); mode.add_argument("--dry-run", action="store_true"); mode.add_argument("--execute", action="store_true")
    ap.add_argument("--state-database-url", default=os.environ.get("ABSURD_SYSTEM_DATABASE_URL", "postgresql:///lucidota_state"))
    ap.add_argument("--storage-database-url", default=os.environ.get("KORPUS_DATABASE_URL", "postgresql:///lucidota_storage"))
    ap.add_argument("--queue", default=QUEUE_NAME)
    ap.add_argument("--idempotency-key")
    ap.add_argument("--priority", type=int, default=55)
    ap.add_argument("--max-attempts", type=int, default=3)
    ap.add_argument("--worker-id")
    args=ap.parse_args(); execute=bool(args.execute)
    try:
        if args.action=="init-schema": action_result, blockers=init_schema(args, execute)
        elif args.action=="enqueue-health-check": action_result, blockers=enqueue(args, execute)
        elif args.action=="worker-once": action_result, blockers=worker_once(args, execute)
        else:
            health, hblockers=krampus_health(args); action_result={"health":health,"execute_performed":False}; blockers=hblockers
    except Exception as exc:
        action_result={}; blockers=[f"exception:{exc}"]
    report={"schema":"lucidota.absurd_krampus_wrapper.report.v1","generated_at":now_iso(),"action":args.action,"mode":"execute" if execute else "dry_run","execute_requested":execute,"state_database_url":redact(state_url(args)),"storage_database_url":redact(storage_url(args)),"action_result":action_result,"db_writes_performed":bool(action_result.get("execute_performed")) if isinstance(action_result, dict) else False,"korpus_custody_mutated_by_wrapper":False,"temporal_claims_mutated_by_wrapper":False,"canonical_graph_writes_performed":False,"blockers":blockers}
    write_report(args.action, report)
    return 0 if not blockers else 1

if __name__ == "__main__":
    raise SystemExit(main())
