#!/usr/bin/env python3
"""Bridge Rust lucidota-intake UnifiedMetadata JSONL custody records into DBOS jobs."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psycopg

ROOT = Path(__file__).resolve().parents[1]
SCHEMAS = [ROOT / "06_SCHEMA/035_dbos_queue_spine.sql", ROOT / "06_SCHEMA/039_dbos_real_work_loop.sql", ROOT / "06_SCHEMA/079_intake_custody_job_bridge.sql"]
OUT = ROOT / "05_OUTPUTS/intake_custody"
QUEUE = "intake_custody"
WORKFLOW = "rust-intake-custody-derived-task-bridge"


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def rel(path: str | Path) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def sha_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def state_db(args: argparse.Namespace) -> str:
    return args.state_database_url or os.environ.get("DBOS_SYSTEM_DATABASE_URL") or "postgresql:///lucidota_state"


def write_report(name: str, payload: dict[str, Any]) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"intake_custody_bridge_{name}_{stamp()}.json"
    payload.setdefault("generated_at", now())
    payload["report_path"] = rel(path)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"REPORT_PATH={rel(path)}")
    return path


def init_schema(args: argparse.Namespace) -> int:
    if args.execute:
        with psycopg.connect(state_db(args)) as conn:
            with conn.cursor() as cur:
                for schema in SCHEMAS:
                    cur.execute(schema.read_text(encoding="utf-8"))
            conn.commit()
    write_report("init_schema_execute" if args.execute else "init_schema_dry_run", {"action": "init_schema", "execute_performed": bool(args.execute), "schemas": [rel(s) for s in SCHEMAS]})
    return 0


def iter_records(path: Path):
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        rec = json.loads(line)
        yield line_no, line, rec


def map_task_queue(task: dict[str, Any]) -> str:
    t = str(task.get("task_type", ""))
    if t == "river_replay_component":
        return "dbos.phase05_streaming_brain"
    if t == "near_duplicate_scan":
        return "dbos.phase05_streaming_brain"
    if t == "media_deep_extract":
        return "document_parse"
    if t == "graph_promote_file":
        return "graph_promotion"
    return QUEUE


def bridge(args: argparse.Namespace) -> int:
    path = Path(args.jsonl).expanduser().resolve()
    blockers: list[str] = []
    if not path.exists():
        blockers.append("jsonl_missing")
    prepared: list[dict[str, Any]] = []
    if not blockers:
        for line_no, line, rec in iter_records(path):
            if rec.get("status") != "indexed":
                continue
            tasks = rec.get("derived_tasks") or []
            for idx, task in enumerate(tasks):
                if args.limit and len(prepared) >= args.limit:
                    break
                queue = map_task_queue(task)
                idem = f"intake:{rec.get('file_hash_sha256','')}:{task.get('task_type','')}:{idx}"
                payload = {
                    "source": "lucidota-intake",
                    "source_jsonl_path": rel(path),
                    "line_no": line_no,
                    "record_sha256": sha_text(line),
                    "file_hash_sha256": rec.get("file_hash_sha256", ""),
                    "source_path": rec.get("source_path", ""),
                    "locked_relative_path": rec.get("locked_relative_path", ""),
                    "file_kind": rec.get("file_kind", ""),
                    "lane": rec.get("lane", ""),
                    "task": task,
                    "canonical_graph_write_allowed": False,
                }
                prepared.append({"queue_name": queue, "workflow_name": WORKFLOW, "job_kind": task.get("task_type", "derived_task"), "idempotency_key": idem, "priority": int(task.get("priority", 100)), "payload": payload, "task": task, "record_sha256": sha_text(line), "file_hash_sha256": rec.get("file_hash_sha256", ""), "source_path": rec.get("source_path", "")})
            if args.limit and len(prepared) >= args.limit:
                break
    inserted = 0
    receipts: list[dict[str, Any]] = []
    if args.execute and not blockers:
        with psycopg.connect(state_db(args)) as conn:
            with conn.cursor() as cur:
                cur.execute("SET LOCAL lucidota.actor_role='foreman'")
                for item in prepared:
                    cur.execute(
                        """
                        INSERT INTO lucidota_control.dbos_queue_job(queue_name,workflow_name,job_kind,idempotency_key,payload,priority,detail)
                        VALUES (%s,%s,%s,%s,%s::jsonb,%s,%s::jsonb)
                        ON CONFLICT (queue_name,idempotency_key) DO UPDATE SET updated_at=lucidota_control.dbos_queue_job.updated_at
                        RETURNING job_uuid::text,(xmax=0) AS inserted_new
                        """,
                        (item["queue_name"], item["workflow_name"], item["job_kind"], item["idempotency_key"], json.dumps(item["payload"]), item["priority"], json.dumps({"source":"scripts/intake_custody_job_bridge.py"})),
                    )
                    job_uuid, inserted_new = cur.fetchone()
                    inserted += 1 if inserted_new else 0
                    cur.execute(
                        """
                        INSERT INTO lucidota_control.intake_custody_bridge_receipt(
                          source_jsonl_path, source_record_sha256, file_hash_sha256, source_path, task_type, target_table, queue_name, job_uuid, idempotency_key, inserted_new, detail
                        ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s::uuid,%s,%s,%s::jsonb)
                        ON CONFLICT(queue_name,idempotency_key) DO UPDATE SET job_uuid=EXCLUDED.job_uuid, inserted_new=EXCLUDED.inserted_new, detail=lucidota_control.intake_custody_bridge_receipt.detail || EXCLUDED.detail
                        RETURNING receipt_uuid::text
                        """,
                        (rel(path), item["record_sha256"], item["file_hash_sha256"], item["source_path"], item["job_kind"], str(item["task"].get("target_table", "")), item["queue_name"], job_uuid, item["idempotency_key"], bool(inserted_new), json.dumps({"payload_sha256": sha_text(json.dumps(item["payload"], sort_keys=True, default=str))})),
                    )
                    receipt_uuid = cur.fetchone()[0]
                    receipts.append({"job_uuid": job_uuid, "receipt_uuid": receipt_uuid, "queue_name": item["queue_name"], "job_kind": item["job_kind"], "inserted_new": bool(inserted_new)})
                    if inserted_new:
                        cur.execute("INSERT INTO lucidota_control.dbos_queue_event(job_uuid,queue_name,event_kind,event_source,detail) VALUES (%s::uuid,%s,'enqueued','intake_custody_job_bridge',%s::jsonb)", (job_uuid, item["queue_name"], json.dumps({"receipt_uuid": receipt_uuid, "source_jsonl_path": rel(path)})))
            conn.commit()
    report = {"action":"bridge","execute_performed":bool(args.execute),"db_writes_performed":bool(args.execute and prepared),"graph_writes_performed":False,"source_jsonl_path":rel(path),"prepared_jobs":len(prepared),"inserted_jobs":inserted,"receipts":receipts,"sample_prepared":prepared[:5],"blockers":blockers}
    write_report("execute" if args.execute else "dry_run", report)
    print(f"PREPARED_JOBS={len(prepared)}")
    print(f"INSERTED_JOBS={inserted}")
    return 0 if not blockers else 2


def main() -> int:
    p = argparse.ArgumentParser(description="Bridge Rust intake custody JSONL into DBOS jobs")
    p.add_argument("--state-database-url")
    sub = p.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("init-schema"); sp.add_argument("--execute", action="store_true"); sp.set_defaults(func=init_schema)
    sp = sub.add_parser("bridge"); sp.add_argument("--execute", action="store_true"); sp.add_argument("--jsonl", required=True); sp.add_argument("--limit", type=int, default=50); sp.set_defaults(func=bridge)
    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
