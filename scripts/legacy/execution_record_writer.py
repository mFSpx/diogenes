#!/usr/bin/env python3
"""Standalone execution record writer for worker outputs.

Writes structured execution receipts into lucidota_control.boring_execution_record
using the same table Boring Beast workers use. Optional Chrono append records the
worker-output event as temporal evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psycopg

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "05_OUTPUTS/execution_records"


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def sha(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def rel(path: Path | str) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def state_db(args: argparse.Namespace) -> str:
    return args.state_database_url or os.environ.get("DBOS_SYSTEM_DATABASE_URL") or "postgresql:///lucidota_state"


def storage_db(args: argparse.Namespace) -> str:
    return args.storage_database_url or os.environ.get("KORPUS_DATABASE_URL") or os.environ.get("DATABASE_URL") or "postgresql:///lucidota_storage"


def write_report(name: str, payload: dict[str, Any]) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"execution_record_writer_{name}_{stamp()}.json"
    payload.setdefault("generated_at", now())
    payload["report_path"] = rel(p)
    p.write_text(json.dumps(payload, indent=2, sort_keys=False, default=str), encoding="utf-8")
    print(f"REPORT_PATH={rel(p)}")
    return p


def split_json_or_csv(raw: str | None) -> list[str]:
    if not raw:
        return []
    raw = raw.strip()
    if raw.startswith("["):
        data = json.loads(raw)
        return [str(x) for x in data]
    return [x.strip() for x in raw.split(",") if x.strip()]


def validation_records(commands: list[str], run_validations: bool) -> list[dict[str, Any]]:
    records = []
    for cmd in commands:
        rec = {"command": cmd, "executed": False, "returncode": None, "stdout_tail": "", "stderr_tail": ""}
        if run_validations:
            proc = subprocess.run(cmd, cwd=ROOT, shell=True, text=True, capture_output=True, timeout=120)
            rec.update({"executed": True, "returncode": proc.returncode, "stdout_tail": proc.stdout[-1000:], "stderr_tail": proc.stderr[-1000:]})
        records.append(rec)
    return records


def append_chrono(args: argparse.Namespace, event: dict[str, Any]) -> str | None:
    source_sha = sha(event)
    with psycopg.connect(storage_db(args)) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO lucidota_korpus.temporal_claim(
                  candidate_timestamp, evidence_source, trust_weight, raw_evidence,
                  extractor, extractor_version, source_path, source_sha256, detail
                )
                VALUES (now(), 'execution_record_writer_event', 0.50, %s, 'execution_record_writer', 'v1',
                        'scripts/execution_record_writer.py', %s, %s::jsonb)
                RETURNING claim_uuid::text
                """,
                (json.dumps({"task_id": event.get("task_id"), "idempotency_key": event.get("idempotency_key")}), source_sha, json.dumps(event)),
            )
            claim_uuid = cur.fetchone()[0]
        conn.commit()
    return claim_uuid


def write(args: argparse.Namespace) -> int:
    files_changed = split_json_or_csv(args.files_changed)
    validations = validation_records(split_json_or_csv(args.validation_command), args.run_validations)
    result_obj = {"message": args.result, "worker": args.worker_name, "validations": validations}
    idem = args.idempotency_key or f"{args.task_id}:{sha(result_obj)[:24]}"
    audit = {
        "verdict": args.audit_verdict,
        "required_fields_ok": True,
        "remediation": args.remediation,
        "evidence_refs": split_json_or_csv(args.evidence_ref),
    }
    blockers = []
    if args.audit_verdict in {"FAIL", "PARTIAL_FAIL"} and not args.remediation:
        blockers.append("remediation_required_for_fail_or_partial_fail")
    if args.audit_verdict == "PASS" and not audit["evidence_refs"]:
        blockers.append("pass_requires_evidence_ref")
    report = {
        "action": "write",
        "execute_performed": bool(args.execute),
        "db_writes_performed": False,
        "graph_writes_performed": False,
        "task_id": args.task_id,
        "idempotency_key": idem,
        "files_changed": files_changed,
        "validation_commands": validations,
        "status": args.status,
        "audit_verdict": audit,
        "blockers": blockers,
    }
    if blockers:
        write_report("blocked", report)
        return 2
    if args.execute:
        with psycopg.connect(state_db(args)) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO lucidota_control.boring_execution_record(
                      task_id, job_uuid, idempotency_key, files_changed, validation_commands,
                      result, status, audit_verdict, detail
                    )
                    VALUES (%s,NULL,%s,%s::jsonb,%s::jsonb,%s,%s,%s::jsonb,%s::jsonb)
                    ON CONFLICT (idempotency_key) DO UPDATE SET
                      files_changed=EXCLUDED.files_changed,
                      validation_commands=EXCLUDED.validation_commands,
                      result=EXCLUDED.result,
                      status=EXCLUDED.status,
                      audit_verdict=EXCLUDED.audit_verdict,
                      detail=EXCLUDED.detail
                    RETURNING execution_uuid::text
                    """,
                    (args.task_id, idem, json.dumps(files_changed), json.dumps(validations), json.dumps(result_obj), args.status, json.dumps(audit), json.dumps({"script": "scripts/execution_record_writer.py"})),
                )
                report["execution_uuid"] = cur.fetchone()[0]
            conn.commit()
        report["db_writes_performed"] = True
        if args.append_chrono:
            report["chrono_claim_uuid"] = append_chrono(args, report)
    write_report("execute" if args.execute else "dry_run", report)
    if "execution_uuid" in report:
        print(f"EXECUTION_UUID={report['execution_uuid']}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Write structured execution record from worker output")
    p.add_argument("--state-database-url")
    p.add_argument("--storage-database-url")
    p.add_argument("--execute", action="store_true")
    p.add_argument("--task-id", required=True)
    p.add_argument("--worker-name", default="operator_cli")
    p.add_argument("--idempotency-key")
    p.add_argument("--files-changed", default="[]")
    p.add_argument("--validation-command", default="[]")
    p.add_argument("--run-validations", action="store_true")
    p.add_argument("--result", required=True)
    p.add_argument("--status", choices=["succeeded", "failed", "partial_fail", "rejected", "dead_lettered"], default="succeeded")
    p.add_argument("--audit-verdict", choices=["PASS", "FAIL", "PARTIAL_FAIL"], default="PASS")
    p.add_argument("--remediation", default="")
    p.add_argument("--evidence-ref", default="[]")
    p.add_argument("--append-chrono", action="store_true")
    args = p.parse_args()
    return write(args)


if __name__ == "__main__":
    raise SystemExit(main())
