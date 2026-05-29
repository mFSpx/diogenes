#!/usr/bin/env python3
"""Register ABSURD worker contracts and workflow registry entries for the indy_polycareer subsystem.

Mutation class: queue_writer (contracts + queue rows only; no canonical graph tables).
Receipt mode: ABSURD_POSTGRES_RUNTIME

Registers:
  - 3 worker contracts in lucidota_control.absurd_worker_contract
  - 3 absurd_queue rows in lucidota_control.absurd_queue (blocked receipt if table missing)
  - 3 workflow_registry rows in lucidota_control.workflow_registry (blocked receipt if table missing)

CLI:
  --dry-run   Print what would be inserted; no DB writes (default)
  --execute   Write for real
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import psycopg
    from psycopg.rows import dict_row
except Exception:  # pragma: no cover
    psycopg = None  # type: ignore

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "05_OUTPUTS" / "indy_polycareer"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def db_url(args: argparse.Namespace) -> str:
    return (
        getattr(args, "database_url", None)
        or os.environ.get("LUCIDOTA_GO_STATE_DSN")
        or os.environ.get("DATABASE_URL")
        or "postgresql:///lucidota_state"
    )


def redact(url: str) -> str:
    if url.startswith("postgresql:///"):
        return "postgresql:///<database>"
    if "@" in url:
        return "postgresql://<redacted>@" + url.split("@", 1)[1]
    return "set_redacted"


def table_exists(cur: Any, qualified_name: str) -> bool:
    cur.execute("SELECT to_regclass(%s)", (qualified_name,))
    row = cur.fetchone()
    val = row["to_regclass"] if isinstance(row, dict) else row[0]
    return val is not None


def write_receipt(report: dict[str, Any]) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "contract_registration_receipt.json"
    report["report_path"] = str(out.relative_to(ROOT))
    out.write_text(json.dumps(report, indent=2, sort_keys=False, default=str), encoding="utf-8")
    return out


# ---------------------------------------------------------------------------
# Contract definitions
# ---------------------------------------------------------------------------

WORKER_CONTRACTS: list[dict[str, Any]] = [
    {
        "worker_key": "polycareer_intake_v1",
        "queue_name": "indy_polycareer",
        "script_path": "scripts/polycareer_intake_worker.py",
        "input_contract": {
            "job_kind": "intake_custody",
            "required_keys": ["source_path", "source_kind", "case_key", "idempotency_key"],
        },
        "output_contract": {
            "writes": [
                "lucidota_control.absurd_queue_event",
                "lucidota_control.workflow_event",
                "03_VAULT/cas/<sha256_prefix>/<sha256>",
                "05_OUTPUTS/indy_polycareer/<job_uuid>/receipt.json",
            ],
            "truth_status": "not_truth_claim_candidate",
        },
        "idempotency_rule": "sha256(queue_name + job_kind + idempotency_key)",
        "retry_policy": "retry intake idempotently; no canonical graph mutation",
        "dead_letter_policy": "dead-letter intake metadata only",
        "canonical_graph_write_allowed": False,
        "status": "implemented",
        "evidence_refs": ["scripts/polycareer_intake_worker.py"],
        "detail": {
            "subsystem": "indy_polycareer",
            "registered_by": "polycareer_register_contracts.py",
            "strict_dequeue_enforced": True,
        },
    },
    {
        "worker_key": "polycareer_extract_v1",
        "queue_name": "indy_polycareer",
        "script_path": "scripts/polycareer_extract_worker.py",
        "input_contract": {
            "job_kind": "extract_claims",
            "required_keys": [
                "intake_receipt_uuid",
                "cas_path",
                "sha256",
                "file_kind",
                "case_key",
                "idempotency_key",
            ],
        },
        "output_contract": {
            "writes": [
                "lucidota_control.absurd_queue_event",
                "lucidota_control.workflow_event",
                "05_OUTPUTS/indy_polycareer/<job_uuid>/receipt.json",
            ],
            "truth_status": "model_computed_finding_candidate",
        },
        "idempotency_rule": "sha256(queue_name + job_kind + idempotency_key)",
        "retry_policy": "retry extraction idempotently; no temporal or graph mutation",
        "dead_letter_policy": "dead-letter extraction metadata only",
        "canonical_graph_write_allowed": False,
        "status": "implemented",
        "evidence_refs": ["scripts/polycareer_extract_worker.py"],
        "detail": {
            "subsystem": "indy_polycareer",
            "registered_by": "polycareer_register_contracts.py",
            "groq_fallback_allowed": True,
            "groq_models": [
                "llama-4-scout-17b-16e-instruct",
                "llama-3.3-70b-versatile",
            ],
            "strict_dequeue_enforced": True,
        },
    },
    {
        "worker_key": "polycareer_glow_v1",
        "queue_name": "indy_polycareer",
        "script_path": "scripts/polycareer_glow_watch_worker.py",
        "input_contract": {
            "job_kind": "glow_watch_stage",
            "required_keys": ["since_hours", "limit", "threshold", "idempotency_key"],
        },
        "output_contract": {
            "writes": [
                "lucidota_control.absurd_queue_event",
                "lucidota_control.workflow_event",
                "05_OUTPUTS/indy_polycareer/glow_watch_findings.jsonl",
                "05_OUTPUTS/indy_polycareer/glow_watch_latest.md",
                "05_OUTPUTS/indy_polycareer/<job_uuid>/receipt.json",
            ],
            "truth_status": "not_truth_claim_candidate",
        },
        "idempotency_rule": "sha256(queue_name + job_kind + idempotency_key)",
        "retry_policy": "retry watch cycle idempotently; read-only observation",
        "dead_letter_policy": "dead-letter watch metadata only",
        "canonical_graph_write_allowed": False,
        "status": "implemented",
        "evidence_refs": ["scripts/polycareer_glow_watch_worker.py"],
        "detail": {
            "subsystem": "indy_polycareer",
            "registered_by": "polycareer_register_contracts.py",
            "groq_fallback_allowed": False,
            "strict_dequeue_enforced": True,
        },
    },
]

QUEUE_ROWS: list[dict[str, Any]] = [
    {
        "queue_name": "indy_polycareer",
        "owner_subsystem": "indy_polycareer",
        "status": "active",
        "max_attempts": 3,
        "visibility_timeout_seconds": 300,
        "notes": (
            "Indy Polycareer ABSURD queue. Handles intake_custody, extract_claims, "
            "and glow_watch_stage job kinds. Never writes canonical graph tables."
        ),
    },
]

WORKFLOW_ROWS: list[dict[str, Any]] = [
    {
        "workflow_name": "indy-polycareer-intake",
        "owner": "indy_polycareer",
        "phase": "020",
        "status": "active",
        "command": "scripts/polycareer_intake_worker.py",
        "inputs": {"queue_name": "indy_polycareer", "job_kind": "intake_custody"},
        "outputs": {
            "writes": ["cas", "absurd_queue_event", "workflow_event", "receipt"],
            "enqueues": "indy_polycareer/extract_claims",
        },
        "notes": (
            "Receive raw drops (files, text, agent artifacts). "
            "Hash, classify, dedupe by sha256. Emits routing decision."
        ),
    },
    {
        "workflow_name": "indy-polycareer-extract",
        "owner": "indy_polycareer",
        "phase": "020",
        "status": "active",
        "command": "scripts/polycareer_extract_worker.py",
        "inputs": {"queue_name": "indy_polycareer", "job_kind": "extract_claims"},
        "outputs": {
            "writes": ["absurd_queue_event", "workflow_event", "receipt"],
            "groq_fallback_allowed": True,
        },
        "notes": (
            "Per-document extraction: GLiNER NER, role routing, glow scoring, "
            "OCR sidecar for images/PDF."
        ),
    },
    {
        "workflow_name": "indy-polycareer-glow-watch",
        "owner": "indy_polycareer",
        "phase": "020",
        "status": "active",
        "command": "scripts/polycareer_glow_watch_worker.py",
        "inputs": {"queue_name": "indy_polycareer", "job_kind": "glow_watch_stage"},
        "outputs": {
            "writes": [
                "glow_watch_findings.jsonl",
                "glow_watch_latest.md",
                "absurd_queue_event",
                "workflow_event",
                "receipt",
            ],
            "groq_fallback_allowed": False,
        },
        "notes": (
            "Periodic artifact/workflow-event observation. "
            "Scores for glow. Writes findings JSONL + workflow_event."
        ),
    },
]


# ---------------------------------------------------------------------------
# SQL operations
# ---------------------------------------------------------------------------

def upsert_worker_contracts(cur: Any, execute: bool) -> list[dict[str, Any]]:
    results = []
    for c in WORKER_CONTRACTS:
        row = {
            "worker_key": c["worker_key"],
            "queue_name": c["queue_name"],
            "script_path": c["script_path"],
            "input_contract": json.dumps(c["input_contract"]),
            "output_contract": json.dumps(c["output_contract"]),
            "idempotency_rule": c["idempotency_rule"],
            "retry_policy": c["retry_policy"],
            "dead_letter_policy": c["dead_letter_policy"],
            "canonical_graph_write_allowed": c["canonical_graph_write_allowed"],
            "status": c["status"],
            "evidence_refs": json.dumps(c["evidence_refs"]),
            "detail": json.dumps(c["detail"]),
        }
        if execute:
            cur.execute(
                """
                INSERT INTO lucidota_control.absurd_worker_contract
                  (worker_key, queue_name, script_path,
                   input_contract, output_contract,
                   idempotency_rule, retry_policy, dead_letter_policy,
                   canonical_graph_write_allowed, status,
                   evidence_refs, detail, updated_at)
                VALUES
                  (%(worker_key)s, %(queue_name)s, %(script_path)s,
                   %(input_contract)s::jsonb, %(output_contract)s::jsonb,
                   %(idempotency_rule)s, %(retry_policy)s, %(dead_letter_policy)s,
                   %(canonical_graph_write_allowed)s, %(status)s,
                   %(evidence_refs)s::jsonb, %(detail)s::jsonb, now())
                ON CONFLICT (worker_key) DO UPDATE SET
                  queue_name                  = EXCLUDED.queue_name,
                  script_path                 = EXCLUDED.script_path,
                  input_contract              = EXCLUDED.input_contract,
                  output_contract             = EXCLUDED.output_contract,
                  idempotency_rule            = EXCLUDED.idempotency_rule,
                  retry_policy                = EXCLUDED.retry_policy,
                  dead_letter_policy          = EXCLUDED.dead_letter_policy,
                  canonical_graph_write_allowed = EXCLUDED.canonical_graph_write_allowed,
                  status                      = EXCLUDED.status,
                  evidence_refs               = EXCLUDED.evidence_refs,
                  detail                      = EXCLUDED.detail,
                  updated_at                  = now()
                """,
                row,
            )
            action = "upserted"
        else:
            action = "dry_run"
        results.append({"worker_key": c["worker_key"], "queue_name": c["queue_name"], "action": action})
    return results


def upsert_queues(cur: Any, execute: bool) -> list[dict[str, Any]]:
    results = []
    for q in QUEUE_ROWS:
        if execute:
            cur.execute(
                """
                INSERT INTO lucidota_control.absurd_queue
                  (queue_name, owner_subsystem, status, max_attempts,
                   visibility_timeout_seconds, notes, updated_at)
                VALUES
                  (%(queue_name)s, %(owner_subsystem)s, %(status)s,
                   %(max_attempts)s, %(visibility_timeout_seconds)s,
                   %(notes)s, now())
                ON CONFLICT (queue_name) DO UPDATE SET
                  owner_subsystem            = EXCLUDED.owner_subsystem,
                  status                     = EXCLUDED.status,
                  max_attempts               = EXCLUDED.max_attempts,
                  visibility_timeout_seconds = EXCLUDED.visibility_timeout_seconds,
                  notes                      = EXCLUDED.notes,
                  updated_at                 = now()
                """,
                q,
            )
            action = "upserted"
        else:
            action = "dry_run"
        results.append({"queue_name": q["queue_name"], "action": action})
    return results


def upsert_workflows(cur: Any, execute: bool) -> list[dict[str, Any]]:
    results = []
    for w in WORKFLOW_ROWS:
        row = {
            "workflow_name": w["workflow_name"],
            "owner": w["owner"],
            "phase": w["phase"],
            "status": w["status"],
            "command": w["command"],
            "inputs": json.dumps(w["inputs"]),
            "outputs": json.dumps(w["outputs"]),
            "notes": w["notes"],
        }
        if execute:
            cur.execute(
                """
                INSERT INTO lucidota_control.workflow_registry
                  (workflow_name, owner, phase, status, command,
                   inputs, outputs, notes, updated_at)
                VALUES
                  (%(workflow_name)s, %(owner)s, %(phase)s, %(status)s,
                   %(command)s, %(inputs)s::jsonb, %(outputs)s::jsonb,
                   %(notes)s, now())
                ON CONFLICT (workflow_name) DO UPDATE SET
                  owner     = EXCLUDED.owner,
                  phase     = EXCLUDED.phase,
                  status    = EXCLUDED.status,
                  command   = EXCLUDED.command,
                  inputs    = EXCLUDED.inputs,
                  outputs   = EXCLUDED.outputs,
                  notes     = EXCLUDED.notes,
                  updated_at = now()
                """,
                row,
            )
            action = "upserted"
        else:
            action = "dry_run"
        results.append({"workflow_name": w["workflow_name"], "action": action})
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Register indy_polycareer ABSURD worker contracts and workflow registry entries."
    )
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", dest="dry_run", action="store_true", default=True,
                      help="Show what would be inserted without writing (default)")
    mode.add_argument("--execute", dest="dry_run", action="store_false",
                      help="Write contracts and registry rows for real")
    p.add_argument("--database-url", dest="database_url", default=None,
                   help="Override DB URL (default: LUCIDOTA_GO_STATE_DSN env or postgresql:///lucidota_state)")
    p.add_argument("--json", dest="json_output", action="store_true", default=False,
                   help="Print JSON receipt to stdout instead of human summary")
    return p


def run(args: argparse.Namespace) -> int:
    execute = not args.dry_run
    url = db_url(args)
    started = now()

    report: dict[str, Any] = {
        "schema": "lucidota.indy_polycareer.contract_registration.v1",
        "receipt_mode": "ABSURD_POSTGRES_RUNTIME",
        "subsystem": "indy_polycareer",
        "db_url": redact(url),
        "execute": execute,
        "started_at": started,
        "finished_at": None,
        "ok": False,
        "contracts": [],
        "queues": [],
        "workflows": [],
        "blocked": [],
        "errors": [],
        "report_path": None,
    }

    if psycopg is None:
        msg = "psycopg not available; cannot connect to DB"
        report["errors"].append(msg)
        report["finished_at"] = now()
        out = write_receipt(report)
        print(f"BLOCKED: {msg}", file=sys.stderr)
        print(f"RECEIPT_PATH={out.relative_to(ROOT)}")
        return 1

    try:
        conn = psycopg.connect(url, row_factory=dict_row)
    except Exception as exc:
        msg = f"db_connect_failed: {exc}"
        report["errors"].append(msg)
        report["finished_at"] = now()
        out = write_receipt(report)
        print(f"BLOCKED: {msg}", file=sys.stderr)
        print(f"RECEIPT_PATH={out.relative_to(ROOT)}")
        return 1

    with conn.cursor() as cur:
        # --- Queues first (absurd_worker_contract has FK to absurd_queue) ---
        cur.execute("SAVEPOINT sp_queues")
        if not table_exists(cur, "lucidota_control.absurd_queue"):
            report["blocked"].append("lucidota_control.absurd_queue table missing — queue rows not registered")
            cur.execute("RELEASE SAVEPOINT sp_queues")
        else:
            try:
                report["queues"] = upsert_queues(cur, execute)
                cur.execute("RELEASE SAVEPOINT sp_queues")
            except Exception as exc:
                cur.execute("ROLLBACK TO SAVEPOINT sp_queues")
                cur.execute("RELEASE SAVEPOINT sp_queues")
                report["errors"].append(f"queue_upsert_failed: {exc}")

        # --- Worker contracts (after queue rows exist) ---
        cur.execute("SAVEPOINT sp_contracts")
        if not table_exists(cur, "lucidota_control.absurd_worker_contract"):
            report["blocked"].append("lucidota_control.absurd_worker_contract table missing — contracts not registered")
            cur.execute("RELEASE SAVEPOINT sp_contracts")
        else:
            try:
                report["contracts"] = upsert_worker_contracts(cur, execute)
                cur.execute("RELEASE SAVEPOINT sp_contracts")
            except Exception as exc:
                cur.execute("ROLLBACK TO SAVEPOINT sp_contracts")
                cur.execute("RELEASE SAVEPOINT sp_contracts")
                report["errors"].append(f"contract_upsert_failed: {exc}")

        # --- Workflow registry ---
        cur.execute("SAVEPOINT sp_workflows")
        if not table_exists(cur, "lucidota_control.workflow_registry"):
            report["blocked"].append("lucidota_control.workflow_registry table missing — workflows not registered")
            cur.execute("RELEASE SAVEPOINT sp_workflows")
        else:
            try:
                report["workflows"] = upsert_workflows(cur, execute)
                cur.execute("RELEASE SAVEPOINT sp_workflows")
            except Exception as exc:
                cur.execute("ROLLBACK TO SAVEPOINT sp_workflows")
                cur.execute("RELEASE SAVEPOINT sp_workflows")
                report["errors"].append(f"workflow_upsert_failed: {exc}")

        if execute and not report["errors"]:
            conn.commit()
        elif execute and report["errors"]:
            conn.rollback()
        elif not execute:
            conn.rollback()

    report["ok"] = not report["errors"]
    report["finished_at"] = now()
    out = write_receipt(report)
    report["report_path"] = str(out.relative_to(ROOT))

    if args.json_output:
        print(json.dumps(report, indent=2, default=str))
        return 0 if report["ok"] else 1

    # Human summary
    mode_label = "EXECUTE" if execute else "DRY-RUN"
    print(f"\n[polycareer_register_contracts] {mode_label} — {report['db_url']}")
    print(f"  started_at : {report['started_at']}")
    print(f"  finished_at: {report['finished_at']}")
    print()

    if report["contracts"]:
        print(f"  WORKER CONTRACTS ({len(report['contracts'])} rows):")
        for c in report["contracts"]:
            print(f"    {c['action']:12s}  worker_key={c['worker_key']}  queue={c['queue_name']}")
    else:
        print("  WORKER CONTRACTS: none registered (see blocked/errors)")

    if report["queues"]:
        print(f"\n  ABSURD QUEUES ({len(report['queues'])} rows):")
        for q in report["queues"]:
            print(f"    {q['action']:12s}  queue_name={q['queue_name']}")
    else:
        print("\n  ABSURD QUEUES: none registered (see blocked/errors)")

    if report["workflows"]:
        print(f"\n  WORKFLOW REGISTRY ({len(report['workflows'])} rows):")
        for w in report["workflows"]:
            print(f"    {w['action']:12s}  workflow_name={w['workflow_name']}")
    else:
        print("\n  WORKFLOW REGISTRY: none registered (see blocked/errors)")

    if report["blocked"]:
        print("\n  BLOCKED:")
        for b in report["blocked"]:
            print(f"    ! {b}")

    if report["errors"]:
        print("\n  ERRORS:")
        for e in report["errors"]:
            print(f"    ERROR: {e}")

    status = "OK" if report["ok"] else "FAILED"
    print(f"\n  status: {status}")
    print(f"  RECEIPT_PATH={out.relative_to(ROOT)}")
    return 0 if report["ok"] else 1


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    sys.exit(run(args))


if __name__ == "__main__":
    main()
