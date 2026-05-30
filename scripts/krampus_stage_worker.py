#!/usr/bin/env python3
"""ABSURD worker #3 for krampus_ingest: stages extracted claims into lucidota_go.staging_packet.

Receives a job from queue_name='krampus_ingest' with job_kind='stage'. For each claim
in payload['claims'], maps to lucidota_go.staging_packet columns (source_id, parser_name,
proposed_term, raw_anchor, claim, proposed_item, proposed_edges, status, confidence_bps).
Falls back to 05_OUTPUTS/krampus_ingest/ JSONL only when the table is absent. Validates
the worker contract at dequeue via absurd_worker_contracts.validate_worker_contract().
Writes a per-job receipt and appends one row to the ingest_log TSV. Never writes
canonical graph tables (graph_item, graph_edge, graph_journal).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psycopg
from psycopg.rows import dict_row

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
try:
    from absurd_worker_contracts import validate_worker_contract, record_worker_contract_rejection
    _HAS_CONTRACT_HELPER = True
except Exception:
    _HAS_CONTRACT_HELPER = False
OUT_BASE = ROOT / "05_OUTPUTS" / "krampus_ingest"
QUEUE_NAME = "krampus_ingest"
JOB_KIND = "stage"
WORKER_KEY = "krampus_stage_v1"
WORKFLOW_NAME = "absurd-krampus-ingest-stage"
CANONICAL_GRAPH_TABLES = [
    "lucidota_go.graph_item",
    "lucidota_go.graph_edge",
    "lucidota_go.graph_journal",
]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def sha256_obj(obj: Any) -> str:
    return hashlib.sha256(dumps(obj).encode()).hexdigest()


def sha256_str(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def redact(url: str | None) -> str:
    if not url:
        return "unset"
    if url.startswith("postgresql:///"):
        return "postgresql:///<database>"
    if "@" in url:
        return "postgresql://<redacted>@" + url.split("@", 1)[1]
    return "set_redacted"


def first_value(row: Any) -> Any:
    if row is None:
        return None
    if isinstance(row, dict):
        return next(iter(row.values()))
    return row[0]


def state_url(args: argparse.Namespace) -> str:
    return (
        getattr(args, "state_database_url", None)
        or os.environ.get("ABSURD_SYSTEM_DATABASE_URL")
        or "postgresql:///lucidota_state"
    )


def storage_url(args: argparse.Namespace) -> str:
    return (
        getattr(args, "storage_database_url", None)
        or os.environ.get("KORPUS_DATABASE_URL")
        or "postgresql:///lucidota_storage"
    )


def count_canonical_graph(conn) -> dict[str, int | None]:
    result = {}
    for table in CANONICAL_GRAPH_TABLES:
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT to_regclass(%s)", (table,))
                if first_value(cur.fetchone()) is None:
                    result[table] = None
                    continue
                cur.execute(f"SELECT count(*) FROM {table}")
                result[table] = int(first_value(cur.fetchone()))
        except Exception:
            result[table] = None
    return result


# ---------------------------------------------------------------------------
# Ternary scoring
# ---------------------------------------------------------------------------

def ternary_score(claim: Any) -> int:
    """Return +1 / 0 / -1 based on claim confidence.

    Accepts a dict with a 'confidence' key or a bare string/None.
    Empty, None, or clearly noise claims always score -1.
    """
    if not claim:
        return -1
    if isinstance(claim, dict):
        text = str(claim.get("text") or claim.get("claim_text") or "").strip()
        confidence = claim.get("confidence")
    else:
        text = str(claim).strip()
        confidence = None

    if not text or len(text) < 3:
        return -1

    if confidence is not None:
        try:
            c = float(confidence)
        except (TypeError, ValueError):
            c = None
        if c is not None:
            if c > 0.7:
                return 1
            if c < 0.3:
                return -1
            return 0

    # No explicit confidence: default to neutral
    return 0


def claim_text(claim: Any) -> str:
    if isinstance(claim, dict):
        return str(claim.get("text") or claim.get("claim_text") or "").strip()
    return str(claim or "").strip()


def claim_confidence(claim: Any) -> float | None:
    if isinstance(claim, dict):
        v = claim.get("confidence")
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                pass
    return None


# ---------------------------------------------------------------------------
# Staging table target: lucidota_go.staging_packet
# ---------------------------------------------------------------------------

STAGING_TABLE = "lucidota_go.staging_packet"

# Confidence_bps mapping from ternary: +1 -> 10, 0 -> 4, -1 -> 2
# All values are in the allowed set {0,2,4,6,10,50,69,150}.
_TERNARY_TO_BPS: dict[int, int] = {1: 10, 0: 4, -1: 2}
# Default proposed_term for krampus claims when none is specified
_DEFAULT_TERM = "CLAIM"


def table_exists(conn, table: str) -> bool:
    with conn.cursor() as cur:
        cur.execute("SELECT to_regclass(%s)", (table,))
        return first_value(cur.fetchone()) is not None


def claim_proposed_term(claim: Any) -> str:
    """Extract proposed_term from claim dict; default to CLAIM."""
    if isinstance(claim, dict):
        t = str(claim.get("proposed_term") or claim.get("term") or "").strip().upper()
        if t:
            return t
    return _DEFAULT_TERM


def try_stage_to_db(
    conn,
    *,
    source_id: str,
    raw_anchor: str,
    claim_str: str,
    proposed_term: str,
    proposed_item: dict[str, Any],
    confidence_bps: int,
) -> tuple[bool, str]:
    """Attempt INSERT into lucidota_go.staging_packet. Returns (ok, error_str)."""
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO lucidota_go.staging_packet
                  (source_id, parser_name, proposed_term, raw_anchor, claim,
                   proposed_item, proposed_edges, status, confidence_bps)
                VALUES (%s, 'krampus_stage', %s, %s, %s,
                        %s::jsonb, '[]'::jsonb, 'pending', %s)
                ON CONFLICT DO NOTHING
                RETURNING packet_uuid::text
                """,
                (
                    source_id,
                    proposed_term,
                    raw_anchor[:200],
                    claim_str[:500],
                    json.dumps(proposed_item, sort_keys=True, default=str),
                    confidence_bps,
                ),
            )
            row = cur.fetchone()
        return True, str(first_value(row) or "conflict_skip")
    except Exception as exc:
        return False, str(exc)


# ---------------------------------------------------------------------------
# Fallback JSONL writer
# ---------------------------------------------------------------------------

def write_fallback_jsonl(records: list[dict[str, Any]]) -> Path:
    OUT_BASE.mkdir(parents=True, exist_ok=True)
    path = OUT_BASE / "staging_candidates.jsonl"
    with path.open("a", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, sort_keys=True, default=str) + "\n")
    return path


def write_blocked_receipt(job_id: str, records: list[dict[str, Any]], reason: str) -> Path:
    OUT_BASE.mkdir(parents=True, exist_ok=True)
    job_dir = OUT_BASE / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    path = job_dir / "blocked_receipt.json"
    path.write_text(
        json.dumps(
            {
                "schema": "lucidota.krampus_stage.blocked_receipt.v1",
                "generated_at": now_iso(),
                "job_id": job_id,
                "blocked_reason": reason,
                "blocked_count": len(records),
                "fallback_lane": "jsonl",
                "canonical_graph_writes_performed": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


# ---------------------------------------------------------------------------
# Receipt + log
# ---------------------------------------------------------------------------

def write_staged_receipt(
    job_id: str,
    *,
    staged_count: int,
    blocked_count: int,
    ternary_breakdown: dict[str, int],
    evidence_ref: str,
    lane: str,
    groq_called: bool,
    fallback_jsonl_path: str | None,
    blocked_reason: str | None,
) -> Path:
    job_dir = OUT_BASE / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    path = job_dir / "staged.json"
    path.write_text(
        json.dumps(
            {
                "schema": "lucidota.krampus_stage.staged_receipt.v1",
                "generated_at": now_iso(),
                "job_id": job_id,
                "staged_count": staged_count,
                "blocked_count": blocked_count,
                "ternary_breakdown": ternary_breakdown,
                "evidence_ref": evidence_ref,
                "lane": lane,
                "groq_called": groq_called,
                "fallback_jsonl_path": fallback_jsonl_path,
                "blocked_reason": blocked_reason,
                "canonical_graph_writes_performed": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


def append_ingest_log(
    *,
    job_id: str,
    file_path: str,
    staged_count: int,
    lane: str,
    groq_called: bool,
) -> None:
    OUT_BASE.mkdir(parents=True, exist_ok=True)
    log_path = OUT_BASE / "ingest_log.tsv"
    line = "\t".join(
        [
            now_iso(),
            job_id,
            file_path,
            str(staged_count),
            lane,
            "true" if groq_called else "false",
        ]
    )
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


# ---------------------------------------------------------------------------
# Core staging logic (called inside or outside a DB transaction)
# ---------------------------------------------------------------------------

def stage_claims(
    args: argparse.Namespace,
    job_id: str,
    payload: dict[str, Any],
    storage_conn,
    *,
    dry_run: bool,
) -> dict[str, Any]:
    """Process all claims in payload and return a result summary.

    Each claim is mapped to lucidota_go.staging_packet:
      source_id     <- job_id (or payload['source_id'])
      parser_name   <- 'krampus_stage' (hardcoded)
      proposed_term <- claim['proposed_term'] or claim['term'] or 'CLAIM'
      raw_anchor    <- claim text (truncated to 200)
      claim         <- claim text (truncated to 500)
      proposed_item <- jsonb envelope with full claim metadata
      proposed_edges<- '[]' (krampus does not propose edges)
      status        <- 'pending'
      confidence_bps<- ternary_score +1->10, 0->4, -1->2
    """
    claims = payload.get("claims") or []
    source_file = str(payload.get("source_file") or payload.get("file_path") or job_id)
    # source_id for staging_packet: prefer explicit, fall back to job_id
    source_id = str(payload.get("source_id") or job_id)

    # Probe staging table once
    db_available = False
    db_error = ""
    if not dry_run:
        try:
            db_available = table_exists(storage_conn, STAGING_TABLE)
        except Exception as exc:
            db_error = str(exc)

    staged_db: list[dict[str, Any]] = []
    fallback_records: list[dict[str, Any]] = []
    ternary: dict[str, int] = {"positive": 0, "neutral": 0, "negative": 0}

    for claim in claims:
        text = claim_text(claim)
        if not text:
            continue
        score = ternary_score(claim)
        if score == 1:
            ternary["positive"] += 1
        elif score == -1:
            ternary["negative"] += 1
        else:
            ternary["neutral"] += 1

        conf_bps = _TERNARY_TO_BPS.get(score, 4)
        pterm = claim_proposed_term(claim)
        ev_ref = f"krampus_ingest:{sha256_str(text)[:16]}"

        # proposed_item envelope stores full metadata
        proposed_item: dict[str, Any] = {
            "claim_text": text,
            "ternary_score": score,
            "evidence_ref": ev_ref,
            "source_job_id": job_id,
            "source_file": source_file,
        }
        if isinstance(claim, dict):
            for k in ("confidence", "metadata", "tags", "origin"):
                if k in claim:
                    proposed_item[k] = claim[k]

        rec = {
            "source_id": source_id,
            "parser_name": "krampus_stage",
            "proposed_term": pterm,
            "raw_anchor": text[:200],
            "claim": text[:500],
            "proposed_item": proposed_item,
            "proposed_edges": [],
            "status": "pending",
            "confidence_bps": conf_bps,
            "evidence_ref": ev_ref,
        }

        if dry_run:
            staged_db.append(rec)
            continue

        if db_available:
            ok, err_or_uuid = try_stage_to_db(
                storage_conn,
                source_id=source_id,
                raw_anchor=text,
                claim_str=text,
                proposed_term=pterm,
                proposed_item=proposed_item,
                confidence_bps=conf_bps,
            )
            if ok:
                rec["packet_uuid"] = err_or_uuid
                staged_db.append(rec)
            else:
                db_available = False
                db_error = err_or_uuid
                fallback_records.append(rec)
        else:
            fallback_records.append(rec)

    lane: str
    fallback_jsonl_path: str | None = None
    blocked_reason: str | None = None

    if dry_run:
        lane = "dry_run"
    elif fallback_records:
        jsonl_path = write_fallback_jsonl(fallback_records)
        fallback_jsonl_path = str(jsonl_path.relative_to(ROOT))
        blocked_reason = db_error or "table_missing"
        write_blocked_receipt(job_id, fallback_records, blocked_reason)
        lane = "jsonl_fallback"
    else:
        lane = "db"

    staged_count = len(staged_db) if not dry_run else len(staged_db)
    blocked_count = len(fallback_records)

    evidence_ref = f"krampus_ingest:{sha256_str(job_id)[:16]}"

    receipt_path = write_staged_receipt(
        job_id,
        staged_count=staged_count,
        blocked_count=blocked_count,
        ternary_breakdown=ternary,
        evidence_ref=evidence_ref,
        lane=lane,
        groq_called=False,
        fallback_jsonl_path=fallback_jsonl_path,
        blocked_reason=blocked_reason,
    )

    append_ingest_log(
        job_id=job_id,
        file_path=source_file,
        staged_count=staged_count,
        lane=lane,
        groq_called=False,
    )

    return {
        "staged_count": staged_count,
        "blocked_count": blocked_count,
        "ternary_breakdown": ternary,
        "evidence_ref": evidence_ref,
        "lane": lane,
        "groq_called": False,
        "receipt_path": str(receipt_path.relative_to(ROOT)),
        "fallback_jsonl_path": fallback_jsonl_path,
        "blocked_reason": blocked_reason,
        "canonical_graph_writes_performed": False,
        "outcome": "succeeded",
        "ok": True,
    }


# ---------------------------------------------------------------------------
# ABSURD worker-once
# ---------------------------------------------------------------------------

def worker_once(args: argparse.Namespace, execute: bool) -> tuple[dict[str, Any], list[str]]:
    url = state_url(args); blockers: list[str] = []
    worker_id = getattr(args, "worker_id", None) or f"krampus_stage:{socket.gethostname()}:{os.getpid()}"
    result: dict[str, Any] = {
        "state_database_url": redact(url),
        "queue": QUEUE_NAME,
        "worker_id": worker_id,
        "execute_performed": False,
        "job_processed": False,
        "canonical_graph_writes_performed": False,
    }

    with psycopg.connect(url, row_factory=dict_row) as state_conn:
        with state_conn.cursor() as cur:
            # Peek / FOR UPDATE SKIP LOCKED
            if not execute:
                cur.execute(
                    """
                    SELECT job_uuid::text, workflow_name, job_kind, idempotency_key, status::text, payload
                    FROM lucidota_control.absurd_queue_job
                    WHERE queue_name=%s AND status='queued' AND run_after <= now()
                    ORDER BY priority ASC, created_at ASC
                    LIMIT 1
                    """,
                    (QUEUE_NAME,),
                )
                row = cur.fetchone()
                result["would_process"] = dict(row) if row else None
                return result, blockers

            cur.execute(
                """
                SELECT job_uuid::text, workflow_name, job_kind, idempotency_key,
                       payload, attempt_count, max_attempts
                FROM lucidota_control.absurd_queue_job
                WHERE queue_name=%s AND status='queued' AND run_after <= now()
                  AND job_kind=%s
                ORDER BY priority ASC, created_at ASC
                FOR UPDATE SKIP LOCKED
                LIMIT 1
                """,
                (QUEUE_NAME, JOB_KIND),
            )
            row = cur.fetchone()
            if not row:
                result["no_job_available"] = True
                return result, blockers

            job_uuid = row["job_uuid"]
            payload = dict(row["payload"] or {})
            attempt_count = int(row["attempt_count"])
            max_attempts = int(row["max_attempts"])

            # --- Contract validation at dequeue ---
            if _HAS_CONTRACT_HELPER:
                contract = validate_worker_contract(
                    cur, queue_name=QUEUE_NAME, job_kind=JOB_KIND, worker_key=WORKER_KEY
                )
                if not contract.ok:
                    gate_result, ek = record_worker_contract_rejection(
                        cur,
                        job_uuid=job_uuid,
                        queue_name=QUEUE_NAME,
                        payload=payload,
                        contract=contract,
                        event_source="krampus_stage_worker",
                    )
                    state_conn.commit()
                    blockers.append(ek)
                    result.update({"worker_contract_rejected": True, "error_kind": ek, "error_message": contract.error_message})
                    return result, blockers

            # Claim it
            cur.execute(
                """
                UPDATE lucidota_control.absurd_queue_job
                SET status='running', leased_by=%s, locked_by=%s,
                    locked_at=now(), lease_expires_at=now()+interval '5 minutes',
                    last_heartbeat_at=now(), attempt_count=attempt_count+1, updated_at=now()
                WHERE job_uuid=%s::uuid
                """,
                (worker_id, worker_id, job_uuid),
            )
            cur.execute(
                """
                INSERT INTO lucidota_control.absurd_queue_event
                  (job_uuid, queue_name, event_kind, event_source, detail)
                VALUES (%s::uuid, %s, 'started', 'krampus_stage_worker', %s::jsonb)
                """,
                (job_uuid, QUEUE_NAME, json.dumps({"worker_id": worker_id})),
            )
            state_conn.commit()

        # Stage claims (uses separate storage connection; rolls back cleanly)
        job_result: dict[str, Any] = {}
        ok = False
        error_kind = ""
        error_message = ""

        try:
            surl = storage_url(args)
            with psycopg.connect(surl) as storage_conn:
                # Guard: canonical graph must not change
                before_graph = count_canonical_graph(storage_conn)

                job_result = stage_claims(
                    args,
                    job_uuid,
                    payload,
                    storage_conn,
                    dry_run=False,
                )

                after_graph = count_canonical_graph(storage_conn)
                if before_graph != after_graph:
                    storage_conn.rollback()
                    blockers.append("canonical_graph_counts_changed")
                    job_result["canonical_graph_writes_performed"] = True
                    ok = False
                    error_kind = "canonical_graph_write_detected"
                    error_message = "staging write mutated canonical graph; rolled back"
                else:
                    storage_conn.commit()
                    ok = True

        except Exception as exc:
            ok = False
            error_kind = "stage_exception"
            error_message = str(exc)
            job_result = {"error": error_kind, "error_message": error_message, "outcome": "failed", "ok": False}

        # Finalize queue state
        final_attempt = attempt_count + 1 >= max_attempts
        if ok:
            status = "succeeded"
        elif final_attempt:
            status = "dead_lettered"
        else:
            status = "failed"

        with state_conn.cursor() as cur:
            cur.execute(
                """
                UPDATE lucidota_control.absurd_queue_job
                SET status=%s, result=%s::jsonb,
                    error_kind=%s, error_message=%s, last_error=%s,
                    completed_at=CASE WHEN %s='succeeded' THEN now() ELSE completed_at END,
                    updated_at=now()
                WHERE job_uuid=%s::uuid
                """,
                (
                    status,
                    json.dumps(job_result, default=str),
                    error_kind,
                    error_message,
                    error_message,
                    status,
                    job_uuid,
                ),
            )
            cur.execute(
                """
                INSERT INTO lucidota_control.absurd_queue_event
                  (job_uuid, queue_name, event_kind, event_source, detail)
                VALUES (%s::uuid, %s, %s, 'krampus_stage_worker', %s::jsonb)
                """,
                (
                    job_uuid,
                    QUEUE_NAME,
                    status if status in {"succeeded", "failed", "dead_lettered"} else "failed",
                    json.dumps(job_result, default=str),
                ),
            )
            cur.execute(
                """
                INSERT INTO lucidota_control.workflow_event
                  (workflow_id, run_id, phase, status, source, detail)
                VALUES (%s, %s, 'krampus_stage', %s, 'krampus_stage_worker', %s::jsonb)
                RETURNING event_id::text
                """,
                (
                    WORKFLOW_NAME,
                    job_uuid,
                    status,
                    json.dumps({"queue": QUEUE_NAME, "job_uuid": job_uuid, "result": job_result}, default=str),
                ),
            )
            event_id = first_value(cur.fetchone())

            if status == "dead_lettered":
                cur.execute(
                    """
                    INSERT INTO lucidota_control.absurd_queue_dead_letter
                      (job_uuid, queue_name, workflow_name, job_kind, idempotency_key,
                       error_kind, error_message, attempt_count, payload_sha256, context)
                    SELECT job_uuid, queue_name, workflow_name, job_kind, idempotency_key,
                           %s, %s, attempt_count, %s, %s::jsonb
                    FROM lucidota_control.absurd_queue_job WHERE job_uuid=%s::uuid
                    ON CONFLICT (job_uuid) WHERE resolved=false DO UPDATE SET
                      error_message=EXCLUDED.error_message,
                      attempt_count=EXCLUDED.attempt_count,
                      last_seen_at=now(),
                      context=EXCLUDED.context
                    """,
                    (
                        error_kind or "stage_failed",
                        error_message,
                        sha256_obj(payload),
                        json.dumps(job_result, default=str),
                        job_uuid,
                    ),
                )
        state_conn.commit()

    result.update(
        {
            "execute_performed": True,
            "job_processed": True,
            "job_uuid": job_uuid,
            "workflow_event_id": event_id,
            "status": status,
            "staged_count": job_result.get("staged_count", 0),
            "blocked_count": job_result.get("blocked_count", 0),
            "ternary_breakdown": job_result.get("ternary_breakdown", {}),
            "lane": job_result.get("lane", "unknown"),
            "receipt_path": job_result.get("receipt_path"),
            "canonical_graph_writes_performed": bool(job_result.get("canonical_graph_writes_performed", False)),
        }
    )
    if not ok:
        blockers.append(error_kind or "stage_failed")
    return result, blockers


# ---------------------------------------------------------------------------
# Dry-run preview (no DB, just shows what would be staged)
# ---------------------------------------------------------------------------

def dry_run_preview(args: argparse.Namespace) -> tuple[dict[str, Any], list[str]]:
    blockers: list[str] = []
    url = state_url(args)
    result: dict[str, Any] = {
        "state_database_url": redact(url),
        "queue": QUEUE_NAME,
        "execute_performed": False,
        "canonical_graph_writes_performed": False,
    }
    try:
        with psycopg.connect(url, row_factory=dict_row) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT job_uuid::text, workflow_name, job_kind, idempotency_key, status::text, payload
                    FROM lucidota_control.absurd_queue_job
                    WHERE queue_name=%s AND status='queued' AND run_after <= now()
                    ORDER BY priority ASC, created_at ASC
                    LIMIT 1
                    """,
                    (QUEUE_NAME,),
                )
                row = cur.fetchone()
    except Exception as exc:
        blockers.append(f"state_db_error:{exc}")
        return result, blockers

    if not row:
        result["would_process"] = None
        return result, blockers

    payload = dict(row["payload"] or {})
    claims = payload.get("claims") or []
    preview: list[dict[str, Any]] = []
    for claim in claims[:10]:
        text = claim_text(claim)
        if not text:
            continue
        score = ternary_score(claim)
        preview.append(
            {
                "claim_text": text[:120],
                "ternary_score": score,
                "evidence_ref": f"krampus_ingest:{sha256_str(text)[:16]}",
            }
        )
    result.update(
        {
            "would_process": {"job_uuid": row["job_uuid"], "job_kind": row["job_kind"]},
            "claim_count": len(claims),
            "preview_count": len(preview),
            "preview": preview,
        }
    )
    return result, blockers


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------

def write_report(action: str, report: dict[str, Any]) -> Path:
    OUT_BASE.mkdir(parents=True, exist_ok=True)
    path = OUT_BASE / f"krampus_stage_worker_{action}_{stamp()}.json"
    report.setdefault("generated_at", now_iso())
    report["report_path"] = str(path.relative_to(ROOT))
    path.write_text(json.dumps(report, indent=2, sort_keys=False, default=str), encoding="utf-8")
    print(f"REPORT_PATH={path.relative_to(ROOT)}")
    return path


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description="ABSURD krampus_ingest stage worker — stages extracted claims into graph_staging_candidates."
    )
    ap.add_argument("--state-database-url", default=os.environ.get("ABSURD_SYSTEM_DATABASE_URL", "postgresql:///lucidota_state"))
    ap.add_argument("--storage-database-url", default=os.environ.get("KORPUS_DATABASE_URL", "postgresql:///lucidota_storage"))
    ap.add_argument("--worker-id")

    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--once", action="store_true", help="Process one job and exit.")
    mode.add_argument("--loop", action="store_true", help="Poll continuously until interrupted.")
    mode.add_argument("--dry-run", action="store_true", help="Preview without executing.")

    args = ap.parse_args()

    if args.dry_run:
        action_result, blockers = dry_run_preview(args)
        report = {
            "schema": "lucidota.krampus_stage_worker.report.v1",
            "action": "dry_run",
            "mode": "dry_run",
            "execute_performed": False,
            "canonical_graph_writes_performed": False,
            "action_result": action_result,
            "blockers": blockers,
        }
        write_report("dry_run", report)
        return 0

    if args.loop:
        idle = 0
        while True:
            action_result, blockers = worker_once(args, execute=True)
            if action_result.get("no_job_available"):
                idle += 1
                time.sleep(min(2 ** idle, 30))
            else:
                idle = 0
                report = {
                    "schema": "lucidota.krampus_stage_worker.report.v1",
                    "action": "worker_once",
                    "mode": "loop",
                    "execute_performed": True,
                    "canonical_graph_writes_performed": bool(action_result.get("canonical_graph_writes_performed", False)),
                    "action_result": action_result,
                    "blockers": blockers,
                }
                write_report("loop", report)
        # unreachable
        return 0

    # Default: --once (also triggered with no flags as a safe default)
    execute = bool(args.once) or (not args.dry_run and not args.loop)
    action_result, blockers = worker_once(args, execute=execute)
    report = {
        "schema": "lucidota.krampus_stage_worker.report.v1",
        "action": "worker_once",
        "mode": "execute" if execute else "dry_run",
        "execute_performed": bool(action_result.get("execute_performed", False)),
        "canonical_graph_writes_performed": bool(action_result.get("canonical_graph_writes_performed", False)),
        "action_result": action_result,
        "blockers": blockers,
    }
    write_report("worker_once", report)
    return 0 if not blockers else 2


if __name__ == "__main__":
    raise SystemExit(main())
