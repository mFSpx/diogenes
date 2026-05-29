#!/usr/bin/env python3
"""ABSURD worker: krampus_ingest / route_extract.

Treelite gate → text extract by lane → receipt → enqueue stage job.

Safety laws:
- Writes ABSURD job/event/workflow receipts and 05_OUTPUTS/krampus_ingest receipts only.
- Does not mutate canonical graph tables.
- Does not mutate Chrono temporal claims or KORPUS custody rows.
- Candidates are staged via enqueue; graph promotion gate is downstream.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import socket
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psycopg
from psycopg.rows import dict_row

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "05_OUTPUTS" / "krampus_ingest"

QUEUE_NAME = "krampus_ingest"
JOB_KIND = "route_extract"
WORKER_KEY = "krampus_extract_v1"
WORKFLOW_NAME = "krampus-ingest-route-extract"
STAGE_JOB_KIND = "stage"
CANONICAL_GRAPH_TABLES = ["lucidota_go.graph_item", "lucidota_go.graph_edge", "lucidota_go.graph_journal"]

ARTIFACT_PATH = ROOT / "03_VAULT" / "router" / "treelite_router_v0.tl"
MAX_FAST_CHARS = 2000
MAX_SLOW_BYTES = 50 * 1024  # 50KB
MAX_GROQ_CHARS = 3000
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL_TEXT = "llama-3.3-70b-versatile"
GROQ_MODEL_IMAGE = "meta-llama/llama-4-scout-17b-16e-instruct"
GROQ_MAX_TOKENS = 800

_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")
_DATE_RE = re.compile(
    r"\b(?:\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{4}|"
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4})\b"
)
_URL_RE = re.compile(r"https?://[^\s\"'<>]+")
_NAME_RE = re.compile(r"\b([A-Z][a-z]+ [A-Z][a-z]+(?:\s[A-Z][a-z]+)?)\b")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def sha256_obj(obj: Any) -> str:
    return hashlib.sha256(dumps(obj).encode()).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


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


def db_url(args: argparse.Namespace) -> str:
    return (
        getattr(args, "state_database_url", None)
        or os.environ.get("ABSURD_SYSTEM_DATABASE_URL")
        or "postgresql:///lucidota_state"
    )


def count_table(conn: Any, table: str) -> int | None:
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT to_regclass(%s)", (table,))
            if first_value(cur.fetchone()) is None:
                return None
            cur.execute(f"SELECT count(*) FROM {table}")  # noqa: S608
            return int(first_value(cur.fetchone()))
    except Exception:
        return None


def count_tables(conn: Any, tables: list[str]) -> dict[str, int | None]:
    return {t: count_table(conn, t) for t in tables}


def write_report(action: str, report: dict[str, Any]) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"krampus_route_extract_{action}_{stamp()}.json"
    report["report_path"] = str(path.relative_to(ROOT))
    report.setdefault("generated_at", now_iso())
    path.write_text(json.dumps(report, indent=2, sort_keys=False, default=str), encoding="utf-8")
    print(f"REPORT_PATH={path.relative_to(ROOT)}")
    return path


def write_extract_receipt(job_id: str, data: dict[str, Any]) -> Path:
    job_dir = OUT_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    path = job_dir / "extract.json"
    data.setdefault("generated_at", now_iso())
    path.write_text(json.dumps(data, indent=2, sort_keys=False, default=str), encoding="utf-8")
    print(f"EXTRACT_RECEIPT={path.relative_to(ROOT)}")
    return path


def read_groq_key() -> str | None:
    key_path = Path("/tmp/lucidota_groq_key")
    if key_path.exists():
        text = key_path.read_text(encoding="utf-8").strip()
        if text:
            return text
    return os.environ.get("GROQ_API_KEY")


# ---------------------------------------------------------------------------
# Treelite gate
# ---------------------------------------------------------------------------

def treelite_gate(
    token_count_norm: float,
    is_image: bool,
    is_text: bool,
) -> tuple[str, float]:
    """Return (lane, score). Falls back gracefully when treelite is missing."""
    try:
        import numpy as np
        import treelite.gtil as gtil
        from treelite.frontend import deserialize_bytes

        raw = ARTIFACT_PATH.read_bytes()
        model = deserialize_bytes(raw)
        features = np.array(
            [[token_count_norm, 0.0, float(is_image or not is_text), 0.0, 0.0]],
            dtype=np.float32,
        )
        result = gtil.predict(model, features)
        score = float(result.reshape(-1)[0])
    except Exception:
        # treelite unavailable or artifact unreadable — default to slow lane
        score = 0.5

    if score < 0.3:
        lane = "fast"
    elif score < 0.6:
        lane = "slow"
    else:
        lane = "external"
    return lane, score


# ---------------------------------------------------------------------------
# Extract lanes
# ---------------------------------------------------------------------------

def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _split_paragraphs(text: str) -> list[str]:
    parts = re.split(r"\n\s*\n", text.strip())
    return [p.strip() for p in parts if p.strip()]


def extract_fast(text: str) -> list[dict[str, Any]]:
    snippet = text[:MAX_FAST_CHARS]
    sentences = _split_sentences(snippet)
    return [{"segment": s, "type": "sentence"} for s in sentences]


def extract_slow(text: str) -> list[dict[str, Any]]:
    truncated = text[:MAX_SLOW_BYTES]
    paragraphs = _split_paragraphs(truncated)
    claims: list[dict[str, Any]] = []
    for para in paragraphs:
        claims.append({"segment": para, "type": "paragraph"})
    for match in _EMAIL_RE.finditer(truncated):
        claims.append({"segment": match.group(), "type": "email", "entity_type": "EMAIL"})
    for match in _DATE_RE.finditer(truncated):
        claims.append({"segment": match.group(), "type": "date", "entity_type": "DATE"})
    for match in _URL_RE.finditer(truncated):
        claims.append({"segment": match.group(), "type": "url", "entity_type": "URL"})
    for match in _NAME_RE.finditer(truncated):
        claims.append({"segment": match.group(), "type": "name", "entity_type": "PERSON"})
    return claims


def extract_audit(text: str) -> list[dict[str, Any]]:
    """Same as slow but every claim is flagged for human review."""
    claims = extract_slow(text)
    for c in claims:
        c["audit_flag"] = True
    return claims


def extract_external_groq(
    text: str, is_image: bool, file_path: str | None
) -> tuple[list[dict[str, Any]], bool]:
    """Call Groq API for extraction. Returns (claims, groq_called)."""
    key = read_groq_key()
    if not key:
        # Fall back to slow lane — no credentials available
        return extract_slow(text), False

    if is_image and file_path:
        # Try base64 image for vision model
        try:
            img_bytes = Path(file_path).read_bytes()
            b64 = base64.b64encode(img_bytes).decode("ascii")
            ext = Path(file_path).suffix.lower().lstrip(".")
            mime = f"image/{ext if ext in ('png', 'jpeg', 'jpg', 'gif', 'webp') else 'jpeg'}"
            payload = {
                "model": GROQ_MODEL_IMAGE,
                "max_tokens": GROQ_MAX_TOKENS,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "Extract key claims, entities, and facts from this image. "
                                    'Return a JSON array of {"claim": "...", "entity_type": "...", "confidence": 0.0}.'
                                ),
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:{mime};base64,{b64}"},
                            },
                        ],
                    }
                ],
            }
        except Exception:
            # Can't read image bytes cleanly — fall back to text path
            is_image = False

    if not is_image:
        snippet = text[:MAX_GROQ_CHARS]
        payload = {
            "model": GROQ_MODEL_TEXT,
            "max_tokens": GROQ_MAX_TOKENS,
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Extract key claims, entities, and facts from this document. "
                        'Return a JSON array of {"claim": "...", "entity_type": "...", "confidence": 0.0}. '
                        f"Document: {snippet}"
                    ),
                }
            ],
        }

    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        GROQ_URL,
        data=body,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        content = data["choices"][0]["message"]["content"].strip()
        # Strip markdown fences if present
        if content.startswith("```"):
            content = re.sub(r"^```[a-z]*\n?", "", content)
            content = re.sub(r"```$", "", content).strip()
        parsed = json.loads(content)
        if not isinstance(parsed, list):
            parsed = [parsed]
        claims = [
            {
                "segment": str(item.get("claim", "")),
                "entity_type": str(item.get("entity_type", "CLAIM")),
                "confidence": float(item.get("confidence", 0.5)),
                "type": "groq_claim",
            }
            for item in parsed
            if isinstance(item, dict) and item.get("claim")
        ]
        return claims, True
    except Exception:
        return extract_slow(text), False


# ---------------------------------------------------------------------------
# Core handler
# ---------------------------------------------------------------------------

def handle_route_extract(
    payload: dict[str, Any],
    job_uuid: str,
    execute: bool,
) -> tuple[bool, dict[str, Any]]:
    file_path: str | None = payload.get("file_path")
    sha256: str | None = payload.get("sha256")
    job_id: str = payload.get("job_id") or job_uuid
    is_image: bool = bool(payload.get("is_image", False))
    is_text: bool = bool(payload.get("is_text", True))

    # Read source text
    text = ""
    read_error: str | None = None
    if file_path and Path(file_path).exists():
        try:
            raw_bytes = Path(file_path).read_bytes()
            if sha256 and sha256_bytes(raw_bytes) != sha256:
                return False, {"error": "sha256_mismatch", "file_path": file_path}
            text = raw_bytes.decode("utf-8", errors="replace")
        except Exception as exc:
            read_error = f"file_read_error:{type(exc).__name__}:{exc}"
    elif payload.get("text"):
        text = str(payload["text"])
    else:
        read_error = "no_text_source"

    if read_error:
        return False, {"error": read_error, "job_id": job_id}

    # Token count normalisation (rough: chars / 4 / 1000, capped at 1.0)
    token_count_norm = min(1.0, len(text) / 4.0 / 1000.0)

    t0 = time.monotonic()
    lane, score = treelite_gate(token_count_norm, is_image, is_text)

    groq_called = False
    if lane == "fast":
        claims = extract_fast(text)
        method = "sentence_split"
    elif lane == "slow":
        claims = extract_slow(text)
        method = "paragraph_ner"
    elif lane == "external":
        claims, groq_called = extract_external_groq(text, is_image, file_path)
        method = "groq_api" if groq_called else "paragraph_ner_fallback"
    else:
        # audit or any unexpected value
        claims = extract_audit(text)
        method = "paragraph_ner_audit"

    extract_ms = int((time.monotonic() - t0) * 1000)

    receipt_data: dict[str, Any] = {
        "job_id": job_id,
        "job_uuid": job_uuid,
        "lane": lane,
        "score": score,
        "method": method,
        "groq_called": groq_called,
        "segments_count": len(claims),
        "extract_ms": extract_ms,
        "claims": claims,
        "file_path": file_path,
        "sha256": sha256,
        "is_image": is_image,
        "is_text": is_text,
        "ok": True,
        "status": "succeeded",
    }

    if execute:
        write_extract_receipt(job_id, receipt_data)

    return True, receipt_data


# ---------------------------------------------------------------------------
# Enqueue next stage job
# ---------------------------------------------------------------------------

def enqueue_stage(
    cur: Any,
    *,
    job_id: str,
    claims: list[dict[str, Any]],
    sha256: str | None,
    file_path: str | None,
    origin_job_uuid: str,
) -> str:
    stage_payload: dict[str, Any] = {
        "job_id": job_id,
        "claims": claims,
        "sha256": sha256,
        "file_path": file_path,
        "origin_job_uuid": origin_job_uuid,
    }
    idem = sha256_obj({"queue": QUEUE_NAME, "job_kind": STAGE_JOB_KIND, "job_id": job_id, "origin": origin_job_uuid})
    cur.execute(
        """
        INSERT INTO lucidota_control.absurd_queue_job
          (queue_name, workflow_name, job_kind, idempotency_key, payload, priority, max_attempts, detail)
        VALUES (%s,%s,%s,%s,%s::jsonb,100,3,%s::jsonb)
        ON CONFLICT (queue_name, idempotency_key) DO UPDATE SET updated_at=now()
        RETURNING job_uuid, (xmax=0) AS inserted_new
        """,
        (
            QUEUE_NAME,
            WORKFLOW_NAME,
            STAGE_JOB_KIND,
            idem,
            json.dumps(stage_payload),
            json.dumps({"source": "krampus_route_extract_worker", "origin_job_uuid": origin_job_uuid}),
        ),
    )
    row = cur.fetchone()
    if isinstance(row, dict):
        return str(row["job_uuid"])
    return str(row[0])


# ---------------------------------------------------------------------------
# worker_once
# ---------------------------------------------------------------------------

def worker_once(args: argparse.Namespace, execute: bool) -> tuple[dict[str, Any], list[str]]:
    url = db_url(args)
    blockers: list[str] = []
    worker_id = getattr(args, "worker_id", None) or f"krampus_extract:{socket.gethostname()}:{os.getpid()}"
    result: dict[str, Any] = {
        "state_database_url": redact(url),
        "queue": QUEUE_NAME,
        "job_kind": JOB_KIND,
        "worker_key": WORKER_KEY,
        "worker_id": worker_id,
        "execute_performed": False,
        "job_processed": False,
        "canonical_graph_writes_performed": False,
    }

    with psycopg.connect(url, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            lock_clause = "FOR UPDATE SKIP LOCKED" if execute else ""
            cur.execute(
                f"""
                SELECT job_uuid::text, workflow_name, job_kind, idempotency_key, status::text, payload,
                       attempt_count, max_attempts
                FROM lucidota_control.absurd_queue_job
                WHERE queue_name=%s AND status='queued' AND run_after <= now()
                      AND job_kind=%s
                ORDER BY priority ASC, created_at ASC
                {lock_clause}
                LIMIT 1
                """,  # noqa: S608
                (QUEUE_NAME, JOB_KIND),
            )
            fetched = cur.fetchone()

            if not execute:
                result["would_process"] = dict(fetched) if fetched else None
                return result, blockers

            if not fetched:
                result["no_job_available"] = True
                return result, blockers

            row = dict(fetched)
            job_uuid: str = row["job_uuid"]
            payload: dict[str, Any] = row["payload"] or {}
            attempt_count: int = int(row.get("attempt_count", 0))
            max_attempts: int = int(row.get("max_attempts", 3))

            # Check canonical graph counts before
            graph_before = count_tables(conn, CANONICAL_GRAPH_TABLES)

            cur.execute(
                """
                UPDATE lucidota_control.absurd_queue_job
                SET status='running', leased_by=%s, lease_expires_at=now()+interval '5 minutes',
                    locked_by=%s, locked_at=now(), attempt_count=attempt_count+1, updated_at=now()
                WHERE job_uuid=%s::uuid
                """,
                (worker_id, worker_id, job_uuid),
            )
            cur.execute(
                """
                INSERT INTO lucidota_control.absurd_queue_event
                  (job_uuid, queue_name, event_kind, event_source, detail)
                VALUES (%s::uuid,%s,'started','krampus_route_extract_worker',%s::jsonb)
                """,
                (job_uuid, QUEUE_NAME, json.dumps({"worker_id": worker_id})),
            )
        conn.commit()

    # Run handler outside the lock window (no graph mutation possible)
    ok, handler_result = handle_route_extract(payload, job_uuid, execute)

    stage_job_uuid: str | None = None
    with psycopg.connect(url) as conn:
        with conn.cursor() as cur:
            graph_after = count_tables(conn, CANONICAL_GRAPH_TABLES)
            graph_writes = graph_before != graph_after

            if ok and execute:
                # Enqueue downstream stage job
                try:
                    stage_job_uuid = enqueue_stage(
                        cur,
                        job_id=str(payload.get("job_id") or job_uuid),
                        claims=handler_result.get("claims", []),
                        sha256=payload.get("sha256"),
                        file_path=payload.get("file_path"),
                        origin_job_uuid=job_uuid,
                    )
                    cur.execute(
                        """
                        INSERT INTO lucidota_control.absurd_queue_event
                          (job_uuid, queue_name, event_kind, event_source, detail)
                        VALUES (%s::uuid,%s,'enqueued','krampus_route_extract_worker',%s::jsonb)
                        """,
                        (
                            stage_job_uuid,
                            QUEUE_NAME,
                            json.dumps({"stage": "route_extract_to_stage", "origin_job_uuid": job_uuid}),
                        ),
                    )
                except Exception as exc:
                    blockers.append(f"stage_enqueue_failed:{type(exc).__name__}:{exc}")
                    ok = False

            status = "succeeded" if ok else (
                "dead_lettered" if attempt_count + 1 >= max_attempts else "failed"
            )
            error_kind = "" if ok else "route_extract_failed"
            error_message = "" if ok else json.dumps(handler_result, default=str)[:2000]

            cur.execute(
                """
                UPDATE lucidota_control.absurd_queue_job
                SET status=%s, result=%s::jsonb,
                    completed_at=CASE WHEN %s='succeeded' THEN now() ELSE completed_at END,
                    updated_at=now(), last_error=%s, error_kind=%s, error_message=%s
                WHERE job_uuid=%s::uuid
                """,
                (
                    status,
                    json.dumps(handler_result, default=str),
                    status,
                    error_message,
                    error_kind,
                    error_message,
                    job_uuid,
                ),
            )
            cur.execute(
                """
                INSERT INTO lucidota_control.absurd_queue_event
                  (job_uuid, queue_name, event_kind, event_source, detail)
                VALUES (%s::uuid,%s,%s,'krampus_route_extract_worker',%s::jsonb)
                """,
                (
                    job_uuid,
                    QUEUE_NAME,
                    status,
                    json.dumps({
                        "lane": handler_result.get("lane"),
                        "method": handler_result.get("method"),
                        "groq_called": handler_result.get("groq_called"),
                        "segments_count": handler_result.get("segments_count"),
                        "stage_job_uuid": stage_job_uuid,
                        "blockers": blockers,
                    }),
                ),
            )
            cur.execute(
                """
                INSERT INTO lucidota_control.workflow_event
                  (workflow_id, run_id, phase, status, source, detail)
                VALUES (%s,%s,'krampus_route_extract',%s,'krampus_route_extract_worker',%s::jsonb)
                RETURNING event_id::text
                """,
                (
                    WORKFLOW_NAME,
                    job_uuid,
                    status,
                    json.dumps({"job_uuid": job_uuid, "stage_job_uuid": stage_job_uuid, "result": handler_result}, default=str),
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
                        error_kind or "route_extract_failed",
                        error_message,
                        sha256_obj(payload),
                        json.dumps(handler_result, default=str),
                        job_uuid,
                    ),
                )

            if graph_writes:
                blockers.append("canonical_graph_counts_changed")
                conn.rollback()
            else:
                conn.commit()

    result.update({
        "execute_performed": True,
        "job_processed": True,
        "job_uuid": job_uuid,
        "workflow_event_id": event_id,
        "status": status,
        "stage_job_uuid": stage_job_uuid,
        "lane": handler_result.get("lane"),
        "score": handler_result.get("score"),
        "method": handler_result.get("method"),
        "groq_called": handler_result.get("groq_called"),
        "segments_count": handler_result.get("segments_count"),
        "extract_ms": handler_result.get("extract_ms"),
        "canonical_graph_writes_performed": graph_writes,
        "canonical_graph_counts_before": graph_before,
        "canonical_graph_counts_after": graph_after,
    })
    blockers.extend([b for b in (["route_extract_failed"] if not ok and "route_extract_failed" not in blockers else []) if b not in blockers])
    return result, blockers


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="ABSURD worker: krampus_ingest / route_extract")
    ap.add_argument("--state-database-url", default=os.environ.get("ABSURD_SYSTEM_DATABASE_URL", "postgresql:///lucidota_state"))
    ap.add_argument("--worker-id")

    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Inspect queue without processing (default)")
    mode.add_argument("--execute", action="store_true", help="Claim and process one job")
    mode.add_argument("--once", action="store_true", help="Alias for --execute")
    mode.add_argument("--loop", action="store_true", help="Process jobs in a loop until queue is empty")

    ap.add_argument("--json", action="store_true", help="Print report as JSON to stdout")
    args = ap.parse_args()

    execute = bool(args.execute or args.once)
    do_loop = bool(args.loop)
    if do_loop:
        execute = True

    iterations = 0
    last_result: dict[str, Any] = {}
    last_blockers: list[str] = []

    while True:
        try:
            action_result, blockers = worker_once(args, execute)
        except Exception as exc:
            action_result = {}
            blockers = [f"exception:{type(exc).__name__}:{exc}"]

        last_result = action_result
        last_blockers = blockers
        iterations += 1

        report: dict[str, Any] = {
            "schema": "lucidota.krampus_route_extract.report.v1",
            "generated_at": now_iso(),
            "queue": QUEUE_NAME,
            "job_kind": JOB_KIND,
            "worker_key": WORKER_KEY,
            "mode": "execute" if execute else "dry_run",
            "iteration": iterations,
            "action_result": action_result,
            "blockers": blockers,
            "execute_performed": bool(action_result.get("execute_performed")),
            "canonical_graph_writes_performed": bool(action_result.get("canonical_graph_writes_performed")),
        }

        write_report("worker_once", report)

        if args.json:
            print(json.dumps(report, indent=2, sort_keys=False, default=str))

        if action_result.get("no_job_available") or not execute:
            break
        if not do_loop:
            break
        if blockers and "no_job_available" not in str(blockers):
            # Stop looping on hard errors
            break

    print(f"STATUS={'succeeded' if not last_blockers else 'failed'}")
    if last_result.get("job_uuid"):
        print(f"JOB_UUID={last_result['job_uuid']}")
    return 0 if not last_blockers else 1


if __name__ == "__main__":
    raise SystemExit(main())
