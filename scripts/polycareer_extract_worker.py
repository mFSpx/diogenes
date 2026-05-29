#!/usr/bin/env python3
"""Polycareer extract worker — ABSURD queue consumer.

queue_name : indy_polycareer
job_kind   : extract_claims
worker_key : polycareer_extract_v1

Dequeues with SELECT FOR UPDATE SKIP LOCKED.
Runs treelite gate → routes to regex / GLiNER / Groq extraction.
Writes receipt to 05_OUTPUTS/indy_polycareer/<job_uuid>/extract_receipt.json.
Enqueues downstream job_kind=glow_watch_stage with claims list.
Dead-letters on 3rd failed attempt.

mutation_class : receipt_only   (never writes canonical graph tables)
canonical_graph_write_allowed : false
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import socket
import sys
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psycopg

# ---------------------------------------------------------------------------
# Root + sys.path
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from absurd_worker_contracts import (
    WorkerContractDecision,
    gate_worker_payload_hygiene,
    record_worker_contract_rejection,
    validate_worker_contract,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
QUEUE_NAME  = "indy_polycareer"
JOB_KIND    = "extract_claims"
WORKER_KEY  = "polycareer_extract_v1"
MAX_ATTEMPTS = 3

GROQ_BASE_URL   = "https://api.groq.com/openai/v1"
GROQ_MODEL      = "llama-3.3-70b-versatile"
GROQ_MAX_TOKENS = 500
GROQ_TIMEOUT_SEC = 45.0

TREELITE_MODEL_PATH = ROOT / "03_VAULT" / "router" / "treelite_router_v0.tl"
OUT_ROOT = ROOT / "05_OUTPUTS" / "indy_polycareer"

# Risk terms matching the topology design §4
RISK_TERMS = {"delete", "secret", "token", "canonical", "materialize", "legal", "external"}

# Regex patterns for fast-lane extraction
_CLAIM_SENTENCE_RE = re.compile(
    r"\b(is|are|was|were|must|should|needs?|has|have|can|cannot|will|would|asserts?|claims?|states?|shows?|proves?|indicates?)\b",
    re.I,
)
_ENTITY_RE = re.compile(
    r"\b([A-Z][a-z]{1,30}(?:\s+[A-Z][a-z]{1,30}){0,3})\b|"
    r"\b(\d{4}-\d{2}-\d{2})\b|"
    r"(https?://[^\s\"'<>]{4,120})"
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def sha256_obj(obj: Any) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def db_url() -> str:
    return (
        os.environ.get("LUCIDOTA_CONTROL_DATABASE_URL")
        or os.environ.get("ABSURD_SYSTEM_DATABASE_URL")
        or os.environ.get("DATABASE_URL")
        or "postgresql:///lucidota_state"
    )


def groq_api_key() -> str | None:
    # Prefer /tmp key file (set by ./claw launcher), then env, then secrets.env
    key_file = Path(os.environ.get("LUCIDOTA_GROQ_KEY_FILE", "/tmp/lucidota_groq_key"))
    if key_file.exists():
        val = key_file.read_text(encoding="utf-8").strip()
        if val:
            return val
    val = os.environ.get("GROQ_API_KEY", "").strip()
    if val:
        return val
    secret_env = Path(os.environ.get("LUCIDOTA_SECRET_ENV", Path.home() / ".config" / "lucidota" / "secrets.env"))
    if secret_env.exists():
        for line in secret_env.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("GROQ_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def write_receipt(job_uuid: str, data: dict[str, Any]) -> Path:
    out_dir = OUT_ROOT / job_uuid
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "extract_receipt.json"
    data.setdefault("generated_at", now_iso())
    data["report_path"] = str(path.relative_to(ROOT))
    path.write_text(json.dumps(data, indent=2, sort_keys=False, default=str), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Feature computation (5 features per topology §4)
# ---------------------------------------------------------------------------

def compute_features(text: str, *, ocr_required: bool = False, file_kind: str = "text") -> dict[str, float]:
    tc = max(1, math.ceil(len(text) / 4))
    token_count_norm = min(tc / 1000.0, 1.0)
    mutation_requested = 0.0   # extraction worker never mutates
    cloud_kinds = {"image", "audio", "video"}
    needs_cloud = (
        1.0 if file_kind in cloud_kinds
        else (1.0 if ocr_required else (1.0 if tc > 4000 else 0.0))
    )
    needs_graph_write = 0.0  # polycareer workers never write canonical graph
    low = text.lower()
    risk_count = sum(1 for t in RISK_TERMS if t in low)
    risk_of_slop = min(1.0, 0.05 + risk_count * 0.12 + (0.18 if tc > 500 else 0.0))
    return {
        "token_count_norm": token_count_norm,
        "mutation_requested": mutation_requested,
        "needs_cloud_model": needs_cloud,
        "needs_graph_write": needs_graph_write,
        "risk_of_slop": risk_of_slop,
        "token_count": tc,
    }


# ---------------------------------------------------------------------------
# Treelite gate
# ---------------------------------------------------------------------------

def run_treelite_gate(features: dict[str, float]) -> dict[str, Any]:
    """Load compiled model and score the 5-feature vector.

    Falls back to deterministic rules-only lane assignment if treelite /
    numpy are unavailable or the model file is missing.
    """
    vector = [
        features["token_count_norm"],
        features["mutation_requested"],
        features["needs_cloud_model"],
        features["needs_graph_write"],
        features["risk_of_slop"],
    ]
    try:
        import numpy as np
        import treelite.gtil as gtil
        import treelite

        if not TREELITE_MODEL_PATH.exists():
            raise FileNotFoundError(f"treelite model missing: {TREELITE_MODEL_PATH}")

        model = treelite.frontend.load_xgboost_model(str(TREELITE_MODEL_PATH))
        arr = np.array([vector], dtype=np.float32)
        score = float(gtil.predict(model, arr).reshape(-1)[0])
        verdict = (
            "audit_bias" if score >= 0.9
            else ("slow_bias" if score >= 0.55 else "fast_bias")
        )
        return {
            "available": True,
            "score": round(score, 4),
            "verdict": verdict,
            "features": vector,
            "model_path": str(TREELITE_MODEL_PATH.relative_to(ROOT)),
        }
    except Exception as exc:
        # Deterministic fallback: inline single-tree logic from topology §4
        # node 0: feature[3] (needs_graph_write) < 0.5
        #   left  → node 1: feature[1] (mutation_requested) < 0.5
        #     left  → 0.12 (fast_bias)
        #     right → 0.66 (slow_bias)
        #   right → 0.95 (audit_bias)
        f3, f1 = vector[3], vector[1]
        if f3 >= 0.5:
            score = 0.95
            verdict = "audit_bias"
        elif f1 >= 0.5:
            score = 0.66
            verdict = "slow_bias"
        else:
            score = 0.12
            verdict = "fast_bias"
        return {
            "available": False,
            "score": round(score, 4),
            "verdict": verdict,
            "features": vector,
            "fallback_error": f"{type(exc).__name__}:{exc}",
        }


def lane_from_gate(gate_result: dict[str, Any], features: dict[str, float]) -> str:
    """Apply deterministic overrides then let treelite verdict upgrade lane."""
    # Deterministic rules run first (topology §4)
    if features["needs_graph_write"] >= 0.5:
        return "audit"
    if features["needs_cloud_model"] >= 0.5:
        return "external"
    if features["mutation_requested"] >= 0.5:
        return "slow"

    # token_count_norm <= 0.08 means <= 80 tokens → fast lane eligible
    can_fast = (
        features["token_count_norm"] <= 0.08
        and features["mutation_requested"] < 0.5
        and features["needs_graph_write"] < 0.5
    )
    if can_fast:
        base_lane = "fast"
    else:
        base_lane = "slow"

    # treelite upgrade: slow → audit when audit_bias
    verdict = gate_result.get("verdict", "fast_bias")
    if base_lane == "slow" and verdict == "audit_bias":
        return "audit"
    return base_lane


# ---------------------------------------------------------------------------
# Extraction methods
# ---------------------------------------------------------------------------

def extract_regex(text: str) -> list[dict[str, Any]]:
    """Fast-lane: regex + keyword claim extraction."""
    claims: list[dict[str, Any]] = []
    sentences = re.split(r"(?<=[.!?])\s+|\n{2,}", text.strip())
    for sent in sentences[:40]:
        sent = sent.strip()
        if not sent:
            continue
        if _CLAIM_SENTENCE_RE.search(sent):
            claims.append({
                "text": sent[:400],
                "kind": "sentence_claim",
                "authority_class": "raw_evidence",
                "confidence_bps": 50,
                "method": "regex",
            })
    # Named-entity-like regex pass
    for m in _ENTITY_RE.finditer(text[:8000]):
        entity_text = (m.group(1) or m.group(2) or m.group(3) or "").strip()
        if entity_text and len(entity_text) > 3:
            claims.append({
                "text": entity_text,
                "kind": "entity_candidate",
                "authority_class": "raw_evidence",
                "confidence_bps": 40,
                "method": "regex_entity",
            })
    # Deduplicate by lowercased text
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for c in claims:
        key = c["text"].lower()[:200]
        if key not in seen:
            seen.add(key)
            deduped.append(c)
    return deduped[:30]


def extract_gliner(text: str) -> tuple[list[dict[str, Any]], str]:
    """Slow-lane: attempt GLiNER zero-shot extraction.

    Returns (claims, method_used).  Falls back to regex on import/runtime failure.
    """
    try:
        sys.path.insert(0, str(ROOT / "ALGOS"))
        from gliner_zero_shot_extractor import extract as gliner_extract, parse_labels  # type: ignore

        labels_path = ROOT / "05_OUTPUTS" / "contracts" / "operator_ontology_labels.json"
        if not labels_path.exists():
            raise FileNotFoundError(f"labels file missing: {labels_path}")

        labels = parse_labels(str(labels_path))
        result = gliner_extract(text[:6000], labels, no_fallback=False)
        spans = result.get("spans", [])
        if not spans:
            return extract_regex(text), "gliner_empty_fallback_regex"

        claims: list[dict[str, Any]] = []
        for span in spans[:30]:
            claims.append({
                "text": span.get("text", "")[:400],
                "kind": span.get("label", "entity"),
                "authority_class": "model_computed_finding",
                "confidence_bps": max(0, min(10000, int(round(span.get("score", 0.69) * 10000)))),
                "method": "gliner",
                "start": span.get("start"),
                "end": span.get("end"),
            })
        return claims, "gliner"
    except ImportError:
        return extract_regex(text), "gliner_unavailable_fallback_regex"
    except Exception as exc:
        return extract_regex(text), f"gliner_error_fallback_regex:{type(exc).__name__}"


def extract_groq(text: str, api_key: str) -> tuple[list[dict[str, Any]], int]:
    """External-lane: call Groq llama-3.3-70b-versatile for claim extraction.

    Returns (claims, tokens_used).
    Bounded to GROQ_MAX_TOKENS output tokens.
    """
    excerpt = text[:6000]
    system_msg = (
        "You are a precise claim and entity extractor. "
        "Extract factual claims and named entities from the text. "
        "Return a JSON array of objects with keys: text (string, <=300 chars), "
        "kind (one of: person, org, place, event, claim, date, url, other), "
        "confidence (float 0..1). "
        "Return ONLY the JSON array. No prose. No explanation. Max 20 items."
    )
    user_msg = f"Extract claims and entities from:\n\n{excerpt}"
    request_payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        "max_tokens": GROQ_MAX_TOKENS,
        "temperature": 0.0,
    }
    endpoint = GROQ_BASE_URL.rstrip("/") + "/chat/completions"
    req = urllib.request.Request(
        endpoint,
        data=json.dumps(request_payload).encode("utf-8"),
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "lucidota-polycareer-extract/1.0",
        },
    )
    with urllib.request.urlopen(req, timeout=GROQ_TIMEOUT_SEC) as resp:
        response = json.loads(resp.read().decode("utf-8"))

    tokens_used = response.get("usage", {}).get("completion_tokens", 0)
    raw_content = ""
    try:
        raw_content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError):
        return [], tokens_used

    # Parse the JSON array from model output
    raw_content = raw_content.strip()
    # Strip markdown code fence if present
    if raw_content.startswith("```"):
        raw_content = re.sub(r"^```[a-z]*\n?", "", raw_content)
        raw_content = re.sub(r"\n?```$", "", raw_content.strip())

    try:
        items = json.loads(raw_content)
        if not isinstance(items, list):
            return [], tokens_used
    except json.JSONDecodeError:
        # Last-chance: try to extract JSON array from within prose
        m = re.search(r"\[.*\]", raw_content, re.DOTALL)
        if m:
            try:
                items = json.loads(m.group(0))
            except Exception:
                return [], tokens_used
        else:
            return [], tokens_used

    claims: list[dict[str, Any]] = []
    for item in items[:20]:
        if not isinstance(item, dict):
            continue
        text_val = str(item.get("text", "")).strip()[:400]
        if not text_val:
            continue
        raw_conf = item.get("confidence", 0.69)
        try:
            conf_bps = max(0, min(10000, int(round(float(raw_conf) * 10000))))
        except (TypeError, ValueError):
            conf_bps = 690
        claims.append({
            "text": text_val,
            "kind": str(item.get("kind", "other")),
            "authority_class": "model_computed_finding",
            "confidence_bps": conf_bps,
            "method": "groq",
        })
    return claims, tokens_used


# ---------------------------------------------------------------------------
# Dead-letter helpers
# ---------------------------------------------------------------------------

def dead_letter(cur: Any, *, job_uuid: str, queue_name: str, workflow_name: str,
                job_kind: str, idempotency_key: str, attempt_count: int,
                payload: Any, error_kind: str, error_message: str) -> None:
    result_obj = {
        "ok": False,
        "error": error_kind,
        "error_message": error_message,
        "status": "dead_lettered",
    }
    cur.execute(
        """
        UPDATE lucidota_control.absurd_queue_job
        SET status='dead_lettered', result=%s::jsonb, updated_at=now(), last_error=%s
        WHERE job_uuid=%s::uuid
        """,
        (json.dumps(result_obj, default=str), error_kind, job_uuid),
    )
    cur.execute(
        """
        INSERT INTO lucidota_control.absurd_queue_event
          (job_uuid, queue_name, event_kind, event_source, detail)
        VALUES (%s::uuid, %s, 'dead_lettered', %s, %s::jsonb)
        """,
        (job_uuid, queue_name, WORKER_KEY,
         json.dumps({"error_kind": error_kind, "error_message": error_message}, default=str)),
    )
    cur.execute(
        """
        INSERT INTO lucidota_control.absurd_queue_dead_letter
          (job_uuid, queue_name, workflow_name, job_kind, idempotency_key,
           error_kind, error_message, attempt_count, payload_sha256, context)
        VALUES (%s::uuid, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
        ON CONFLICT (job_uuid) WHERE resolved=false
        DO UPDATE SET
          error_message = EXCLUDED.error_message,
          attempt_count = EXCLUDED.attempt_count,
          last_seen_at  = now(),
          context       = EXCLUDED.context
        """,
        (
            job_uuid, queue_name, workflow_name, job_kind, idempotency_key,
            error_kind, error_message[:2000], attempt_count,
            sha256_obj(payload or {}),
            json.dumps(result_obj, default=str),
        ),
    )


def fail_attempt(cur: Any, *, job_uuid: str, queue_name: str,
                 error_kind: str, error_message: str) -> None:
    result_obj = {
        "ok": False,
        "error": error_kind,
        "error_message": error_message,
        "status": "failed",
    }
    cur.execute(
        """
        UPDATE lucidota_control.absurd_queue_job
        SET status='failed', result=%s::jsonb, updated_at=now(), last_error=%s
        WHERE job_uuid=%s::uuid
        """,
        (json.dumps(result_obj, default=str), error_kind, job_uuid),
    )
    cur.execute(
        """
        INSERT INTO lucidota_control.absurd_queue_event
          (job_uuid, queue_name, event_kind, event_source, detail)
        VALUES (%s::uuid, %s, 'failed', %s, %s::jsonb)
        """,
        (job_uuid, queue_name, WORKER_KEY,
         json.dumps({"error_kind": error_kind, "error_message": error_message}, default=str)),
    )


# ---------------------------------------------------------------------------
# Downstream enqueue helper
# ---------------------------------------------------------------------------

def enqueue_glow_watch_stage(cur: Any, *, job_uuid: str, payload: dict[str, Any],
                              claims: list[dict[str, Any]],
                              source_sha256: str, case_key: str,
                              lane: str, treelite_score: float) -> str:
    next_uuid = str(uuid.uuid4())
    next_idem = "glow_watch_stage:" + sha256_text(
        json.dumps({"parent_uuid": job_uuid, "source_sha256": source_sha256,
                    "case_key": case_key}, sort_keys=True)
    )
    next_payload = {
        "job_kind": "glow_watch_stage",
        "parent_extract_uuid": job_uuid,
        "source_sha256": source_sha256,
        "case_key": case_key,
        "claims": claims,
        "claims_count": len(claims),
        "parent_lane": lane,
        "parent_treelite_score": treelite_score,
        "idempotency_key": next_idem,
        "enqueued_by": WORKER_KEY,
        "enqueued_at": now_iso(),
        "ok": True,
        "status": "queued",
        "result": "downstream_stage_job",
    }
    cur.execute(
        """
        INSERT INTO lucidota_control.absurd_queue_job
          (job_uuid, queue_name, workflow_name, job_kind, idempotency_key, payload, priority, max_attempts)
        VALUES (%s::uuid, %s, %s, %s, %s, %s::jsonb, %s, %s)
        ON CONFLICT (queue_name, idempotency_key) DO UPDATE SET updated_at = now()
        RETURNING job_uuid::text
        """,
        (
            next_uuid, QUEUE_NAME, "indy-polycareer-workflow-wizard",
            "glow_watch_stage", next_idem,
            json.dumps(next_payload, default=str),
            100, MAX_ATTEMPTS,
        ),
    )
    row = cur.fetchone()
    return str(row[0]) if row else next_uuid


# ---------------------------------------------------------------------------
# Core worker logic
# ---------------------------------------------------------------------------

def load_document_text(payload: dict[str, Any]) -> tuple[str, str]:
    """Return (text, source_description).  Tries CAS path then source_path then inline."""
    cas_path = payload.get("cas_path", "")
    if cas_path:
        p = Path(cas_path)
        if not p.is_absolute():
            p = ROOT / cas_path
        if p.exists():
            return p.read_text(encoding="utf-8", errors="replace"), f"cas:{cas_path}"

    source_path = payload.get("source_path", "")
    if source_path:
        p = Path(source_path)
        if p.exists():
            return p.read_text(encoding="utf-8", errors="replace"), f"source:{source_path}"

    inline_text = payload.get("text_excerpt") or payload.get("text") or payload.get("content") or ""
    if inline_text:
        return str(inline_text), "inline"

    return "", "none"


def run_worker_once() -> int:
    """Dequeue and process one job.  Returns exit code (0 = processed ok, 1 = error, 2 = no job)."""
    worker_id = f"{socket.gethostname()}:{os.getpid()}"
    database_url = db_url()

    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            # --- Dequeue with SKIP LOCKED ---
            cur.execute(
                """
                SELECT job_uuid::text, workflow_name, job_kind, idempotency_key,
                       payload, attempt_count, max_attempts
                FROM lucidota_control.absurd_queue_job
                WHERE queue_name = %s
                  AND job_kind   = %s
                  AND status     = 'queued'
                  AND run_after  <= now()
                ORDER BY priority ASC, created_at ASC
                FOR UPDATE SKIP LOCKED
                LIMIT 1
                """,
                (QUEUE_NAME, JOB_KIND),
            )
            row = cur.fetchone()
            if not row:
                print(f"[{WORKER_KEY}] no job available", flush=True)
                return 2

            (job_uuid, workflow_name, job_kind, idempotency_key,
             payload, attempt_count, max_attempts) = row
            attempt_count = int(attempt_count or 0)
            max_attempts_val = int(max_attempts or MAX_ATTEMPTS)

            # Mark running
            cur.execute(
                """
                UPDATE lucidota_control.absurd_queue_job
                SET status         = 'running',
                    leased_by      = %s,
                    lease_expires_at = now() + interval '10 minutes',
                    attempt_count  = attempt_count + 1,
                    updated_at     = now()
                WHERE job_uuid = %s::uuid
                """,
                (worker_id, job_uuid),
            )
            cur.execute(
                """
                INSERT INTO lucidota_control.absurd_queue_event
                  (job_uuid, queue_name, event_kind, event_source, detail)
                VALUES (%s::uuid, %s, 'started', %s, %s::jsonb)
                """,
                (job_uuid, QUEUE_NAME, WORKER_KEY,
                 json.dumps({"worker_id": worker_id, "attempt_count": attempt_count + 1})),
            )
            conn.commit()

        # ---------------------------------------------------------------
        # Process outside the first cursor block so we can handle errors
        # ---------------------------------------------------------------
        started_at = now_iso()
        t0 = time.monotonic()

        # Validate worker contract
        with conn.cursor() as cur:
            contract: WorkerContractDecision = validate_worker_contract(
                cur, queue_name=QUEUE_NAME, job_kind=job_kind, worker_key=WORKER_KEY
            )
        if not contract.ok:
            with conn.cursor() as cur:
                record_worker_contract_rejection(
                    cur,
                    job_uuid=job_uuid,
                    queue_name=QUEUE_NAME,
                    payload=payload,
                    contract=contract,
                    event_source=WORKER_KEY,
                )
            conn.commit()
            print(f"[{WORKER_KEY}] contract rejected: {contract.error_kind}", flush=True)
            return 1

        # Extract inputs from payload
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except Exception:
                payload = {}

        source_sha256 = payload.get("sha256") or payload.get("source_sha256") or ""
        case_key      = payload.get("case_key", "")
        file_kind     = payload.get("file_kind", "text")
        ocr_required  = bool(payload.get("ocr_required", False))

        # Load text
        text, text_source = load_document_text(payload)
        if not text:
            # Nothing to extract — dead-letter immediately
            with conn.cursor() as cur:
                dead_letter(
                    cur,
                    job_uuid=job_uuid,
                    queue_name=QUEUE_NAME,
                    workflow_name=workflow_name or "indy-polycareer-workflow-wizard",
                    job_kind=job_kind,
                    idempotency_key=idempotency_key,
                    attempt_count=attempt_count + 1,
                    payload=payload,
                    error_kind="no_text_content",
                    error_message="payload contained no loadable text (cas_path, source_path, text_excerpt all empty or missing)",
                )
            conn.commit()
            write_receipt(str(job_uuid), {
                "schema": "lucidota.polycareer_extract.receipt.v1",
                "receipt_mode": "ABSURD_POSTGRES_RUNTIME",
                "job_uuid": str(job_uuid),
                "queue_name": QUEUE_NAME,
                "job_kind": JOB_KIND,
                "worker_key": WORKER_KEY,
                "case_key": case_key,
                "idempotency_key": idempotency_key,
                "status": "dead_lettered",
                "error_kind": "no_text_content",
                "error_message": "no loadable text",
                "ok": False,
                "started_at": started_at,
                "finished_at": now_iso(),
            })
            return 1

        # Compute treelite features
        features = compute_features(text, ocr_required=ocr_required, file_kind=file_kind)
        gate_result = run_treelite_gate(features)
        lane = lane_from_gate(gate_result, features)
        treelite_score = gate_result["score"]
        treelite_verdict = gate_result["verdict"]

        print(
            f"[{WORKER_KEY}] job={job_uuid} lane={lane} score={treelite_score:.4f} "
            f"verdict={treelite_verdict} tokens={features['token_count']}",
            flush=True,
        )

        # --- Extraction path ---
        claims: list[dict[str, Any]] = []
        method_used = "none"
        groq_called = False
        groq_model_used: str | None = None
        groq_tokens_used = 0
        extraction_error: str = ""

        if lane == "fast":
            claims = extract_regex(text)
            method_used = "regex"

        elif lane == "slow":
            claims, method_used = extract_gliner(text)

        elif lane in {"external", "audit"}:
            # external or audit: Groq fallback
            api_key_val = groq_api_key()
            if not api_key_val:
                # No key — fall back to regex and note it
                claims = extract_regex(text)
                method_used = "regex_groq_key_missing"
                extraction_error = "groq_api_key_not_available"
            else:
                try:
                    claims, groq_tokens_used = extract_groq(text, api_key_val)
                    groq_called = True
                    groq_model_used = GROQ_MODEL
                    method_used = "groq"
                except urllib.error.HTTPError as exc:
                    extraction_error = f"groq_http_error:{exc.code}:{exc.reason}"
                    claims = extract_regex(text)
                    method_used = f"regex_groq_failed:{exc.code}"
                except Exception as exc:
                    extraction_error = f"groq_exception:{type(exc).__name__}:{exc}"
                    claims = extract_regex(text)
                    method_used = f"regex_groq_failed:{type(exc).__name__}"

        else:
            # dead_letter lane — shouldn't happen in normal operation
            claims = extract_regex(text)
            method_used = "regex_dead_letter_lane"

        # If GLiNER returned a "fallback" method but did produce claims, we may
        # still want Groq for the external lane upgrade — already handled above.

        finished_at = now_iso()
        latency_ms = round((time.monotonic() - t0) * 1000.0, 1)

        # --- Build receipt ---
        receipt_data: dict[str, Any] = {
            "schema": "lucidota.polycareer_extract.receipt.v1",
            "receipt_mode": "ABSURD_POSTGRES_RUNTIME",
            "job_uuid": str(job_uuid),
            "queue_name": QUEUE_NAME,
            "job_kind": JOB_KIND,
            "worker_key": WORKER_KEY,
            "case_key": case_key,
            "idempotency_key": idempotency_key,
            "status": "succeeded",
            "lane": lane,
            "treelite_score": treelite_score,
            "treelite_verdict": treelite_verdict,
            "treelite_available": gate_result.get("available", False),
            "groq_called": groq_called,
            "groq_model": groq_model_used,
            "groq_tokens_used": groq_tokens_used,
            "method_used": method_used,
            "extraction_error": extraction_error,
            "source_sha256": source_sha256,
            "text_source": text_source,
            "token_count": features["token_count"],
            "features": {k: v for k, v in features.items() if k != "token_count"},
            "claims_count": len(claims),
            "claims": claims,
            "ok": True,
            "error": "none",
            "error_kind": "",
            "error_message": extraction_error,
            "started_at": started_at,
            "finished_at": finished_at,
            "latency_ms": latency_ms,
            "worker_id": worker_id,
        }

        receipt_path = write_receipt(str(job_uuid), receipt_data)

        # --- Payload hygiene gate on result ---
        hygiene_ok, hygiene_detail = gate_worker_payload_hygiene(
            receipt_data,
            queue_name=QUEUE_NAME,
            worker_key=WORKER_KEY,
            job_kind=job_kind,
        )
        if not hygiene_ok:
            with conn.cursor() as cur:
                is_final = (attempt_count + 1) >= max_attempts_val
                if is_final:
                    dead_letter(
                        cur,
                        job_uuid=job_uuid,
                        queue_name=QUEUE_NAME,
                        workflow_name=workflow_name or "indy-polycareer-workflow-wizard",
                        job_kind=job_kind,
                        idempotency_key=idempotency_key,
                        attempt_count=attempt_count + 1,
                        payload=payload,
                        error_kind="hygiene_failed",
                        error_message=str(hygiene_detail.get("error_message", "")),
                    )
                else:
                    fail_attempt(
                        cur,
                        job_uuid=job_uuid,
                        queue_name=QUEUE_NAME,
                        error_kind="hygiene_failed",
                        error_message=str(hygiene_detail.get("error_message", "")),
                    )
            conn.commit()
            print(f"[{WORKER_KEY}] hygiene failed: {hygiene_detail.get('error_message')}", flush=True)
            return 1

        # --- Persist success + enqueue downstream ---
        with conn.cursor() as cur:
            # Mark succeeded
            cur.execute(
                """
                UPDATE lucidota_control.absurd_queue_job
                SET status       = 'succeeded',
                    result       = %s::jsonb,
                    completed_at = now(),
                    updated_at   = now(),
                    last_error   = ''
                WHERE job_uuid   = %s::uuid
                """,
                (json.dumps(receipt_data, default=str), job_uuid),
            )
            cur.execute(
                """
                INSERT INTO lucidota_control.absurd_queue_event
                  (job_uuid, queue_name, event_kind, event_source, detail)
                VALUES (%s::uuid, %s, 'succeeded', %s, %s::jsonb)
                """,
                (
                    job_uuid, QUEUE_NAME, WORKER_KEY,
                    json.dumps({
                        "lane": lane,
                        "method_used": method_used,
                        "claims_count": len(claims),
                        "groq_called": groq_called,
                        "latency_ms": latency_ms,
                        "receipt_path": str(receipt_path.relative_to(ROOT)),
                    }, default=str),
                ),
            )

            # workflow_event
            cur.execute(
                """
                INSERT INTO lucidota_control.workflow_event
                  (workflow_id, run_id, phase, status, source, detail)
                VALUES (%s, %s, 'extract_claims', 'succeeded', %s, %s::jsonb)
                RETURNING event_id::text
                """,
                (
                    "indy-polycareer-workflow-wizard",
                    str(job_uuid),
                    WORKER_KEY,
                    json.dumps({
                        "job_uuid": str(job_uuid),
                        "queue_name": QUEUE_NAME,
                        "lane": lane,
                        "method_used": method_used,
                        "claims_count": len(claims),
                        "groq_called": groq_called,
                        "groq_model": groq_model_used,
                        "groq_tokens_used": groq_tokens_used,
                        "receipt_path": str(receipt_path.relative_to(ROOT)),
                    }, default=str),
                ),
            )
            workflow_event_id = cur.fetchone()[0]

            # Groq fallback workflow_event (required by topology §8)
            if groq_called:
                cur.execute(
                    """
                    INSERT INTO lucidota_control.workflow_event
                      (workflow_id, run_id, phase, status, source, detail)
                    VALUES (%s, %s, 'groq_fallback', 'succeeded', 'groq_fallback', %s::jsonb)
                    """,
                    (
                        "indy-polycareer-workflow-wizard",
                        str(job_uuid),
                        json.dumps({
                            "model": groq_model_used,
                            "tokens_used": groq_tokens_used,
                            "job_uuid": str(job_uuid),
                            "queue_name": QUEUE_NAME,
                            "job_kind": JOB_KIND,
                            "authority_class": "model_computed_finding",
                        }, default=str),
                    ),
                )

            # Enqueue downstream glow_watch_stage if we extracted any claims
            next_job_uuid: str | None = None
            if claims:
                next_job_uuid = enqueue_glow_watch_stage(
                    cur,
                    job_uuid=str(job_uuid),
                    payload=payload,
                    claims=claims,
                    source_sha256=source_sha256,
                    case_key=case_key,
                    lane=lane,
                    treelite_score=treelite_score,
                )

        conn.commit()

        print(
            f"[{WORKER_KEY}] succeeded job={job_uuid} "
            f"claims={len(claims)} method={method_used} "
            f"next={next_job_uuid} workflow_event={workflow_event_id}",
            flush=True,
        )
        return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(
        description=f"ABSURD worker: {QUEUE_NAME}/{JOB_KIND} ({WORKER_KEY})"
    )
    ap.add_argument(
        "--once", action="store_true", default=True,
        help="Process one job then exit (default). Always true in current impl.",
    )
    ap.add_argument(
        "--loop", action="store_true", default=False,
        help="Loop continuously until no job is available.",
    )
    ap.add_argument(
        "--loop-sleep-sec", type=float, default=2.0,
        help="Sleep between loop iterations when no job was found.",
    )
    args = ap.parse_args()

    if args.loop:
        while True:
            rc = run_worker_once()
            if rc == 2:
                time.sleep(args.loop_sleep_sec)
            elif rc != 0:
                return rc
    else:
        return run_worker_once()


if __name__ == "__main__":
    raise SystemExit(main())
