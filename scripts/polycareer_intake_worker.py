#!/usr/bin/env python3
"""ABSURD polycareer intake worker.

queue_name : indy_polycareer
job_kind   : intake_custody
worker_key : polycareer_intake_v1

Dequeues one ``intake_custody`` job at a time, validates the payload, hashes
the artifact, writes an intake receipt to
``05_OUTPUTS/indy_polycareer/<job_id>/intake_receipt.json``, and enqueues the
follow-on ``extract_claims`` job on the same queue.

Safety laws (never violated):
- Never writes to canonical graph tables (lucidota_go.graph_item / graph_edge /
  graph_journal).
- All mutations are queue-spine state, workflow_event, and receipt files only.
- Groq is never called; this stage is deterministic.
- Dead-letter on any unhandled exception, respecting max_attempts.

CLI:
    python3 scripts/polycareer_intake_worker.py --once            # process one job
    python3 scripts/polycareer_intake_worker.py --loop            # keep running
    python3 scripts/polycareer_intake_worker.py --loop --dry-run  # inspect without writes
"""
from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import os
import shutil
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

from absurd_worker_contracts import (
    gate_worker_payload_hygiene,
    record_worker_contract_rejection,
    validate_worker_contract,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

QUEUE_NAME = "indy_polycareer"
JOB_KIND = "intake_custody"
NEXT_JOB_KIND = "extract_claims"
WORKER_KEY = "polycareer_intake_v1"
WORKFLOW_ID = "indy-polycareer-glow-watch"
RECEIPT_SCHEMA = "lucidota.indy_polycareer.receipt.v1"

OUT_DIR = ROOT / "05_OUTPUTS" / "indy_polycareer"
CAS_ROOT = ROOT / "03_VAULT" / "cas"

CANONICAL_GRAPH_TABLES = [
    "lucidota_go.graph_item",
    "lucidota_go.graph_edge",
    "lucidota_go.graph_journal",
]

# Treelite feature indices (mirrors topology spec §4)
# feature[0] = token_count_norm
# feature[1] = mutation_requested
# feature[2] = needs_cloud_model
# feature[3] = needs_graph_write
# feature[4] = risk_of_slop
RISK_TERMS = {"delete", "secret", "token", "canonical", "materialize", "legal", "external"}

LOOP_SLEEP_SECONDS = 2.0
LOOP_IDLE_SLEEP_SECONDS = 5.0

# ---------------------------------------------------------------------------
# Tiny utilities
# ---------------------------------------------------------------------------


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def sha256_obj(obj: Any) -> str:
    return hashlib.sha256(dumps(obj).encode()).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def rel(path: Path | str) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def db_url(args: argparse.Namespace) -> str:
    return (
        args.database_url
        or os.environ.get("ABSURD_SYSTEM_DATABASE_URL")
        or os.environ.get("DATABASE_URL")
        or "postgresql:///lucidota_state"
    )


def worker_id(args: argparse.Namespace) -> str:
    wid = getattr(args, "worker_id", None) or ""
    return wid or f"{WORKER_KEY}@{socket.gethostname()}:{os.getpid()}"


# ---------------------------------------------------------------------------
# File classification helpers
# ---------------------------------------------------------------------------

_TEXT_SUFFIXES = {
    ".txt", ".md", ".csv", ".json", ".yaml", ".yml", ".log",
    ".rst", ".toml", ".ini", ".cfg", ".py", ".rs", ".js", ".ts",
}
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tif", ".tiff", ".webp"}
_AUDIO_SUFFIXES = {".mp3", ".wav", ".flac", ".m4a", ".ogg"}
_VIDEO_SUFFIXES = {".mp4", ".mkv", ".mov", ".avi", ".webm"}
_ARCHIVE_SUFFIXES = {".zip", ".tar", ".gz", ".bz2", ".xz", ".7z", ".rar"}


def detect_mime(path: Path) -> str:
    mime, _ = mimetypes.guess_type(str(path))
    return mime or "application/octet-stream"


def detect_file_kind(path: Path, mime: str) -> str:
    suffix = path.suffix.lower()
    m = (mime or "").lower()
    if m.startswith("image/") or suffix in _IMAGE_SUFFIXES:
        return "image"
    if m == "application/pdf" or suffix == ".pdf":
        return "document"
    if m.startswith("video/") or suffix in _VIDEO_SUFFIXES:
        return "video"
    if m.startswith("audio/") or suffix in _AUDIO_SUFFIXES:
        return "audio"
    if suffix in _ARCHIVE_SUFFIXES:
        return "archive"
    if m.startswith("text/") or suffix in _TEXT_SUFFIXES:
        return "text"
    return "binary"


# ---------------------------------------------------------------------------
# Treelite gate (deterministic inline tree, mirrors spec §4)
# ---------------------------------------------------------------------------

def _compute_features(
    *,
    artifact_path: str,
    operator_note: str,
    source_type: str,
    token_count: int = 0,
) -> dict[str, Any]:
    """Compute the 5-vector features for the inline treelite gate."""
    combined = f"{artifact_path} {operator_note} {source_type}".lower()
    risk_hit_count = sum(1 for term in RISK_TERMS if term in combined)
    tc_norm = min(float(token_count) / 1000.0, 1.0)
    mutation_words = {"stage", "write", "ingest", "mutate", "promote", "materialize"}
    mutation_requested = 1.0 if any(w in combined for w in mutation_words) else 0.0
    needs_cloud_model = 0.0  # intake never uses cloud model
    needs_graph_write = 0.0  # intake never writes canonical graph
    risk_of_slop = min(
        1.0,
        0.05 + risk_hit_count * 0.12 + (0.18 if token_count > 500 else 0.0),
    )
    return {
        "token_count_norm": round(tc_norm, 4),
        "mutation_requested": mutation_requested,
        "needs_cloud_model": needs_cloud_model,
        "needs_graph_write": needs_graph_write,
        "risk_of_slop": round(risk_of_slop, 4),
        "can_fast_lane": tc_norm <= 0.08 and mutation_requested == 0.0 and needs_graph_write == 0.0,
    }


def treelite_gate(features: dict[str, Any]) -> dict[str, Any]:
    """
    Inline single-tree gate (spec §4 tree structure):
      node 0: feature[3] (needs_graph_write) < 0.5
        left  → node 1: feature[1] (mutation_requested) < 0.5
          left  → leaf 0.12  (fast_bias)
          right → leaf 0.66  (slow_bias)
        right → leaf 0.95   (audit_bias)
    """
    vector = [
        float(features.get("token_count_norm") or 0.0),
        float(features.get("mutation_requested") or 0.0),
        float(features.get("needs_cloud_model") or 0.0),
        float(features.get("needs_graph_write") or 0.0),
        float(features.get("risk_of_slop") or 0.0),
    ]
    try:
        import numpy as np
        from treelite import gtil, model_builder as mb

        meta = mb.Metadata(
            num_feature=5, task_type="kRegressor", average_tree_output=False,
            num_target=1, num_class=[1], leaf_vector_shape=(1, 1),
        )
        ann = mb.TreeAnnotation(num_tree=1, target_id=[0], class_id=[0])
        builder = mb.ModelBuilder(
            threshold_type="float32",
            leaf_output_type="float32",
            metadata=meta,
            tree_annotation=ann,
            postprocessor=mb.PostProcessorFunc(name="identity"),
            base_scores=[0.0],
        )
        builder.start_tree()
        builder.start_node(0)
        builder.numerical_test(3, 0.5, default_left=True, opname="<", left_child_key=1, right_child_key=2)
        builder.end_node()
        builder.start_node(1)
        builder.numerical_test(1, 0.5, default_left=True, opname="<", left_child_key=3, right_child_key=4)
        builder.end_node()
        builder.start_node(2); builder.leaf(0.95); builder.end_node()
        builder.start_node(3); builder.leaf(0.12); builder.end_node()
        builder.start_node(4); builder.leaf(0.66); builder.end_node()
        builder.end_tree()
        model = builder.commit()
        score = float(gtil.predict(model, np.array([vector], dtype=np.float32)).reshape(-1)[0])
        verdict = "audit_bias" if score >= 0.9 else ("slow_bias" if score >= 0.55 else "fast_bias")
        return {"available": True, "score": round(score, 4), "features": vector, "verdict": verdict}
    except Exception as exc:
        # Deterministic fallback: score the tree manually
        ngw = vector[3]
        mut = vector[1]
        if ngw >= 0.5:
            score = 0.95
        elif mut >= 0.5:
            score = 0.66
        else:
            score = 0.12
        verdict = "audit_bias" if score >= 0.9 else ("slow_bias" if score >= 0.55 else "fast_bias")
        return {
            "available": False,
            "fallback": "deterministic_inline",
            "error": f"{type(exc).__name__}:{exc}",
            "score": round(score, 4),
            "features": vector,
            "verdict": verdict,
        }


def assign_lane(features: dict[str, Any], treelite: dict[str, Any]) -> str:
    """Assign the execution lane from deterministic rules first, treelite second."""
    if features.get("needs_graph_write"):
        return "audit"
    if features.get("mutation_requested"):
        return "slow"
    if features.get("can_fast_lane"):
        return "fast"
    if treelite.get("verdict") == "audit_bias":
        return "audit"
    return "slow"


# ---------------------------------------------------------------------------
# Role routing (ported from legacy lucidota_indy_polycareer)
# ---------------------------------------------------------------------------

ROLE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "INTAKE_CLERK": ("drop", "incoming", "intake", "classify", "route", "manifest", "received", "file", "folder", "archive"),
    "EVIDENCE_VAULT": ("evidence", "hash", "sha", "cas", "custody", "source", "preserve", "archive", "receipt", "chain"),
    "OSINT_ANALYST": ("osint", "verify", "geolocate", "source", "pivot", "open source", "archive", "claim", "confidence"),
    "FRAUD_EXAMINER": ("fraud", "transaction", "incident", "finding", "interview", "loss", "scheme", "misrepresentation"),
    "LEGAL_CLERK": ("legal", "case", "court", "filing", "discovery", "exhibit", "authority", "deadline", "paralegal"),
    "NEWS_EDITOR": ("story", "publish", "editor", "headline", "source", "fact check", "right of reply", "correction"),
    "RESEARCH_LIBRARIAN": ("research", "literature", "screen", "include", "exclude", "bibliography", "systematic", "study"),
    "EXEC_ASSISTANT": ("calendar", "meeting", "follow up", "task", "brief", "schedule", "inbox", "reminder"),
    "MAILROOM_TECH": ("scan", "ocr", "index", "mail", "routing slip", "document", "metadata", "queue"),
    "RISK_ANALYST": ("risk", "threat", "vulnerability", "mitigation", "exposure", "trigger", "safety", "security"),
    "GLOW_HUNTER": ("glow", "anomaly", "weird", "unique", "invention", "better", "method", "pattern"),
}

GLOW_POSITIVE: dict[str, int] = {
    "anomaly": 10, "anomalous": 10, "weird": 6, "unique": 7, "novel": 7, "unusual": 7,
    "invention": 10, "inventions": 10, "hypersystemic": 12,
    "better": 8, "faster": 5, "elegant": 6, "clever": 5, "brilliant": 8, "dope": 7,
    "method": 6, "workflow": 6, "protocol": 5, "pattern": 5, "playbook": 5, "system": 4,
    "receipt": 7, "receipts": 7, "evidence": 5, "reproducible": 8, "repeatable": 8,
    "operator": 4, "rust": 4, "ternary": 5,
}
GLOW_PHRASES: dict[str, int] = {
    "operator control": 10, "rust core": 10, "python wet clay": 10,
    "db is the os": 14, "db as os": 14, "go-25": 8,
    "palantir on a shitty laptop": 14, "low resources": 8,
    "krampus express": 12, "diogenes kernel": 12,
    "indy_reads": 10, "indy reads": 10,
}
GLOW_NEGATIVE: dict[str, int] = {
    "credential": -10, "missing": -6, "failed": -8, "error": -5,
    "traceback": -8, "exception": -6,
}
METHOD_CUES = (
    "because", "by ", "using", "instead", "unlike", "method",
    "workflow", "pattern", "protocol", "trick", "move",
)


def _approx_words(text: str) -> list[str]:
    import re
    return re.findall(r"[A-Za-z0-9_'-]+", text.lower())


def route_text(text: str, top_n: int = 5) -> list[dict[str, Any]]:
    low = text.lower()
    scores: list[dict[str, Any]] = []
    for mode, kws in ROLE_KEYWORDS.items():
        score = 0
        hits: list[str] = []
        for kw in kws:
            if kw in low:
                score += 3 + min(4, len(kw.split()))
                hits.append(kw)
        if mode in {"INTAKE_CLERK", "EVIDENCE_VAULT"} and any(x in low for x in ("file", "source", "evidence", "drop")):
            score += 2
        if score:
            scores.append({"mode": mode, "score": score, "hits": hits[:8]})
    if not scores:
        scores.append({"mode": "INTAKE_CLERK", "score": 1, "hits": ["default"]})
    scores.sort(key=lambda r: (r["score"], r["mode"]), reverse=True)
    return scores[:top_n]


def compute_glow(text: str) -> dict[str, Any]:
    words = _approx_words(text)
    counts: dict[str, int] = {}
    score = 0
    for w in words:
        if w in GLOW_POSITIVE:
            counts[w] = counts.get(w, 0) + 1
            score += GLOW_POSITIVE[w]
        if w in GLOW_NEGATIVE:
            counts[w] = counts.get(w, 0) + 1
            score += GLOW_NEGATIVE[w]
    low = text.lower()
    for phrase, weight in GLOW_PHRASES.items():
        if phrase in low:
            counts[phrase] = counts.get(phrase, 0) + 1
            score += weight
    has_method = any(cue in low for cue in METHOD_CUES)
    has_receipt = any(x in low for x in ("evidence", "receipt", "source", "hash", "archive", "proof"))
    if has_method:
        score += 10
    if has_receipt:
        score += 10
    score = max(0, min(100, score))
    return {
        "glow_score": score,
        "signals": sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:12],
        "has_method": has_method,
        "has_receipt": has_receipt,
    }


# ---------------------------------------------------------------------------
# CAS helper
# ---------------------------------------------------------------------------

def cas_store(source: Path, sha256: str, *, dry_run: bool = False) -> Path:
    """Copy artifact into CAS layout. Returns relative CAS path."""
    dest = CAS_ROOT / sha256[:2] / sha256
    if not dry_run and not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(source), str(dest))
    return dest


# ---------------------------------------------------------------------------
# Receipt writer
# ---------------------------------------------------------------------------

def write_intake_receipt(receipt: dict[str, Any]) -> Path:
    job_id = receipt.get("job_uuid", stamp())
    out_dir = OUT_DIR / str(job_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "intake_receipt.json"
    receipt["report_path"] = rel(out_path)
    out_path.write_text(json.dumps(receipt, indent=2, sort_keys=False, default=str), encoding="utf-8")
    print(f"RECEIPT_PATH={rel(out_path)}")
    return out_path


# ---------------------------------------------------------------------------
# Payload validation
# ---------------------------------------------------------------------------

REQUIRED_PAYLOAD_KEYS = ("artifact_path", "source_type")


def validate_intake_payload(payload: dict[str, Any]) -> tuple[bool, str]:
    """Check that the intake payload has minimum required fields."""
    for key in REQUIRED_PAYLOAD_KEYS:
        val = payload.get(key, "")
        if not val or not str(val).strip():
            return False, f"missing_required_field:{key}"
    artifact_path = str(payload["artifact_path"])
    if not artifact_path.startswith("/"):
        return False, "artifact_path_must_be_absolute"
    return True, ""


# ---------------------------------------------------------------------------
# Dead-letter helper
# ---------------------------------------------------------------------------

def _dead_letter(
    cur: Any,
    *,
    job_uuid: str,
    queue_name: str,
    workflow_name: str,
    job_kind: str,
    idempotency_key: str,
    error_kind: str,
    error_message: str,
    attempt_count: int,
    payload: dict[str, Any],
    context: dict[str, Any],
) -> None:
    cur.execute(
        """
        INSERT INTO lucidota_control.absurd_queue_dead_letter
          (job_uuid, queue_name, workflow_name, job_kind, idempotency_key,
           error_kind, error_message, attempt_count, payload_sha256, context)
        VALUES (%s::uuid, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
        ON CONFLICT (job_uuid) WHERE resolved=false DO UPDATE SET
          error_kind    = EXCLUDED.error_kind,
          error_message = EXCLUDED.error_message,
          attempt_count = EXCLUDED.attempt_count,
          last_seen_at  = now(),
          context       = EXCLUDED.context
        """,
        (
            job_uuid, queue_name, workflow_name, job_kind, idempotency_key,
            error_kind, error_message, attempt_count,
            sha256_obj(payload), dumps(context),
        ),
    )


# ---------------------------------------------------------------------------
# Core handler
# ---------------------------------------------------------------------------

def _handle_intake_custody(
    *,
    job_uuid: str,
    payload: dict[str, Any],
    attempt_count: int,
    dry_run: bool,
    wid: str,
) -> tuple[bool, dict[str, Any], dict[str, Any] | None]:
    """
    Process one ``intake_custody`` job.

    Returns: (ok, result_dict, receipt_dict_or_None)
    receipt_dict is written to disk only when ok=True and not dry_run.
    """
    started_at = now()
    artifact_path = str(payload.get("artifact_path", ""))
    source_type = str(payload.get("source_type", "file"))
    operator_note = str(payload.get("operator_note", ""))

    # --- payload validation ---
    ok_p, err_p = validate_intake_payload(payload)
    if not ok_p:
        return False, {
            "error": "intake_payload_invalid",
            "error_kind": "intake_payload_invalid",
            "error_message": err_p,
            "status": "failed",
        }, None

    path = Path(artifact_path)
    if not path.exists():
        return False, {
            "error": "artifact_not_found",
            "error_kind": "artifact_not_found",
            "error_message": f"path does not exist: {artifact_path}",
            "status": "failed",
        }, None
    if not path.is_file():
        return False, {
            "error": "artifact_not_a_file",
            "error_kind": "artifact_not_a_file",
            "error_message": f"path is not a regular file: {artifact_path}",
            "status": "failed",
        }, None

    # --- file metadata ---
    size_bytes = path.stat().st_size
    mime = detect_mime(path)
    fkind = detect_file_kind(path, mime)

    # --- sha256 ---
    digest = sha256_file(path)

    # --- token count estimate (text files only; others get 0) ---
    token_count_est = 0
    if fkind in {"text", "document"}:
        token_count_est = max(1, size_bytes // 4)

    # --- treelite features + gate ---
    features = _compute_features(
        artifact_path=artifact_path,
        operator_note=operator_note,
        source_type=source_type,
        token_count=token_count_est,
    )
    treelite = treelite_gate(features)
    lane = assign_lane(features, treelite)

    # --- role routing + glow (from filename + operator_note) ---
    probe_text = f"{path.name} {operator_note}"
    routes = route_text(probe_text, top_n=5)
    glow_result = compute_glow(probe_text)
    primary_mode = routes[0]["mode"] if routes else "INTAKE_CLERK"

    # --- CAS copy ---
    cas_path_obj = cas_store(path, digest, dry_run=dry_run)
    cas_path_rel = rel(cas_path_obj) if not dry_run else f"03_VAULT/cas/{digest[:2]}/{digest}"

    # --- idempotency key for next job ---
    case_key = str(payload.get("case_key", ""))
    next_idem = sha256_text(f"indy_extract:{digest}:{case_key}")
    next_job_uuid = str(uuid.uuid4())

    receipt: dict[str, Any] = {
        "schema": RECEIPT_SCHEMA,
        "receipt_mode": "ABSURD_POSTGRES_RUNTIME",
        "job_uuid": job_uuid,
        "queue_name": QUEUE_NAME,
        "job_kind": JOB_KIND,
        "worker_key": WORKER_KEY,
        "case_key": case_key,
        "idempotency_key": str(payload.get("idempotency_key", "")),
        "status": "succeeded",
        "lane": lane,
        "treelite_score": treelite.get("score"),
        "treelite_verdict": treelite.get("verdict"),
        "groq_used": False,
        "groq_model": None,
        "groq_tokens_used": 0,
        "source_path": artifact_path,
        "source_sha256": digest,
        "cas_path": cas_path_rel,
        "mime": mime,
        "file_kind": fkind,
        "size_bytes": size_bytes,
        "token_count_estimate": token_count_est,
        "primary_mode": primary_mode,
        "routes": routes,
        "glow_score": glow_result["glow_score"],
        "glow_signals": glow_result["signals"],
        "entities_count": 0,
        "claims_count": 0,
        "staging_packet_uuids": [],
        "graph_promotion_packet_uuids": [],
        "next_job_uuid": next_job_uuid,
        "next_queue": QUEUE_NAME,
        "next_job_kind": NEXT_JOB_KIND,
        "next_idempotency_key": next_idem,
        "error_kind": "",
        "error_message": "",
        "attempt_count": attempt_count,
        "started_at": started_at,
        "finished_at": now(),
        "worker_id": wid,
        "dry_run": dry_run,
        "canonical_graph_writes_performed": False,
        "report_path": "",  # filled by write_intake_receipt
    }

    result: dict[str, Any] = {
        "status": "succeeded",
        "sha256": digest,
        "file_kind": fkind,
        "mime": mime,
        "size_bytes": size_bytes,
        "cas_path": cas_path_rel,
        "primary_mode": primary_mode,
        "glow_score": glow_result["glow_score"],
        "lane": lane,
        "treelite_score": treelite.get("score"),
        "next_job_uuid": next_job_uuid,
        "next_job_kind": NEXT_JOB_KIND,
        "receipt_path": "",  # filled after write
        "worker_key": WORKER_KEY,
        "error": "none",
        "error_kind": "",
        "error_message": "",
    }
    return True, result, receipt


# ---------------------------------------------------------------------------
# Dequeue + execute
# ---------------------------------------------------------------------------

def consume_one(args: argparse.Namespace) -> int:
    """
    Select one queued job FOR UPDATE SKIP LOCKED, run the handler,
    write the receipt, enqueue the follow-on job, and commit.

    Returns 0 on success, 1 if no job found, 2 on failure/dead-letter.
    """
    url = db_url(args)
    wid = worker_id(args)
    dry_run: bool = bool(args.dry_run)

    with psycopg.connect(url, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            # --- dry-run inspection ---
            if dry_run:
                cur.execute(
                    """
                    SELECT job_uuid::text, idempotency_key, payload, job_kind, status::text,
                           attempt_count, max_attempts, workflow_name
                    FROM lucidota_control.absurd_queue_job
                    WHERE queue_name = %s
                      AND status = 'queued'
                      AND run_after <= now()
                    ORDER BY priority, created_at
                    LIMIT 1
                    """,
                    (QUEUE_NAME,),
                )
                row = cur.fetchone()
                if not row:
                    print("DRY_RUN: no queued job")
                    return 1
                payload = dict(row["payload"] or {})
                contract = validate_worker_contract(
                    cur,
                    queue_name=QUEUE_NAME,
                    job_kind=str(row.get("job_kind") or JOB_KIND),
                    worker_key=WORKER_KEY,
                )
                print(f"DRY_RUN: would process job_uuid={row['job_uuid']}")
                print(f"DRY_RUN: contract.ok={contract.ok} {contract.error_kind}")
                print(f"DRY_RUN: payload_keys={list(payload.keys())}")
                return 0

            # --- execute path ---
            cur.execute("SET LOCAL lucidota.actor_role='worker'")
            cur.execute(
                """
                SELECT job_uuid::text, idempotency_key, payload, job_kind,
                       attempt_count, max_attempts, workflow_name
                FROM lucidota_control.absurd_queue_job
                WHERE queue_name = %s
                  AND status = 'queued'
                  AND run_after <= now()
                ORDER BY priority, created_at
                FOR UPDATE SKIP LOCKED
                LIMIT 1
                """,
                (QUEUE_NAME,),
            )
            row = cur.fetchone()
            if not row:
                return 1  # nothing to do; caller decides whether to sleep/loop

            job_uuid: str = row["job_uuid"]
            payload: dict[str, Any] = dict(row["payload"] or {})
            attempt_count: int = int(row.get("attempt_count") or 0) + 1
            max_attempts: int = int(row.get("max_attempts") or 3)
            workflow_name: str = str(row.get("workflow_name") or WORKFLOW_ID)
            job_kind: str = str(row.get("job_kind") or JOB_KIND)

            # --- lock row as running ---
            cur.execute(
                """
                UPDATE lucidota_control.absurd_queue_job
                SET status           = 'running',
                    locked_by        = %s,
                    leased_by        = %s,
                    locked_at        = now(),
                    lease_expires_at = now() + interval '5 minutes',
                    last_heartbeat_at = now(),
                    attempt_count    = %s
                WHERE job_uuid = %s::uuid
                """,
                (wid, wid, attempt_count, job_uuid),
            )
            cur.execute(
                """
                INSERT INTO lucidota_control.absurd_queue_event
                  (job_uuid, queue_name, event_kind, event_source, detail)
                VALUES (%s::uuid, %s, 'started', %s, %s::jsonb)
                """,
                (job_uuid, QUEUE_NAME, WORKER_KEY, dumps({"worker_id": wid, "attempt": attempt_count})),
            )
            conn.commit()  # release lock update so other workers can proceed on different jobs

            # --- worker contract validation ---
            contract = validate_worker_contract(
                cur,
                queue_name=QUEUE_NAME,
                job_kind=job_kind,
                worker_key=WORKER_KEY,
            )
            if not contract.ok:
                result, error_kind = record_worker_contract_rejection(
                    cur,
                    job_uuid=job_uuid,
                    queue_name=QUEUE_NAME,
                    payload=payload,
                    contract=contract,
                    event_source=WORKER_KEY,
                )
                conn.commit()
                print(f"CONTRACT_REJECTED job_uuid={job_uuid} error={error_kind}", file=sys.stderr)
                return 2

            # --- payload hygiene gate ---
            hyg_ok, hyg_result = gate_worker_payload_hygiene(
                payload,
                queue_name=QUEUE_NAME,
                worker_key=WORKER_KEY,
                job_kind=job_kind,
                required_keys=("artifact_path", "source_type"),
                min_score=0,  # intake payloads are short; hygiene enforced by validate_intake_payload
            )
            # hygiene gate logs only — we enforce structural requirements in the handler
            # (gate_worker_payload_hygiene min_score=0 means no score block, just structural check)

            # --- run handler ---
            ok = False
            result: dict[str, Any] = {}
            receipt: dict[str, Any] | None = None
            error_kind = ""
            error_message = ""

            try:
                ok, result, receipt = _handle_intake_custody(
                    job_uuid=job_uuid,
                    payload=payload,
                    attempt_count=attempt_count,
                    dry_run=False,
                    wid=wid,
                )
                if not ok:
                    error_kind = result.get("error_kind") or "handler_error"
                    error_message = result.get("error_message") or dumps(result)
            except Exception as exc:
                ok = False
                error_kind = "handler_exception"
                error_message = f"{type(exc).__name__}: {exc}"
                result = {
                    "error": "handler_exception",
                    "error_kind": error_kind,
                    "error_message": error_message,
                    "status": "failed",
                }

            # --- write receipt to disk (before DB commit so path lands in result) ---
            if ok and receipt is not None:
                try:
                    receipt_path = write_intake_receipt({**receipt, "job_uuid": job_uuid})
                    result["receipt_path"] = rel(receipt_path)
                    receipt["report_path"] = rel(receipt_path)
                except Exception as exc:
                    ok = False
                    error_kind = "receipt_write_failed"
                    error_message = f"{type(exc).__name__}: {exc}"
                    result = {
                        "error": "receipt_write_failed",
                        "error_kind": error_kind,
                        "error_message": error_message,
                        "status": "failed",
                    }

            # --- determine final job status ---
            if ok:
                final_status = "succeeded"
            elif attempt_count >= max_attempts:
                final_status = "dead_lettered"
            else:
                final_status = "failed"

            # --- update job row ---
            cur.execute(
                """
                UPDATE lucidota_control.absurd_queue_job
                SET status        = %s,
                    result        = %s::jsonb,
                    last_heartbeat_at = now(),
                    error_kind    = %s,
                    error_message = %s,
                    last_error    = %s,
                    completed_at  = CASE WHEN %s = 'succeeded' THEN now() ELSE completed_at END
                WHERE job_uuid = %s::uuid
                """,
                (
                    final_status,
                    dumps(result),
                    error_kind,
                    error_message,
                    error_message,
                    final_status,
                    job_uuid,
                ),
            )

            # --- event ---
            event_kind = final_status if final_status in {"succeeded", "failed", "dead_lettered"} else "failed"
            cur.execute(
                """
                INSERT INTO lucidota_control.absurd_queue_event
                  (job_uuid, queue_name, event_kind, event_source, detail)
                VALUES (%s::uuid, %s, %s, %s, %s::jsonb)
                """,
                (job_uuid, QUEUE_NAME, event_kind, WORKER_KEY, dumps(result)),
            )

            # --- workflow_event ---
            cur.execute(
                """
                INSERT INTO lucidota_control.workflow_event
                  (workflow_id, run_id, phase, status, source, detail)
                VALUES (%s, %s, 'intake_custody', %s, %s, %s::jsonb)
                RETURNING event_id::text
                """,
                (
                    WORKFLOW_ID,
                    job_uuid,
                    "succeeded" if ok else "failed",
                    WORKER_KEY,
                    dumps({
                        "worker_id": wid,
                        "sha256": result.get("sha256", ""),
                        "file_kind": result.get("file_kind", ""),
                        "lane": result.get("lane", ""),
                        "glow_score": result.get("glow_score", 0),
                    }),
                ),
            )
            workflow_event_row = cur.fetchone()
            workflow_event_id = (workflow_event_row or {}).get("event_id", "")

            # --- dead letter ---
            if final_status == "dead_lettered":
                _dead_letter(
                    cur,
                    job_uuid=job_uuid,
                    queue_name=QUEUE_NAME,
                    workflow_name=workflow_name,
                    job_kind=job_kind,
                    idempotency_key=str(row.get("idempotency_key") or ""),
                    error_kind=error_kind or "handler_error",
                    error_message=error_message,
                    attempt_count=attempt_count,
                    payload=payload,
                    context=result,
                )

            # --- enqueue follow-on extract_claims job ---
            next_job_uuid_inserted: str | None = None
            if ok and receipt is not None:
                next_idem = receipt.get("next_idempotency_key") or sha256_text(
                    f"indy_extract:{receipt.get('source_sha256', '')}:{receipt.get('case_key', '')}"
                )
                next_uuid = receipt.get("next_job_uuid") or str(uuid.uuid4())
                next_payload: dict[str, Any] = {
                    "job_kind": NEXT_JOB_KIND,
                    "intake_receipt_uuid": job_uuid,
                    "cas_path": receipt.get("cas_path", ""),
                    "source_path": receipt.get("source_path", ""),
                    "sha256": receipt.get("source_sha256", ""),
                    "file_kind": receipt.get("file_kind", ""),
                    "mime": receipt.get("mime", ""),
                    "case_key": receipt.get("case_key", ""),
                    "role_hint": receipt.get("primary_mode", "INTAKE_CLERK"),
                    "ocr_required": receipt.get("file_kind") in {"image", "document"},
                    "token_count_estimate": receipt.get("token_count_estimate", 0),
                    "idempotency_key": next_idem,
                    "intake_receipt_path": receipt.get("report_path", ""),
                }
                cur.execute(
                    """
                    INSERT INTO lucidota_control.absurd_queue_job
                      (job_uuid, queue_name, workflow_name, job_kind, idempotency_key, payload)
                    VALUES (%s::uuid, %s, %s, %s, %s, %s::jsonb)
                    ON CONFLICT (queue_name, idempotency_key) DO UPDATE
                      SET updated_at = lucidota_control.absurd_queue_job.updated_at
                    RETURNING job_uuid::text, (xmax = 0) AS inserted_new
                    """,
                    (
                        next_uuid,
                        QUEUE_NAME,
                        WORKFLOW_ID,
                        NEXT_JOB_KIND,
                        next_idem,
                        dumps(next_payload),
                    ),
                )
                enq_row = cur.fetchone()
                if enq_row:
                    next_job_uuid_inserted = enq_row.get("job_uuid") or enq_row[0]
                    print(f"NEXT_JOB_UUID={next_job_uuid_inserted}")

            conn.commit()

        # end cursor

    # end connection

    print(f"JOB_UUID={job_uuid}")
    print(f"STATUS={final_status}")
    if workflow_event_id:
        print(f"WORKFLOW_EVENT_ID={workflow_event_id}")

    return 0 if final_status == "succeeded" else 2


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="ABSURD polycareer intake worker (queue=indy_polycareer, kind=intake_custody).",
    )
    p.add_argument("--database-url", default="", help="Postgres DSN (default: env or postgresql:///lucidota_state)")
    p.add_argument("--worker-id", default="", help="Unique worker identity string")
    p.add_argument("--dry-run", action="store_true", help="Inspect next job without executing or writing anything")

    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--once", action="store_true", help="Process exactly one job and exit")
    mode.add_argument("--loop", action="store_true", help="Keep running until interrupted")

    return p


def main() -> int:
    args = build_parser().parse_args()

    if args.once:
        rc = consume_one(args)
        if rc == 1:
            print("NO_JOB: nothing queued")
        return rc

    # --loop
    consecutive_empty = 0
    while True:
        try:
            rc = consume_one(args)
            if rc == 1:
                # no job available
                consecutive_empty += 1
                sleep_sec = min(LOOP_IDLE_SLEEP_SECONDS * consecutive_empty, 60.0)
                time.sleep(sleep_sec)
            else:
                consecutive_empty = 0
                if rc != 0:
                    # failed/dead-lettered but not fatal — keep running
                    time.sleep(LOOP_SLEEP_SECONDS)
        except KeyboardInterrupt:
            print("\nINTERRUPTED")
            return 0
        except Exception as exc:
            print(f"LOOP_ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
            time.sleep(LOOP_SLEEP_SECONDS)

    return 0  # unreachable


if __name__ == "__main__":
    raise SystemExit(main())
