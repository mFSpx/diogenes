#!/usr/bin/env python3
"""ABSURD queue worker: indy_polycareer / glow_watch_stage.

Queue:      indy_polycareer
Job kind:   glow_watch_stage
Worker key: polycareer_glow_v1
Mutation class: candidate_writer (staging_packet / graph_staging_candidates ONLY)

Safety laws:
- Never writes lucidota_go.graph_item, graph_edge, or graph_journal.
- Never writes lucidota_korpus.temporal_claim.
- Staging packet writes are candidates only (status='pending').
- All Groq/external model calls are forbidden in this worker.
- If lucidota_korpus.graph_staging_candidates is absent, output falls back to
  05_OUTPUTS/indy_polycareer/staging_candidates.jsonl with a blocked receipt flag.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import socket
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psycopg
from psycopg.rows import dict_row

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from absurd_worker_contracts import (
    gate_worker_payload_hygiene,
    record_worker_contract_rejection,
    validate_worker_contract,
)

# ── identity ─────────────────────────────────────────────────────────────────
QUEUE_NAME   = "indy_polycareer"
JOB_KIND     = "glow_watch_stage"
WORKER_KEY   = "polycareer_glow_v1"
WORKFLOW_ID  = "indy-polycareer-glow-watch"
SCHEMA_VER   = "lucidota.indy_polycareer.glow_watch_worker.v1"

# ── paths ─────────────────────────────────────────────────────────────────────
OUT_ROOT      = ROOT / "05_OUTPUTS" / "indy_polycareer"
FALLBACK_JSONL = OUT_ROOT / "staging_candidates.jsonl"
LATEST_MD     = OUT_ROOT / "glow_watch_latest.md"

# ── canonical graph tables we must never touch ────────────────────────────────
CANONICAL_GRAPH_TABLES = [
    "lucidota_go.graph_item",
    "lucidota_go.graph_edge",
    "lucidota_go.graph_journal",
]
TEMPORAL_TABLES = ["lucidota_korpus.temporal_claim"]

# ── GLOW detection patterns (deterministic regex / keyword) ───────────────────
# Each rule is (pattern_name, compiled_regex, ternary_hint)
# ternary_hint is +1 for glow, 0 for neutral, -1 for noise
# Patterns use leading \b for word-start anchoring.
# Stems (e.g. "contradic") intentionally lack a trailing \b so they match
# inflected forms like "contradicts", "contradicting", "contradiction".
_GLOW_RULES: list[tuple[str, re.Pattern[str], int]] = [
    (
        "ANOMALY",
        re.compile(
            r"\b(?:unexpect\w*|anomal\w*|contradic\w*|inconsist\w*|deviat\w*|"
            r"surpis\w*|sudden\w*|discontinu\w*|irregular\w*|outlier\w*|"
            r"aberrant\w*|reversal\w*|mismatch\w*|break\s+from|unlike(?:ly)?|"
            r"without\s+precedent|never\s+before|first\s+time|against\s+prior)",
            re.I,
        ),
        1,
    ),
    (
        "NOVEL_PATTERN",
        re.compile(
            r"\b(?:novel\w*|first\s+occurrence|new\s+method|unique\s+approach|"
            r"unprecedented\w*|original(?:ly)?|innovati\w*|never\s+seen|"
            r"distinct\s+from|introduces?\w*|invent\w*|pioneer\w*|breakthrough\w*)",
            re.I,
        ),
        1,
    ),
    (
        "OPERATOR_SIGNAL",
        re.compile(
            r"\b(?:operator\s+note|operator\s+signal|north(?:ern)?\.strike|"
            r"northern\s+says|instruction\s+from\s+operator|direct(?:ly)?\s+instruct\w*|"
            r"operator\s+feedback|as\s+directed|per\s+operator|explicit\s+order|"
            r"operator\s+confirm\w*|operator\s+approv\w*)",
            re.I,
        ),
        1,
    ),
    (
        "HIGH_CONFIDENCE",
        re.compile(
            r"\b(?:confidence[:\s]+(?:0\.\d{2,}|[89]\d|100)|"
            r"confidence_bps[:\s]+(?:8[5-9]\d{2}|9\d{3}|10000)|"
            r"high\s+confidence|strongly\s+supported|well[- ]document\w*|"
            r"receipt[- ]backed|verified\s+claim|confirmed\s+finding|proven\b)",
            re.I,
        ),
        1,
    ),
    (
        "CONTRADICTION",
        re.compile(
            r"\b(?:contradict\w*|conflict\w*|inconsistent\s+with|"
            r"at\s+odds\s+with|disput\w*|refut\w*|negat\w*|overturn\w*|"
            r"contrary\s+to|counter\w*\s+(?:to\s+)?(?:prior|existing|"
            r"previous|staged|established)|disagree\w*\s+with|challeng\w*\s+(?:the\s+)?"
            r"(?:claim|finding|assertion))",
            re.I,
        ),
        1,
    ),
    # noise patterns — vague / unverified language
    (
        "NOISE_VAGUE",
        re.compile(
            r"\b(?:(?:some|various|certain|many)\s+(?:things?|stuff|issues?|matters?)|"
            r"it\s+is\s+(?:unclear|unknown|uncertain)|(?:could|might|may)\s+be\b|"
            r"potentially\b|presumably\b|supposedly\b|allegedly\s+unclear|unclear\s+if\b|"
            r"no\s+evidence\b|unverified\b|unconfirmed\b|speculative\b|rumor\b|hearsay\b)",
            re.I,
        ),
        -1,
    ),
]


# ── helpers ───────────────────────────────────────────────────────────────────

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def sha256_obj(obj: Any) -> str:
    return hashlib.sha256(dumps(obj).encode()).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


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
        args.state_database_url
        or os.environ.get("ABSURD_SYSTEM_DATABASE_URL")
        or "postgresql:///lucidota_state"
    )


def storage_url(args: argparse.Namespace) -> str:
    return (
        args.storage_database_url
        or os.environ.get("LUCIDOTA_GO_STORAGE_DSN")
        or "postgresql:///lucidota_storage"
    )


def table_exists(cur: Any, qualified_name: str) -> bool:
    cur.execute("SELECT to_regclass(%s)", (qualified_name,))
    return first_value(cur.fetchone()) is not None


def count_table(conn: psycopg.Connection[Any], table: str) -> int | None:
    with conn.cursor() as cur:
        if not table_exists(cur, table):
            return None
        cur.execute(f"SELECT count(*) FROM {table}")  # noqa: S608 – table name is a constant
        return int(first_value(cur.fetchone()))


def count_tables_snapshot(conn: psycopg.Connection[Any], tables: list[str]) -> dict[str, int | None]:
    return {t: count_table(conn, t) for t in tables}


def rel(path: Path | str) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


# ── GLOW DETECTION ────────────────────────────────────────────────────────────

def detect_glow_patterns(text: str) -> list[dict[str, Any]]:
    """Return list of fired pattern dicts for a claim text."""
    fired: list[dict[str, Any]] = []
    for name, pattern, hint in _GLOW_RULES:
        hits = pattern.findall(text)
        if hits:
            fired.append({"pattern": name, "hits": hits[:8], "ternary_hint": hint})
    return fired


def ternary_score(patterns_fired: list[dict[str, Any]]) -> int:
    """Collapse pattern hits to +1 / 0 / -1.

    +1 (glow/interesting): any positive-hint pattern fired AND no noise domination.
    -1 (noise): only noise patterns fired OR noise count >= 2*positive count.
     0 (neutral): everything else.
    """
    positives = sum(1 for p in patterns_fired if p["ternary_hint"] == 1)
    negatives = sum(1 for p in patterns_fired if p["ternary_hint"] == -1)
    if positives == 0 and negatives == 0:
        return 0
    if positives == 0 and negatives > 0:
        return -1
    if negatives >= 2 * positives:
        return -1
    return 1


def score_claim(claim_text: str, confidence_bps: int) -> dict[str, Any]:
    """Full glow scoring for a single claim string."""
    patterns = detect_glow_patterns(claim_text)
    base_score = ternary_score(patterns)
    # High-confidence extract score (>= 8500 bps = 0.85) promotes neutral to +1
    if base_score == 0 and confidence_bps >= 8500:
        patterns.append({"pattern": "HIGH_CONFIDENCE_BPS_BOOST", "hits": [f"bps={confidence_bps}"], "ternary_hint": 1})
        base_score = 1
    return {
        "ternary_score": base_score,
        "patterns_fired": patterns,
        "confidence_bps": confidence_bps,
        "claim_sha256": sha256_text(claim_text),
    }


# ── RECEIPT + OUTPUT WRITERS ─────────────────────────────────────────────────

def write_receipt(job_uuid: str, payload: dict[str, Any]) -> Path:
    out_dir = OUT_ROOT / job_uuid
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "glow_receipt.json"
    payload.setdefault("generated_at", now_iso())
    payload["report_path"] = rel(path)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False, ensure_ascii=False), encoding="utf-8")
    return path


def append_glow_latest_md(job_uuid: str, case_key: str, scored_claims: list[dict[str, Any]]) -> None:
    LATEST_MD.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = [
        f"\n## Glow Watch Run — {now_iso()}",
        f"- job_uuid: `{job_uuid}`",
        f"- case_key: `{case_key}`",
        f"- claims scored: {len(scored_claims)}",
        "",
    ]
    glow_claims = [c for c in scored_claims if c["glow"]["ternary_score"] == 1]
    noise_claims = [c for c in scored_claims if c["glow"]["ternary_score"] == -1]
    if glow_claims:
        lines.append(f"### Glow (+1) — {len(glow_claims)} claim(s)")
        for c in glow_claims[:10]:
            patterns = ", ".join(p["pattern"] for p in c["glow"]["patterns_fired"])
            lines.append(f"- `{c['claim_text'][:120]}` → {patterns}")
        lines.append("")
    if noise_claims:
        lines.append(f"### Noise (-1) — {len(noise_claims)} claim(s)")
        for c in noise_claims[:5]:
            lines.append(f"- `{c['claim_text'][:80]}`")
        lines.append("")
    with LATEST_MD.open("a", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def write_fallback_jsonl(candidates: list[dict[str, Any]]) -> None:
    """Fallback path when graph_staging_candidates table does not exist."""
    FALLBACK_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with FALLBACK_JSONL.open("a", encoding="utf-8") as fh:
        for c in candidates:
            fh.write(json.dumps(c, default=str) + "\n")


# ── STAGING WRITE (storage DB) ────────────────────────────────────────────────

def write_staging_candidates(
    storage_conn: psycopg.Connection[Any],
    job_uuid: str,
    source_sha256: str,
    case_key: str,
    primary_mode: str,
    evidence_ref: str,
    scored_claims: list[dict[str, Any]],
) -> tuple[list[str], bool, str]:
    """
    Attempt INSERT into lucidota_korpus.graph_staging_candidates.
    Falls back to JSONL if the table is absent.

    Returns (candidate_uuids, used_db, blocked_reason).
    """
    candidate_uuids: list[str] = []
    staged_at = now_iso()

    with storage_conn.cursor() as cur:
        has_table = table_exists(cur, "lucidota_korpus.graph_staging_candidates")

    if not has_table:
        candidates = [
            {
                "candidate_uuid": str(uuid.uuid4()),
                "source_job_id": job_uuid,
                "claim_text": sc["claim_text"],
                "ternary_score": sc["glow"]["ternary_score"],
                "evidence_ref": evidence_ref,
                "staged_at": staged_at,
                "case_key": case_key,
                "primary_mode": primary_mode,
                "source_sha256": source_sha256,
                "patterns_fired": sc["glow"]["patterns_fired"],
                "confidence_bps": sc["glow"]["confidence_bps"],
            }
            for sc in scored_claims
        ]
        write_fallback_jsonl(candidates)
        candidate_uuids = [c["candidate_uuid"] for c in candidates]
        return candidate_uuids, False, "graph_staging_candidates_table_absent"

    with storage_conn.cursor() as cur:
        for sc in scored_claims:
            cid = str(uuid.uuid4())
            cur.execute(
                """
                INSERT INTO lucidota_korpus.graph_staging_candidates
                  (candidate_uuid, source_job_id, claim_text, ternary_score,
                   evidence_ref, staged_at)
                VALUES (%s::uuid, %s, %s, %s, %s, %s::timestamptz)
                ON CONFLICT DO NOTHING
                """,
                (
                    cid,
                    job_uuid,
                    sc["claim_text"],
                    sc["glow"]["ternary_score"],
                    evidence_ref,
                    staged_at,
                ),
            )
            candidate_uuids.append(cid)
    storage_conn.commit()
    return candidate_uuids, True, ""


# ── STAGING PACKET WRITE ──────────────────────────────────────────────────────

def write_staging_packets(
    storage_conn: psycopg.Connection[Any],
    job_uuid: str,
    source_sha256: str,
    case_key: str,
    primary_mode: str,
    scored_claims: list[dict[str, Any]],
) -> list[str]:
    """
    Write glow-positive claims to lucidota_go.staging_packet.
    Only writes where ternary_score == +1.
    Never promotes to canonical graph (status always 'pending').
    """
    packet_uuids: list[str] = []
    glow_claims = [sc for sc in scored_claims if sc["glow"]["ternary_score"] == 1]
    if not glow_claims:
        return packet_uuids
    with storage_conn.cursor() as cur:
        has_table = table_exists(cur, "lucidota_go.staging_packet")
    if not has_table:
        return packet_uuids
    with storage_conn.cursor() as cur:
        for sc in glow_claims:
            patterns = [p["pattern"] for p in sc["glow"]["patterns_fired"]]
            proposed_item = {
                "term": "CANDIDATE",
                "label": sc["claim_text"][:200],
                "source_sha256": source_sha256,
                "case_key": case_key,
                "confidence_bps": sc["glow"]["confidence_bps"],
                "indy_routes": [primary_mode],
                "glow_patterns": patterns,
            }
            cur.execute(
                """
                INSERT INTO lucidota_go.staging_packet
                  (source_id, parser_name, raw_anchor, claim,
                   proposed_item, proposed_edges, status, confidence_bps)
                VALUES (%s, %s, %s, %s, %s::jsonb, %s::jsonb, %s, %s)
                RETURNING packet_uuid::text
                """,
                (
                    source_sha256,
                    "polycareer_glow_watch_worker",
                    sc["claim_text"][:500],
                    f"glow_candidate:{sc['claim_text'][:200]}",
                    json.dumps(proposed_item),
                    "[]",
                    "pending",
                    sc["glow"]["confidence_bps"],
                ),
            )
            row = cur.fetchone()
            if row:
                packet_uuids.append(first_value(row))
    storage_conn.commit()
    return packet_uuids


# ── WORKER CORE ───────────────────────────────────────────────────────────────

def process_job(
    payload: dict[str, Any],
    job_uuid: str,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[str]]:
    """
    Core glow-watch logic. Scores each claim in the payload claims list,
    writes receipts and staging candidates, returns (result_dict, blockers).
    """
    blockers: list[str] = []

    claims_raw: list[Any] = payload.get("claims", [])
    source_sha256: str = payload.get("source_sha256", "")
    case_key: str = payload.get("case_key", "")
    primary_mode: str = payload.get("primary_mode", "GLOW_HUNTER")
    evidence_refs: list[str] = payload.get("evidence_refs", [])

    # gate: evidence_refs must be non-empty for staging
    if not evidence_refs:
        blockers.append("evidence_refs_empty")

    # normalize claims: each entry may be str or {"text":..., "confidence_bps":...}
    normalized_claims: list[dict[str, Any]] = []
    for raw in claims_raw:
        if isinstance(raw, str):
            normalized_claims.append({"text": raw, "confidence_bps": 50})
        elif isinstance(raw, dict):
            normalized_claims.append({
                "text": str(raw.get("text", raw.get("claim_text", ""))),
                "confidence_bps": int(raw.get("confidence_bps", raw.get("confidence", 50))),
            })

    if not normalized_claims:
        blockers.append("claims_list_empty")

    if blockers:
        return {
            "ok": False,
            "error": "input_validation_failed",
            "blockers": blockers,
            "claims_count": 0,
            "scored_claims": [],
        }, blockers

    # score all claims
    scored_claims: list[dict[str, Any]] = []
    for nc in normalized_claims:
        glow = score_claim(nc["text"], nc["confidence_bps"])
        scored_claims.append({"claim_text": nc["text"], "glow": glow})

    # write human-readable append
    append_glow_latest_md(job_uuid, case_key, scored_claims)

    evidence_ref = evidence_refs[0] if evidence_refs else f"05_OUTPUTS/indy_polycareer/{job_uuid}/receipt.json"

    # write staging candidates (storage DB)
    storage = storage_url(args)
    candidate_uuids: list[str] = []
    staging_packet_uuids: list[str] = []
    staging_used_db = False
    staging_blocked_reason = ""

    try:
        with psycopg.connect(storage) as sconn:
            candidate_uuids, staging_used_db, staging_blocked_reason = write_staging_candidates(
                sconn, job_uuid, source_sha256, case_key, primary_mode, evidence_ref, scored_claims,
            )
            staging_packet_uuids = write_staging_packets(
                sconn, job_uuid, source_sha256, case_key, primary_mode, scored_claims,
            )
    except Exception as exc:
        blockers.append(f"storage_write_error:{exc}")

    glow_count  = sum(1 for sc in scored_claims if sc["glow"]["ternary_score"] == 1)
    noise_count = sum(1 for sc in scored_claims if sc["glow"]["ternary_score"] == -1)

    result: dict[str, Any] = {
        "ok": not bool([b for b in blockers if "storage_write_error" not in b]),
        "outcome": "succeeded",
        "claims_count": len(scored_claims),
        "glow_count": glow_count,
        "noise_count": noise_count,
        "neutral_count": len(scored_claims) - glow_count - noise_count,
        "candidate_uuids": candidate_uuids,
        "staging_packet_uuids": staging_packet_uuids,
        "staging_used_db": staging_used_db,
        "staging_blocked_reason": staging_blocked_reason,
        "case_key": case_key,
        "source_sha256": source_sha256,
        "primary_mode": primary_mode,
        "canonical_graph_write_performed": False,
        "groq_used": False,
        "scored_claims": scored_claims,
    }
    if blockers:
        result["ok"] = False
        result["outcome"] = "failed"
        result["blockers"] = blockers
    return result, blockers


def worker_once(args: argparse.Namespace, execute: bool) -> tuple[dict[str, Any], list[str]]:
    blockers: list[str] = []
    surl = state_url(args)
    worker_id = args.worker_id or f"{WORKER_KEY}:{socket.gethostname()}:{os.getpid()}"
    result: dict[str, Any] = {
        "state_database_url": redact(surl),
        "queue": QUEUE_NAME,
        "worker_id": worker_id,
        "execute_performed": False,
        "job_processed": False,
    }

    with psycopg.connect(surl) as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            # SELECT FOR UPDATE SKIP LOCKED dequeue
            cur.execute(
                """
                SELECT job_uuid::text, workflow_name, job_kind, idempotency_key,
                       status::text, payload, attempt_count, max_attempts
                FROM lucidota_control.absurd_queue_job
                WHERE queue_name = %s
                  AND status = 'queued'
                  AND run_after <= now()
                ORDER BY priority ASC, created_at ASC
                LIMIT 1
                FOR UPDATE SKIP LOCKED
                """,
                (QUEUE_NAME,),
            )
            row = cur.fetchone()

            if not execute:
                result["would_process"] = dict(row) if row else None
                if row:
                    result["worker_contract"] = validate_worker_contract(
                        cur, queue_name=QUEUE_NAME, job_kind=str(row["job_kind"]), worker_key=WORKER_KEY
                    ).as_result()
                return result, blockers

            if not row:
                result["no_job_available"] = True
                return result, blockers

            job_uuid   = str(row["job_uuid"])
            job_kind   = str(row["job_kind"])
            payload    = row["payload"] if isinstance(row["payload"], dict) else {}
            attempt_count = int(row["attempt_count"])
            max_attempts  = int(row["max_attempts"])

            # validate contract
            contract = validate_worker_contract(
                cur, queue_name=QUEUE_NAME, job_kind=job_kind, worker_key=WORKER_KEY
            )
            result["worker_contract"] = contract.as_result()
            if not contract.ok:
                gate_result, error_kind = record_worker_contract_rejection(
                    cur,
                    job_uuid=job_uuid,
                    queue_name=QUEUE_NAME,
                    payload=payload,
                    contract=contract,
                    event_source=WORKER_KEY,
                )
                conn.commit()
                result.update({
                    "execute_performed": True,
                    "job_processed": False,
                    "job_uuid": job_uuid,
                    "status": "failed",
                    "result": gate_result,
                })
                blockers.append(error_kind)
                return result, blockers

            if job_kind != JOB_KIND:
                blockers.append(f"unexpected_job_kind:{job_kind}")
                return result, blockers

            # snapshot canonical graph counts before work
            graph_before: dict[str, Any] = {}
            try:
                graph_before = count_tables_snapshot(conn, CANONICAL_GRAPH_TABLES)
            except Exception as exc:
                graph_before = {"error": str(exc)}

            # lease job
            cur.execute(
                """
                UPDATE lucidota_control.absurd_queue_job
                SET status = 'running',
                    leased_by = %s,
                    lease_expires_at = now() + interval '10 minutes',
                    attempt_count = attempt_count + 1,
                    updated_at = now()
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
                (job_uuid, QUEUE_NAME, WORKER_KEY, json.dumps({"worker_id": worker_id})),
            )
            conn.commit()

        # --- work outside the cursor lock ---
        started_at = now_iso()
        work_result, work_blockers = process_job(payload, job_uuid, args)
        finished_at = now_iso()

        # payload hygiene gate on result
        gate_payload = {**work_result, "outcome": work_result.get("outcome", "succeeded"), "ok": work_result.get("ok", False)}
        payload_ok, hygiene = gate_worker_payload_hygiene(
            gate_payload,
            queue_name=QUEUE_NAME,
            worker_key=WORKER_KEY,
            job_kind=job_kind,
            required_keys=("ok", "outcome"),
            min_score=0,
        )
        if not payload_ok:
            work_blockers.append(hygiene.get("hygiene", {}).get("error", "hygiene_failed"))

        ok = not work_blockers
        status = "succeeded" if ok else "failed"

        # build receipt
        receipt: dict[str, Any] = {
            "schema": SCHEMA_VER,
            "receipt_mode": "ABSURD_POSTGRES_RUNTIME",
            "job_uuid": job_uuid,
            "queue_name": QUEUE_NAME,
            "job_kind": job_kind,
            "worker_key": WORKER_KEY,
            "case_key": work_result.get("case_key", ""),
            "idempotency_key": str(payload.get("idempotency_key", "")),
            "status": status,
            "groq_used": False,
            "canonical_graph_write_performed": False,
            "claims_count": work_result.get("claims_count", 0),
            "glow_count": work_result.get("glow_count", 0),
            "noise_count": work_result.get("noise_count", 0),
            "neutral_count": work_result.get("neutral_count", 0),
            "candidate_uuids": work_result.get("candidate_uuids", []),
            "staging_packet_uuids": work_result.get("staging_packet_uuids", []),
            "staging_used_db": work_result.get("staging_used_db", False),
            "staging_blocked_reason": work_result.get("staging_blocked_reason", ""),
            "error_kind": ";".join(work_blockers) if work_blockers else "",
            "error_message": "",
            "attempt_count": attempt_count + 1,
            "started_at": started_at,
            "finished_at": finished_at,
        }
        receipt_path = write_receipt(job_uuid, receipt)
        receipt["report_path"] = rel(receipt_path)

        with conn.cursor(row_factory=dict_row) as cur:
            # snapshot canonical graph after work
            graph_after: dict[str, Any] = {}
            try:
                graph_after = count_tables_snapshot(conn, CANONICAL_GRAPH_TABLES)
            except Exception as exc:
                graph_after = {"error": str(exc)}

            if graph_before != graph_after:
                work_blockers.append("canonical_graph_counts_changed_FORBIDDEN")
                status = "failed"
                ok = False
                receipt["status"] = status
                receipt["error_kind"] = ";".join(work_blockers)

            cur.execute(
                """
                UPDATE lucidota_control.absurd_queue_job
                SET status = %s,
                    result = %s::jsonb,
                    completed_at = CASE WHEN %s THEN now() ELSE completed_at END,
                    updated_at = now(),
                    last_error = %s
                WHERE job_uuid = %s::uuid
                """,
                (
                    status,
                    json.dumps(receipt, default=str),
                    ok,
                    receipt["error_kind"],
                    job_uuid,
                ),
            )
            cur.execute(
                """
                INSERT INTO lucidota_control.absurd_queue_event
                  (job_uuid, queue_name, event_kind, event_source, detail)
                VALUES (%s::uuid, %s, %s, %s, %s::jsonb)
                """,
                (
                    job_uuid,
                    QUEUE_NAME,
                    status,
                    WORKER_KEY,
                    json.dumps({
                        "worker_id": worker_id,
                        "blockers": work_blockers,
                        "glow_count": receipt["glow_count"],
                        "claims_count": receipt["claims_count"],
                    }),
                ),
            )
            cur.execute(
                """
                INSERT INTO lucidota_control.workflow_event
                  (workflow_id, run_id, phase, status, source, detail)
                VALUES (%s, %s, 'glow_watch_stage', %s, %s, %s::jsonb)
                RETURNING event_id::text
                """,
                (
                    WORKFLOW_ID,
                    job_uuid,
                    status,
                    WORKER_KEY,
                    json.dumps({
                        "queue": QUEUE_NAME,
                        "job_uuid": job_uuid,
                        "glow_count": receipt["glow_count"],
                        "claims_count": receipt["claims_count"],
                        "case_key": receipt["case_key"],
                        "canonical_graph_write_performed": False,
                        "groq_used": False,
                    }, default=str),
                ),
            )
            event_id = first_value(cur.fetchone())

            if not ok:
                # dead-letter on failure or max retries
                cur.execute(
                    """
                    INSERT INTO lucidota_control.absurd_queue_dead_letter
                      (job_uuid, queue_name, workflow_name, job_kind,
                       idempotency_key, error_kind, error_message,
                       attempt_count, payload_sha256, context)
                    SELECT job_uuid, queue_name, workflow_name, job_kind,
                           idempotency_key, %s, %s, attempt_count, %s, %s::jsonb
                    FROM lucidota_control.absurd_queue_job
                    WHERE job_uuid = %s::uuid
                    ON CONFLICT (job_uuid) WHERE resolved = false
                    DO UPDATE SET
                      error_message = EXCLUDED.error_message,
                      last_seen_at  = now(),
                      context       = EXCLUDED.context
                    """,
                    (
                        receipt["error_kind"],
                        receipt["error_kind"],
                        sha256_obj(payload),
                        json.dumps(receipt, default=str),
                        job_uuid,
                    ),
                )
            conn.commit()

        result.update({
            "execute_performed": True,
            "job_processed": True,
            "job_uuid": job_uuid,
            "workflow_event_id": event_id,
            "status": status,
            "receipt": receipt,
            "canonical_graph_counts_before": graph_before,
            "canonical_graph_counts_after": graph_after,
            "canonical_graph_writes_performed": graph_before != graph_after,
        })
        blockers.extend(work_blockers)

    return result, blockers


# ── ENQUEUE helper (for testing / integration) ────────────────────────────────

def enqueue(args: argparse.Namespace, execute: bool) -> tuple[dict[str, Any], list[str]]:
    blockers: list[str] = []
    surl = state_url(args)
    payload = {
        "job_kind": JOB_KIND,
        "claims": args.claims_json and json.loads(args.claims_json) or [],
        "source_sha256": args.source_sha256 or "",
        "case_key": args.case_key or "",
        "primary_mode": args.primary_mode or "GLOW_HUNTER",
        "evidence_refs": args.evidence_refs and json.loads(args.evidence_refs) or [],
        "idempotency_key": args.idempotency_key
            or f"glow_watch_stage:{args.source_sha256 or 'nohash'}:{args.case_key or 'nocase'}",
    }
    idempotency_key = str(payload["idempotency_key"])
    result: dict[str, Any] = {
        "state_database_url": redact(surl),
        "queue": QUEUE_NAME,
        "job_kind": JOB_KIND,
        "idempotency_key": idempotency_key,
        "execute_performed": False,
        "job_uuid": None,
        "inserted_new": False,
    }
    if not execute:
        result["would_enqueue"] = payload
        return result, blockers
    with psycopg.connect(surl) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO lucidota_control.absurd_queue_job
                  (queue_name, workflow_name, job_kind, idempotency_key,
                   payload, priority, max_attempts, detail)
                VALUES (%s, %s, %s, %s, %s::jsonb, %s, %s, %s::jsonb)
                ON CONFLICT (queue_name, idempotency_key)
                DO UPDATE SET updated_at = now()
                RETURNING job_uuid, (xmax = 0) AS inserted_new
                """,
                (
                    QUEUE_NAME,
                    WORKFLOW_ID,
                    JOB_KIND,
                    idempotency_key,
                    json.dumps(payload),
                    args.priority,
                    args.max_attempts,
                    json.dumps({"source": WORKER_KEY}),
                ),
            )
            job_uuid, inserted_new = cur.fetchone()
            if inserted_new:
                cur.execute(
                    """
                    INSERT INTO lucidota_control.absurd_queue_event
                      (job_uuid, queue_name, event_kind, event_source, detail)
                    VALUES (%s, %s, 'enqueued', %s, %s::jsonb)
                    """,
                    (job_uuid, QUEUE_NAME, WORKER_KEY, json.dumps({"job_kind": JOB_KIND})),
                )
        conn.commit()
        result.update({
            "execute_performed": True,
            "job_uuid": str(job_uuid),
            "inserted_new": bool(inserted_new),
        })
    return result, blockers


# ── OUTPUT ────────────────────────────────────────────────────────────────────

def write_report(action: str, report: dict[str, Any]) -> Path:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    path = OUT_ROOT / f"glow_watch_worker_{action}_{stamp()}.json"
    report.setdefault("generated_at", now_iso())
    report["report_path"] = rel(path)
    path.write_text(json.dumps(report, indent=2, sort_keys=False, default=str), encoding="utf-8")
    print(f"REPORT_PATH={rel(path)}", flush=True)
    return path


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(
        description="ABSURD worker: indy_polycareer / glow_watch_stage"
    )
    ap.add_argument(
        "--action",
        choices=["worker-once", "enqueue", "audit"],
        default="worker-once",
        help="worker-once: dequeue and process one job; enqueue: inject a test job; audit: dry-run inspect",
    )
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", default=True)
    mode.add_argument("--execute", action="store_true")
    ap.add_argument("--state-database-url", default=os.environ.get("ABSURD_SYSTEM_DATABASE_URL", "postgresql:///lucidota_state"))
    ap.add_argument("--storage-database-url", default=os.environ.get("LUCIDOTA_GO_STORAGE_DSN", "postgresql:///lucidota_storage"))
    ap.add_argument("--worker-id")
    ap.add_argument("--priority", type=int, default=50)
    ap.add_argument("--max-attempts", type=int, default=2)
    ap.add_argument("--idempotency-key")
    # enqueue helper args
    ap.add_argument("--claims-json", help="JSON array of claim strings or objects")
    ap.add_argument("--source-sha256", default="")
    ap.add_argument("--case-key", default="")
    ap.add_argument("--primary-mode", default="GLOW_HUNTER")
    ap.add_argument("--evidence-refs", help="JSON array of evidence ref paths")
    ap.add_argument("--json", action="store_true", help="Print JSON output to stdout")
    args = ap.parse_args()
    execute = bool(args.execute)

    blockers: list[str] = []
    action_result: dict[str, Any] = {}
    try:
        if args.action == "worker-once" or args.action == "audit":
            action_result, blockers = worker_once(args, execute)
        elif args.action == "enqueue":
            action_result, blockers = enqueue(args, execute)
    except Exception as exc:
        action_result = {"error": str(exc)}
        blockers = [f"exception:{exc}"]

    report = {
        "schema": SCHEMA_VER,
        "generated_at": now_iso(),
        "action": args.action,
        "mode": "execute" if execute else "dry_run",
        "execute_requested": execute,
        "state_database_url": redact(state_url(args)),
        "storage_database_url": redact(storage_url(args)),
        "queue": QUEUE_NAME,
        "job_kind": JOB_KIND,
        "worker_key": WORKER_KEY,
        "action_result": action_result,
        "db_writes_performed": bool(action_result.get("execute_performed")),
        "canonical_graph_writes_performed": bool(action_result.get("canonical_graph_writes_performed")),
        "groq_used": False,
        "blockers": blockers,
    }
    out_path = write_report(args.action, report)
    if args.json:
        print(json.dumps(report, indent=2, default=str))
    return 0 if not blockers else 1


if __name__ == "__main__":
    sys.exit(main())
