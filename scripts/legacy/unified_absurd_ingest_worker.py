#!/usr/bin/env python3
"""LUCIDOTA unified Absurd ingest worker.

Native ABSURD/Postgres worker for local investigative artifact ingest.

Properties:
- Uses lucidota_control.absurd_workflow, not new ABSURD architecture.
- Claims exactly one row with UPDATE ... FOR UPDATE SKIP LOCKED ... RETURNING.
- Holds database locks only for claim/finish transactions; file and parser work run
  outside row locks.
- Preserves raw local evidence fidelity inside lucidota_storage/lucidota_go.
- Streams an inline Amalgamated Text Surface to stdout for operator review.
- Logs successes, anomalies, policy gates, and terminal failures as COMMENT
  primitives using existing GO staging/evidence tables.
- Uses ALGOS.serpentina_self_righting as the recovery-cost math for retries.
- Never sends email, files public records, or publishes; outbound vectors are
  hard-gated to draft_only receipts.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import html
import json
import mimetypes
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

import psycopg
from psycopg.rows import dict_row

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ALGOS.serpentina_self_righting import Morphology, recovery_priority, righting_time_index
from ALGOS.poikilotherm_schoolfield import normalized_activity
from ALGOS.thanatosis import decide as thanatosis_decide
from ALGOS.rete_bandit_gate import apply_rete_bandit
from ALGOS.omni_chaotic_sprint import ChaoticOmniEngine
from ALGOS.percyphon import procedural_entity_generator
from core.runtime_dsns import resolve_state_dsn, resolve_storage_dsn
from core.telemetry.diogenes import compress_activity, staple_activity
from core.language_membrane import route_inbound_text, weave_output
from pypeline.math.epistemic_certainty import (
    comment_prior,
    filesystem_observation,
    parser_extraction,
)
from pypeline.math.model_vram_scheduler import (
    evict_transient_embedding_context,
    neo_knows_kung_fu_context,
    persist_governor_decision,
    plan_lora_preemption,
)
from pypeline.math.prompt_injection import (
    detect_prompt_injection,
    neutralize_for_display,
)

CAS_ROOT = ROOT / "03_VAULT" / "cas"
OUT_DIR = ROOT / "05_OUTPUTS" / "absurd"
MATH_IMPORT_DIR = ROOT / "02_RECORDS_OFFICE" / "imports" / "math_zip_2026_05_13"
OFFICIAL_ONTOLOGY = ROOT / "OFFICIAL_ONTOLOGY.json"
GO_ACTIVE_TERMS = ROOT / "BOOKS" / "GO_ACTIVE_TERMS.json"
ACTIVE_ONTOLOGY = ROOT / "BOOKS" / "ACTIVE_ONTOLOGY.json"

WORKFLOW_NAME = "unified_ingest"
WORKER_SOURCE = "scripts/unified_absurd_ingest_worker.py"
DEFAULT_LEASE_SECONDS = 3600
DEFAULT_MAX_FAILURES = 4
DEFAULT_STREAM_BYTES = 256 * 1024
DEFAULT_FRAME_CHARS = 4000
DEFAULT_IDLE_SLEEP_SECONDS = 2.0
DEFAULT_BYTEWAX_TICK_LIMIT = 40
DEFAULT_POIKILOTHERM_FLOOR = float(os.environ.get("LUCIDOTA_POIKILOTHERM_ACTIVITY_FLOOR", "0.35"))
DEFAULT_THANATOSIS_FLOOR = float(os.environ.get("LUCIDOTA_THANATOSIS_DORMANCY_FLOOR", "0.05"))
DEFAULT_HARD_TIMEOUT_SECONDS = float(os.environ.get("LUCIDOTA_ABSURD_STEP_TIMEOUT_SECONDS", "120"))

GO25: list[str] = [
    "ENTITY",
    "ATTRIBUTE",
    "RELATIONSHIP",
    "FRICTION",
    "LEVERAGE",
    "VISIBILITY",
    "ACTION",
    "EVENT",
    "TIME",
    "PATTERN",
    "HYPOTHESIS",
    "CLAIM",
    "EVIDENCE",
    "ATOMIC_ID",
    "SIGNAL",
    "GLOW",
    "TERM",
    "TOOL",
    "ALGORITHM",
    "NAUGHTY",
    "NICE",
    "GROUP",
    "OPERATOR",
    "MODE",
    "COMMENT",
]

STATE_SCHEMA_SQL = """
CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE SCHEMA IF NOT EXISTS lucidota_control;

CREATE TABLE IF NOT EXISTS lucidota_control.workflow_event (
    event_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    workflow_id text NOT NULL,
    run_id text NOT NULL,
    phase text NOT NULL,
    status text NOT NULL CHECK (status IN (
        'queued',
        'running',
        'blocked',
        'waiting_user',
        'succeeded',
        'failed',
        'cancelled'
    )),
    source text NOT NULL,
    detail jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS workflow_event_run_idx
    ON lucidota_control.workflow_event (workflow_id, run_id, created_at DESC);

CREATE TABLE IF NOT EXISTS lucidota_control.absurd_workflow (
    workflow_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workflow_name TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    state JSONB NOT NULL DEFAULT '{}'::jsonb,
    payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    error_log TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

ALTER TABLE lucidota_control.absurd_workflow
    ADD COLUMN IF NOT EXISTS priority INTEGER NOT NULL DEFAULT 100,
    ADD COLUMN IF NOT EXISTS attempt_count INTEGER NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS failure_count INTEGER NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS max_attempts INTEGER NOT NULL DEFAULT 4,
    ADD COLUMN IF NOT EXISTS locked_by TEXT,
    ADD COLUMN IF NOT EXISTS locked_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS lease_expires_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS completed_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS last_comment_uuid UUID;

CREATE INDEX IF NOT EXISTS idx_absurd_workflow_status
    ON lucidota_control.absurd_workflow(status);
CREATE INDEX IF NOT EXISTS idx_absurd_workflow_claim_pending
    ON lucidota_control.absurd_workflow(priority ASC, created_at ASC)
    WHERE status = 'pending';
CREATE INDEX IF NOT EXISTS idx_absurd_workflow_running_lease
    ON lucidota_control.absurd_workflow(lease_expires_at)
    WHERE status = 'running';
"""

STORAGE_SCHEMA_SQL = """
CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE SCHEMA IF NOT EXISTS lucidota_go;

CREATE TABLE IF NOT EXISTS lucidota_go.term_registry (
    term text PRIMARY KEY,
    term_number integer,
    status text NOT NULL DEFAULT 'active',
    definition text NOT NULL DEFAULT '',
    parent_term text,
    detail jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS lucidota_go.staging_packet (
    packet_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    source_id text NOT NULL DEFAULT '',
    parser_name text NOT NULL DEFAULT 'go_fast_parser',
    proposed_term text REFERENCES lucidota_go.term_registry(term),
    raw_anchor text NOT NULL DEFAULT '',
    claim text NOT NULL DEFAULT '',
    proposed_item jsonb NOT NULL DEFAULT '{}'::jsonb,
    proposed_edges jsonb NOT NULL DEFAULT '[]'::jsonb,
    status text NOT NULL DEFAULT 'pending',
    confidence_bps integer NOT NULL DEFAULT 0,
    operator_uuid uuid,
    created_at timestamptz NOT NULL DEFAULT now(),
    decided_at timestamptz
);

CREATE TABLE IF NOT EXISTS lucidota_go.graph_promotion_evidence_resolution (
    resolution_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    evidence_ref text NOT NULL,
    ref_kind text NOT NULL,
    resolved boolean NOT NULL,
    resolver text NOT NULL DEFAULT 'scripts/graph_promotion_packet_reviewer.py',
    source_table text,
    source_uuid uuid,
    source_path text,
    detail jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_graph_promotion_evidence_resolution_ref_created
    ON lucidota_go.graph_promotion_evidence_resolution(evidence_ref, created_at DESC);
"""

TERM_DEFINITIONS: dict[str, str] = {
    "ENTITY": "Universal node / target; anything nameable or functionally acting.",
    "ATTRIBUTE": "Static property or metadata attached to an entity.",
    "RELATIONSHIP": "Connection between entities; directional or undirected hook/leash.",
    "FRICTION": "Relational stress between entities; not moral judgment.",
    "LEVERAGE": "Pressure, influence, liability, centrality, or usable force.",
    "VISIBILITY": "Public/common exposure; what all or most observers can see.",
    "ACTION": "Something taken/done by an actor.",
    "EVENT": "Something that happens/occurs; may be non-agentic.",
    "TIME": "Order, timestamp, duration, decay, velocity, pacing.",
    "PATTERN": "Repeated structure, motif, recurrence, detectable form.",
    "HYPOTHESIS": "Provisional explanatory model awaiting evidence/judgment.",
    "CLAIM": "Asserted proposition that can be supported, disputed, or tested.",
    "EVIDENCE": "Artifact, observation, citation, page, hash, or source support.",
    "ATOMIC_ID": "Stable unique identifier for a thing, source, record, page, or node.",
    "SIGNAL": "Incoming indication, weak or strong, before final interpretation.",
    "GLOW": "Private/special salience only some observers perceive.",
    "TERM": "Vocabulary item, label, alias, controlled word, or named concept.",
    "TOOL": "Instrument or executable mechanism used by the system/user.",
    "ALGORITHM": "Named procedure or operation with behavior.",
    "NAUGHTY": "Krampus/Santa-class hunted/judged condition; distinct from friction.",
    "NICE": "Santa/Krampus-class approved/reward/mercy condition; distinct from positive valence.",
    "GROUP": "Cluster, community, set, faction, or collection of entities.",
    "OPERATOR": "Symbolic/logical/modifying operation applied to terms, claims, or data.",
    "MODE": "Active lens, parser pack, state, or interpretive posture.",
    "COMMENT": "Human note, margin mark, correction, judgment, or aside.",
}

TEXT_SUFFIXES = {
    ".txt",
    ".md",
    ".json",
    ".jsonl",
    ".csv",
    ".tsv",
    ".log",
    ".py",
    ".sql",
    ".toml",
    ".yaml",
    ".yml",
    ".html",
    ".htm",
    ".xml",
    ".rst",
}

OUTBOUND_KEYS = {
    "send_email",
    "email_to",
    "smtp_send",
    "publish",
    "publish_url",
    "public_filing",
    "file_public",
    "external_delivery",
    "outbound_action",
    "delivery_mode",
    "recipient",
    "recipients",
}

TERMINAL_ERROR_KINDS = (
    FileNotFoundError,
    PermissionError,
)


class TerminalPolicyError(RuntimeError):
    """Non-retryable policy violation."""


@dataclass(frozen=True)
class RuntimeContext:
    state_dsn: str
    storage_dsn: str
    worker_id: str
    lease_seconds: int
    max_failures: int


@dataclass(frozen=True)
class SerpentinaDecision:
    worth_retry: bool
    righting_priority: float
    righting_time: float
    force_remaining: int
    mutation: dict[str, Any]
    reason: str


def now_z() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def jdump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def rel(path: Path | str) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def state_db_url() -> str:
    return resolve_state_dsn()


def storage_db_url() -> str:
    return resolve_storage_dsn()


def emit_frame(kind: str, detail: dict[str, Any], body: str | None = None) -> None:
    print(f"\n===== {kind} :: {now_z()} =====", flush=True)
    print(jdump(detail), flush=True)
    if body:
        print(body, flush=True)
    print(f"===== /{kind} =====\n", flush=True)


def run_cmd(cmd: list[str], timeout: int = 120) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, timeout=timeout)


def daily_telemetry_wal_path() -> Path:
    return OUT_DIR / f"absurd_telemetry_{datetime.now(timezone.utc).strftime('%Y-%m-%d')}.jsonl"


def append_telemetry_wal(event_name: str, payload: dict[str, Any]) -> Path:
    """Append one telemetry/receipt event to the daily WAL.

    Phase 6 hardening rule: no one-file-per-status micro-receipts from this
    worker. Init/verify/status reports append compact JSONL records behind an
    exclusive flock so downstream ledgers can stream one file.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = daily_telemetry_wal_path()
    generated_at = payload.setdefault("generated_at", now_z())
    payload["receipt_name"] = event_name
    payload["report_path"] = rel(path)
    payload["telemetry_wal_path"] = rel(path)
    payload_hash = canonical_sha256(payload)
    event_id = canonical_sha256({"event_name": event_name, "generated_at": generated_at, "payload_sha256": payload_hash})
    payload["payload_sha256"] = payload_hash
    payload["telemetry_event_id"] = event_id
    record = {
        "schema": "lucidota.absurd.telemetry_event.v1",
        "event_id": event_id,
        "event_name": event_name,
        "generated_at": generated_at,
        "payload_sha256": payload_hash,
        "payload": payload,
    }
    with path.open("a", encoding="utf-8") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        fh.write(canonical_json(record) + "\n")
        fh.flush()
        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
    return path


def workflow_event_status(status: str) -> str:
    """Map ABSURD-native status words onto lucidota_control.workflow_event law."""
    s = (status or "").lower()
    if s in {"queued", "pending"}:
        return "queued"
    if s in {"running", "processing"}:
        return "running"
    if s in {"completed", "complete", "succeeded", "success", "ok"}:
        return "succeeded"
    if s in {"dead_lettered", "failed", "error"}:
        return "failed"
    if s in {"cancelled", "canceled"}:
        return "cancelled"
    if s in {"blocked", "waiting_user"}:
        return s
    return "failed"


def ensure_state_schema(dsn: str) -> None:
    with psycopg.connect(dsn) as conn:
        conn.execute(STATE_SCHEMA_SQL)
        conn.commit()


def ensure_storage_schema(dsn: str) -> None:
    with psycopg.connect(dsn) as conn:
        conn.execute(STORAGE_SCHEMA_SQL)
        for idx, term in enumerate(GO25, 1):
            conn.execute(
                """
                INSERT INTO lucidota_go.term_registry(term, term_number, status, definition, detail)
                VALUES (%s, %s, 'active', %s, %s::jsonb)
                ON CONFLICT (term) DO UPDATE
                SET term_number = COALESCE(lucidota_go.term_registry.term_number, EXCLUDED.term_number),
                    definition = CASE
                        WHEN lucidota_go.term_registry.definition = '' THEN EXCLUDED.definition
                        ELSE lucidota_go.term_registry.definition
                    END,
                    updated_at = now()
                """,
                (term, idx, TERM_DEFINITIONS.get(term, ""), jdump({"source": WORKER_SOURCE, "ontology": "GO-25"})),
            )
        conn.commit()


def load_json_file(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def active_terms_from_go_file(path: Path) -> list[str]:
    data = load_json_file(path)
    rows = data.get("terms") or []
    out: list[str] = []
    for row in rows:
        if isinstance(row, dict) and isinstance(row.get("term"), str):
            out.append(row["term"])
    return out


def verify_ontology_contract(strict_active_file: bool = False) -> dict[str, Any]:
    blockers: list[str] = []
    details: dict[str, Any] = {"go25_expected": GO25, "files": {}}

    try:
        official = load_json_file(OFFICIAL_ONTOLOGY)
        official_terms = list(official.get("active_terms") or [])[:25]
        details["files"][rel(OFFICIAL_ONTOLOGY)] = official_terms
        if official_terms != GO25:
            blockers.append("OFFICIAL_ONTOLOGY active_terms[0:25] does not match GO-25 exact labels")
        if official.get("official_ontology") != "GO-25":
            blockers.append("OFFICIAL_ONTOLOGY official_ontology is not GO-25")
    except Exception as exc:
        blockers.append(f"OFFICIAL_ONTOLOGY unreadable: {exc}")

    try:
        book_terms = active_terms_from_go_file(GO_ACTIVE_TERMS)[:25]
        details["files"][rel(GO_ACTIVE_TERMS)] = book_terms
        if book_terms != GO25:
            blockers.append("BOOKS/GO_ACTIVE_TERMS terms[0:25] does not match GO-25 exact labels")
    except Exception as exc:
        blockers.append(f"GO_ACTIVE_TERMS unreadable: {exc}")

    if ACTIVE_ONTOLOGY.exists():
        try:
            active = load_json_file(ACTIVE_ONTOLOGY)
            details["files"][rel(ACTIVE_ONTOLOGY)] = active
            active_name = active.get("active_ontology") or active.get("official_ontology")
            if active_name and active_name != "GO-25":
                msg = f"BOOKS/ACTIVE_ONTOLOGY advertises {active_name!r}, expected 'GO-25'"
                if strict_active_file:
                    blockers.append(msg)
                else:
                    details.setdefault("warnings", []).append(msg)
        except Exception as exc:
            blockers.append(f"ACTIVE_ONTOLOGY unreadable: {exc}")

    details["ok"] = not blockers
    details["blockers"] = blockers
    return details


def verify_math_package() -> dict[str, Any]:
    manifest = MATH_IMPORT_DIR / "MANIFEST.txt"
    extracted = MATH_IMPORT_DIR / "extracted" / "math"
    zip_path = MATH_IMPORT_DIR / "math.zip"
    blockers: list[str] = []
    manifest_files: list[str] = []

    if not MATH_IMPORT_DIR.exists():
        blockers.append(f"missing math import dir: {rel(MATH_IMPORT_DIR)}")
    if not manifest.exists():
        blockers.append(f"missing manifest: {rel(manifest)}")
    else:
        manifest_files = [line.strip() for line in manifest.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
    if not zip_path.exists():
        blockers.append(f"missing math.zip: {rel(zip_path)}")
    if not extracted.exists():
        blockers.append(f"missing extracted dir: {rel(extracted)}")

    extracted_files = sorted(str(p.relative_to(extracted)) for p in extracted.rglob("*") if p.is_file()) if extracted.exists() else []
    manifest_base = sorted(Path(p).name for p in manifest_files)
    extracted_base = sorted(Path(p).name for p in extracted_files)
    missing_extracted = sorted(set(manifest_base) - set(extracted_base))
    if missing_extracted:
        blockers.append("manifest files missing from extracted/math: " + ", ".join(missing_extracted))

    pypeline_import_ok = False
    pypeline_import_error = None
    try:
        import pypeline.math as pypeline_math  # noqa: F401
        from pypeline.math.validators import validate
        import pypeline.math.validators_extended  # noqa: F401

        smoke = validate("EQ-030", signature_hits_last_24h=2)
        pypeline_import_ok = bool(smoke.passed)
        if not pypeline_import_ok:
            blockers.append("pypeline.math validator smoke failed")
    except Exception as exc:
        pypeline_import_error = str(exc)
        blockers.append(f"pypeline.math import smoke failed: {exc}")

    return {
        "ok": not blockers,
        "math_import_dir": rel(MATH_IMPORT_DIR),
        "math_zip_exists": zip_path.exists(),
        "math_zip_size": zip_path.stat().st_size if zip_path.exists() else None,
        "manifest_count": len(manifest_files),
        "extracted_count": len(extracted_files),
        "missing_extracted": missing_extracted,
        "pypeline_import_ok": pypeline_import_ok,
        "pypeline_import_error": pypeline_import_error,
        "blockers": blockers,
    }


def write_report(name: str, payload: dict[str, Any]) -> Path:
    return append_telemetry_wal(name, payload)


def rust_offload_candidates() -> list[dict[str, Any]]:
    return [
        {
            "surface": "sha256_file_with_progress + lock_into_cas",
            "reason": "byte hashing/copy/chmod are deterministic hot-path I/O and already fit the Rust intake/CAS lane",
            "rust_workspace": "01_REPOS/lucidota_etl",
            "suggested_boundary": "Rust binary/PyO3 function returns sha256, byte_size, cas_relative_path, mime_guess",
        },
        {
            "surface": "extract_text_surface/read_epub_text/read_pdf_text/strip_html",
            "reason": "bounded text extraction and HTML stripping are memory-sensitive parsing work; Python should orchestrate only",
            "rust_workspace": "01_REPOS/lucidota_etl",
            "suggested_boundary": "Rust extractor returns text frames plus extract_meta with byte/page caps enforced before Python sees text",
        },
        {
            "surface": "text_frames + analyze_text_signals",
            "reason": "large string slicing, regex scans, and term counting are startup/memory taxes for ephemeral workers",
            "rust_workspace": "01_REPOS/lucidota_etl",
            "suggested_boundary": "Rust analyzer returns frame offsets, text_sha256, counts, route hints, and GO term hits",
        },
        {
            "surface": "verify_ontology_contract + JSON/schema validation",
            "reason": "validation should be precompiled and streamable instead of reparsed per short-lived worker",
            "rust_workspace": "01_REPOS/lucidota_etl",
            "suggested_boundary": "jsonschema-rs validator or Rust CLI emits compact PASS/FAIL diagnostics",
        },
    ]


def verify_runtime(strict_active_file: bool = False) -> dict[str, Any]:
    ontology = verify_ontology_contract(strict_active_file=strict_active_file)
    math_pkg = verify_math_package()
    blockers = []
    blockers.extend(ontology.get("blockers") or [])
    blockers.extend(math_pkg.get("blockers") or [])
    return {
        "schema": "lucidota.unified_absurd_ingest_worker.verify.v1",
        "ok": not blockers,
        "ontology": ontology,
        "math_package": math_pkg,
        "rust_offload_candidates": rust_offload_candidates(),
        "blockers": blockers,
    }


def resolve_source_path(value: str | None) -> Path:
    if not value:
        raise ValueError("payload.source_path is required")
    p = Path(value).expanduser()
    if not p.is_absolute():
        p = (ROOT / p).resolve()
    else:
        p = p.resolve()
    return p


def sha256_file_with_progress(path: Path) -> tuple[str, int]:
    h = hashlib.sha256()
    total_size = path.stat().st_size
    processed = 0
    emit_frame("SUBTLE_KNIFE_HASH_START", {"source_path": str(path), "byte_size": total_size})
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
            processed += len(chunk)
            if total_size > 0:
                pct = min(100, int(processed * 100 / total_size))
                filled = pct // 4
                sys.stdout.write(f"\r  -> CAS hash [{('█' * filled)}{('-' * (25 - filled))}] {pct}%")
                sys.stdout.flush()
    if total_size > 0:
        print("", flush=True)
    digest = h.hexdigest()
    emit_frame("SUBTLE_KNIFE_HASH_DONE", {"source_path": str(path), "sha256": digest, "byte_size": total_size})
    return digest, total_size


def sha256_file_quiet(path: Path) -> tuple[str, int]:
    h = hashlib.sha256()
    total_size = path.stat().st_size
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest(), total_size


def build_input_fingerprint(source_path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    source_digest, byte_size = sha256_file_quiet(source_path)
    stable_payload = {
        k: v
        for k, v in payload.items()
        if k
        not in {
            "source_sha256",
            "source_byte_size",
            "input_sha256",
            "input_fingerprint",
            "idempotency_version",
        }
    }
    fingerprint = {
        "version": "lucidota.absurd.input_fingerprint.v1",
        "workflow_name": WORKFLOW_NAME,
        "source_path": str(source_path),
        "source_sha256": source_digest,
        "source_byte_size": byte_size,
        "payload": stable_payload,
    }
    input_sha256 = canonical_sha256(fingerprint)
    return {
        "source_sha256": source_digest,
        "source_byte_size": byte_size,
        "input_sha256": input_sha256,
        "input_fingerprint": fingerprint,
        "idempotency_version": fingerprint["version"],
    }


def lock_into_cas(path: Path, digest: str) -> Path:
    target_dir = CAS_ROOT / digest[:2] / digest[2:4]
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / digest
    if not target.exists():
        tmp = target.with_suffix(".tmp")
        shutil.copyfile(path, tmp)
        os.chmod(tmp, 0o440)
        os.replace(tmp, target)
    return target


def insert_staging_packet(
    conn: psycopg.Connection[Any],
    *,
    source_id: str,
    parser_name: str,
    proposed_term: str,
    raw_anchor: str,
    claim: str,
    proposed_item: dict[str, Any],
    status: str = "pending",
    confidence_bps: int = 50,
) -> str:
    row = conn.execute(
        """
        INSERT INTO lucidota_go.staging_packet(
            source_id, parser_name, proposed_term, raw_anchor, claim,
            proposed_item, proposed_edges, status, confidence_bps
        )
        VALUES (%s, %s, %s, %s, %s, %s::jsonb, '[]'::jsonb, %s, %s)
        RETURNING packet_uuid::text
        """,
        (source_id, parser_name, proposed_term, raw_anchor, claim, jdump(proposed_item), status, confidence_bps),
    ).fetchone()
    return str(row[0])


def log_comment_primitive(
    storage_dsn: str,
    *,
    source_path: str,
    digest: str | None,
    note: str,
    detail: dict[str, Any] | None = None,
    resolved: bool = True,
    evidence_ref: str = "INDY_SYNTHESIS_ENTRY",
) -> str | None:
    payload = dict(detail or {})
    payload.setdefault("epistemic_certainty", comment_prior(evidence_refs=[evidence_ref]).as_dict())
    payload.setdefault("epistemic_flag", payload["epistemic_certainty"]["label"])
    payload.update(
        {
            "go_term": "COMMENT",
            "note": note,
            "source_path": source_path,
            "sha256": digest,
            "created_by": WORKER_SOURCE,
            "created_at": now_z(),
        }
    )
    try:
        with psycopg.connect(storage_dsn) as conn:
            row = conn.execute(
                """
                INSERT INTO lucidota_go.graph_promotion_evidence_resolution(
                    evidence_ref, ref_kind, resolved, resolver, source_table,
                    source_uuid, source_path, detail
                )
                VALUES (%s, 'comment_primitive', %s, 'indy_reads_cooperator',
                        'staging_packet', gen_random_uuid(), %s, %s::jsonb)
                RETURNING resolution_uuid::text
                """,
                (evidence_ref, resolved, source_path, jdump(payload)),
            ).fetchone()
            resolution_uuid = str(row[0])
            insert_staging_packet(
                conn,
                source_id=digest or "no-digest",
                parser_name="unified_absurd_indy_reads_comment",
                proposed_term="COMMENT",
                raw_anchor=source_path,
                claim=note[:2000],
                proposed_item={**payload, "resolution_uuid": resolution_uuid},
                status="comment",
                confidence_bps=50,
            )
            conn.commit()
            return resolution_uuid
    except Exception as exc:
        emit_frame(
            "COMMENT_PRIMITIVE_LOG_FAILURE",
            {"source_path": source_path, "sha256": digest, "error": str(exc), "note": note[:500]},
        )
        return None


def log_state_event(
    state_dsn: str,
    *,
    workflow_id: str,
    phase: str,
    status: str,
    detail: dict[str, Any],
) -> None:
    try:
        with psycopg.connect(state_dsn) as conn:
            exists = conn.execute(
                """
                SELECT to_regclass('lucidota_control.workflow_event') IS NOT NULL
                """
            ).fetchone()[0]
            if not exists:
                return
            event_status = workflow_event_status(status)
            event_detail = dict(detail or {})
            event_detail.setdefault("raw_absurd_status", status)
            event_detail.setdefault("epistemic_flag", event_detail.get("epistemic_certainty", {}).get("label", "SURE_MAYBE"))
            conn.execute(
                """
                INSERT INTO lucidota_control.workflow_event(workflow_id, run_id, phase, status, source, detail)
                VALUES ('unified_absurd_ingest_worker', %s, %s, %s, %s, %s::jsonb)
                """,
                (workflow_id, phase, event_status, WORKER_SOURCE, jdump(event_detail)),
            )
            conn.commit()
    except Exception:
        return


def run_bytewax_tick_best_effort(limit: int = DEFAULT_BYTEWAX_TICK_LIMIT, *, include_activitywatch: bool = False, timeout: int = 30) -> dict[str, Any]:
    """Run one bounded abductive blender tick outside row locks."""
    script = ROOT / "scripts" / "bytewax_abductive_blender.py"
    if not script.exists():
        return {"ok": False, "error": "missing scripts/bytewax_abductive_blender.py"}
    cmd = [sys.executable, str(script), "tick", "--limit", str(limit)]
    if not include_activitywatch:
        cmd.append("--no-activitywatch")
    try:
        cp = run_cmd(cmd, timeout=timeout)
        return {
            "ok": cp.returncode == 0,
            "returncode": cp.returncode,
            "stdout_tail": cp.stdout[-4000:],
            "stderr_tail": cp.stderr[-2000:],
            "include_activitywatch": include_activitywatch,
        }
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}", "include_activitywatch": include_activitywatch}


def maybe_payload_bytewax_tick(ctx: RuntimeContext, payload: dict[str, Any], workflow_id: str, phase: str) -> None:
    if not payload.get("bytewax_tick_after_finish"):
        return
    limit = int(payload.get("bytewax_tick_limit") or DEFAULT_BYTEWAX_TICK_LIMIT)
    timeout = int(payload.get("bytewax_tick_timeout_seconds") or 30)
    include_aw = bool(payload.get("bytewax_tick_activitywatch"))
    report = run_bytewax_tick_best_effort(limit=limit, include_activitywatch=include_aw, timeout=timeout)
    emit_frame("BYTEWAX_ABDUCTIVE_TICK_AFTER_FINISH", {"workflow_id": workflow_id, "phase": phase, **report})
    log_state_event(ctx.state_dsn, workflow_id=workflow_id, phase="bytewax_abductive_tick_after_finish", status="succeeded" if report.get("ok") else "failed", detail=report)


def strip_html(raw: str) -> str:
    raw = re.sub(r"<script.*?</script>", " ", raw, flags=re.I | re.S)
    raw = re.sub(r"<style.*?</style>", " ", raw, flags=re.I | re.S)
    raw = re.sub(r"</p>|<br\s*/?>|</h\d>|</div>", "\n\n", raw, flags=re.I)
    raw = re.sub(r"<[^>]+>", " ", raw)
    raw = html.unescape(raw)
    raw = re.sub(r"[ \t\r\f\v]+", " ", raw)
    raw = re.sub(r"\n\s+", "\n", raw)
    raw = re.sub(r"\n{3,}", "\n\n", raw)
    return raw.strip()


def read_epub_text(path: Path, max_bytes: int) -> tuple[str, dict[str, Any]]:
    parts: list[str] = []
    consumed = 0
    with zipfile.ZipFile(path) as zf:
        names = sorted(n for n in zf.namelist() if n.lower().endswith((".xhtml", ".html", ".htm")))
        for name in names:
            if consumed >= max_bytes:
                break
            try:
                raw = zf.read(name)[: max_bytes - consumed].decode("utf-8", errors="replace")
            except Exception:
                continue
            txt = strip_html(raw)
            if txt:
                parts.append(txt)
                consumed += len(txt.encode("utf-8", errors="ignore"))
    return "\n\n".join(parts), {"extract_method": "epub-zip-html", "stream_truncated": consumed >= max_bytes}


def read_pdf_text(path: Path, max_pages: int, timeout: int) -> tuple[str, dict[str, Any]]:
    if shutil.which("pdftotext"):
        cmd = ["pdftotext", "-layout", "-enc", "UTF-8", "-f", "1", "-l", str(max_pages), str(path), "-"]
        cp = run_cmd(cmd, timeout=timeout)
        if cp.returncode == 0:
            return cp.stdout, {"extract_method": "pdftotext", "max_pages": max_pages, "stderr": cp.stderr[-1000:]}
        return "", {"extract_method": "pdftotext_failed", "max_pages": max_pages, "stderr": cp.stderr[-2000:], "returncode": cp.returncode}
    return "", {"extract_method": "pdftotext_missing", "max_pages": max_pages}


def read_strings_text(path: Path, max_bytes: int, timeout: int) -> tuple[str, dict[str, Any]]:
    if shutil.which("strings"):
        cp = run_cmd(["strings", "-n", "5", str(path)], timeout=timeout)
        text = cp.stdout[:max_bytes]
        return text, {"extract_method": "strings", "returncode": cp.returncode, "stderr": cp.stderr[-1000:], "stream_truncated": len(cp.stdout) > max_bytes}
    with path.open("rb") as fh:
        raw = fh.read(max_bytes)
    return raw.decode("utf-8", errors="replace"), {"extract_method": "binary-decode-fallback", "stream_truncated": path.stat().st_size > max_bytes}


def extract_text_surface(path: Path, state: dict[str, Any], payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    max_bytes = int(state.get("stream_max_bytes") or payload.get("max_stream_bytes") or DEFAULT_STREAM_BYTES)
    max_pages = int(payload.get("max_pdf_pages") or state.get("max_pdf_pages") or 10)
    timeout = int(payload.get("extract_timeout_seconds") or 120)
    suffix = path.suffix.lower()

    if suffix in TEXT_SUFFIXES:
        raw = path.read_bytes()[:max_bytes]
        text = raw.decode("utf-8", errors="replace")
        return text, {"extract_method": "direct-text", "stream_truncated": path.stat().st_size > max_bytes, "max_bytes": max_bytes}

    if suffix == ".pdf":
        text, meta = read_pdf_text(path, max_pages=max_pages, timeout=timeout)
        if text.strip():
            if len(text.encode("utf-8", errors="ignore")) > max_bytes:
                text = text.encode("utf-8", errors="ignore")[:max_bytes].decode("utf-8", errors="replace")
                meta["stream_truncated"] = True
            return text, {**meta, "max_bytes": max_bytes}
        fallback_text, fallback_meta = read_strings_text(path, max_bytes=max_bytes, timeout=timeout)
        return fallback_text, {**meta, "fallback": fallback_meta, "max_bytes": max_bytes}

    if suffix == ".epub":
        return read_epub_text(path, max_bytes=max_bytes)

    return read_strings_text(path, max_bytes=max_bytes, timeout=timeout)


def text_frames(text: str, frame_chars: int) -> Iterator[tuple[int, str]]:
    if not text:
        return
    i = 0
    for start in range(0, len(text), frame_chars):
        i += 1
        yield i, text[start : start + frame_chars]


def sentenceish(text: str, limit: int = 12) -> list[str]:
    cleaned = re.sub(r"\s+", " ", text).strip()
    bits = re.split(r"(?<=[.!?])\s+", cleaned)
    return [b.strip() for b in bits if len(b.strip()) > 20][:limit]


def analyze_text_signals(text: str, source_path: Path) -> dict[str, Any]:
    low = text.lower()
    terms: list[str] = ["EVIDENCE", "CLAIM", "COMMENT"]
    rules: list[tuple[str, Iterable[str]]] = [
        ("ENTITY", ["name", "person", "company", "organization", "tenant", "landlord"]),
        ("ATTRIBUTE", ["address", "phone", "email", "metadata", "size", "mime"]),
        ("RELATIONSHIP", ["because", "therefore", "linked", "connected", "with", "between"]),
        ("FRICTION", ["dispute", "conflict", "risk", "breach", "failure", "complaint"]),
        ("LEVERAGE", ["pressure", "authority", "deadline", "liability", "order"]),
        ("VISIBILITY", ["public", "published", "visible", "notice", "record"]),
        ("ACTION", ["sent", "called", "filed", "paid", "moved", "signed", "submitted"]),
        ("EVENT", ["occurred", "happened", "meeting", "hearing", "inspection"]),
        ("TIME", ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday", "202", "date"]),
        ("PATTERN", ["again", "repeated", "pattern", "cycle", "trend"]),
        ("HYPOTHESIS", ["maybe", "possibly", "hypothesis", "suggests", "could"]),
        ("SIGNAL", ["signal", "indicates", "marker", "hint", "clue"]),
        ("GLOW", ["salience", "strange", "notable", "weird"]),
        ("TERM", ["defined", "term", "label", "category"]),
        ("TOOL", ["script", "tool", "worker", "parser", "service"]),
        ("ALGORITHM", ["algorithm", "matrix", "vector", "hash", "model", "kinematic"]),
        ("GROUP", ["group", "team", "committee", "board", "family"]),
        ("OPERATOR", ["operator", "manual", "human", "user"]),
        ("MODE", ["draft", "mode", "state", "posture", "runtime"]),
    ]
    for term, needles in rules:
        if any(n in low for n in needles):
            terms.append(term)
    if re.search(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b", text):
        terms.extend(["ENTITY", "ATTRIBUTE", "SIGNAL"])
    if re.search(r"\b(?:\+?1[-. ]?)?\(?\d{3}\)?[-. ]?\d{3}[-. ]?\d{4}\b", text):
        terms.extend(["ENTITY", "ATTRIBUTE", "SIGNAL"])
    if re.search(r"\b\d{3,6}\s+[A-Z][A-Za-z0-9 .'-]+\b", text):
        terms.extend(["ENTITY", "ATTRIBUTE", "RELATIONSHIP"])

    seen: set[str] = set()
    unique_terms = [t for t in terms if t in GO25 and not (t in seen or seen.add(t))]
    sents = sentenceish(text)
    return {
        "terms": unique_terms,
        "route": " ∩ ".join(unique_terms[:8]) + " → AMALGAMATED_TEXT_SURFACE",
        "sentence_count_sampled": len(sents),
        "first_sentences": sents[:3],
        "source_suffix": source_path.suffix.lower(),
        "text_sha256": hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest(),
        "character_count": len(text),
        "byte_count": len(text.encode("utf-8", errors="ignore")),
    }



def absurd_rete_bandit_decision(*, payload: dict[str, Any], state: dict[str, Any], workflow_id: str, source_path: Path, text: str, analysis: dict[str, Any], epistemic: dict[str, Any], prompt_injection: dict[str, Any]) -> dict[str, Any]:
    compressed_activity = analysis.get("compressed_activity") or compress_activity(payload, state)
    return apply_rete_bandit({
        "source": "absurd_workflow",
        "source_ref": workflow_id,
        "event_time": now_z(),
        "text_surface": text[:20000],
        "payload": {
            **(payload or {}),
            "state_step": state.get("step"),
            "source_path": str(source_path),
            "mime_type": state.get("mime_type"),
            "sha256": state.get("digest"),
        },
        "compressed_activity": compressed_activity,
        "ontology_terms": analysis.get("terms") or [],
        "epistemic_flag": epistemic.get("label") or "SURE_MAYBE",
        "injection_flag": bool(prompt_injection.get("detected")),
    })

def outbound_requested(payload: dict[str, Any]) -> bool:
    for key in OUTBOUND_KEYS:
        value = payload.get(key)
        if value in (None, False, "", [], {}):
            continue
        if key == "delivery_mode" and str(value).lower() in {"draft", "draft_only", "none", "local"}:
            continue
        return True
    return False


def enforce_draft_only(payload: dict[str, Any]) -> dict[str, Any]:
    application_state = str(payload.get("application_state") or payload.get("outbound_state") or "draft_only").lower()
    delivery_mode = str(payload.get("delivery_mode") or "draft_only").lower()
    requested = outbound_requested(payload)
    manual_token_present = bool(payload.get("manual_receipt_token"))

    detail = {
        "application_state": application_state,
        "delivery_mode": delivery_mode,
        "outbound_requested": requested,
        "manual_receipt_token_present": manual_token_present,
        "send_performed": False,
        "publish_performed": False,
        "public_filing_performed": False,
        "outbound_status": "draft_only",
        "requires_operator_signoff": True,
    }

    if application_state != "draft_only":
        raise TerminalPolicyError(f"outbound application_state must remain draft_only, got {application_state!r}")
    if requested and delivery_mode not in {"draft", "draft_only", "none", "local"}:
        raise TerminalPolicyError(f"outbound delivery_mode blocked by draft_only gate: {delivery_mode!r}")
    return detail


def classify_error(error_msg: str, exc: BaseException | None = None) -> str:
    low = error_msg.lower()
    if exc and isinstance(exc, TERMINAL_ERROR_KINDS):
        return "terminal_local_resource"
    if isinstance(exc, TerminalPolicyError):
        return "terminal_policy"
    if "could not obtain lock" in low or "deadlock" in low or "lock timeout" in low:
        return "db_lock"
    if "out of memory" in low or "cannot allocate memory" in low or "killed" in low:
        return "memory_pressure"
    if "timeout" in low or "timed out" in low:
        return "external_hook_timeout"
    if "pdftotext" in low or "parser" in low or "decode" in low:
        return "parser_failure"
    if "no such file" in low or "not found" in low or "missing" in low:
        return "terminal_local_resource"
    return "generic_exception"


def serpentina_decide(
    *,
    failure_count: int,
    max_failures: int,
    error_msg: str,
    state: dict[str, Any],
    exc: BaseException | None = None,
) -> SerpentinaDecision:
    error_kind = classify_error(error_msg, exc)
    force_remaining = max(0, max_failures - failure_count)
    current_stream = int(state.get("stream_max_bytes") or DEFAULT_STREAM_BYTES)
    error_complexity = min(6.0, max(1.0, len(error_msg) / 160.0))
    morphology = Morphology(
        length=max(1.0, float(failure_count + 1) * error_complexity),
        width=max(1.0, float(max_failures)),
        height=max(0.5, float(force_remaining + 1) / 2.0),
        mass=max(1.0, float(current_stream) / 65536.0),
    )
    righting = recovery_priority(morphology, max_index=12.0)
    righting_time = righting_time_index(morphology)

    terminal = error_kind.startswith("terminal_")
    worth_retry = (not terminal) and force_remaining > 0 and righting < 0.98

    next_stream = max(16 * 1024, current_stream // 2)
    mutation = {
        "error_kind": error_kind,
        "chunk_boundary_bytes_before": current_stream,
        "chunk_boundary_bytes_after": next_stream,
        "clear_resident_llama_context": True,
        "network_layer": "proxychains_reserved_no_external_call",
        "db_lock_strategy": "release_transaction_and_retry_with_skip_locked",
        "parser_strategy": "fallback_to_strings_or_smaller_pdf_window",
        "retry_backoff_seconds": min(60, 2 ** min(failure_count, 5)),
    }
    reason = (
        f"Serpentina righting_priority={righting:.3f}, righting_time={righting_time:.3f}, "
        f"force_remaining={force_remaining}, error_kind={error_kind}; "
        + ("retry with mutated methodology" if worth_retry else "terminal/depleted; dead-letter safely")
    )
    return SerpentinaDecision(worth_retry, righting, righting_time, force_remaining, mutation, reason)


def apply_serpentina_state(state: dict[str, Any], decision: SerpentinaDecision, error_msg: str) -> dict[str, Any]:
    new_state = dict(state or {})
    history = list(new_state.get("serpentina_history") or [])
    history.append({**asdict(decision), "error": error_msg, "at": now_z()})
    new_state["serpentina_history"] = history[-20:]
    new_state["stream_max_bytes"] = decision.mutation["chunk_boundary_bytes_after"]
    if decision.mutation["error_kind"] in {"parser_failure", "external_hook_timeout"}:
        new_state["max_pdf_pages"] = max(1, int(new_state.get("max_pdf_pages") or 10) // 2)
    new_state["_last_error"] = error_msg
    new_state["_last_recovery_decision"] = asdict(decision)
    return new_state


def execute_serpentina_mutation(decision: SerpentinaDecision, *, workflow_id: str, reason: str) -> dict[str, Any]:
    """Execute safe local mutation hooks requested by Serpentina recovery math.

    The hook deliberately does not kill daemon processes.  It only clears
    transient embedding/Python/CUDA allocator state and returns a receipt that is
    persisted in workflow events before the row is released for SKIP LOCKED retry.
    """
    detail: dict[str, Any] = {"workflow_id": workflow_id, "reason": reason, "mutation": decision.mutation, "executed": []}
    if decision.mutation.get("clear_resident_llama_context"):
        receipt = evict_transient_embedding_context(reason=f"serpentina:{workflow_id}:{decision.mutation.get('error_kind')}")
        detail["executed"].append(receipt.as_dict())
    detail["status"] = "completed"
    detail["master_daemon_killed"] = False
    return detail


def current_gpu_temp_c(default: float = 58.0) -> float:
    try:
        cp = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=5,
        )
        if cp.returncode == 0 and cp.stdout.strip():
            return float(cp.stdout.strip().splitlines()[0].split(",")[0].strip())
    except Exception:
        pass
    return float(default)


def poikilotherm_throttle_gate(
    *,
    state: dict[str, Any],
    payload: dict[str, Any],
    elapsed_seconds: float,
    timeout_seconds: float = DEFAULT_HARD_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    temp_c = float(payload.get("gpu_temp_c") or state.get("gpu_temp_c") or current_gpu_temp_c())
    activity = normalized_activity(temp_c)
    friction_score = float(payload.get("friction_score") or state.get("friction_score") or 0.0)
    timeout_hit = elapsed_seconds >= timeout_seconds
    freeze = timeout_hit or friction_score >= 1.0 or activity < DEFAULT_POIKILOTHERM_FLOOR
    pace_seconds = 0.0 if not freeze else min(8.0, max(0.5, (DEFAULT_POIKILOTHERM_FLOOR - activity + max(0.0, friction_score)) * 6.0))
    dormancy = thanatosis_decide(
        delta_e=max(0.0, friction_score + (1.0 - activity)),
        k=int(state.get("attempt_count") or 0),
        dormancy_floor=DEFAULT_THANATOSIS_FLOOR,
        seed=state.get("workflow_id") or payload.get("workflow_id") or "lucidota",
    )
    return {
        "schema": "lucidota.poikilotherm_throttle_gate.v1",
        "gpu_temp_c": temp_c,
        "normalized_activity": round(activity, 6),
        "friction_score": friction_score,
        "timeout_hit": timeout_hit,
        "freeze_context": bool(freeze or dormancy.dormant),
        "pace_seconds": round(pace_seconds, 3),
        "dormancy": asdict(dormancy),
        "needle_swarm_throttle_tok_per_sec": 7200,
        "outbound_state": "draft_only",
    }


def run_chaotic_sprint_best_effort() -> dict[str, Any]:
    try:
        engine = ChaoticOmniEngine()
        engine.verify_environment()
        synthetic_stream = [
            {"voronoi_cell_id": "runtime_chaos", "epistemic_flag": "FACT"},
            {"voronoi_cell_id": "runtime_chaos", "epistemic_flag": "POSSIBLE"},
            {"voronoi_cell_id": "runtime_chaos", "epistemic_flag": "BULLSHIT"},
        ]
        engine.run_sprint_cycle(synthetic_stream)
        return {"ok": True, "status": "completed"}
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


class UnifiedAbsurdEngine:
    def __init__(self, ctx: RuntimeContext):
        self.ctx = ctx

    def init(self) -> None:
        ensure_state_schema(self.ctx.state_dsn)
        ensure_storage_schema(self.ctx.storage_dsn)

    def submit(self, source_path: Path, payload_extra: dict[str, Any] | None = None, priority: int = 100) -> str:
        self.init()
        payload = dict(payload_extra or {})
        payload["source_path"] = str(source_path)
        payload.setdefault("application_state", "draft_only")
        payload.setdefault("draft_only", True)
        fingerprint = build_input_fingerprint(source_path, payload)
        payload.update(fingerprint)
        with psycopg.connect(self.ctx.state_dsn) as conn:
            existing = conn.execute(
                """
                SELECT workflow_id::text, status
                FROM lucidota_control.absurd_workflow
                WHERE workflow_name=%s
                  AND payload->>'input_sha256'=%s
                  AND status IN ('pending','running','completed')
                ORDER BY completed_at DESC NULLS LAST, updated_at DESC
                LIMIT 1
                """,
                (WORKFLOW_NAME, payload["input_sha256"]),
            ).fetchone()
            if existing:
                workflow_id = str(existing[0])
                emit_frame(
                    "ABSURD_SUBMIT_IDEMPOTENT_HIT",
                    {
                        "workflow_id": workflow_id,
                        "existing_status": str(existing[1]),
                        "source_path": str(source_path),
                        "workflow_name": WORKFLOW_NAME,
                        "input_sha256": payload["input_sha256"],
                        "source_sha256": payload["source_sha256"],
                    },
                )
                return workflow_id
            row = conn.execute(
                """
                INSERT INTO lucidota_control.absurd_workflow(
                    workflow_name, status, payload, state, priority, max_attempts
                )
                VALUES (%s, 'pending', %s::jsonb, %s::jsonb, %s, %s)
                RETURNING workflow_id::text
                """,
                (
                    WORKFLOW_NAME,
                    jdump(payload),
                    jdump(
                        {
                            "step": "cas_lock",
                            "created_by": WORKER_SOURCE,
                            "created_at": now_z(),
                            "input_sha256": payload["input_sha256"],
                            "source_sha256": payload["source_sha256"],
                        }
                    ),
                    priority,
                    self.ctx.max_failures,
                ),
            ).fetchone()
            conn.commit()
        workflow_id = str(row[0])
        emit_frame(
            "ABSURD_SUBMIT",
            {
                "workflow_id": workflow_id,
                "source_path": str(source_path),
                "workflow_name": WORKFLOW_NAME,
                "input_sha256": payload["input_sha256"],
                "source_sha256": payload["source_sha256"],
            },
        )
        return workflow_id

    def reset_stalled(self, *, all_nonterminal: bool = False, include_terminal: bool = False) -> int:
        self.init()
        with psycopg.connect(self.ctx.state_dsn) as conn:
            if include_terminal:
                row = conn.execute(
                    """
                    UPDATE lucidota_control.absurd_workflow
                    SET status='pending', error_log=NULL, locked_by=NULL, locked_at=NULL,
                        lease_expires_at=NULL, completed_at=NULL, updated_at=now()
                    RETURNING workflow_id
                    """
                ).fetchall()
            elif all_nonterminal:
                row = conn.execute(
                    """
                    UPDATE lucidota_control.absurd_workflow
                    SET status='pending', error_log=NULL, locked_by=NULL, locked_at=NULL,
                        lease_expires_at=NULL, updated_at=now()
                    WHERE status IN ('pending','running','failed')
                    RETURNING workflow_id
                    """
                ).fetchall()
            else:
                row = conn.execute(
                    """
                    UPDATE lucidota_control.absurd_workflow
                    SET status='pending', error_log=NULL, locked_by=NULL, locked_at=NULL,
                        lease_expires_at=NULL, updated_at=now()
                    WHERE status='running' AND lease_expires_at IS NOT NULL AND lease_expires_at < now()
                    RETURNING workflow_id
                    """
                ).fetchall()
            conn.commit()
        count = len(row)
        emit_frame("ABSURD_RESET_STALLED", {"rows_reset": count, "all_nonterminal": all_nonterminal, "include_terminal": include_terminal})
        return count

    def requeue_expired_leases(self) -> int:
        with psycopg.connect(self.ctx.state_dsn) as conn:
            rows = conn.execute(
                """
                UPDATE lucidota_control.absurd_workflow
                SET status = CASE WHEN failure_count >= max_attempts THEN 'dead_lettered' ELSE 'pending' END,
                    error_log = COALESCE(error_log, 'lease expired before worker finish'),
                    locked_by = NULL,
                    locked_at = NULL,
                    lease_expires_at = NULL,
                    updated_at = now()
                WHERE status='running' AND lease_expires_at IS NOT NULL AND lease_expires_at < now()
                RETURNING workflow_id::text, status
                """
            ).fetchall()
            conn.commit()
        return len(rows)

    def claim_next(self) -> dict[str, Any] | None:
        with psycopg.connect(self.ctx.state_dsn, row_factory=dict_row) as conn:
            row = conn.execute(
                """
                WITH candidate AS (
                    SELECT workflow_id
                    FROM lucidota_control.absurd_workflow
                    WHERE status='pending'
                    ORDER BY priority ASC, created_at ASC
                    LIMIT 1
                    FOR UPDATE SKIP LOCKED
                )
                UPDATE lucidota_control.absurd_workflow w
                SET status='running',
                    locked_by=%s,
                    locked_at=now(),
                    lease_expires_at=now() + (%s::text || ' seconds')::interval,
                    attempt_count=attempt_count + 1,
                    error_log=NULL,
                    updated_at=now()
                FROM candidate
                WHERE w.workflow_id = candidate.workflow_id
                RETURNING w.workflow_id::text,
                          w.workflow_name,
                          w.status,
                          w.state,
                          w.payload,
                          w.attempt_count,
                          w.failure_count,
                          w.max_attempts,
                          w.created_at::text
                """,
                (self.ctx.worker_id, self.ctx.lease_seconds),
            ).fetchone()
            conn.commit()
            return dict(row) if row else None

    def finish(
        self,
        workflow_id: str,
        *,
        status: str,
        state: dict[str, Any],
        error_log: str | None = None,
        increment_failure: bool = False,
        last_comment_uuid: str | None = None,
    ) -> None:
        if status == "running":
            persisted = "pending"
            state = dict(state or {})
            state["_absurd_last_requested_status"] = "running"
        else:
            persisted = status
        if persisted not in {"pending", "completed", "failed", "dead_lettered", "cancelled"}:
            raise ValueError(f"invalid finish status {status!r}")

        with psycopg.connect(self.ctx.state_dsn) as conn:
            conn.execute(
                """
                UPDATE lucidota_control.absurd_workflow
                SET status=%s,
                    state=%s::jsonb,
                    error_log=%s,
                    failure_count=failure_count + CASE WHEN %s THEN 1 ELSE 0 END,
                    locked_by=NULL,
                    locked_at=NULL,
                    lease_expires_at=NULL,
                    completed_at=CASE WHEN %s IN ('completed','dead_lettered','cancelled') THEN now() ELSE completed_at END,
                    last_comment_uuid=COALESCE(%s::uuid, last_comment_uuid),
                    updated_at=now()
                WHERE workflow_id=%s::uuid
                """,
                (persisted, jdump(state), error_log, increment_failure, persisted, last_comment_uuid, workflow_id),
            )
            conn.commit()

    def process_one(self) -> bool:
        self.requeue_expired_leases()
        row = self.claim_next()
        if not row:
            return False

        workflow_id = row["workflow_id"]
        workflow_name = row["workflow_name"]
        state = dict(row.get("state") or {})
        payload = dict(row.get("payload") or {})
        failure_count = int(row.get("failure_count") or 0)
        max_failures = int(row.get("max_attempts") or self.ctx.max_failures or DEFAULT_MAX_FAILURES)
        source_path_for_logs = str(payload.get("source_path") or state.get("source_path_absolute") or "")
        digest_for_logs = state.get("digest")

        emit_frame(
            "ABSURD_CLAIM",
            {
                "workflow_id": workflow_id,
                "workflow_name": workflow_name,
                "step": state.get("step", "cas_lock"),
                "attempt_count": row.get("attempt_count"),
                "failure_count": failure_count,
                "max_failures": max_failures,
            },
        )
        log_state_event(self.ctx.state_dsn, workflow_id=workflow_id, phase="claim", status="running", detail={"step": state.get("step")})
        throttle_gate = poikilotherm_throttle_gate(state={**state, "workflow_id": workflow_id, "attempt_count": row.get("attempt_count")}, payload=payload, elapsed_seconds=0.0)
        state["poikilotherm_gate"] = throttle_gate
        emit_frame("ABSURD_POIKILOTHERM_GATE", throttle_gate)
        if throttle_gate.get("freeze_context"):
            time.sleep(float(throttle_gate.get("pace_seconds") or 0.5))

        if workflow_name != WORKFLOW_NAME:
            err = f"No unified ingest handler for workflow_name={workflow_name!r}"
            comment = log_comment_primitive(
                self.ctx.storage_dsn,
                source_path=source_path_for_logs,
                digest=digest_for_logs,
                note=err,
                detail={"workflow_id": workflow_id, "status": "dead_lettered", "reason": "unknown_handler"},
                resolved=False,
                evidence_ref="ABSURD_UNKNOWN_HANDLER",
            )
            state["_last_error"] = err
            self.finish(workflow_id, status="dead_lettered", state=state, error_log=err, increment_failure=True, last_comment_uuid=comment)
            log_state_event(self.ctx.state_dsn, workflow_id=workflow_id, phase="unknown_handler_dead_letter", status="failed", detail={"error": err, "epistemic_certainty": comment_prior(evidence_refs=["ABSURD_UNKNOWN_HANDLER"]).as_dict()})
            return True

        try:
            started = time.perf_counter()
            requested_status, new_state, comment_uuid = handle_unified_ingest(payload, state, self.ctx, workflow_id)
            elapsed = time.perf_counter() - started
            post_gate = poikilotherm_throttle_gate(
                state={**new_state, "workflow_id": workflow_id, "attempt_count": row.get("attempt_count")},
                payload=payload,
                elapsed_seconds=elapsed,
            )
            new_state["poikilotherm_gate"] = post_gate
            emit_frame("ABSURD_POIKILOTHERM_POST_GATE", {**post_gate, "elapsed_seconds": round(elapsed, 3)})
            if post_gate.get("freeze_context"):
                time.sleep(float(post_gate.get("pace_seconds") or 0.5))
            self.finish(workflow_id, status=requested_status, state=new_state, last_comment_uuid=comment_uuid)
            log_state_event(self.ctx.state_dsn, workflow_id=workflow_id, phase="finish", status=requested_status, detail={"step": new_state.get("step")})
            maybe_payload_bytewax_tick(self.ctx, payload, workflow_id, phase=str(new_state.get("step") or "finish"))
            chaotic = run_chaotic_sprint_best_effort()
            new_state["chaotic_omni_front"] = chaotic
            log_state_event(self.ctx.state_dsn, workflow_id=workflow_id, phase="chaotic_omni_front", status="succeeded" if chaotic.get("ok") else "failed", detail=chaotic)
            emit_frame("CHAOTIC_OMNI_FRONT", chaotic)
            emit_frame("ABSURD_FINISH", {"workflow_id": workflow_id, "requested_status": requested_status, "persisted_status": "pending" if requested_status == "running" else requested_status, "step": new_state.get("step")})
            return True
        except Exception as exc:
            error_msg = f"{type(exc).__name__}: {exc}"
            trace = traceback.format_exc()
            decision = serpentina_decide(
                failure_count=failure_count + 1,
                max_failures=max_failures,
                error_msg=error_msg,
                state=state,
                exc=exc,
            )
            mutated_state = apply_serpentina_state(state, decision, error_msg)
            mutation_receipt = execute_serpentina_mutation(decision, workflow_id=workflow_id, reason=error_msg)
            mutated_state["serpentina_mutation_receipt"] = mutation_receipt
            source_path_for_logs = source_path_for_logs or str(mutated_state.get("source_path_absolute") or "")
            digest_for_logs = digest_for_logs or mutated_state.get("digest")
            note = f"Absurd workflow exception. {decision.reason}. {error_msg}"
            comment = log_comment_primitive(
                self.ctx.storage_dsn,
                source_path=source_path_for_logs,
                digest=digest_for_logs,
                note=note,
                detail={
                    "workflow_id": workflow_id,
                    "traceback": trace[-6000:],
                    "serpentina_decision": asdict(decision),
                    "serpentina_mutation_receipt": mutation_receipt,
                    "status": "pending" if decision.worth_retry else "dead_lettered",
                    "epistemic_certainty": comment_prior(evidence_refs=["ABSURD_EXCEPTION_SERPENTINA"], rationale="Exception traceback and Serpentina recovery decision are operational telemetry, not semantic proof.").as_dict(),
                },
                resolved=not decision.worth_retry,
                evidence_ref="ABSURD_EXCEPTION_SERPENTINA",
            )
            emit_frame("SERPENTINA_RECOVERY_DECISION", {"workflow_id": workflow_id, **asdict(decision), "error": error_msg})
            chaotic = run_chaotic_sprint_best_effort()
            emit_frame("CHAOTIC_OMNI_FRONT", chaotic)
            if decision.worth_retry:
                self.finish(
                    workflow_id,
                    status="pending",
                    state=mutated_state,
                    error_log=note,
                    increment_failure=True,
                    last_comment_uuid=comment,
                )
                log_state_event(self.ctx.state_dsn, workflow_id=workflow_id, phase="serpentina_retry", status="queued", detail={"error": error_msg, "serpentina_decision": asdict(decision), "serpentina_mutation_receipt": mutation_receipt})
                time.sleep(float(decision.mutation.get("retry_backoff_seconds") or 1))
            else:
                self.finish(
                    workflow_id,
                    status="dead_lettered",
                    state=mutated_state,
                    error_log=note,
                    increment_failure=True,
                    last_comment_uuid=comment,
                )
                log_state_event(self.ctx.state_dsn, workflow_id=workflow_id, phase="serpentina_dead_letter", status="failed", detail={"error": error_msg, "serpentina_decision": asdict(decision), "serpentina_mutation_receipt": mutation_receipt})
            return True

    def work_loop(self, *, limit: int = 0, idle_sleep: float = DEFAULT_IDLE_SLEEP_SECONDS) -> int:
        self.init()
        emit_frame(
            "ABSURD_CONTINUOUS_DAEMON_ONLINE",
            {
                "worker_id": self.ctx.worker_id,
                "workflow_name": WORKFLOW_NAME,
                "state_dsn": self.ctx.state_dsn,
                "storage_dsn": self.ctx.storage_dsn,
                "safety_fence": "draft_only; no email/public filing/publish send path exists in this worker",
                "limit": limit,
            },
        )
        processed = 0
        inline_blender_every = float(os.environ.get("LUCIDOTA_ABSURD_INLINE_BYTEWAX_TICK_SECONDS") or 0)
        last_blender_tick = 0.0
        while True:
            try:
                did_work = self.process_one()
                if did_work:
                    processed += 1
                    if limit and processed >= limit:
                        return processed
                    continue
                if limit:
                    return processed
                if inline_blender_every > 0 and time.time() - last_blender_tick >= inline_blender_every:
                    report = run_bytewax_tick_best_effort(limit=DEFAULT_BYTEWAX_TICK_LIMIT, include_activitywatch=False, timeout=30)
                    last_blender_tick = time.time()
                    emit_frame("ABSURD_INLINE_BYTEWAX_IDLE_TICK", report)
                time.sleep(idle_sleep)
            except KeyboardInterrupt:
                emit_frame("ABSURD_OPERATOR_INTERRUPT", {"processed": processed})
                return processed
            except Exception as exc:
                emit_frame("ABSURD_MASTER_LOOP_SURVIVED_EXCEPTION", {"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()[-4000:]})
                time.sleep(max(idle_sleep, 5.0))


def handle_unified_ingest(
    payload: dict[str, Any],
    state: dict[str, Any],
    ctx: RuntimeContext,
    workflow_id: str,
) -> tuple[str, dict[str, Any], str | None]:
    step = state.get("step") or "cas_lock"
    if step == "cas_lock":
        return step_cas_lock(payload, state, ctx, workflow_id)
    if step in {"indy_reads_synthesis", "stream_text", "amalgamated_text_surface"}:
        return step_indy_reads_synthesis(payload, state, ctx, workflow_id)
    if step in {"draft_only_gate", "outbound_gate"}:
        return step_draft_only_gate(payload, state, ctx, workflow_id)
    if step == "completed":
        return "completed", state, None
    raise ValueError(f"unknown unified ingest step: {step!r}")


def step_cas_lock(
    payload: dict[str, Any],
    state: dict[str, Any],
    ctx: RuntimeContext,
    workflow_id: str,
) -> tuple[str, dict[str, Any], str | None]:
    source_path = resolve_source_path(payload.get("source_path"))
    if not source_path.exists():
        raise FileNotFoundError(f"Investigative target resource missing from drive: {source_path}")
    if not source_path.is_file():
        raise ValueError(f"source_path is not a file: {source_path}")

    digest, byte_size = sha256_file_with_progress(source_path)
    cas_path = lock_into_cas(source_path, digest)
    mime = mimetypes.guess_type(str(source_path))[0] or "application/octet-stream"
    fs_mtime = datetime.fromtimestamp(source_path.stat().st_mtime, tz=timezone.utc).isoformat().replace("+00:00", "Z")
    epistemic = filesystem_observation(sha256=digest, path=str(source_path), mtime_utc=fs_mtime).as_dict()

    with psycopg.connect(ctx.storage_dsn) as conn:
        packet_uuid = insert_staging_packet(
            conn,
            source_id=digest,
            parser_name="subtle_knife_cas_lock",
            proposed_term="EVIDENCE",
            raw_anchor=str(source_path),
            claim="file_cas_lock",
            proposed_item={
                "entity_type": "file_artifact",
                "file_path": str(source_path),
                "cas_path": rel(cas_path),
                "sha256": digest,
                "mime_type": mime,
                "byte_size": byte_size,
                "mtime_utc": fs_mtime,
                "workflow_id": workflow_id,
                "internal_visibility": "unredacted_local_vault",
                "epistemic_certainty": epistemic,
                "epistemic_flag": epistemic["label"],
            },
            status="pending",
            confidence_bps=50,
        )
        conn.execute(
            """
            INSERT INTO lucidota_go.graph_promotion_evidence_resolution(
                evidence_ref, ref_kind, resolved, resolver, source_table,
                source_uuid, source_path, detail
            )
            VALUES (%s, 'filesystem_mtime', true, 'unified_absurd_ingest_worker',
                    'staging_packet', %s::uuid, %s, %s::jsonb)
            """,
            (fs_mtime, packet_uuid, str(source_path), jdump({"sha256": digest, "trust_weight": 1.0, "workflow_id": workflow_id, "epistemic_certainty": epistemic})),
        )
        conn.commit()

    new_state = dict(state)
    new_state.update(
        {
            "step": "indy_reads_synthesis",
            "source_path_absolute": str(source_path),
            "source_path_repo_relative": rel(source_path),
            "digest": digest,
            "cas_path": rel(cas_path),
            "mime_type": mime,
            "byte_size": byte_size,
            "mtime_utc": fs_mtime,
            "cas_packet_uuid": packet_uuid,
            "cas_epistemic_certainty": epistemic,
            "stream_max_bytes": int(new_state.get("stream_max_bytes") or payload.get("max_stream_bytes") or DEFAULT_STREAM_BYTES),
            "max_pdf_pages": int(new_state.get("max_pdf_pages") or payload.get("max_pdf_pages") or 10),
        }
    )
    comment = log_comment_primitive(
        ctx.storage_dsn,
        source_path=str(source_path),
        digest=digest,
        note="CAS lock complete; raw local evidence preserved without internal redaction.",
        detail={"workflow_id": workflow_id, "cas_path": rel(cas_path), "packet_uuid": packet_uuid, "mime_type": mime, "byte_size": byte_size, "epistemic_certainty": epistemic},
        evidence_ref="ABSURD_CAS_LOCK_COMPLETE",
    )
    event_detail = {"workflow_id": workflow_id, "sha256": digest, "cas_path": rel(cas_path), "packet_uuid": packet_uuid, "epistemic_certainty": epistemic}
    log_state_event(ctx.state_dsn, workflow_id=workflow_id, phase="cas_lock_complete", status="succeeded", detail=event_detail)
    emit_frame("CAS_LOCK_COMPLETE", event_detail)
    return "running", new_state, comment


def step_indy_reads_synthesis(
    payload: dict[str, Any],
    state: dict[str, Any],
    ctx: RuntimeContext,
    workflow_id: str,
) -> tuple[str, dict[str, Any], str | None]:
    source_path = Path(state.get("source_path_absolute") or resolve_source_path(payload.get("source_path")))
    digest = state.get("digest")
    if not digest:
        digest, _ = sha256_file_with_progress(source_path)

    text, extract_meta = extract_text_surface(source_path, state, payload)
    analysis = analyze_text_signals(text, source_path)
    analysis["compressed_activity"] = staple_activity({"source_path": str(source_path)}, payload, state)["compressed_activity"]
    analysis["language_membrane"] = route_inbound_text(text[:4000], enclave_id=workflow_id)
    analysis["percyphon"] = procedural_entity_generator(
        villagers=list(payload.get("villagers") or []),
        psyche_wrath_velocity=float(payload.get("psyche_wrath_velocity") or 0.0),
        psyche_forensic_shield_ratio=float(payload.get("psyche_forensic_shield_ratio") or 0.0),
    )
    prompt_injection = detect_prompt_injection(text)
    epistemic = parser_extraction(
        sha256=str(digest),
        extract_method=str(extract_meta.get("extract_method") or "unknown"),
        injection_detected=bool(prompt_injection.get("detected")),
    ).as_dict()
    rete_bandit = absurd_rete_bandit_decision(
        payload=payload,
        state=state,
        workflow_id=workflow_id,
        source_path=source_path,
        text=text,
        analysis=analysis,
        epistemic=epistemic,
        prompt_injection=prompt_injection,
    )
    lora_plan = plan_lora_preemption(payload, {**state, "digest": digest, "source_path_absolute": str(source_path)})
    neo_context: dict[str, Any] | None = None
    if payload.get("vram_scheduler") or payload.get("adapter_id") or payload.get("lora_adapter_id") or lora_plan.get("adapter_candidates"):
        neo_context = neo_knows_kung_fu_context(payload, {**state, "digest": digest, "source_path_absolute": str(source_path)})
        lora_plan = dict(neo_context.get("plan") or lora_plan)
        try:
            with psycopg.connect(ctx.state_dsn) as conn:
                decision_id = persist_governor_decision(conn, lora_plan)
                if decision_id:
                    lora_plan["decision_id"] = decision_id
                    if neo_context:
                        neo_context["decision_id"] = decision_id
                conn.commit()
        except Exception as exc:
            lora_plan["persist_warning"] = f"{type(exc).__name__}: {exc}"
    if payload.get("require_vram_allow") and lora_plan.get("decision") != "allow":
        raise TerminalPolicyError(f"VRAM scheduler decision blocked required loadout: {lora_plan.get('decision')}")
    analysis["epistemic_certainty"] = epistemic
    analysis["epistemic_flag"] = epistemic["label"]
    analysis["prompt_injection"] = prompt_injection
    analysis["immunization_policy"] = "untrusted_source_text_is_data_only_never_instructions"
    analysis["lora_preemption_plan"] = lora_plan
    analysis["neo_knows_kung_fu_context"] = neo_context
    analysis["rete_bandit_decision"] = rete_bandit
    frame_chars = int(payload.get("frame_chars") or DEFAULT_FRAME_CHARS)
    display_text = neutralize_for_display(text, prompt_injection)
    woven_surface = weave_output(
        deterministic_template=f"workflow={workflow_id}\nsource={source_path}\nterms={analysis['terms'][:8]}",
        rag_quotes=[{"doc_id": str(source_path.name), "quote": display_text[:400], "score": 1.0}],
        deepseek_synthesis=display_text[:1200],
        fairyfuse_context={"workflow_id": workflow_id, "source_path": str(source_path)},
    )

    emit_frame(
        "AMALGAMATED_TEXT_SURFACE_HEADER",
        {
            "workflow_id": workflow_id,
            "cooperator_identity": "INDY_READs",
            "source_path": str(source_path),
            "sha256": digest,
            "extract_meta": extract_meta,
            "go_route": analysis["route"],
            "terms": analysis["terms"],
            "epistemic_certainty": epistemic,
            "prompt_injection": {k: v for k, v in prompt_injection.items() if k != "findings"} | {"findings": prompt_injection.get("findings", [])[:5]},
            "lora_preemption_decision": lora_plan.get("decision"),
            "neo_knows_kung_fu": bool(neo_context),
            "rete_bandit_selected_algorithm": rete_bandit.get("selected_algorithm"),
            "rete_bandit_selected_engine": rete_bandit.get("selected_engine"),
            "rete_bandit_reward": rete_bandit.get("reward"),
            "language_membrane": analysis["language_membrane"],
            "percyphon": {"slot_count": analysis["percyphon"]["slot_count"], "ternary_offset": analysis["percyphon"]["ternary_offset"]},
            "woven_surface": woven_surface["smoothing"]["status"],
            "internal_redaction": False,
            "display_immunization": "data_only_line_markers; raw text hash preserved separately",
        },
    )
    if display_text:
        for frame_no, frame in text_frames(display_text, frame_chars):
            emit_frame(
                "AMALGAMATED_TEXT_SURFACE_FRAME",
                {"workflow_id": workflow_id, "frame": frame_no, "chars": len(frame), "sha256": digest, "untrusted_source_text": True, "immunized_for_display": True},
                frame,
            )
    else:
        emit_frame("AMALGAMATED_TEXT_SURFACE_EMPTY", {"workflow_id": workflow_id, "sha256": digest, "extract_meta": extract_meta})

    with psycopg.connect(ctx.storage_dsn) as conn:
        packet_uuid = insert_staging_packet(
            conn,
            source_id=digest,
            parser_name="indy_reads_amalgamated_text_surface",
            proposed_term="COMMENT",
            raw_anchor=str(source_path),
            claim="indy_reads_synthesis",
            proposed_item={
                "workflow_id": workflow_id,
                "source_path": str(source_path),
                "sha256": digest,
                "analysis": analysis,
                "extract_meta": extract_meta,
                "text_sample": text[:20000],
                "display_sample": display_text[:20000],
                "internal_visibility": "unredacted_local_vault",
                "untrusted_source_text": True,
                "epistemic_certainty": epistemic,
                "epistemic_flag": epistemic["label"],
            },
            status="comment",
            confidence_bps=50,
        )
        conn.commit()

    new_state = dict(state)
    new_state.update(
        {
            "step": "draft_only_gate",
            "indy_reads_packet_uuid": packet_uuid,
            "text_sha256": analysis["text_sha256"],
            "streamed_chars": analysis["character_count"],
            "streamed_bytes": analysis["byte_count"],
            "stream_truncated": bool(extract_meta.get("stream_truncated")),
            "extract_meta": extract_meta,
            "go_terms": analysis["terms"],
            "go_route": analysis["route"],
            "indy_epistemic_certainty": epistemic,
            "prompt_injection": prompt_injection,
            "lora_preemption_plan": lora_plan,
            "neo_knows_kung_fu_context": neo_context,
            "rete_bandit_decision": rete_bandit,
        }
    )
    comment = log_comment_primitive(
        ctx.storage_dsn,
        source_path=str(source_path),
        digest=digest,
        note="INDY_READs synthesis pass complete; inline Amalgamated Text Surface streamed and staged as COMMENT.",
        detail={"workflow_id": workflow_id, "packet_uuid": packet_uuid, "analysis": analysis, "extract_meta": extract_meta, "epistemic_certainty": epistemic},
        evidence_ref="INDY_SYNTHESIS_COMPLETE",
    )
    event_detail = {"workflow_id": workflow_id, "packet_uuid": packet_uuid, "terms": analysis["terms"], "streamed_chars": analysis["character_count"], "epistemic_certainty": epistemic, "prompt_injection_detected": prompt_injection.get("detected"), "lora_preemption_decision": lora_plan.get("decision"), "neo_knows_kung_fu": bool(neo_context), "rete_bandit_selected_algorithm": rete_bandit.get("selected_algorithm"), "rete_bandit_selected_engine": rete_bandit.get("selected_engine"), "rete_bandit_reward": rete_bandit.get("reward")}
    log_state_event(ctx.state_dsn, workflow_id=workflow_id, phase="indy_synthesis_complete", status="succeeded", detail=event_detail)
    emit_frame("INDY_SYNTHESIS_COMPLETE", event_detail)
    return "running", new_state, comment


def step_draft_only_gate(
    payload: dict[str, Any],
    state: dict[str, Any],
    ctx: RuntimeContext,
    workflow_id: str,
) -> tuple[str, dict[str, Any], str | None]:
    source_path = str(state.get("source_path_absolute") or payload.get("source_path") or "")
    digest = state.get("digest")
    detail = enforce_draft_only(payload)
    detail["workflow_id"] = workflow_id
    detail["source_path"] = source_path
    detail["sha256"] = digest
    epistemic = filesystem_observation(sha256=str(digest or "no-digest"), path=source_path or "no-source").as_dict()
    detail["epistemic_certainty"] = epistemic
    detail["epistemic_flag"] = epistemic["label"]

    with psycopg.connect(ctx.storage_dsn) as conn:
        conn.execute(
            """
            INSERT INTO lucidota_go.graph_promotion_evidence_resolution(
                evidence_ref, ref_kind, resolved, resolver, source_table,
                source_uuid, source_path, detail
            )
            VALUES ('OUTBOUND_SEND_GATE', 'communique_policy', true, 'security_validator',
                    'staging_packet', gen_random_uuid(), %s, %s::jsonb)
            """,
            (source_path, jdump(detail)),
        )
        conn.commit()

    new_state = dict(state)
    new_state.update({"step": "completed", "draft_only_gate": detail, "completed_at": now_z()})
    comment = log_comment_primitive(
        ctx.storage_dsn,
        source_path=source_path,
        digest=digest,
        note="Outbound vectors hard-gated to draft_only; no send, publish, or public filing action executed.",
        detail=detail,
        evidence_ref="OUTBOUND_SEND_GATE_COMMENT",
    )
    log_state_event(ctx.state_dsn, workflow_id=workflow_id, phase="draft_only_gate_complete", status="succeeded", detail=detail)
    emit_frame("DRAFT_ONLY_GATE_COMPLETE", detail)
    return "completed", new_state, comment


def status_summary(ctx: RuntimeContext) -> dict[str, Any]:
    ensure_state_schema(ctx.state_dsn)
    with psycopg.connect(ctx.state_dsn, row_factory=dict_row) as conn:
        rows = conn.execute(
            """
            SELECT workflow_name, status, count(*) AS n,
                   min(created_at)::text AS oldest_created_at,
                   max(updated_at)::text AS newest_updated_at
            FROM lucidota_control.absurd_workflow
            GROUP BY workflow_name, status
            ORDER BY workflow_name, status
            """
        ).fetchall()
    return {"schema": "lucidota.unified_absurd_ingest_worker.status.v1", "generated_at": now_z(), "rows": [dict(r) for r in rows]}


def parse_payload_json(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    data = json.loads(value)
    if not isinstance(data, dict):
        raise ValueError("--payload-json must decode to an object")
    return data


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="LUCIDOTA unified ABSURD ingest worker")
    p.add_argument("action", nargs="?", default="work", choices=["init", "verify", "submit", "work", "work-once", "reset-stalled", "status"])
    p.add_argument("target", nargs="?", help="source file for submit")
    p.add_argument("--source-path", help="source file for submit")
    p.add_argument("--payload-json", help="extra JSON object merged into submit payload")
    p.add_argument("--priority", type=int, default=100)
    p.add_argument("--worker-id", default=f"unified-absurd-{os.getpid()}")
    p.add_argument("--lease-seconds", type=int, default=DEFAULT_LEASE_SECONDS)
    p.add_argument("--max-failures", type=int, default=DEFAULT_MAX_FAILURES)
    p.add_argument("--limit", type=int, default=0, help="for work: 0 means forever; positive exits after N processed rows")
    p.add_argument("--idle-sleep", type=float, default=DEFAULT_IDLE_SLEEP_SECONDS)
    p.add_argument("--all-nonterminal", action="store_true", help="reset pending/running/failed rows to pending")
    p.add_argument("--include-terminal", action="store_true", help="reset even completed/dead_lettered/cancelled rows; dangerous")
    p.add_argument("--strict-active-ontology-file", action="store_true", help="treat BOOKS/ACTIVE_ONTOLOGY mismatch as fatal")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    ctx = RuntimeContext(
        state_dsn=state_db_url(),
        storage_dsn=storage_db_url(),
        worker_id=args.worker_id,
        lease_seconds=args.lease_seconds,
        max_failures=args.max_failures,
    )
    engine = UnifiedAbsurdEngine(ctx)

    if args.action == "init":
        engine.init()
        report = verify_runtime(strict_active_file=args.strict_active_ontology_file)
        path = write_report("unified_absurd_init", report)
        print("REPORT_PATH=" + rel(path))
        print("UNIFIED_ABSURD_INIT=OK" if report["ok"] else "UNIFIED_ABSURD_INIT=BLOCKED")
        return 0 if report["ok"] else 4

    if args.action == "verify":
        report = verify_runtime(strict_active_file=args.strict_active_ontology_file)
        path = write_report("unified_absurd_verify", report)
        print("REPORT_PATH=" + rel(path))
        print("UNIFIED_ABSURD_VERIFY=OK" if report["ok"] else "UNIFIED_ABSURD_VERIFY=BLOCKED")
        return 0 if report["ok"] else 4

    if args.action == "submit":
        source = args.source_path or args.target
        if not source:
            raise SystemExit("submit requires --source-path or positional target")
        payload = parse_payload_json(args.payload_json)
        workflow_id = engine.submit(resolve_source_path(source), payload_extra=payload, priority=args.priority)
        print(f"WORKFLOW_ID={workflow_id}")
        return 0

    if args.action == "work-once":
        engine.init()
        did = engine.process_one()
        print("UNIFIED_ABSURD_WORK_ONCE=" + ("processed" if did else "empty"))
        return 0

    if args.action == "work":
        processed = engine.work_loop(limit=args.limit, idle_sleep=args.idle_sleep)
        print(f"UNIFIED_ABSURD_WORK_PROCESSED={processed}")
        return 0

    if args.action == "reset-stalled":
        count = engine.reset_stalled(all_nonterminal=args.all_nonterminal, include_terminal=args.include_terminal)
        print(f"UNIFIED_ABSURD_RESET_ROWS={count}")
        return 0

    if args.action == "status":
        payload = status_summary(ctx)
        path = write_report("unified_absurd_status", payload)
        print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
        print("REPORT_PATH=" + rel(path))
        return 0

    raise AssertionError(args.action)


if __name__ == "__main__":
    raise SystemExit(main())
