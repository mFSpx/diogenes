#!/usr/bin/env python3
"""RFC-CHRONO-001 lane split, projection, and graph-promotion gate.

This is an Absurd-compatible durable worker step: it preserves temporal_claim as
the append-only ledger, writes only derived classification/projection/candidate
rows, and refuses to promote weak mtime chronology as case-event truth.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psycopg
from psycopg.rows import dict_row

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = ROOT / "06_SCHEMA" / "109_chrono_lane_split_projection_gate.sql"
OUT_DIR = ROOT / "05_OUTPUTS" / "chrono_ledger"

LANE_CASE_EVENT = "LANE_CASE_EVENT"
LANE_ARTIFACT_CUSTODY = "LANE_ARTIFACT_CUSTODY"
LANE_SYSTEM_RUNTIME = "LANE_SYSTEM_RUNTIME"

WEAK_MTIME_SOURCES = {"filesystem_mtime_db_occurrence", "mtime_snapshot_v1"}
TRUST_ORDER = {
    "comm_ingress_time": 100,
    "embedded_json_log_time": 90,
    "filename_strict_iso": 80,
    "filename_date": 70,
    "markdown_header_date": 60,
    "filesystem_mtime_db_occurrence": 50,
    "mtime_snapshot_v1": 40,
    "invalid_fixture": 0,
}
RUNTIME_PATTERNS = [
    r"(^|/)04_RUNTIME(/|$)", r"(^|/)05_OUTPUTS(/|$)", r"(^|/)tests?(/|$)",
    r"runtime/audit", r"classification_run", r"abba_applied", r"abba_dryrun",
    r"orchestrate_", r"runtime/logs", r"tickletrunk_scan_", r"chrono_ledger/",
    r"generated", r"report", r"receipt", r"pytest", r"__pycache__", r"\.venv/",
]
GENERATED_PATTERNS = [
    r"(^|/)05_OUTPUTS(/|$)", r"generated", r"report", r"receipt", r"snapshot", r"projection",
    r"manifest", r"audit", r"classification_run", r"abba_", r"orchestrate_",
]
CASE_EVENT_SOURCES = {
    "comm_ingress_time", "embedded_json_log_time", "official_document_date",
    "authored_document_date", "media_capture_exif_date", "filename_strict_iso",
    "filename_date", "markdown_header_date",
}
ARTIFACT_SOURCES = WEAK_MTIME_SOURCES | {
    "cas_first_seen", "archive_ingest_time", "file_copy_time", "archive_unpack_time",
    "artifact_ingest_time", "file_seen", "file_hash_time",
}
RUNTIME_SOURCE_HINTS = {
    "runtime_audit_time", "classification_run_time", "abba_applied_time", "abba_dryrun_time",
    "orchestrate_time", "test_run_time", "generated_report_time", "pipeline_event_time",
}
EVENT_MAP_CASE = {
    "comm_ingress_time": "MESSAGE_RECEIVED",
    "embedded_json_log_time": "MESSAGE_RECEIVED",
    "official_document_date": "OFFICIAL_RECORD_DATED",
    "authored_document_date": "DOCUMENT_AUTHORED",
    "media_capture_exif_date": "MEDIA_CAPTURED",
    "filename_strict_iso": "CASE_EVENT_OBSERVED",
    "filename_date": "CASE_EVENT_OBSERVED",
    "markdown_header_date": "DOCUMENT_AUTHORED",
}


@dataclass
class NormalizedClaim:
    claim_uuid: str
    file_uuid: str | None
    artifact_sha256: str | None
    source_path: str | None
    candidate_timestamp: str
    evidence_source: str
    extractor: str | None
    extractor_version: str | None
    trust_weight: float | None
    epistemic_flag: str
    invalid: bool
    chrono_lane: str
    source_family: str
    path_family: str
    is_runtime_artifact: bool
    is_generated_artifact: bool
    is_batch_candidate: bool
    weak_pair_group_id: str | None
    graph_event_type: str
    promotion_block_reason: list[str]
    classification_detail: dict[str, Any]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def rel(path: str | Path) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def db_url(args: argparse.Namespace) -> str:
    return args.database_url or os.environ.get("KORPUS_DATABASE_URL") or os.environ.get("DATABASE_URL") or "postgresql:///lucidota_storage"


def write_report(name: str, payload: dict[str, Any]) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"chrono_lane_split_projection_gate_{name}_{stamp()}.json"
    payload.setdefault("generated_at", now_iso())
    payload["report_path"] = rel(out)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"REPORT_PATH={rel(out)}")
    return out


def has_any(patterns: list[str], text: str) -> bool:
    return any(re.search(p, text, re.I) for p in patterns)


def source_family(evidence_source: str, invalid: bool) -> str:
    if invalid:
        return "invalid_fixture"
    es = evidence_source.strip().lower()
    if es in WEAK_MTIME_SOURCES:
        return "weak_mtime"
    if es in RUNTIME_SOURCE_HINTS or any(token in es for token in ("runtime", "classification", "abba", "orchestrate", "audit", "test", "generated", "pipeline")):
        return "system_runtime"
    if es in {"comm_ingress_time", "embedded_json_log_time"}:
        return es
    if es == "filename_strict_iso":
        return "filename_strict_iso"
    if es == "filename_date":
        return "filename_date"
    if es == "markdown_header_date":
        return "markdown_header_date"
    if es in {"official_document_date", "authored_document_date", "media_capture_exif_date"}:
        return es
    if es in ARTIFACT_SOURCES or any(token in es for token in ("mtime", "cas", "archive", "ingest", "copy", "unpack", "file_seen", "hash")):
        return "artifact_custody"
    return "unknown_temporal_source"


def classify_path(path: str | None) -> tuple[str, bool, bool]:
    if not path:
        return "missing_path", False, False
    p = path.replace("\\", "/")
    runtime = has_any(RUNTIME_PATTERNS, p)
    generated = has_any(GENERATED_PATTERNS, p)
    if runtime:
        return "system_runtime_artifact", True, generated
    if generated:
        return "generated_artifact", False, True
    if re.search(r"(^|/)(03_VAULT|02_RECORDS_OFFICE|KRAMPUSCHEWING)(/|$)", p):
        return "case_or_corpus_artifact", False, False
    return "source_artifact", False, False


def epistemic_flag(trust_weight: Any, invalid: bool) -> str:
    if invalid:
        return "BULLSHIT"
    try:
        tw = float(trust_weight)
    except Exception:
        return "SURE_MAYBE"
    if tw >= 0.99:
        return "FACT"
    if tw >= 0.80:
        return "PROBABLE"
    if tw >= 0.50:
        return "POSSIBLE"
    return "SURE_MAYBE"


def lane_for(sf: str, es: str, path_family: str, is_runtime: bool, is_generated: bool) -> str:
    if is_runtime or path_family == "system_runtime_artifact":
        return LANE_SYSTEM_RUNTIME
    if sf == "system_runtime":
        return LANE_SYSTEM_RUNTIME
    if sf == "weak_mtime" or sf == "artifact_custody":
        return LANE_ARTIFACT_CUSTODY
    if es in CASE_EVENT_SOURCES and not is_generated:
        return LANE_CASE_EVENT
    if is_generated:
        return LANE_SYSTEM_RUNTIME
    # Conservative default: unknown file chronology is custody, not case truth.
    return LANE_ARTIFACT_CUSTODY


def event_type_for(lane: str, evidence_source: str, source_path: str | None) -> str:
    es = evidence_source.lower()
    sp = (source_path or "").lower()
    if lane == LANE_CASE_EVENT:
        return EVENT_MAP_CASE.get(es, "CASE_EVENT_OBSERVED")
    if lane == LANE_ARTIFACT_CUSTODY:
        if es in WEAK_MTIME_SOURCES or "mtime" in es:
            return "FILE_MODIFIED"
        if "cas" in es:
            return "CAS_MATERIALIZED"
        if "unpack" in es:
            return "ARCHIVE_UNPACKED"
        if "ingest" in es:
            return "ARTIFACT_INGESTED"
        if "hash" in es:
            return "ARTIFACT_HASHED"
        return "FILE_SEEN"
    if "classification" in es or "classification_run" in sp:
        return "CLASSIFICATION_RUN"
    if "abba_applied" in es or "abba_applied" in sp:
        return "ABBA_APPLIED"
    if "abba_dryrun" in es or "abba_dryrun" in sp:
        return "ABBA_DRYRUN"
    if "orchestrate" in es or "orchestrate" in sp:
        return "ORCHESTRATION_RUN"
    if "report" in sp:
        return "REPORT_GENERATED"
    if "batch" in es or "batch" in sp:
        return "PIPELINE_BATCH_CREATED"
    return "AUDIT_RECORD_CREATED"


def weak_pair_id(file_uuid: str | None, timestamp: str, source_family_value: str) -> str | None:
    if source_family_value != "weak_mtime" or not file_uuid:
        return None
    raw = f"{file_uuid}|{timestamp}"
    return hashlib.sha256(raw.encode()).hexdigest()


def classify_claim(row: dict[str, Any]) -> NormalizedClaim:
    invalid = bool(row.get("invalid"))
    es = str(row.get("evidence_source") or "unknown")
    sf = source_family(es, invalid)
    pf, is_runtime, is_generated = classify_path(row.get("source_path"))
    lane = lane_for(sf, es.lower(), pf, is_runtime, is_generated)
    flag = epistemic_flag(row.get("trust_weight"), invalid)
    graph_event_type = event_type_for(lane, es, row.get("source_path"))
    blocks: list[str] = []
    if invalid:
        blocks.append("invalid=true")
    if flag == "BULLSHIT":
        blocks.append("epistemic_flag=BULLSHIT")
    if not row.get("source_path"):
        blocks.append("missing_source_path")
    if not row.get("file_uuid") and lane != LANE_SYSTEM_RUNTIME:
        blocks.append("missing_file_uuid")
    if sf == "weak_mtime" and lane == LANE_CASE_EVENT:
        blocks.append("weak_mtime_attempting_case_event")
    if (is_runtime or is_generated) and lane == LANE_CASE_EVENT:
        blocks.append("runtime_or_generated_artifact_attempting_case_event")
    timestamp = row["candidate_timestamp"].isoformat() if hasattr(row["candidate_timestamp"], "isoformat") else str(row["candidate_timestamp"])
    return NormalizedClaim(
        claim_uuid=str(row["claim_uuid"]),
        file_uuid=str(row["file_uuid"]) if row.get("file_uuid") else None,
        artifact_sha256=str(row["source_sha256"]) if row.get("source_sha256") else None,
        source_path=str(row["source_path"]) if row.get("source_path") else None,
        candidate_timestamp=timestamp,
        evidence_source=es,
        extractor=str(row["extractor"]) if row.get("extractor") else None,
        extractor_version=str(row["extractor_version"]) if row.get("extractor_version") else None,
        trust_weight=float(row["trust_weight"]) if row.get("trust_weight") is not None else None,
        epistemic_flag=flag,
        invalid=invalid,
        chrono_lane=lane,
        source_family=sf,
        path_family=pf,
        is_runtime_artifact=is_runtime,
        is_generated_artifact=is_generated,
        is_batch_candidate=sf == "weak_mtime",
        weak_pair_group_id=weak_pair_id(str(row["file_uuid"]) if row.get("file_uuid") else None, timestamp, sf),
        graph_event_type=graph_event_type,
        promotion_block_reason=blocks,
        classification_detail={
            "rfc": "RFC-CHRONO-001",
            "rule": "mtime_is_custody_not_case_truth; runtime_path_overrides_case_event",
            "trust_order_rank": TRUST_ORDER.get(sf, TRUST_ORDER.get(es.lower(), 10)),
        },
    )


def fetch_claims(conn: psycopg.Connection[Any], limit: int | None = None) -> list[dict[str, Any]]:
    sql = """
        SELECT claim_uuid::text, artifact_uuid::text, file_uuid::text, candidate_timestamp,
               evidence_source, trust_weight, raw_evidence, extractor, extractor_version,
               source_path, source_sha256, created_at, detail, coalesce(invalid,false) AS invalid,
               invalidation_reason
        FROM lucidota_korpus.temporal_claim
        ORDER BY created_at ASC, claim_uuid ASC
    """
    if limit:
        sql += " LIMIT %s"
        return list(conn.execute(sql, (limit,)).fetchall())
    return list(conn.execute(sql).fetchall())


def upsert_normalized(conn: psycopg.Connection[Any], claims: list[NormalizedClaim]) -> int:
    rows = [(
        c.claim_uuid, c.file_uuid, c.artifact_sha256, c.source_path, c.candidate_timestamp,
        c.evidence_source, c.extractor, c.extractor_version, c.trust_weight, c.epistemic_flag,
        c.invalid, c.chrono_lane, c.source_family, c.path_family, c.is_runtime_artifact,
        c.is_generated_artifact, c.is_batch_candidate, c.weak_pair_group_id,
        c.graph_event_type, c.promotion_block_reason, json.dumps(c.classification_detail),
    ) for c in claims]
    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO lucidota_korpus.chrono_claim_normalized(
              claim_uuid,file_uuid,artifact_sha256,source_path,candidate_timestamp,evidence_source,
              extractor,extractor_version,trust_weight,epistemic_flag,invalid,chrono_lane,
              source_family,path_family,is_runtime_artifact,is_generated_artifact,is_batch_candidate,
              weak_pair_group_id,graph_event_type,promotion_block_reason,classification_detail,normalized_at
            ) VALUES (%s::uuid,%s::uuid,%s,%s,%s::timestamptz,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb,now())
            ON CONFLICT(claim_uuid) DO UPDATE SET
              file_uuid=EXCLUDED.file_uuid, artifact_sha256=EXCLUDED.artifact_sha256,
              source_path=EXCLUDED.source_path, candidate_timestamp=EXCLUDED.candidate_timestamp,
              evidence_source=EXCLUDED.evidence_source, extractor=EXCLUDED.extractor,
              extractor_version=EXCLUDED.extractor_version, trust_weight=EXCLUDED.trust_weight,
              epistemic_flag=EXCLUDED.epistemic_flag, invalid=EXCLUDED.invalid,
              chrono_lane=EXCLUDED.chrono_lane, source_family=EXCLUDED.source_family,
              path_family=EXCLUDED.path_family, is_runtime_artifact=EXCLUDED.is_runtime_artifact,
              is_generated_artifact=EXCLUDED.is_generated_artifact, is_batch_candidate=EXCLUDED.is_batch_candidate,
              weak_pair_group_id=EXCLUDED.weak_pair_group_id, graph_event_type=EXCLUDED.graph_event_type,
              promotion_block_reason=EXCLUDED.promotion_block_reason,
              classification_detail=EXCLUDED.classification_detail, normalized_at=now()
            """,
            rows,
        )
    return len(rows)


def prefix_for(path: str | None) -> str:
    if not path:
        return "missing"
    parts = Path(path.replace("\\", "/")).parts
    if not parts:
        return path[:80]
    if len(parts) >= 2:
        return "/".join(parts[:2])
    return parts[0]


def detect_batches(claims: list[NormalizedClaim]) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str], list[NormalizedClaim]] = defaultdict(list)
    for c in claims:
        if c.is_batch_candidate and not c.invalid:
            buckets[(c.candidate_timestamp, c.chrono_lane)].append(c)
    clusters: list[dict[str, Any]] = []
    for (ts, lane), rows in buckets.items():
        file_ids = {r.file_uuid or r.artifact_sha256 or r.claim_uuid for r in rows}
        file_count = len(file_ids)
        if file_count < 25:
            continue
        clusters.append({
            "timestamp": ts,
            "lane": lane,
            "evidence_sources": sorted({r.evidence_source for r in rows}),
            "file_count": file_count,
            "path_prefixes": sorted({prefix_for(r.source_path) for r in rows})[:32],
            "sample_paths": sorted({r.source_path or "" for r in rows if r.source_path})[:20],
            "classification": "BATCH_OPERATION_TIME" if file_count < 50 else "BATCH_OPERATION_TIME_HARD_CASE_BLOCK",
            "graph_allowed_as": ["ARCHIVE_UNPACKED", "FILES_TOUCHED", "PIPELINE_BATCH_CREATED", "GENERATED_ARTIFACT_BURST"],
            "case_event_blocked": file_count >= 50,
            "detail": {"rfc": "RFC-CHRONO-001", "threshold_25": True, "threshold_50": file_count >= 50},
        })
    return clusters


def upsert_batches(conn: psycopg.Connection[Any], clusters: list[dict[str, Any]]) -> int:
    if not clusters:
        return 0
    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO lucidota_korpus.chrono_batch_cluster(
              timestamp,lane,evidence_sources,file_count,path_prefixes,sample_paths,classification,
              graph_allowed_as,case_event_blocked,detected_at,detail
            ) VALUES (%s::timestamptz,%s,%s,%s,%s,%s,%s,%s,%s,now(),%s::jsonb)
            ON CONFLICT(timestamp,lane,classification) DO UPDATE SET
              evidence_sources=EXCLUDED.evidence_sources, file_count=EXCLUDED.file_count,
              path_prefixes=EXCLUDED.path_prefixes, sample_paths=EXCLUDED.sample_paths,
              graph_allowed_as=EXCLUDED.graph_allowed_as, case_event_blocked=EXCLUDED.case_event_blocked,
              detected_at=now(), detail=EXCLUDED.detail
            """,
            [(
                c["timestamp"], c["lane"], c["evidence_sources"], c["file_count"],
                c["path_prefixes"], c["sample_paths"], c["classification"],
                c["graph_allowed_as"], c["case_event_blocked"], json.dumps(c["detail"]),
            ) for c in clusters],
        )
    return len(clusters)


def selection_rank(c: NormalizedClaim) -> tuple[int, float, str, str]:
    # Stronger internal/source-derived timestamps beat weak mtime. Timestamp itself
    # is only a deterministic tie-breaker after source strength/confidence.
    rank = TRUST_ORDER.get(c.source_family, TRUST_ORDER.get(c.evidence_source.lower(), 10))
    tw = c.trust_weight if c.trust_weight is not None else 0.0
    return (-rank, -tw, c.candidate_timestamp, c.claim_uuid)


def build_file_projection(claims: list[NormalizedClaim], clusters: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_file: dict[str, list[NormalizedClaim]] = defaultdict(list)
    cluster_keys = {(c["timestamp"], c["lane"]) for c in clusters}
    for c in claims:
        if c.file_uuid and not c.invalid:
            by_file[c.file_uuid].append(c)
    projections: list[dict[str, Any]] = []
    for file_uuid, rows in by_file.items():
        rows_sorted = sorted(rows, key=selection_rank)
        selected = rows_sorted[0]
        weak_groups = {r.weak_pair_group_id or r.claim_uuid for r in rows if r.source_family == "weak_mtime"}
        strong_rows = [r for r in rows if r.source_family != "weak_mtime"]
        timestamps = {r.candidate_timestamp for r in rows}
        has_batch = (selected.candidate_timestamp, selected.chrono_lane) in cluster_keys
        reason_bits = [
            "projection_not_archive",
            f"selected_by_source_rank:{selected.source_family}",
            f"lane:{selected.chrono_lane}",
        ]
        if weak_groups:
            reason_bits.append(f"weak_mtime_pairs_collapsed:{len(weak_groups)}")
        if strong_rows:
            reason_bits.append(f"strong_claims_present:{len(strong_rows)}")
        if has_batch:
            reason_bits.append("selected_timestamp_in_batch_cluster")
        projections.append({
            "file_uuid": file_uuid,
            "selected_claim_uuid": selected.claim_uuid,
            "selected_timestamp": selected.candidate_timestamp,
            "selected_lane": selected.chrono_lane,
            "selected_evidence_source": selected.evidence_source,
            "selected_confidence": selected.trust_weight,
            "competing_claim_count": len(rows),
            "weak_claim_count": len(weak_groups),
            "strong_claim_count": len(strong_rows),
            "has_conflict": len(timestamps) > 1,
            "has_batch_cluster": has_batch,
            "projection_reason": ";".join(reason_bits),
            "detail": {"rfc": "RFC-CHRONO-001", "candidate_claim_count": len(rows)},
        })
    return projections


def upsert_file_projection(conn: psycopg.Connection[Any], projections: list[dict[str, Any]]) -> int:
    if not projections:
        return 0
    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO lucidota_korpus.chrono_file_projection(
              file_uuid,selected_claim_uuid,selected_timestamp,selected_lane,selected_evidence_source,
              selected_confidence,competing_claim_count,weak_claim_count,strong_claim_count,
              has_conflict,has_batch_cluster,projection_reason,projection_refreshed_at,detail
            ) VALUES (%s::uuid,%s::uuid,%s::timestamptz,%s,%s,%s,%s,%s,%s,%s,%s,%s,now(),%s::jsonb)
            ON CONFLICT(file_uuid) DO UPDATE SET
              selected_claim_uuid=EXCLUDED.selected_claim_uuid, selected_timestamp=EXCLUDED.selected_timestamp,
              selected_lane=EXCLUDED.selected_lane, selected_evidence_source=EXCLUDED.selected_evidence_source,
              selected_confidence=EXCLUDED.selected_confidence, competing_claim_count=EXCLUDED.competing_claim_count,
              weak_claim_count=EXCLUDED.weak_claim_count, strong_claim_count=EXCLUDED.strong_claim_count,
              has_conflict=EXCLUDED.has_conflict, has_batch_cluster=EXCLUDED.has_batch_cluster,
              projection_reason=EXCLUDED.projection_reason, projection_refreshed_at=now(), detail=EXCLUDED.detail
            """,
            [(
                p["file_uuid"], p["selected_claim_uuid"], p["selected_timestamp"], p["selected_lane"],
                p["selected_evidence_source"], p["selected_confidence"], p["competing_claim_count"],
                p["weak_claim_count"], p["strong_claim_count"], p["has_conflict"], p["has_batch_cluster"],
                p["projection_reason"], json.dumps(p["detail"]),
            ) for p in projections],
        )
    return len(projections)


def build_candidates(claim_map: dict[str, NormalizedClaim], projections: list[dict[str, Any]], clusters: list[dict[str, Any]]) -> list[dict[str, Any]]:
    hard_case_cluster_keys = {(c["timestamp"], c["lane"]) for c in clusters if c["case_event_blocked"]}
    candidates: list[dict[str, Any]] = []
    for p in projections:
        c = claim_map[p["selected_claim_uuid"]]
        block = list(c.promotion_block_reason)
        if (c.candidate_timestamp, c.chrono_lane) in hard_case_cluster_keys and c.chrono_lane == LANE_CASE_EVENT:
            block.append("batch_cluster_attempting_case_event")
        if c.source_family == "weak_mtime" and c.chrono_lane == LANE_CASE_EVENT:
            block.append("weak_mtime_attempting_case_event")
        if c.chrono_lane == LANE_CASE_EVENT and (c.is_runtime_artifact or c.is_generated_artifact):
            block.append("generated_or_runtime_file_attempting_human_event")
        if not c.candidate_timestamp:
            block.append("candidate_timestamp_missing")
        if not c.graph_event_type:
            block.append("graph_event_type_missing")
        # Weak mtime custody candidates are allowed as FILE_MODIFIED etc.; they are not case truth.
        blocked = bool(block)
        candidates.append({
            "source_claim_uuid": c.claim_uuid,
            "file_uuid": c.file_uuid,
            "graph_event_type": c.graph_event_type,
            "chrono_lane": c.chrono_lane,
            "timestamp": c.candidate_timestamp,
            "confidence": c.trust_weight,
            "blocked": blocked,
            "block_reason": sorted(set(block)),
            "promotion_reason": "blocked_by_gate" if blocked else f"lane_qualified:{c.chrono_lane}:{c.graph_event_type};projection_not_archive",
            "detail": {
                "rfc": "RFC-CHRONO-001",
                "projection_reason": p["projection_reason"],
                "source_family": c.source_family,
                "path_family": c.path_family,
            },
        })
    return candidates


def upsert_candidates(conn: psycopg.Connection[Any], candidates: list[dict[str, Any]]) -> int:
    if not candidates:
        return 0
    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO lucidota_go.graph_promotion_candidate(
              source_claim_uuid,file_uuid,graph_event_type,chrono_lane,timestamp,confidence,
              blocked,block_reason,promotion_reason,created_at,detail
            ) VALUES (%s::uuid,%s::uuid,%s,%s,%s::timestamptz,%s,%s,%s,%s,now(),%s::jsonb)
            ON CONFLICT(source_claim_uuid) DO UPDATE SET
              file_uuid=EXCLUDED.file_uuid, graph_event_type=EXCLUDED.graph_event_type,
              chrono_lane=EXCLUDED.chrono_lane, timestamp=EXCLUDED.timestamp,
              confidence=EXCLUDED.confidence, blocked=EXCLUDED.blocked,
              block_reason=EXCLUDED.block_reason, promotion_reason=EXCLUDED.promotion_reason,
              detail=EXCLUDED.detail
            """,
            [(
                c["source_claim_uuid"], c["file_uuid"], c["graph_event_type"], c["chrono_lane"],
                c["timestamp"], c["confidence"], c["blocked"], c["block_reason"],
                c["promotion_reason"], json.dumps(c["detail"]),
            ) for c in candidates],
        )
    return len(candidates)


def validate(conn: psycopg.Connection[Any]) -> dict[str, Any]:
    def q(sql, params=()):
        row = conn.execute(sql, params).fetchone()
        if isinstance(row, dict):
            return next(iter(row.values()))
        return row[0]
    return {
        "temporal_claim_count": int(q("SELECT count(*) FROM lucidota_korpus.temporal_claim")),
        "normalized_count": int(q("SELECT count(*) FROM lucidota_korpus.chrono_claim_normalized")),
        "claims_without_lane": int(q("""
            SELECT count(*) FROM lucidota_korpus.temporal_claim tc
            LEFT JOIN lucidota_korpus.chrono_claim_normalized n ON n.claim_uuid=tc.claim_uuid
            WHERE n.claim_uuid IS NULL OR n.chrono_lane IS NULL
        """)),
        "weak_mtime_case_event_count": int(q("""
            SELECT count(*) FROM lucidota_korpus.chrono_claim_normalized
            WHERE source_family='weak_mtime' AND chrono_lane='LANE_CASE_EVENT'
        """)),
        "runtime_case_event_count": int(q("""
            SELECT count(*) FROM lucidota_korpus.chrono_claim_normalized
            WHERE (is_runtime_artifact OR is_generated_artifact) AND chrono_lane='LANE_CASE_EVENT'
        """)),
        "batch_cluster_count": int(q("SELECT count(*) FROM lucidota_korpus.chrono_batch_cluster")),
        "hard_case_block_cluster_count": int(q("SELECT count(*) FROM lucidota_korpus.chrono_batch_cluster WHERE case_event_blocked")),
        "projection_rows": int(q("SELECT count(*) FROM lucidota_korpus.chrono_file_projection")),
        "promotion_candidates": int(q("SELECT count(*) FROM lucidota_go.graph_promotion_candidate")),
        "blocked_candidates": int(q("SELECT count(*) FROM lucidota_go.graph_promotion_candidate WHERE blocked")),
        "unblocked_candidates": int(q("SELECT count(*) FROM lucidota_go.graph_promotion_candidate WHERE NOT blocked")),
        "candidate_missing_reason_count": int(q("SELECT count(*) FROM lucidota_go.graph_promotion_candidate WHERE promotion_reason='' OR promotion_reason IS NULL")),
    }


def refresh(args: argparse.Namespace) -> int:
    report: dict[str, Any] = {
        "action": "refresh_lane_split_projection_gate",
        "execute_performed": bool(args.execute),
        "db_writes_performed": False,
        "graph_writes_performed": False,
        "schema_path": rel(SCHEMA),
        "rfc": "RFC-CHRONO-001",
        "headers": ["CLAIM_LEDGER", "FILE_PROJECTION", "CASE_EVENT_TIMELINE", "ARTIFACT_CUSTODY_TIMELINE", "SYSTEM_RUNTIME_TIMELINE"],
        "projection_disclaimer": "FILE_PROJECTION is projection, not archive; temporal_claim is the archive.",
        "blockers": [],
    }
    with psycopg.connect(db_url(args), row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute(SCHEMA.read_text(encoding="utf-8"))
        claims_raw = fetch_claims(conn, args.limit)
        claims = [classify_claim(dict(row)) for row in claims_raw]
        clusters = detect_batches(claims)
        projections = build_file_projection(claims, clusters)
        claim_map = {c.claim_uuid: c for c in claims}
        candidates = build_candidates(claim_map, projections, clusters)
        lane_counts = Counter(c.chrono_lane for c in claims)
        source_family_counts = Counter(c.source_family for c in claims)
        if args.execute:
            run_uuid = conn.execute(
                "INSERT INTO lucidota_korpus.chrono_lane_refresh_run(status,report) VALUES ('running',%s::jsonb) RETURNING run_uuid::text",
                (json.dumps({"started_by": "scripts/chrono_lane_split_projection_gate.py"}),),
            ).fetchone()["run_uuid"]
            normalized = upsert_normalized(conn, claims)
            batch_count = upsert_batches(conn, clusters)
            projection_count = upsert_file_projection(conn, projections)
            candidate_count = upsert_candidates(conn, candidates)
            validation = validate(conn)
            blockers = []
            if validation["claims_without_lane"] != 0:
                blockers.append("claims_without_lane")
            if validation["weak_mtime_case_event_count"] != 0:
                blockers.append("weak_mtime_promoted_as_case_event")
            if validation["runtime_case_event_count"] != 0:
                blockers.append("runtime_or_generated_case_event_lane")
            if validation["candidate_missing_reason_count"] != 0:
                blockers.append("candidate_missing_promotion_reason")
            status = "PASS" if not blockers else "FAIL"
            report.update({
                "db_writes_performed": True,
                "run_uuid": run_uuid,
                "claims_seen": len(claims_raw),
                "claims_normalized": normalized,
                "batch_clusters_upserted": batch_count,
                "file_projection_rows_upserted": projection_count,
                "promotion_candidates_upserted": candidate_count,
                "lane_counts": dict(lane_counts),
                "source_family_counts_top": dict(source_family_counts.most_common(20)),
                "validation": validation,
                "blockers": blockers,
                "status": status,
            })
            conn.execute(
                """
                UPDATE lucidota_korpus.chrono_lane_refresh_run
                SET finished_at=now(), claims_seen=%s, claims_normalized=%s,
                    file_projection_rows=%s, batch_clusters_upserted=%s,
                    promotion_candidates_upserted=%s, status=%s, report=%s::jsonb
                WHERE run_uuid=%s::uuid
                """,
                (len(claims_raw), normalized, projection_count, batch_count, candidate_count, status, json.dumps(report, default=str), run_uuid),
            )
            conn.commit()
        else:
            report.update({
                "claims_seen": len(claims_raw),
                "would_normalize": len(claims),
                "would_upsert_batch_clusters": len(clusters),
                "would_upsert_file_projection_rows": len(projections),
                "would_upsert_promotion_candidates": len(candidates),
                "lane_counts": dict(lane_counts),
                "source_family_counts_top": dict(source_family_counts.most_common(20)),
                "sample_clusters": clusters[:5],
                "sample_projection": projections[:5],
                "status": "DRY_RUN",
            })
    write_report("execute" if args.execute else "dry_run", report)
    print("CHRONO_LANE_SPLIT_GATE=" + report["status"])
    print("CLAIMS_NORMALIZED=" + str(report.get("claims_normalized", report.get("would_normalize"))))
    print("BATCH_CLUSTERS=" + str(report.get("batch_clusters_upserted", report.get("would_upsert_batch_clusters"))))
    print("PROMOTION_CANDIDATES=" + str(report.get("promotion_candidates_upserted", report.get("would_upsert_promotion_candidates"))))
    return 0 if report["status"] in {"PASS", "DRY_RUN"} else 4


def main() -> int:
    parser = argparse.ArgumentParser(description="RFC-CHRONO-001 lane split/projection/promotion gate")
    parser.add_argument("--database-url")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    return refresh(args)


if __name__ == "__main__":
    raise SystemExit(main())
