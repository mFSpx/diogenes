#!/usr/bin/env python3
"""Chrono conservation verifier CLI.

Verifies that the current best-time projection links back to temporal_claim
rows and that low-confidence/conflicting temporal evidence remains queryable.
No temporal evidence is deleted, overwritten, or collapsed.
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psycopg
from psycopg.rows import dict_row

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "05_OUTPUTS" / "chrono_ledger"
SCHEMA = ROOT / "06_SCHEMA/071_chrono_conservation_verifier.sql"


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


def db(args: argparse.Namespace) -> str:
    return args.database_url or os.environ.get("DATABASE_URL") or "postgresql:///lucidota_storage"


def q1(cur: psycopg.Cursor[Any], sql: str, params: tuple[Any, ...] = ()) -> Any:
    cur.execute(sql, params)
    row = cur.fetchone()
    if row is None:
        return None
    return next(iter(row.values())) if isinstance(row, dict) else row[0]


def qrows(cur: psycopg.Cursor[Any], sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
    cur.execute(sql, params)
    return [dict(row) for row in cur.fetchall()]


def write_report(name: str, payload: dict[str, Any]) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"chrono_conservation_verify_{name}_{stamp()}.json"
    payload.setdefault("generated_at", now_iso())
    payload["report_path"] = rel(path)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"REPORT_PATH={rel(path)}")
    return path


def init_schema(args: argparse.Namespace) -> int:
    if args.execute:
        with psycopg.connect(db(args)) as conn:
            with conn.cursor() as cur:
                cur.execute(SCHEMA.read_text(encoding="utf-8"))
            conn.commit()
    write_report("init_schema_execute" if args.execute else "init_schema_dry_run", {
        "action": "init_schema",
        "execute_performed": bool(args.execute),
        "schema": rel(SCHEMA),
    })
    return 0


def verify(args: argparse.Namespace) -> int:
    with psycopg.connect(db(args), row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            view_exists = q1(cur, "SELECT to_regclass('lucidota_korpus.resolved_chrono_timeline_with_claim')") is not None
            if not view_exists:
                raise SystemExit("resolved_chrono_timeline_with_claim missing; run init-schema --execute")
            total_claims = int(q1(cur, "SELECT count(*) FROM lucidota_korpus.temporal_claim") or 0)
            low_confidence_claims = int(q1(cur, "SELECT count(*) FROM lucidota_korpus.temporal_claim WHERE trust_weight < 0.80") or 0)
            best_rows = int(q1(cur, "SELECT count(*) FROM lucidota_korpus.resolved_chrono_timeline_with_claim") or 0)
            distinct_claimed_files = int(q1(cur, "SELECT count(DISTINCT file_uuid) FROM lucidota_korpus.temporal_claim WHERE file_uuid IS NOT NULL") or 0)
            best_without_claim = int(q1(cur, """
                SELECT count(*)
                FROM lucidota_korpus.resolved_chrono_timeline_with_claim r
                LEFT JOIN lucidota_korpus.temporal_claim tc ON tc.claim_uuid = r.selected_claim_uuid
                WHERE tc.claim_uuid IS NULL
            """) or 0)
            low_confidence_selected = int(q1(cur, "SELECT count(*) FROM lucidota_korpus.resolved_chrono_timeline_with_claim WHERE confidence_score < 0.80") or 0)
            disputed_files = int(q1(cur, "SELECT count(*) FROM lucidota_korpus.resolved_chrono_timeline_with_claim WHERE has_conflict") or 0)
            conflicting_claims_preserved = int(q1(cur, """
                WITH conflicted AS (
                  SELECT file_uuid
                  FROM lucidota_korpus.temporal_claim
                  WHERE file_uuid IS NOT NULL
                  GROUP BY file_uuid
                  HAVING count(*) > 1
                     AND (count(DISTINCT candidate_timestamp) > 1 OR count(DISTINCT evidence_source) > 1)
                )
                SELECT count(*)
                FROM lucidota_korpus.temporal_claim tc
                JOIN conflicted c ON c.file_uuid = tc.file_uuid
            """) or 0)
            ranking_violations = int(q1(cur, """
                WITH violations AS (
                  SELECT r.file_uuid
                  FROM lucidota_korpus.resolved_chrono_timeline_with_claim r
                  JOIN lucidota_korpus.temporal_claim tc ON tc.file_uuid = r.file_uuid
                  WHERE tc.trust_weight > r.confidence_score
                     OR (tc.trust_weight = r.confidence_score AND tc.candidate_timestamp < r.resolved_timestamp)
                     OR (tc.trust_weight = r.confidence_score AND tc.candidate_timestamp = r.resolved_timestamp
                         AND tc.created_at > (SELECT created_at FROM lucidota_korpus.temporal_claim selected WHERE selected.claim_uuid = r.selected_claim_uuid)
                         AND tc.claim_uuid <> r.selected_claim_uuid)
                )
                SELECT count(*) FROM violations
            """) or 0)
            latest_pass_uuid = q1(cur, "SELECT ranking_pass_uuid::text FROM lucidota_korpus.chrono_ranking_pass ORDER BY created_at DESC LIMIT 1")
            latest_results_without_claim = 0
            if latest_pass_uuid:
                latest_results_without_claim = int(q1(cur, """
                    SELECT count(*)
                    FROM lucidota_korpus.chrono_ranking_result rr
                    LEFT JOIN lucidota_korpus.temporal_claim tc ON tc.claim_uuid = rr.selected_claim_uuid
                    WHERE rr.ranking_pass_uuid=%s::uuid AND tc.claim_uuid IS NULL
                """, (latest_pass_uuid,)) or 0)
            source_distribution = qrows(cur, """
                SELECT evidence_source, trust_weight::text AS trust_weight, count(*)::bigint AS claims
                FROM lucidota_korpus.temporal_claim
                GROUP BY evidence_source, trust_weight
                ORDER BY trust_weight DESC, evidence_source
            """)
            selected_distribution = qrows(cur, """
                SELECT dominant_evidence_source AS evidence_source, confidence_score::text AS trust_weight, count(*)::bigint AS files
                FROM lucidota_korpus.resolved_chrono_timeline_with_claim
                GROUP BY dominant_evidence_source, confidence_score
                ORDER BY confidence_score DESC, dominant_evidence_source
            """)
    checks = [
        {"check": "best_projection_has_claim_uuid", "passed": best_without_claim == 0 and best_rows > 0, "value": best_without_claim},
        {"check": "best_rows_cover_claimed_files", "passed": best_rows == distinct_claimed_files, "value": {"best_rows": best_rows, "distinct_claimed_files": distinct_claimed_files}},
        {"check": "low_confidence_claims_preserved", "passed": low_confidence_claims > 0, "value": low_confidence_claims},
        {"check": "conflicting_claims_preserved", "passed": conflicting_claims_preserved > 0 and disputed_files > 0, "value": {"disputed_files": disputed_files, "conflicting_claims_preserved": conflicting_claims_preserved}},
        {"check": "ranking_violations_zero", "passed": ranking_violations == 0, "value": ranking_violations},
        {"check": "latest_ranking_results_link_to_claims", "passed": latest_results_without_claim == 0, "value": {"latest_pass_uuid": latest_pass_uuid, "latest_results_without_claim": latest_results_without_claim}},
    ]
    passed = all(item["passed"] for item in checks)
    report = {
        "action": "verify",
        "execute_performed": False,
        "db_writes_performed": False,
        "graph_writes_performed": False,
        "temporal_claims_deleted_or_overwritten": False,
        "projection": "lucidota_korpus.resolved_chrono_timeline_with_claim",
        "status": "PASS" if passed else "FAIL",
        "checks": checks,
        "metrics": {
            "total_claims": total_claims,
            "low_confidence_claims": low_confidence_claims,
            "low_confidence_selected_best_times": low_confidence_selected,
            "best_rows": best_rows,
            "distinct_claimed_files": distinct_claimed_files,
            "disputed_files": disputed_files,
            "conflicting_claims_preserved": conflicting_claims_preserved,
            "ranking_violations": ranking_violations,
            "latest_ranking_pass_uuid": latest_pass_uuid,
            "source_distribution": source_distribution,
            "selected_distribution": selected_distribution,
        },
        "blockers": [] if passed else [c["check"] for c in checks if not c["passed"]],
    }
    write_report("pass" if passed else "fail", report)
    print(f"CHRONO_CONSERVATION_STATUS={report['status']}")
    print(f"RANKING_VIOLATIONS={ranking_violations}")
    print(f"LOW_CONFIDENCE_CLAIMS={low_confidence_claims}")
    print(f"DISPUTED_FILES={disputed_files}")
    return 0 if passed else 5


def source_audit(args: argparse.Namespace) -> int:
    with psycopg.connect(db(args), row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            total_claims = int(q1(cur, "SELECT count(*) FROM lucidota_korpus.temporal_claim") or 0)
            claims_by_source = qrows(cur, """
                SELECT evidence_source,
                       count(*)::bigint AS claim_count,
                       count(DISTINCT file_uuid)::bigint AS files_covered,
                       min(trust_weight)::text AS min_trust_weight,
                       max(trust_weight)::text AS max_trust_weight
                FROM lucidota_korpus.temporal_claim
                GROUP BY evidence_source
                ORDER BY claim_count DESC, evidence_source
            """)
            selected_by_source = qrows(cur, """
                SELECT dominant_evidence_source AS evidence_source,
                       count(*)::bigint AS selected_file_count,
                       min(confidence_score)::text AS min_confidence,
                       max(confidence_score)::text AS max_confidence
                FROM lucidota_korpus.resolved_chrono_timeline_with_claim
                GROUP BY dominant_evidence_source
                ORDER BY selected_file_count DESC, dominant_evidence_source
            """)
            invalid_weights = int(q1(cur, "SELECT count(*) FROM lucidota_korpus.temporal_claim WHERE trust_weight < 0 OR trust_weight > 1") or 0)
            missing_extractor_or_source = int(q1(cur, "SELECT count(*) FROM lucidota_korpus.temporal_claim WHERE coalesce(evidence_source,'')='' OR coalesce(extractor,'')=''") or 0)
            mtime_snapshot_claims = int(q1(cur, "SELECT count(*) FROM lucidota_korpus.temporal_claim WHERE evidence_source='mtime_snapshot_v1'") or 0)
            source_sha_invalid = int(q1(cur, "SELECT count(*) FROM lucidota_korpus.temporal_claim WHERE source_sha256 IS NOT NULL AND source_sha256 !~ '^[0-9a-f]{64}$'") or 0)
            ranking_violations = int(q1(cur, """
                WITH violations AS (
                  SELECT r.file_uuid
                  FROM lucidota_korpus.resolved_chrono_timeline_with_claim r
                  JOIN lucidota_korpus.temporal_claim tc ON tc.file_uuid = r.file_uuid
                  WHERE tc.trust_weight > r.confidence_score
                     OR (tc.trust_weight = r.confidence_score AND tc.candidate_timestamp < r.resolved_timestamp)
                )
                SELECT count(*) FROM violations
            """) or 0)
            null_file_claims_by_source = qrows(cur, """
                SELECT evidence_source, count(*)::bigint AS claim_count
                FROM lucidota_korpus.temporal_claim
                WHERE file_uuid IS NULL
                GROUP BY evidence_source
                ORDER BY claim_count DESC, evidence_source
            """)
    checks = [
        {"check": "temporal_claims_exist", "passed": total_claims > 0, "value": total_claims},
        {"check": "source_distribution_nonempty", "passed": len(claims_by_source) > 0, "value": len(claims_by_source)},
        {"check": "selected_distribution_nonempty", "passed": len(selected_by_source) > 0, "value": len(selected_by_source)},
        {"check": "trust_weights_in_bounds", "passed": invalid_weights == 0, "value": invalid_weights},
        {"check": "source_sha256_valid_when_present", "passed": source_sha_invalid == 0, "value": source_sha_invalid},
        {"check": "ranking_violations_zero", "passed": ranking_violations == 0, "value": ranking_violations},
        {"check": "mtime_snapshot_claims_present", "passed": mtime_snapshot_claims >= 18625, "value": mtime_snapshot_claims},
        {"check": "extractor_and_source_present", "passed": missing_extractor_or_source == 0, "value": missing_extractor_or_source},
    ]
    passed = all(item["passed"] for item in checks)
    report = {
        "action": "source_audit",
        "execute_performed": False,
        "db_writes_performed": False,
        "graph_writes_performed": False,
        "status": "PASS" if passed else "FAIL",
        "checks": checks,
        "metrics": {
            "total_claims": total_claims,
            "claims_by_source": claims_by_source,
            "selected_by_source": selected_by_source,
            "null_file_claims_by_source": null_file_claims_by_source,
            "invalid_weights": invalid_weights,
            "missing_extractor_or_source": missing_extractor_or_source,
            "source_sha_invalid": source_sha_invalid,
            "mtime_snapshot_claims": mtime_snapshot_claims,
            "ranking_violations": ranking_violations,
        },
        "blockers": [] if passed else [c["check"] for c in checks if not c["passed"]],
    }
    write_report("source_audit_pass" if passed else "source_audit_fail", report)
    print(f"CHRONO_SOURCE_AUDIT_STATUS={report['status']}")
    print(f"TOTAL_CLAIMS={total_claims}")
    print(f"RANKING_VIOLATIONS={ranking_violations}")
    print(f"MTIME_SNAPSHOT_CLAIMS={mtime_snapshot_claims}")
    return 0 if passed else 6


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify Chrono best-time projection links to claims and preserves low-confidence/conflicting evidence")
    parser.add_argument("--database-url")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("init-schema")
    p.add_argument("--execute", action="store_true")
    p.set_defaults(func=init_schema)
    p = sub.add_parser("verify")
    p.set_defaults(func=verify)
    p = sub.add_parser("source-audit")
    p.set_defaults(func=source_audit)
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
