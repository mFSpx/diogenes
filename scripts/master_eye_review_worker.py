#!/usr/bin/env python3
"""Deterministic Master Eye runtime reviewer for Phase 0.5 artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psycopg
from psycopg.rows import dict_row

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "05_OUTPUTS" / "phase05"
SCHEMAS = [
    ROOT / "06_SCHEMA/030_phase05_brain_archaeology.sql",
    ROOT / "06_SCHEMA/070_master_eye_runtime_review.sql",
]
VERSION = "master_eye_runtime_review_v1"


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


def write_report(name: str, payload: dict[str, Any]) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"master_eye_review_{name}_{stamp()}.json"
    payload.setdefault("generated_at", now_iso())
    payload["report_path"] = rel(path)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"REPORT_PATH={rel(path)}")
    return path


def sha_obj(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def init_schema(args: argparse.Namespace) -> int:
    if args.execute:
        with psycopg.connect(db(args)) as conn:
            with conn.cursor() as cur:
                for schema in SCHEMAS:
                    cur.execute(schema.read_text(encoding="utf-8"))
            conn.commit()
    write_report("init_schema_execute" if args.execute else "init_schema_dry_run", {
        "action": "init_schema",
        "execute_performed": bool(args.execute),
        "schemas": [rel(s) for s in SCHEMAS],
    })
    return 0


def evidence_len(value: Any) -> int:
    if isinstance(value, list):
        return len(value)
    return 0


def steps_len(value: Any) -> int:
    if isinstance(value, list):
        return len(value)
    return 0


def review_design_atom(row: dict[str, Any]) -> dict[str, Any]:
    blockers: list[str] = []
    if not row["normalized_claim"]:
        blockers.append("normalized_claim_missing")
    if not row["raw_excerpt"]:
        blockers.append("raw_excerpt_missing")
    if evidence_len(row["evidence"]) == 0:
        blockers.append("evidence_missing")
    if not row["authority_class"]:
        blockers.append("authority_class_missing")
    if blockers:
        judgment = "needs_more_evidence"
        next_action = "Add evidence refs/raw excerpt before promotion or blueprint use."
        confidence = 7000
    elif row["atom_kind"] in {"workflow", "requirement", "governance_rule", "doctrine", "constraint"}:
        judgment = "syllabus_faithful"
        next_action = "Eligible for workflow blueprint synthesis or Master Eye promotion review."
        confidence = min(9500, int(row["confidence_bps"]) + 1500)
    else:
        judgment = "defer"
        next_action = "Keep as design evidence; review again after related artifacts accumulate."
        confidence = int(row["confidence_bps"])
    return {
        "target_kind": "design_atom",
        "target_uuid": row["atom_uuid"],
        "judgment": judgment,
        "rationale": f"Master Eye v1 checked atom_kind={row['atom_kind']}, evidence_count={evidence_len(row['evidence'])}, authority_class={row['authority_class']}; blockers={blockers or ['none']}.",
        "recommended_next_action": next_action,
        "confidence_bps": confidence,
        "source_detail": {"blockers": blockers, "extractor_version": row["extractor_version"], "reviewer": VERSION},
    }


def review_workflow_blueprint(row: dict[str, Any]) -> dict[str, Any]:
    blockers: list[str] = []
    if steps_len(row["steps"]) == 0:
        blockers.append("steps_missing")
    if not row["source_atom_uuids"]:
        blockers.append("source_atom_uuids_missing")
    if not row["purpose"]:
        blockers.append("purpose_missing")
    if blockers:
        judgment = "workflow_gap"
        next_action = "Add steps/source_atom_uuids before implementation."
        confidence = 7000
    elif row["absurd_target"]:
        judgment = "implementation_ready"
        next_action = "Queue implementation as ABSURD work order after operator priority selection."
        confidence = min(9500, int(row["canonical_confidence_bps"]) + 1000)
    else:
        judgment = "defer"
        next_action = "Keep recovered; no ABSURD implementation target selected."
        confidence = int(row["canonical_confidence_bps"])
    return {
        "target_kind": "workflow_blueprint",
        "target_uuid": row["blueprint_uuid"],
        "judgment": judgment,
        "rationale": f"Master Eye v1 checked workflow_key={row['workflow_key']}, steps={steps_len(row['steps'])}, source_atoms={len(row['source_atom_uuids'] or [])}, absurd_target={row['absurd_target']}; blockers={blockers or ['none']}.",
        "recommended_next_action": next_action,
        "confidence_bps": confidence,
        "source_detail": {"blockers": blockers, "workflow_key": row["workflow_key"], "reviewer": VERSION},
    }


def fetch_targets(conn: psycopg.Connection[Any], target: str, limit: int) -> list[dict[str, Any]]:
    with conn.cursor(row_factory=dict_row) as cur:
        if target in {"design_atom", "all"}:
            cur.execute(
                """
                SELECT atom_uuid::text, atom_kind, normalized_claim, raw_excerpt, authority_class::text,
                       confidence_bps, evidence, extractor_version
                FROM lucidota_archaeology.design_atom
                ORDER BY created_at DESC
                LIMIT %s
                """,
                (limit,),
            )
            atoms = [review_design_atom(dict(row)) for row in cur.fetchall()]
        else:
            atoms = []
        if target in {"workflow_blueprint", "all"}:
            cur.execute(
                """
                SELECT blueprint_uuid::text, workflow_key, purpose, absurd_target, steps,
                       source_atom_uuids, canonical_confidence_bps
                FROM lucidota_archaeology.workflow_blueprint
                ORDER BY updated_at DESC
                LIMIT %s
                """,
                (limit,),
            )
            blueprints = [review_workflow_blueprint(dict(row)) for row in cur.fetchall()]
        else:
            blueprints = []
    return atoms + blueprints


def insert_review(cur: psycopg.Cursor[Any], review: dict[str, Any]) -> bool:
    review_key = sha_obj({
        "version": VERSION,
        "target_kind": review["target_kind"],
        "target_uuid": review["target_uuid"],
        "judgment": review["judgment"],
        "source_detail": review["source_detail"],
    })
    cur.execute(
        """
        INSERT INTO lucidota_archaeology.master_eye_review(
          target_kind, target_uuid, judgment, rationale, recommended_next_action,
          confidence_bps, reviewed_by, review_key, source_detail
        )
        VALUES (%s,%s::uuid,%s,%s,%s,%s,%s,%s,%s::jsonb)
        ON CONFLICT (review_key) WHERE review_key IS NOT NULL DO NOTHING
        RETURNING review_uuid::text
        """,
        (
            review["target_kind"],
            review["target_uuid"],
            review["judgment"],
            review["rationale"],
            review["recommended_next_action"],
            review["confidence_bps"],
            VERSION,
            review_key,
            json.dumps(review["source_detail"]),
        ),
    )
    return cur.fetchone() is not None


def review(args: argparse.Namespace) -> int:
    with psycopg.connect(db(args)) as conn:
        prepared = fetch_targets(conn, args.target, args.limit)
        inserted = 0
        if args.execute and prepared:
            with conn.cursor() as cur:
                for item in prepared:
                    if insert_review(cur, item):
                        inserted += 1
            conn.commit()
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM lucidota_archaeology.master_eye_review")
            total = int(cur.fetchone()[0])
    by_judgment: dict[str, int] = {}
    for item in prepared:
        by_judgment[item["judgment"]] = by_judgment.get(item["judgment"], 0) + 1
    actionable = [item for item in prepared if item["target_kind"] == "workflow_blueprint" and item["judgment"] == "implementation_ready"]
    workflow_gaps = [item for item in prepared if item["target_kind"] == "workflow_blueprint" and item["judgment"] == "workflow_gap"]
    report = {
        "action": "review",
        "target": args.target,
        "workflow_blueprint_actionable_count": len(actionable),
        "workflow_blueprint_gap_count": len(workflow_gaps),
        "execute_performed": bool(args.execute),
        "db_writes_performed": bool(args.execute),
        "graph_writes_performed": False,
        "reviewer_version": VERSION,
        "reviews_prepared": len(prepared),
        "reviews_inserted": inserted,
        "master_eye_reviews_total": total,
        "judgment_counts": by_judgment,
        "sample_reviews": prepared[:12],
        "blockers": [] if prepared else ["no_review_targets"],
    }
    write_report("review_execute" if args.execute else "review_dry_run", report)
    print(f"REVIEWS_PREPARED={len(prepared)}")
    print(f"REVIEWS_INSERTED={inserted if args.execute else 0}")
    return 0 if prepared and (not getattr(args, "require_actionable", False) or bool(actionable)) else 2


def main() -> int:
    parser = argparse.ArgumentParser(description="Run deterministic Master Eye reviews over design atoms/workflow blueprints")
    parser.add_argument("--database-url")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("init-schema")
    p.add_argument("--execute", action="store_true")
    p.set_defaults(func=init_schema)
    p = sub.add_parser("review")
    p.add_argument("--execute", action="store_true")
    p.add_argument("--target", choices=["design_atom", "workflow_blueprint", "all"], default="all")
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--require-actionable", action="store_true", help="fail if no implementation_ready workflow_blueprint is found")
    p.set_defaults(func=review)
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
