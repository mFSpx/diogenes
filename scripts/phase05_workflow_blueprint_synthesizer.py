#!/usr/bin/env python3
"""Synthesize workflow_blueprint rows from existing Phase 0.5 design_atom rows.

This is executable synthesis, not prose. It reads already-captured design_atom
claims, clusters them into concrete ABSURD-target workflow candidates, and writes
workflow_blueprint rows only when --execute is explicit. It does not ingest new
corpus data and does not mutate graph tables.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psycopg
from psycopg.rows import dict_row

ROOT = Path(__file__).resolve().parents[1]
SCHEMA0 = ROOT / "06_SCHEMA/030_phase05_brain_archaeology.sql"
SCHEMA = ROOT / "06_SCHEMA/066_phase05_workflow_blueprint_synthesis.sql"
OUT = ROOT / "05_OUTPUTS" / "phase05"
VERSION = "phase05_workflow_blueprint_synthesis_v1"


@dataclass(frozen=True)
class Bucket:
    key: str
    title: str
    purpose: str
    queues: tuple[str, ...]
    required_tables: tuple[str, ...]
    required_services: tuple[str, ...]
    patterns: tuple[str, ...]


BUCKETS: tuple[Bucket, ...] = (
    Bucket(
        "absurd_queue_spine",
        "ABSURD Queue Spine Execution Workflow",
        "Recover durable queue/worker/retry/dead-letter execution rules from design atoms.",
        ("control",),
        ("lucidota_control.absurd_queue_job", "lucidota_control.absurd_queue_event"),
        ("scripts/spine_job_adapter.py", "scripts/case_pipeline_dispatch.py"),
        (r"\bABSURD\b", r"\bqueue\b", r"\bworker\b", r"\bdaemon\b", r"\bretry\b", r"\bdead[-_ ]letter\b"),
    ),
    Bucket(
        "chrono_ledger",
        "Chrono-Ledger Temporal Conservation Workflow",
        "Recover temporal claim preservation, ranking, replay cursor, and service health rules.",
        ("chrono",),
        ("lucidota_korpus.temporal_claim", "lucidota_korpus.resolved_chrono_timeline"),
        ("lucidota-chrono-ledger", "scripts/check_chrono_ledger_service.sh"),
        (r"\bChrono\b", r"\btemporal\b", r"\bmtime\b", r"\btimeline\b", r"\branking\b"),
    ),
    Bucket(
        "graph_promotion",
        "Evidence-Gated Graph Promotion Workflow",
        "Recover no-direct-graph-write promotion rules, authority classes, decisions, and journal receipts.",
        ("graph_promotion",),
        ("lucidota_go.graph_promotion_packet", "lucidota_go.graph_promotion_decision", "lucidota_go.graph_journal"),
        ("scripts/graph_promotion_gate.py", "scripts/graph_promotion_execute.py"),
        (r"\bgraph\b", r"\bpromotion\b", r"\bcanonical\b", r"\bevidence\b", r"\bauthority_class\b"),
    ),
    Bucket(
        "surface_cep",
        "Surface / CEP Conversation Instruction Workflow",
        "Recover generated-surface-to-plain-language-instruction and conversation command fan-in rules.",
        ("surface_cep",),
        ("lucidota_control.conversation_command", "lucidota_runtime.surface_command_envelope"),
        ("scripts/surface_instruction_compile_dry_run.py", "scripts/spine_surface_cep_worker.py"),
        (r"\bsurface\b", r"\bconversation\b", r"\bcommand envelope\b", r"\bCEP\b", r"\bMarrow\b"),
    ),
    Bucket(
        "phase05_brain_archaeology",
        "Phase 0.5 Brain Archaeology Recovery Workflow",
        "Recover operator ontology, design atoms, workflow blueprints, and fidelity guard execution rules.",
        ("phase05_streaming_brain",),
        ("lucidota_archaeology.design_atom", "lucidota_archaeology.workflow_blueprint"),
        ("scripts/phase05_design_atom_extractor.py", "scripts/phase05_workflow_blueprint_synthesizer.py"),
        (r"\bBrain Archaeology\b", r"\bPhase 0\\.5\b", r"\bdesign_atom\b", r"\bworkflow_blueprint\b", r"\bfidelity\b", r"\bontology\b"),
    ),
    Bucket(
        "security_quarantine",
        "Security Quarantine Gate Workflow",
        "Recover secret/corpus quarantine rules that gate Brain Archaeology full ingest.",
        ("security",),
        ("05_OUTPUTS/security/security_quarantine_manifest_*.json",),
        ("scripts/lucidota_security_quarantine_gate.py",),
        (r"\bsecurity\b", r"\bquarantine\b", r"\bsecret\b", r"\btoken\b", r"\bcredential\b"),
    ),
    Bucket(
        "tickletrunk",
        "TICKLETRUNK Proof Hoard Discovery Workflow",
        "Recover toolbox sovereignty and manifest-update rules before new code is written.",
        ("toolbox",),
        ("00_PROJECT_BRAIN/TICKLETRUNK.json",),
        ("scripts/tickletrunk_scan.py",),
        (r"\bTICKLETRUNK\b", r"\bproof hoard\b", r"\btoolbox\b", r"\bsovereign\b"),
    ),
)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def db_url(args: argparse.Namespace) -> str:
    return args.database_url or os.environ.get("DATABASE_URL") or "postgresql:///lucidota_storage"


def rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def write_report(name: str, payload: dict[str, Any]) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"phase05_workflow_blueprint_{name}_{stamp()}.json"
    payload.setdefault("generated_at", now_iso())
    payload["report_path"] = rel(path)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False, ensure_ascii=False), encoding="utf-8")
    print(f"REPORT_PATH={rel(path)}")
    return path


def init_schema(args: argparse.Namespace) -> int:
    if args.execute:
        with psycopg.connect(db_url(args)) as conn:
            with conn.cursor() as cur:
                cur.execute(SCHEMA0.read_text(encoding="utf-8"))
                cur.execute(SCHEMA.read_text(encoding="utf-8"))
            conn.commit()
    write_report("init_schema_execute" if args.execute else "init_schema_dry_run", {
        "action": "init_schema",
        "execute_performed": bool(args.execute),
        "schemas": [rel(SCHEMA0), rel(SCHEMA)],
    })
    return 0


def fetch_atoms(conn: psycopg.Connection[Any], limit: int) -> list[dict[str, Any]]:
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT atom_uuid::text, atom_kind, title, normalized_claim, authority_class::text,
                   status, confidence_bps, tags, evidence, created_at
            FROM lucidota_archaeology.design_atom
            WHERE status IN ('candidate','promoted','needs_operator_authority')
            ORDER BY created_at DESC, confidence_bps DESC
            LIMIT %s
            """,
            (limit,),
        )
        return list(cur.fetchall())


def match_bucket(atom: dict[str, Any], bucket: Bucket) -> bool:
    haystack = " ".join([
        str(atom.get("atom_kind", "")),
        str(atom.get("title", "")),
        str(atom.get("normalized_claim", "")),
        " ".join(str(x) for x in (atom.get("tags") or [])),
    ])
    return any(re.search(pattern, haystack, re.I) for pattern in bucket.patterns)


def blueprint_from_bucket(bucket: Bucket, atoms: list[dict[str, Any]], min_atoms: int) -> dict[str, Any] | None:
    if len(atoms) < min_atoms:
        return None
    atoms = atoms[:48]
    source_atom_uuids = [a["atom_uuid"] for a in atoms]
    atom_kinds = sorted({str(a["atom_kind"]) for a in atoms})
    steps = [
        {
            "step_key": "load_design_atoms",
            "description": "Load linked design_atom rows and retain evidence refs.",
            "source_atom_count": len(atoms),
        },
        {
            "step_key": "validate_authority",
            "description": "Require authority_class and preserve operator ontology/fidelity rules.",
            "authority_classes": sorted({str(a["authority_class"]) for a in atoms}),
        },
        {
            "step_key": "execute_candidate_workflow",
            "description": f"Execute or stage the recovered {bucket.title} under its queue/worker contract.",
            "queues": list(bucket.queues),
        },
        {
            "step_key": "record_evidence",
            "description": "Write execution/audit receipt and do not promote to graph without promotion packet.",
        },
    ]
    confidence = min(9000, 5000 + (len(atoms) * 250))
    return {
        "workflow_key": f"recovered_{bucket.key}",
        "title": bucket.title,
        "purpose": bucket.purpose,
        "maturity": "recovered",
        "absurd_target": True,
        "input_contract": {
            "source": "lucidota_archaeology.design_atom",
            "atom_kinds": atom_kinds,
            "min_atoms": min_atoms,
            "requires_evidence_refs": True,
        },
        "output_contract": {
            "writes": ["lucidota_archaeology.workflow_blueprint"],
            "forbidden": ["canonical_graph_mutation", "operator_ontology_softening"],
            "requires_status_ledger_evidence": True,
        },
        "steps": steps,
        "queues": list(bucket.queues),
        "required_tables": list(bucket.required_tables),
        "required_services": list(bucket.required_services),
        "required_models": [],
        "source_atom_uuids": source_atom_uuids,
        "authority_class": "model_computed_finding",
        "canonical_confidence_bps": confidence,
        "operator_confirmed": False,
    }


def synthesize(args: argparse.Namespace) -> int:
    with psycopg.connect(db_url(args)) as conn:
        atoms = fetch_atoms(conn, args.limit)
        grouped: dict[str, list[dict[str, Any]]] = {bucket.key: [] for bucket in BUCKETS}
        for atom in atoms:
            for bucket in BUCKETS:
                if match_bucket(atom, bucket):
                    grouped[bucket.key].append(atom)
        blueprints = [
            bp
            for bucket in BUCKETS
            if (bp := blueprint_from_bucket(bucket, grouped[bucket.key], args.min_atoms)) is not None
        ]
        inserted = 0
        updated = 0
        if args.execute:
            with conn.cursor(row_factory=dict_row) as cur:
                for bp in blueprints:
                    cur.execute(
                        """
                        INSERT INTO lucidota_archaeology.workflow_blueprint(
                          workflow_key, title, purpose, maturity, absurd_target, input_contract,
                          output_contract, steps, queues, required_tables, required_services,
                          required_models, source_atom_uuids, authority_class,
                          canonical_confidence_bps, operator_confirmed
                        )
                        VALUES (%s,%s,%s,%s,%s,%s::jsonb,%s::jsonb,%s::jsonb,%s::text[],%s::text[],%s::text[],%s::text[],%s::uuid[],%s::lucidota_archaeology.authority_class,%s,%s)
                        ON CONFLICT (workflow_key) DO UPDATE SET
                          title=EXCLUDED.title,
                          purpose=EXCLUDED.purpose,
                          maturity=EXCLUDED.maturity,
                          absurd_target=EXCLUDED.absurd_target,
                          input_contract=EXCLUDED.input_contract,
                          output_contract=EXCLUDED.output_contract,
                          steps=EXCLUDED.steps,
                          queues=EXCLUDED.queues,
                          required_tables=EXCLUDED.required_tables,
                          required_services=EXCLUDED.required_services,
                          required_models=EXCLUDED.required_models,
                          source_atom_uuids=EXCLUDED.source_atom_uuids,
                          authority_class=EXCLUDED.authority_class,
                          canonical_confidence_bps=EXCLUDED.canonical_confidence_bps,
                          updated_at=now()
                        WHERE lucidota_archaeology.workflow_blueprint.operator_confirmed = false
                        RETURNING (xmax = 0) AS inserted_new
                        """,
                        (
                            bp["workflow_key"],
                            bp["title"],
                            bp["purpose"],
                            bp["maturity"],
                            bp["absurd_target"],
                            json.dumps(bp["input_contract"]),
                            json.dumps(bp["output_contract"]),
                            json.dumps(bp["steps"]),
                            bp["queues"],
                            bp["required_tables"],
                            bp["required_services"],
                            bp["required_models"],
                            bp["source_atom_uuids"],
                            bp["authority_class"],
                            bp["canonical_confidence_bps"],
                            bp["operator_confirmed"],
                        ),
                    )
                    row = cur.fetchone()
                    if row and row["inserted_new"]:
                        inserted += 1
                    elif row:
                        updated += 1
                report_stub = {
                    "workflow_keys": [bp["workflow_key"] for bp in blueprints],
                    "group_counts": {k: len(v) for k, v in grouped.items()},
                }
                cur.execute(
                    """
                    INSERT INTO lucidota_archaeology.workflow_blueprint_synthesis_run(
                      synthesizer_version, source_atom_count, blueprints_prepared,
                      blueprints_inserted, blueprints_updated, dry_run, detail
                    )
                    VALUES (%s,%s,%s,%s,%s,false,%s::jsonb)
                    RETURNING run_uuid::text
                    """,
                    (VERSION, len(atoms), len(blueprints), inserted, updated, json.dumps(report_stub)),
                )
                run_uuid = cur.fetchone()["run_uuid"]
            conn.commit()
        else:
            run_uuid = None
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute("SELECT count(*) AS n FROM lucidota_archaeology.workflow_blueprint")
            total_blueprints = int(cur.fetchone()["n"])
    report = {
        "action": "synthesize",
        "execute_performed": bool(args.execute),
        "db_writes_performed": bool(args.execute),
        "graph_writes_performed": False,
        "synthesizer_version": VERSION,
        "source_atom_count": len(atoms),
        "group_counts": {k: len(v) for k, v in grouped.items()},
        "blueprints_prepared": len(blueprints),
        "blueprints_inserted": inserted,
        "blueprints_updated": updated,
        "workflow_blueprints_total": total_blueprints,
        "run_uuid": run_uuid,
        "blueprints": blueprints,
        "blockers": [] if blueprints else ["no_matching_design_atom_clusters"],
    }
    report_path = write_report("synthesize_execute" if args.execute else "synthesize_dry_run", report)
    if args.execute and run_uuid:
        with psycopg.connect(db_url(args)) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE lucidota_archaeology.workflow_blueprint_synthesis_run SET report_path=%s WHERE run_uuid=%s::uuid",
                    (rel(report_path), run_uuid),
                )
            conn.commit()
    print(f"BLUEPRINTS_PREPARED={len(blueprints)}")
    print(f"BLUEPRINTS_INSERTED={inserted}")
    print(f"BLUEPRINTS_UPDATED={updated}")
    if run_uuid:
        print(f"RUN_UUID={run_uuid}")
    return 0 if blueprints else 2


def main() -> int:
    parser = argparse.ArgumentParser(description="Synthesize workflow_blueprint rows from existing design_atom rows")
    parser.add_argument("--database-url")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("init-schema")
    p.add_argument("--execute", action="store_true")
    p.set_defaults(func=init_schema)
    p = sub.add_parser("synthesize")
    p.add_argument("--execute", action="store_true")
    p.add_argument("--limit", type=int, default=500)
    p.add_argument("--min-atoms", type=int, default=1)
    p.set_defaults(func=synthesize)
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
