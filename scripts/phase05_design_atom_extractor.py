#!/usr/bin/env python3
"""Deterministic Phase 0.5 design atom extractor.

This converts allowlisted custody artifacts into design_atom candidates using
operator-doctrine/design-rule patterns. It does not call LLMs and does not
mutate graph tables.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psycopg
from psycopg.rows import dict_row

ROOT = Path(__file__).resolve().parents[1]
SCHEMA0 = ROOT / "06_SCHEMA/030_phase05_brain_archaeology.sql"
SCHEMA = ROOT / "06_SCHEMA/051_phase05_design_atom_runtime.sql"
OUT = ROOT / "05_OUTPUTS/phase05"

SIGNALS = [
    ("doctrine", re.compile(r"\b(HARD LAW|Hard law|law:|do not|never|must not|forbidden)\b", re.I)),
    ("requirement", re.compile(r"\b(MUST|must|required|requirement|shall|needs to|has to)\b", re.I)),
    ("workflow", re.compile(r"\b(workflow|queue|ABSURD|worker|daemon|pipeline|loop)\b", re.I)),
    ("governance_rule", re.compile(r"\b(authority_class|operator_confirmed|consent|permission|canonical|promotion|graph)\b", re.I)),
    ("constraint", re.compile(r"\b(no direct|blocked|gate|guard|preserve|idempotent|non-destructive|quarantine)\b", re.I)),
    ("component", re.compile(r"\b(KORPUS|KRAMPUSCHEWING|Chrono-Ledger|TICKLETRUNK|FairyFuse|DIOGENES|CatchMe|SimpleMem|DeMem|TRACER)\b", re.I)),
]

TEXT_SUFFIXES = {".md", ".txt", ".json", ".jsonl", ".yaml", ".yml", ".toml", ".ini", ".conf", ".sql", ".py", ".rs", ".sh"}


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


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def write_report(name: str, payload: dict[str, Any]) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"phase05_design_atom_{name}_{stamp()}.json"
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
        "schema": rel(SCHEMA),
    })
    return 0


def fetch_artifacts(conn: psycopg.Connection[Any], limit: int) -> list[dict[str, Any]]:
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT artifact_uuid::text, source_path, source_sha256, size_bytes, mime_guess, ingest_status, manifest_path
            FROM lucidota_archaeology.allowlisted_ingest_artifact
            WHERE ingest_status = 'custody_recorded'
            ORDER BY created_at DESC, source_path
            LIMIT %s
            """,
            (limit,),
        )
        return list(cur.fetchall())


def load_text(source_path: str, max_bytes: int) -> str:
    p = ROOT / source_path
    if not p.exists():
        p = Path(source_path)
    if p.suffix.lower() not in TEXT_SUFFIXES:
        return ""
    raw = p.read_bytes()[:max_bytes]
    return raw.decode("utf-8", errors="replace")


def normalize_claim(line: str) -> str:
    return re.sub(r"\s+", " ", line.strip().strip("-*#> ")).strip()


def atom_kind_for(line: str) -> str:
    for kind, rx in SIGNALS:
        if rx.search(line):
            return kind
    return "requirement"


def candidate_atoms(row: dict[str, Any], text: str, max_per_artifact: int, extractor_version: str) -> list[dict[str, Any]]:
    atoms: list[dict[str, Any]] = []
    seen: set[str] = set()
    for idx, line in enumerate(text.splitlines(), start=1):
        claim = normalize_claim(line)
        if len(claim) < 24 or len(claim) > 420:
            continue
        if not any(rx.search(claim) for _, rx in SIGNALS):
            continue
        key = sha256_text(claim.lower())
        if key in seen:
            continue
        seen.add(key)
        kind = atom_kind_for(claim)
        tags = sorted({kind, "phase05", "deterministic_extraction"})
        atoms.append({
            "source_artifact_uuid": row["artifact_uuid"],
            "source_component_uuid": None,
            "atom_kind": kind,
            "title": claim[:96],
            "normalized_claim": claim,
            "raw_excerpt": line[:1200],
            "authority_class": "raw_evidence",
            "status": "candidate",
            "confidence_bps": 6500,
            "tags": tags,
            "entities": [],
            "evidence": [{
                "artifact_uuid": row["artifact_uuid"],
                "source_path": row["source_path"],
                "source_sha256": row["source_sha256"],
                "line_number": idx,
                "excerpt_sha256": sha256_text(line),
                "manifest_path": row["manifest_path"],
            }],
            "extractor_version": extractor_version,
        })
        if len(atoms) >= max_per_artifact:
            break
    return atoms


def insert_atom(cur: psycopg.Cursor[Any], atom: dict[str, Any]) -> bool:
    cur.execute(
        """
        INSERT INTO lucidota_archaeology.design_atom(
          source_artifact_uuid, source_component_uuid, atom_kind, title,
          normalized_claim, raw_excerpt, authority_class, status, confidence_bps,
          tags, entities, evidence, extractor_version
        )
        VALUES (%s::uuid, %s::uuid, %s, %s, %s, %s, %s::lucidota_archaeology.authority_class,
                %s, %s, %s::text[], %s::jsonb, %s::jsonb, %s)
        ON CONFLICT (source_artifact_uuid, extractor_version, md5(normalized_claim))
        DO NOTHING
        RETURNING atom_uuid::text
        """,
        (
            atom["source_artifact_uuid"], atom["source_component_uuid"], atom["atom_kind"], atom["title"],
            atom["normalized_claim"], atom["raw_excerpt"], atom["authority_class"], atom["status"],
            atom["confidence_bps"], atom["tags"], json.dumps(atom["entities"]), json.dumps(atom["evidence"]),
            atom["extractor_version"],
        ),
    )
    return cur.fetchone() is not None


def extract_atoms(args: argparse.Namespace) -> int:
    blockers: list[str] = []
    prepared: list[dict[str, Any]] = []
    artifacts_seen = 0
    with psycopg.connect(db_url(args)) as conn:
        rows = fetch_artifacts(conn, args.limit)
        for row in rows:
            artifacts_seen += 1
            try:
                text = load_text(row["source_path"], args.max_bytes)
            except Exception as exc:
                blockers.append(f"artifact_unreadable:{row['source_path']}:{type(exc).__name__}")
                continue
            if not text.strip():
                continue
            prepared.extend(candidate_atoms(row, text, args.max_per_artifact, args.extractor_version))
        inserted = 0
        if args.execute and prepared:
            with conn.cursor() as cur:
                for atom in prepared:
                    if insert_atom(cur, atom):
                        inserted += 1
            conn.commit()
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute("SELECT count(*) AS n FROM lucidota_archaeology.design_atom")
            total = cur.fetchone()["n"]
    report = {
        "action": "extract",
        "execute_performed": bool(args.execute),
        "db_writes_performed": bool(args.execute),
        "graph_writes_performed": False,
        "artifacts_seen": artifacts_seen,
        "atoms_prepared": len(prepared),
        "atoms_inserted": inserted,
        "design_atoms_total": total,
        "extractor_version": args.extractor_version,
        "sample_atoms": prepared[:10],
        "blockers": blockers,
    }
    write_report("extract_execute" if args.execute else "extract_dry_run", report)
    print(f"ATOMS_PREPARED={len(prepared)}")
    print(f"ATOMS_INSERTED={inserted if args.execute else 0}")
    return 0 if prepared or not args.require_atoms else 2


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract deterministic design atoms from allowlisted Phase 0.5 custody artifacts")
    parser.add_argument("--database-url")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("init-schema")
    p.add_argument("--execute", action="store_true")
    p.set_defaults(func=init_schema)
    p = sub.add_parser("extract")
    p.add_argument("--execute", action="store_true")
    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--max-bytes", type=int, default=250_000)
    p.add_argument("--max-per-artifact", type=int, default=12)
    p.add_argument("--extractor-version", default="phase05_design_atom_deterministic_v1")
    p.add_argument("--require-atoms", action="store_true")
    p.set_defaults(func=extract_atoms)
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
