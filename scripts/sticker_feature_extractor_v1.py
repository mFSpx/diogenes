#!/usr/bin/env python3
"""Sticker feature vector extractor v1 over allowlisted parsed text custody."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psycopg
from psycopg.rows import dict_row

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ALGOS.krampus_stickers import (  # noqa: E402
    extract_operator_telemetry,
    extract_operator_vibes,
    extract_psyche_vibes,
    extract_rainmaker_vibes,
    extract_resilience_vibes,
)

OUT = ROOT / "05_OUTPUTS/phase05"
SCHEMAS = [ROOT / "06_SCHEMA/030_phase05_brain_archaeology.sql", ROOT / "06_SCHEMA/080_sticker_feature_vector_extractor_v1.sql"]
FEATURE_VERSION = "phase05_stickers_v1"


def now() -> str:
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
    path = OUT / f"sticker_feature_extractor_{name}_{stamp()}.json"
    payload.setdefault("generated_at", now())
    payload["report_path"] = rel(path)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"REPORT_PATH={rel(path)}")
    return path


def init_schema(args: argparse.Namespace) -> int:
    if args.execute:
        with psycopg.connect(db(args)) as conn:
            with conn.cursor() as cur:
                for schema in SCHEMAS:
                    cur.execute(schema.read_text(encoding="utf-8"))
            conn.commit()
    write_report("init_schema_execute" if args.execute else "init_schema_dry_run", {"action":"init_schema","execute_performed":bool(args.execute),"schemas":[rel(s) for s in SCHEMAS]})
    return 0


def deterministic_uuid(namespace: str, value: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"lucidota:{namespace}:{value}"))


def syntax_features(text: str) -> dict[str, float]:
    chars = max(1, len(text))
    words = max(1, len(text.split()))
    return {
        "syntactic_chaos": (text.count("!") + text.count("?") + text.count(";") + text.count(":")) / chars,
        "ellipsis_density": text.count("...") / words,
        "punctuation_velocity": sum(1 for c in text if c in "!?;:,—-") / words,
        "first_person_gravity": sum(1 for w in text.lower().split() if w.strip(".,!?;:") in {"i", "me", "my", "mine", "myself"}) / words,
        "lexical_intent_ratio": sum(1 for w in text.lower().split() if w.strip(".,!?;:") in {"must", "need", "do", "execute", "build", "run"}) / words,
        "structural_entropy": (text.count("#") + text.count("|") + text.count("-") + text.count("*")) / words,
    }


def all_features(text: str) -> dict[str, float]:
    features: dict[str, float] = {}
    features.update(syntax_features(text))
    features.update(extract_operator_vibes(text))
    features.update(extract_psyche_vibes(text))
    features.update(extract_resilience_vibes(text))
    features.update(extract_rainmaker_vibes(text))
    features.update(extract_operator_telemetry(text))
    return features


def vector_columns(features: dict[str, float]) -> dict[str, float]:
    return {
        "syntactic_chaos": features.get("syntactic_chaos", 0.0),
        "ellipsis_density": features.get("ellipsis_density", 0.0),
        "punctuation_velocity": features.get("punctuation_velocity", 0.0),
        "first_person_gravity": features.get("first_person_gravity", 0.0),
        "lexical_intent_ratio": features.get("lexical_intent_ratio", 0.0),
        "structural_entropy": features.get("structural_entropy", 0.0),
        "ledger_density": features.get("operator_ledger_density", 0.0),
        "visceral_ratio": features.get("operator_visceral_ratio", 0.0),
        "recursion_score": features.get("operator_recursion_score", 0.0),
        "directive_ratio": features.get("operator_directive_ratio", 0.0),
        "target_density": features.get("operator_target_density", 0.0),
        "forensic_shield_ratio": features.get("psyche_forensic_shield_ratio", 0.0),
        "dissociative_index": features.get("psyche_dissociative_index", 0.0),
        "poetic_entropy": features.get("psyche_poetic_entropy", 0.0),
        "wrath_velocity": features.get("psyche_wrath_velocity", 0.0),
        "bureaucratic_weaponization_index": features.get("resilience_bureaucratic_weaponization_index", 0.0),
        "resource_exhaustion_metric": features.get("resilience_resource_exhaustion_metric", 0.0),
        "swarm_orchestration_density": features.get("resilience_swarm_orchestration_density", 0.0),
        "conspiracy_grounding_ratio": features.get("resilience_conspiracy_grounding_ratio", 0.0),
        "chaotic_good_tax": features.get("resilience_chaotic_good_tax", 0.0),
        "corporate_grit_tension": features.get("rainmaker_corporate_grit_tension", 0.0),
        "countdown_density": features.get("rainmaker_countdown_density", 0.0),
        "asset_structuring_weight": features.get("rainmaker_asset_structuring_weight", 0.0),
        "pitch_to_prose_ratio": features.get("rainmaker_pitch_formatting_ratio", 0.0),
        "agent_symmetry_ratio": features.get("telemetry_agent_symmetry_ratio", 0.0),
        "subtle_knife_discipline": features.get("telemetry_protocol_discipline", 0.0),
        "manic_velocity": features.get("telemetry_manic_velocity", 0.0),
    }


def is_allowed(source_path: str, prefixes: list[str]) -> bool:
    return any(source_path == p or source_path.startswith(p.rstrip("/") + "/") for p in prefixes)


def extract(args: argparse.Namespace) -> int:
    prefixes = args.allow_prefix or ["00_PROJECT_BRAIN", "05_OUTPUTS/intake_custody"]
    prepared = []
    inserted = 0
    with psycopg.connect(db(args), row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT to_regclass('lucidota_korpus.document_parse_run')")
            if cur.fetchone()["to_regclass"] is None:
                raise SystemExit("document_parse_run_missing")
            cur.execute(
                """
                SELECT run_uuid::text, source_path, source_sha256, output_markdown_path
                FROM lucidota_korpus.document_parse_run r
                WHERE status='parsed'
                  AND NOT EXISTS (
                    SELECT 1 FROM lucidota_archaeology.sticker_feature_source_receipt sr
                    WHERE sr.run_uuid=r.run_uuid AND sr.feature_version=%s
                  )
                ORDER BY created_at DESC
                LIMIT %s
                """,
                (FEATURE_VERSION, args.limit),
            )
            rows = [dict(r) for r in cur.fetchall()]
            for row in rows:
                if not is_allowed(row["source_path"], prefixes):
                    continue
                md_path = ROOT / row["output_markdown_path"]
                if not md_path.exists():
                    continue
                text = md_path.read_text(encoding="utf-8", errors="replace")
                features = all_features(text)
                cols = vector_columns(features)
                prepared.append({"row": row, "cols": cols, "raw_features": features, "text_sha256": hashlib.sha256(text.encode()).hexdigest()})
            if args.execute:
                for item in prepared:
                    row = item["row"]
                    cols = item["cols"]
                    artifact_uuid = deterministic_uuid("artifact", row["source_sha256"])
                    col_names = ["artifact_uuid", "feature_version", *cols.keys(), "raw_features"]
                    values = [artifact_uuid, FEATURE_VERSION, *cols.values(), json.dumps({"all_features": item["raw_features"], "source_run_uuid": row["run_uuid"], "text_sha256": item["text_sha256"]})]
                    placeholders = ["%s::uuid", "%s", *["%s" for _ in cols], "%s::jsonb"]
                    cur.execute(
                        f"INSERT INTO lucidota_archaeology.sticker_feature_vector({','.join(col_names)}) VALUES ({','.join(placeholders)}) RETURNING vector_uuid::text",
                        tuple(values),
                    )
                    vector_uuid = cur.fetchone()["vector_uuid"]
                    cur.execute(
                        """
                        INSERT INTO lucidota_archaeology.sticker_feature_source_receipt(run_uuid, vector_uuid, source_path, source_sha256, feature_version, detail)
                        VALUES (%s::uuid,%s::uuid,%s,%s,%s,%s::jsonb)
                        ON CONFLICT(run_uuid, feature_version) DO NOTHING
                        RETURNING receipt_uuid::text
                        """,
                        (row["run_uuid"], vector_uuid, row["source_path"], row["source_sha256"], FEATURE_VERSION, json.dumps({"script":"scripts/sticker_feature_extractor_v1.py"})),
                    )
                    if cur.fetchone():
                        inserted += 1
                conn.commit()
    report = {"action":"extract","execute_performed":bool(args.execute),"db_writes_performed":bool(args.execute),"graph_writes_performed":False,"allow_prefixes":prefixes,"prepared_vectors":len(prepared),"inserted_vectors":inserted,"sample":prepared[:3],"blockers":[]}
    write_report("extract_execute" if args.execute else "extract_dry_run", report)
    print(f"PREPARED_VECTORS={len(prepared)}")
    print(f"INSERTED_VECTORS={inserted}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Sticker feature vector extractor v1")
    p.add_argument("--database-url")
    sub = p.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("init-schema"); sp.add_argument("--execute", action="store_true"); sp.set_defaults(func=init_schema)
    sp = sub.add_parser("extract"); sp.add_argument("--execute", action="store_true"); sp.add_argument("--limit", type=int, default=25); sp.add_argument("--allow-prefix", action="append", default=[]); sp.set_defaults(func=extract)
    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
