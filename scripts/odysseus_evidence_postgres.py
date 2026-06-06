#!/usr/bin/env python3
"""KRAMPUS EXPRESS EVIDENCE INGESTION — Write odysseus evidence to Postgres.

The evidence ledger lives in lucidota_control.runtime_status_fact, not in
Markdown. This script inserts the odysseus extraction evidence as facts.

Usage:
  python3 scripts/odysseus_evidence_postgres.py [--dry-run] [--json]
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DB_URL = os.environ.get("ABSURD_SYSTEM_DATABASE_URL") or os.environ.get("DATABASE_URL") or "postgresql:///lucidota_state"
STORAGE_URL = os.environ.get("LUCIDOTA_GO_STORAGE_DSN") or "postgresql:///lucidota_storage"


def now_z() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")[:19] + "Z"


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def load_receipt(path: str) -> dict[str, Any]:
    p = ROOT / path if not path.startswith("/") else Path(path)
    if p.exists():
        return json.loads(p.read_text())
    return {}


def insert_fact(cursor, subsystem: str, key: str, value: dict, evidence_refs: list[str]) -> None:
    """Upsert a runtime_status_fact row."""
    cursor.execute("""
        INSERT INTO lucidota_control.runtime_status_fact
            (subsystem, fact_key, fact_value, evidence_refs)
        VALUES (%s, %s, %s::jsonb, %s::jsonb)
        ON CONFLICT (subsystem, fact_key)
        DO UPDATE SET
            fact_value = EXCLUDED.fact_value,
            evidence_refs = EXCLUDED.evidence_refs,
            derived_at = now()
    """, (subsystem, key, json.dumps(value), json.dumps(evidence_refs)))


def insert_evidence(cursor, evidence_ref: str, ref_kind: str, resolved: bool, resolver: str, detail: dict) -> None:
    """Insert a graph_promotion_evidence_resolution row."""
    cursor.execute("""
        INSERT INTO lucidota_go.graph_promotion_evidence_resolution
            (evidence_ref, ref_kind, resolved, resolver, detail)
        VALUES (%s, %s, %s, %s, %s::jsonb)
    """, (evidence_ref, ref_kind, resolved, resolver, json.dumps(detail)))


def storage_conn():
    """Get a connection to the storage DB."""
    import psycopg2
    return psycopg2.connect(STORAGE_URL)


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="KRAMPUS EXPRESS: Write odysseus evidence to Postgres")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    t0 = time.time()

    # Load the latest extraction receipt
    extract = load_receipt("05_OUTPUTS/brag/riverml_extract_receipt_20260606T082654.json")
    manual = load_receipt("05_OUTPUTS/odysseus_manual/manual_build_20260606T082859Z.json")

    print(f"=== KRAMPUS EXPRESS: EVIDENCE INGESTION ===", file=sys.stderr)
    print(f"  DB: {DB_URL}", file=sys.stderr)

    if not extract:
        print("  [ERROR] Extraction receipt not found!", file=sys.stderr)
        return 1

    conn = None
    try:
        import psycopg2
        conn = psycopg2.connect(DB_URL)
        cur = conn.cursor()
    except Exception as e:
        print(f"  [ERROR] DB connection failed: {e}", file=sys.stderr)
        print(f"  [INFO]  Falling back to receipt-only mode.", file=sys.stderr)
        conn = None

    try:
        # ─── FACT 1: Odysseus Extraction Summary ─────────────────
        fact1 = {
            "files_extracted": extract.get("files_extracted", 0),
            "total_chars": extract.get("total_chars", 0),
            "total_shapes": extract.get("total_shapes", 0),
            "go25_chunks": extract.get("go25_chunks", 0),
            "o75_chunks": extract.get("o75_chunks", 0),
            "root414_hashes": extract.get("root414_hashes", 0),
            "riverml_available": extract.get("riverml_available", False),
            "stream_features": extract.get("stream_features_count", 0),
            "extraction_elapsed_s": extract.get("elapsed_s", 0),
            "source": "01_REPOS/odysseus",
            "origin": "https://github.com/pewdiepie-archdaemon/odysseus",
        }

        # ─── FACT 2: Subsystem Breakdown ────────────────────────
        by_subsystem = extract.get("by_subsystem", {})
        fact2 = {
            "api_routes": by_subsystem.get("api", 0),
            "core_modules": by_subsystem.get("core", 0),
            "ui_modules": by_subsystem.get("ui", 0),
            "test_files": by_subsystem.get("test", 0),
            "service_modules": by_subsystem.get("service", 0),
            "script_clis": by_subsystem.get("script", 0),
            "mcp_servers": by_subsystem.get("mcp", 0),
            "integration_bundles": by_subsystem.get("integration", 0),
            "infra_configs": by_subsystem.get("infra", 0),
        }

        # ─── FACT 3: ABSURD Queue Registration ──────────────────
        fact3 = {
            "registered_scripts": [
                "scripts/odysseus_riverml_extract.py",
                "scripts/odysseus_friday_snapshot.py",
            ],
            "scheduled_snapshot": "0 6 * * 5",
            "queue": "external_command (via absurd_queue_spine.py)",
        }

        # ─── FACT 4: Manual Build ───────────────────────────────
        volumes = manual.get("volumes", [])
        fact4 = {
            "volumes": {v["volume"]: {"lines": v.get("lines", 0), "sha256": v.get("sha256", "")[:16]} for v in volumes},
            "total_volumes": len(volumes),
        }

        evidence_refs = [
            "05_OUTPUTS/brag/riverml_extract_receipt_20260606T082654.json",
            "05_OUTPUTS/odysseus_manual/manual_build_20260606T082859Z.json",
            "scripts/absurd_queue_spine.py",
        ]

        if conn and not args.dry_run:
            # Write runtime facts
            insert_fact(cur, "odysseus", "odysseus_extraction", fact1, evidence_refs)
            insert_fact(cur, "odysseus", "odysseus_subsystems", fact2, evidence_refs)
            insert_fact(cur, "odysseus", "odysseus_absurd_registration", fact3, evidence_refs)
            insert_fact(cur, "odysseus", "odysseus_manual_build", fact4, evidence_refs)

            # Write individual evidence resolutions (to storage DB)
            ev_count = 0
            try:
                sconn = storage_conn()
                scur = sconn.cursor()
                insert_evidence(scur,
                    evidence_ref="05_OUTPUTS/brag/riverml_extract_receipt_20260606T082654.json",
                    ref_kind="file",
                    resolved=True,
                    resolver="scripts/odysseus_evidence_postgres.py",
                    detail={"fact_keys": ["odysseus_extraction", "odysseus_subsystems"]}
                )
                insert_evidence(scur,
                    evidence_ref="scripts/absurd_queue_spine.py",
                    ref_kind="source",
                    resolved=True,
                    resolver="scripts/odysseus_evidence_postgres.py",
                    detail={"fact_keys": ["odysseus_absurd_registration"]}
                )
                insert_evidence(scur,
                    evidence_ref="05_OUTPUTS/odysseus_manual/manual_build_20260606T082859Z.json",
                    ref_kind="file",
                    resolved=True,
                    resolver="scripts/odysseus_evidence_postgres.py",
                    detail={"fact_keys": ["odysseus_manual_build"]}
                )
                sconn.commit()
                sconn.close()
                ev_count = 3
            except Exception as e:
                print(f"  [WARN] Storage DB write failed: {e}", file=sys.stderr)

            conn.commit()
            print(f"  Written: 4 runtime facts + {ev_count} evidence resolutions", file=sys.stderr)
        elif args.dry_run:
            print(f"  [DRY-RUN] Would write: 4 runtime facts + 3 evidence resolutions", file=sys.stderr)
        else:
            print(f"  [SKIP] No DB connection", file=sys.stderr)

        # Always write a local receipt
        receipt = {
            "schema": "lucidota.odysseus.evidence_ingestion.v1",
            "status": "PASS",
            "generated_at": now_z(),
            "facts_written": 4 if conn and not args.dry_run else 0,
            "evidence_resolutions": ev_count if conn and not args.dry_run else 0,
            "subsystem": "odysseus",
            "evidence_refs": evidence_refs,
            "elapsed_s": round(time.time() - t0, 2),
            "dry_run": args.dry_run,
            "db_connected": conn is not None,
        }
        receipt_dir = ROOT / "05_OUTPUTS" / "odysseus_evidence"
        receipt_dir.mkdir(parents=True, exist_ok=True)
        receipt_path = receipt_dir / f"evidence_ingestion_{stamp()}.json"
        receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"  Receipt: {receipt_path}", file=sys.stderr)

        if args.json:
            print(json.dumps(receipt, indent=2))

        return 0

    except Exception as e:
        print(f"  [ERROR] {e}", file=sys.stderr)
        if conn:
            conn.rollback()
        return 1
    finally:
        if conn:
            conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
