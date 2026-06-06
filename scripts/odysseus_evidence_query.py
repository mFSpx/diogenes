#!/usr/bin/env python3
"""KRAMPUS EXPRESS EVIDENCE QUERY — Pull the real ledger from Postgres.

The evidence ledger isn't a Markdown file — it lives in:
  - lucidota_control.runtime_status_fact (state DB)
  - lucidota_go.graph_promotion_evidence_resolution (storage DB)

This script queries both and prints the live evidence state.

Usage:
  python3 scripts/odysseus_evidence_query.py [--json] [--subsystem odysseus]
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATE_URL = os.environ.get("ABSURD_SYSTEM_DATABASE_URL") or os.environ.get("DATABASE_URL") or "postgresql:///lucidota_state"
STORAGE_URL = os.environ.get("LUCIDOTA_GO_STORAGE_DSN") or "postgresql:///lucidota_storage"


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="KRAMPUS EXPRESS: Query evidence from Postgres")
    ap.add_argument("--subsystem", default="odysseus", help="Subsystem to query facts for")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    try:
        import psycopg2

        # ─── Query runtime facts ────────────────────────────────
        sconn = psycopg2.connect(STATE_URL)
        scur = sconn.cursor()
        scur.execute("""
            SELECT fact_key, fact_value, evidence_refs, derived_at
            FROM lucidota_control.runtime_status_fact
            WHERE subsystem = %s
            ORDER BY fact_key
        """, (args.subsystem,))
        facts = []
        for row in scur.fetchall():
            facts.append({
                "fact_key": row[0],
                "fact_value": row[1],
                "evidence_refs": row[2],
                "derived_at": row[3].isoformat() if row[3] else None,
            })
        sconn.close()

        # ─── Query evidence resolutions ─────────────────────────
        dconn = psycopg2.connect(STORAGE_URL)
        dcur = dconn.cursor()
        dcur.execute("""
            SELECT evidence_ref, ref_kind, resolved, resolver, detail, created_at
            FROM lucidota_go.graph_promotion_evidence_resolution
            WHERE detail->>'fact_keys' IS NOT NULL
            ORDER BY created_at DESC
            LIMIT 50
        """)
        resolutions = []
        for row in dcur.fetchall():
            resolutions.append({
                "evidence_ref": row[0],
                "ref_kind": row[1],
                "resolved": row[2],
                "resolver": row[3],
                "detail": row[4],
                "created_at": row[5].isoformat() if row[5] else None,
            })
        dconn.close()

        result = {
            "schema": "lucidota.krampus_express.evidence_query.v1",
            "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "subsystem": args.subsystem,
            "runtime_facts": facts,
            "evidence_resolutions": resolutions,
        }

        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(f"\n=== KRAMPUS EXPRESS: EVIDENCE LEDGER ===")
            print(f"  Subsystem: {args.subsystem}")
            print(f"  Runtime facts: {len(facts)}")
            for f in facts:
                fv = f["fact_value"]
                if isinstance(fv, dict):
                    summary = ", ".join(f"{k}: {v}" for k, v in list(fv.items())[:5])
                else:
                    summary = str(fv)[:80]
                print(f"    {f['fact_key']}: {summary}")
                for ref in f.get("evidence_refs", []):
                    print(f"      └─ {ref}")
            print(f"  Evidence resolutions: {len(resolutions)}")
            for r in resolutions[:5]:
                print(f"    {r['evidence_ref']} → resolved={r['resolved']} ({r['ref_kind']})")

        return 0

    except ImportError:
        print("psycopg2 required", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
