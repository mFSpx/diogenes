#!/usr/bin/env python3
"""Emit a Percyphon village scaffold and optionally write it to Postgres."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
for site_packages in sorted(ROOT.glob(".venv/lib/python*/site-packages")):
    site_path = str(site_packages)
    if site_path not in sys.path:
        sys.path.insert(0, site_path)

try:
    import psycopg
except ModuleNotFoundError:  # pragma: no cover - alternate env fallback
    import psycopg2 as psycopg  # type: ignore[assignment]

DEFAULT_DSN = os.environ.get("LUCIDOTA_GO_STATE_DSN") or os.environ.get("DATABASE_URL") or "postgresql:///lucidota_state"


def build_scaffold(*, seed: str | None, villagers: list[str], fluid_slots: int, source: str) -> dict[str, Any]:
    from ALGOS.percyphon import procedural_entity_generator

    base = villagers or ([seed] if seed else [])
    scaffold = procedural_entity_generator(base, fluid_slots=fluid_slots)
    scaffold["source"] = source
    scaffold["seed_override"] = seed
    return scaffold


def write_db(dsn: str, scaffold: dict[str, Any]) -> None:
    slots = scaffold.get("slots") or []
    first = slots[0] if slots else {}
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO lucidota_go.percyphon_village (
                    vuuid,
                    name,
                    persona,
                    alias,
                    ternary_state,
                    slots,
                    relevance_confidence_bps,
                    seed,
                    authority
                ) VALUES (
                    %(vuuid)s,
                    %(name)s,
                    %(persona)s,
                    %(alias)s,
                    %(ternary_state)s,
                    %(slots)s::jsonb,
                    %(relevance_confidence_bps)s,
                    %(seed)s,
                    %(authority)s
                )
                ON CONFLICT (vuuid) DO UPDATE SET
                    name = EXCLUDED.name,
                    persona = EXCLUDED.persona,
                    alias = EXCLUDED.alias,
                    ternary_state = EXCLUDED.ternary_state,
                    slots = EXCLUDED.slots,
                    relevance_confidence_bps = EXCLUDED.relevance_confidence_bps,
                    seed = EXCLUDED.seed,
                    authority = EXCLUDED.authority,
                    updated_at = now()
                """,
                {
                    "vuuid": scaffold.get("uuid"),
                    "name": scaffold.get("name") or first.get("name", ""),
                    "persona": first.get("persona", "witness"),
                    "alias": first.get("alias", "Alias-0000"),
                    "ternary_state": int(first.get("ternary_offset", 0)),
                    "slots": json.dumps(slots),
                    "relevance_confidence_bps": int(scaffold.get("relevance_confidence_bps", 0)),
                    "seed": scaffold.get("seed", ""),
                    "authority": scaffold.get("authority", "procedural_scaffold_candidate_not_truth"),
                },
            )
        conn.commit()


def main() -> int:
    ap = argparse.ArgumentParser(description="Emit a runtime Percyphon village scaffold and optionally write it to Postgres.")
    ap.add_argument("--seed", default="")
    ap.add_argument("--source", default="Runtime")
    ap.add_argument("--villager", action="append", default=[])
    ap.add_argument("--fluid-slots", type=int, default=100)
    ap.add_argument("--dsn", default=DEFAULT_DSN)
    ap.add_argument("--write-db", action="store_true", default=True)
    ap.add_argument("--no-write-db", dest="write_db", action="store_false")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    scaffold = build_scaffold(seed=args.seed or None, villagers=args.villager, fluid_slots=args.fluid_slots, source=args.source)
    if args.write_db:
        write_db(args.dsn, scaffold)
    print(json.dumps(scaffold, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
