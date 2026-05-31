#!/usr/bin/env python3
"""percyphon_village_seed.py — deterministic village seeder.

Given a seed list, generates Percyphon scaffolds and upserts them into
lucidota_go.percyphon_village. Uses xxHash128-compatible coordinates (SHA-256[:16]).

Mutation class: candidate_writer
Receipt mode: LOCAL_FILE_PRODUCT + ABSURD_POSTGRES_RUNTIME
Authority: procedural_scaffold_candidate_not_truth only.

Usage:
    python3 scripts/percyphon_village_seed.py [--count N] [--dry-run]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

# Repo root on sys.path so ALGOS.percyphon imports
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import psycopg2
import psycopg2.extras

from ALGOS.percyphon import (
    ProceduralSlot,
    _sha256_hex,
    _xxhash128_int,
    _uuid_from_sha256,
    _slot_name,
    _ternary_state,
    _FIXED_SLOT_COUNT,
    _PROCEDURAL_SLOT_COUNT,
    procedural_entity_generator,
)


# ---------------------------------------------------------------------------
# Sample seed corpus (50 deterministic entries — domain anchors + ontology)
# ---------------------------------------------------------------------------

_SAMPLE_SEEDS = [
    # Ontology anchors
    "go25:OBJECT",
    "go25:EVENT",
    "go25:EDGE",
    "go25:claim_cluster",
    "go25:evidence_thread",
    "go25:authority_class",
    "go25:graph_promotion",
    "go25:staging_packet",
    "go25:graph_item",
    "go25:graph_edge",
    # Percyphon layer
    "percyphon:scaffold",
    "percyphon:village",
    "percyphon:slot_identity",
    "percyphon:fluid_domain",
    "percyphon:ternary_offset",
    "percyphon:psyche_wrath",
    "percyphon:psyche_forensic",
    "percyphon:persona:ledger",
    "percyphon:persona:runner",
    "percyphon:persona:witness",
    "percyphon:persona:archivist",
    "percyphon:persona:carrier",
    "percyphon:persona:scribe",
    # Domain handles
    "domain:legal_corpus",
    "domain:code_review",
    "domain:claim_extraction",
    "domain:entity_resolution",
    "domain:graph_truth",
    "domain:candidate_layer",
    "domain:routing_decision",
    # Runtime concepts
    "runtime:absurd_queue",
    "runtime:lucidota_state",
    "runtime:lucidota_storage",
    "runtime:term_registry",
    "runtime:soul_registry",
    "runtime:route_decision",
    "runtime:diogenes_mirror",
    "runtime:ckdog1_soul",
    # Workflow nodes
    "workflow:intake",
    "workflow:extract",
    "workflow:score",
    "workflow:promote",
    "workflow:archive",
    "workflow:dispatch",
    "workflow:gate",
    "workflow:receipt",
    # Identity markers
    "identity:lucidota",
    "identity:northern_strike",
    "identity:indy_reads",
    "identity:diogenes",
]

assert len(_SAMPLE_SEEDS) == 50, f"Expected 50 seeds, got {len(_SAMPLE_SEEDS)}"


# ---------------------------------------------------------------------------
# Synthetic seed expansion to fill 5000-row target
# Deterministic: no randomness, all SHA-256 derived namespaces
# ---------------------------------------------------------------------------

_SYNTHETIC_NAMESPACES = [
    "entity", "event", "edge", "claim", "evidence", "witness",
    "regulator", "adversary", "mask", "grip", "snare", "void",
    "archivist", "runner", "carrier", "scribe", "ledger",
    "corpus", "routing", "gate", "absurd", "staging",
]


def _synthetic_seeds(target_total: int = 5000) -> list[str]:
    """Generate deterministic synthetic seeds to reach target_total.
    Anchors come first; synthetic slots fill the remainder."""
    seeds = list(_SAMPLE_SEEDS)
    needed = target_total - len(seeds)
    if needed <= 0:
        return seeds
    per_ns = needed // len(_SYNTHETIC_NAMESPACES) + 1
    idx = 0
    for ns in _SYNTHETIC_NAMESPACES:
        for i in range(per_ns):
            if len(seeds) >= target_total:
                break
            seeds.append(f"synthetic:{ns}:{idx:05d}")
            idx += 1
        if len(seeds) >= target_total:
            break
    return seeds[:target_total]


# ---------------------------------------------------------------------------
# Scaffold → DB row
# ---------------------------------------------------------------------------

def seed_to_row(seed: str) -> dict:
    """Generate a single villager row from a seed string."""
    scaffold = procedural_entity_generator(
        [seed],
        psyche_wrath_velocity=0.0,
        psyche_forensic_shield_ratio=0.0,
        fluid_slots=100,
    )
    slots = scaffold["slots"]
    slot1 = slots[0]  # slot_index=1 is the primary identity handle

    # vuuid is derived from the seed directly (stable, not slot-1 uuid)
    vuuid = _uuid_from_sha256(f"village:{seed}")

    # relevance_confidence_bps: deterministic from seed hash
    h = _sha256_hex(f"relevance:{seed}")
    relevance_bps = int(h[:4], 16) % 10001  # 0..10000

    return {
        "vuuid": vuuid,
        "name": slot1["name"],
        "persona": slot1["persona"],
        "alias": slot1["alias"],
        "ternary_state": slot1["ternary_offset"],
        "slots": json.dumps(slots),
        "relevance_confidence_bps": relevance_bps,
        "seed": seed,
        "authority": "procedural_scaffold_candidate_not_truth",
    }


# ---------------------------------------------------------------------------
# Upsert
# ---------------------------------------------------------------------------

UPSERT_SQL = """
INSERT INTO lucidota_go.percyphon_village
    (vuuid, name, persona, alias, ternary_state, slots,
     relevance_confidence_bps, seed, authority, updated_at)
VALUES
    (%(vuuid)s, %(name)s, %(persona)s, %(alias)s, %(ternary_state)s,
     %(slots)s::jsonb, %(relevance_confidence_bps)s, %(seed)s,
     %(authority)s, now())
ON CONFLICT (vuuid) DO UPDATE SET
    name = EXCLUDED.name,
    persona = EXCLUDED.persona,
    alias = EXCLUDED.alias,
    ternary_state = EXCLUDED.ternary_state,
    slots = EXCLUDED.slots,
    relevance_confidence_bps = EXCLUDED.relevance_confidence_bps,
    seed = EXCLUDED.seed,
    updated_at = now();
"""


def upsert_villagers(seeds: list[str], dsn: str, dry_run: bool = False) -> int:
    rows = [seed_to_row(s) for s in seeds]
    if dry_run:
        print(f"[dry-run] Would upsert {len(rows)} rows")
        print(json.dumps(rows[0], indent=2))
        return len(rows)

    conn = psycopg2.connect(dsn)
    try:
        with conn, conn.cursor() as cur:
            psycopg2.extras.execute_batch(cur, UPSERT_SQL, rows, page_size=50)
        print(f"[percyphon_village_seed] upserted {len(rows)} villagers")
    finally:
        conn.close()
    return len(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Seed percyphon_village table")
    parser.add_argument("--count", type=int, default=5000,
                        help="Total villager rows to seed (default 5000)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print first row; do not write to DB")
    parser.add_argument("--dsn", default=os.environ.get(
        "LUCIDOTA_GO_STORAGE_DSN", "postgresql:///lucidota_storage"),
                        help="Storage DSN")
    args = parser.parse_args()

    seeds = _synthetic_seeds(args.count)
    n = upsert_villagers(seeds, args.dsn, dry_run=args.dry_run)
    print(f"[receipt] percyphon_village upsert={n} dry_run={args.dry_run}")


if __name__ == "__main__":
    main()
