#!/usr/bin/env python3
"""Percyphon.ai: zero-VRAM procedural entity generator.

Architecture:
  Slots 1-28   — fixed identity mask (mirrors CKDOG1 soul positions)
  Slots 29-128 — procedural verbosity expansion (runtime fluid domain slots)
  Total: 128 xxHash128 coordinate slots per villager scaffold.

Constitutional constraint: emits only `procedural_scaffold_candidate_not_truth`.
No DB writes. No model calls. No randomness. Pure SHA-256 arithmetic.
"""
from __future__ import annotations

import hashlib
import json
import psycopg2
from dataclasses import asdict, dataclass
from typing import Any, Sequence

try:
    import xxhash
except ImportError:
    xxhash = None

_PERSONAS = ["ledger", "runner", "witness", "archivist", "carrier", "scribe"]
_FIXED_SLOT_COUNT = 28       # identity mask (mirrors CKDOG1 soul positions)
_PROCEDURAL_SLOT_COUNT = 100  # verbosity expansion slots (indices 29-128)
_TOTAL_SLOTS = _FIXED_SLOT_COUNT + _PROCEDURAL_SLOT_COUNT  # 128


def _sha256_hex(data: str) -> str:
    return hashlib.sha256(data.encode("utf-8", errors="ignore")).hexdigest()


def _xxhash128_int(data: str) -> int:
    """128-bit deterministic hash via xxhash128 (or sha256 fallback)."""
    if xxhash:
        return xxhash.xxh128(data).intdigest()
    raw = hashlib.sha256(data.encode("utf-8", errors="ignore")).digest()
    return int.from_bytes(raw[:16], "big")


def _uuid_from_sha256(seed: str) -> str:
    h = _sha256_hex(seed)
    return f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"


def _slot_name(seed: str, idx: int) -> tuple[str, str, str]:
    h = _sha256_hex(f"{seed}:{idx}")
    name = f"Villager-{int(h[:6], 16) % 5000:04d}"
    alias = f"Alias-{h[6:10]}"
    persona = _PERSONAS[int(h[10:12], 16) % len(_PERSONAS)]
    return name, alias, persona


def _ternary_state(seed: str, idx: int, base_offset: int) -> int:
    """Per-slot ternary modulation: base_offset ± hash spread → {-1, 0, +1}."""
    h = _sha256_hex(f"{seed}:ternary:{idx}")
    spread = int(h[:2], 16) % 3  # 0, 1, or 2
    raw = base_offset + spread - 1  # shift spread to -1..1 range
    return max(-1, min(1, raw))


def relevance_confidence_bps(seed: str) -> int:
    """Deterministic relevance confidence in basis points (0-10000)."""
    h = _sha256_hex(f"relevance:{seed}")
    return int(h[:4], 16) % 10001


def concept_expansion(seed: str, n: int = 8) -> list[str]:
    """Generates n semantically adjacent seed strings by hashing variants."""
    return [f"{seed}:expand:{i}" for i in range(n)]


def village_curator(village_seeds: list[str], top_k: int = 5000) -> list[str]:
    """Ranks seeds by relevance_confidence_bps descending, returns top_k."""
    ranked_seeds = sorted(village_seeds, key=relevance_confidence_bps, reverse=True)
    return ranked_seeds[:top_k]


def scaffold_log_entry(scaffold: dict, dsn: str) -> dict:
    """Inserts into lucidota_go.percyphon_scaffold_log."""
    conn = psycopg2.connect(dsn)
    cur = conn.cursor()
    source_id = scaffold['uuid']
    parser_name = 'percyphon_generator'
    proposed_term = 'ENTITY'
    claim = scaffold['name']
    proposed_item = json.dumps(scaffold)
    confidence_bps = scaffold['relevance_confidence_bps']
    # snap confidence_bps to nearest value in the canonical BPS set
    _BPS_SET = [0, 2, 4, 6, 10, 50, 69, 150]
    confidence_bps = min(_BPS_SET, key=lambda x: abs(x - confidence_bps))
    status = 'pending'
    cur.execute("""
        INSERT INTO lucidota_go.percyphon_scaffold_log (source_id, parser_name, proposed_term, claim, proposed_item, confidence_bps, status)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (source_id) DO NOTHING
        RETURNING *
    """, (source_id, parser_name, proposed_term, claim, proposed_item, confidence_bps, status))
    row = cur.fetchone()
    conn.commit()
    cur.close()
    conn.close()
    if row:
        return dict(row)
    else:
        return None


@dataclass(frozen=True)
class ProceduralSlot:
    slot_index: int          # 1-indexed, 1..128
    name: str
    alias: str
    persona: str
    uuid: str
    ternary_offset: int      # per-slot ternary state: -1, 0, or +1
    coord_128: int           # xxHash128-compatible 128-bit coordinate

    def as_dict(self) -> dict[str, Any]:
        return {
            "slot_index": self.slot_index,
            "name": self.name,
            "alias": self.alias,
            "persona": self.persona,
            "uuid": self.uuid,
            "ternary_offset": self.ternary_offset,
            "coord_128": self.coord_128,
        }

def procedural_entity_generator(villagers, psyche_wrath_velocity=0.0, psyche_forensic_shield_ratio=0.0, fluid_slots=100) -> dict:
    seed = "|".join(str(v) for v in (villagers or [])[:5000]) or "lucidota-villager-baseline"
    slots = []
    for slot_index in range(1, 30):
        name, alias, persona = _slot_name(seed, slot_index)
        ternary = _ternary_state(seed, slot_index, int(psyche_wrath_velocity*100))
        coord = _xxhash128_int(f"{seed}:coord:{slot_index}")
        uuid = _uuid_from_sha256(f"{seed}:fixed:{slot_index}")
        slots.append(ProceduralSlot(slot_index, name, alias, persona, uuid, ternary, coord).as_dict())
    for idx in range(fluid_slots):
        villager_ref = villagers[idx % len(villagers)] if villagers else seed
        slot_index = 29 + idx
        name, alias, persona = _slot_name(f"{seed}:{villager_ref}", slot_index)
        ternary = _ternary_state(seed, slot_index, int(psyche_forensic_shield_ratio*100))
        coord = _xxhash128_int(f"{seed}:fluid:{idx}:{villager_ref}")
        uuid = _uuid_from_sha256(f"{seed}:fluid:{slot_index}:{villager_ref}")
        slots.append(ProceduralSlot(slot_index, name, alias, persona, uuid, ternary, coord).as_dict())
    return {
        "seed": seed,
        "slots": slots,
        "source_count": min(5000, len(villagers) or 5000),
        "uuid": _uuid_from_sha256(seed),
        "name": slots[0]["name"],
        "relevance_confidence_bps": relevance_confidence_bps(seed),
        "authority": "procedural_scaffold_candidate_not_truth"
    }

def villager_scaffold(seed: str) -> dict:
    return procedural_entity_generator([seed], fluid_slots=100)