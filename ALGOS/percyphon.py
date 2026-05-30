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
from dataclasses import asdict, dataclass
from typing import Any, Sequence


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

_PERSONAS = ["ledger", "runner", "witness", "archivist", "carrier", "scribe"]
_FIXED_SLOT_COUNT = 28       # identity mask (mirrors CKDOG1 soul positions)
_PROCEDURAL_SLOT_COUNT = 100  # verbosity expansion slots (indices 29-128)
_TOTAL_SLOTS = _FIXED_SLOT_COUNT + _PROCEDURAL_SLOT_COUNT  # 128


def _sha256_hex(data: str) -> str:
    return hashlib.sha256(data.encode("utf-8", errors="ignore")).hexdigest()


def _xxhash128_int(data: str) -> int:
    """128-bit deterministic hash via sha256 first 16 bytes (zero-dep fallback)."""
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


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Core generator
# ---------------------------------------------------------------------------

def procedural_entity_generator(
    villagers: Sequence[str] | None = None,
    *,
    psyche_wrath_velocity: float = 0.0,
    psyche_forensic_shield_ratio: float = 0.0,
    fluid_slots: int = 100,
) -> dict[str, Any]:
    """Generate a deterministic 128-slot Percyphon scaffold from a Villager seed slice.

    Slots 1-28:   fixed identity mask.
    Slots 29-128: procedural verbosity expansion (fluid_slots capped at 100).
    """
    villagers = list(villagers or [])
    seed = "|".join(villagers[:5000]) or "lucidota-villager-baseline"

    # Scalar ternary base offset from psyche dials
    if psyche_wrath_velocity > psyche_forensic_shield_ratio:
        base_offset = 1
    elif psyche_forensic_shield_ratio > psyche_wrath_velocity:
        base_offset = -1
    else:
        base_offset = 0

    fluid_slots = max(0, min(int(fluid_slots), _PROCEDURAL_SLOT_COUNT))

    slots: list[ProceduralSlot] = []

    # Fixed identity slots (1-28)
    for idx in range(1, _FIXED_SLOT_COUNT + 1):
        name, alias, persona = _slot_name(seed, idx)
        slot = ProceduralSlot(
            slot_index=idx,
            name=name,
            alias=alias,
            persona=persona,
            uuid=_uuid_from_sha256(f"{seed}:{idx}"),
            ternary_offset=_ternary_state(seed, idx, base_offset),
            coord_128=_xxhash128_int(f"{seed}:coord:{idx}"),
        )
        slots.append(slot)

    # Procedural verbosity expansion slots (29-128)
    for local_idx in range(fluid_slots):
        idx = _FIXED_SLOT_COUNT + 1 + local_idx  # 29..128
        villager_ref = (
            villagers[local_idx % len(villagers)]
            if villagers
            else f"baseline-{local_idx:04d}"
        )
        name, alias, persona = _slot_name(f"{seed}:{villager_ref}", idx)
        slot = ProceduralSlot(
            slot_index=idx,
            name=name,
            alias=alias,
            persona=persona,
            uuid=_uuid_from_sha256(f"{seed}:fluid:{idx}:{villager_ref}"),
            ternary_offset=_ternary_state(seed, idx, base_offset),
            coord_128=_xxhash128_int(f"{seed}:coord:{idx}:{villager_ref}"),
        )
        slots.append(slot)

    return {
        "schema": "lucidota.percyphon.procedural_entity_generator.v2",
        "source_count": min(5000, len(villagers) or 5000),
        "fixed_slot_count": _FIXED_SLOT_COUNT,
        "fluid_slot_count": fluid_slots,
        "total_slot_count": len(slots),
        "psyche_wrath_velocity": float(psyche_wrath_velocity),
        "psyche_forensic_shield_ratio": float(psyche_forensic_shield_ratio),
        "base_ternary_offset": base_offset,
        "slots": [s.as_dict() for s in slots],
        "zero_vram": True,
        "authority": "procedural_scaffold_candidate_not_truth",
        "note": (
            "Slots 1-28 fixed identity mask (CKDOG1 mirror). "
            "Slots 29-128 procedural verbosity expansion. "
            "xxHash128-compatible 128-bit coordinates via sha256[:16]. "
            "Per-slot ternary modulation from psyche dials + hash spread."
        ),
    }


# ---------------------------------------------------------------------------
# Convenience: single-villager scaffold (for DB upsert path)
# ---------------------------------------------------------------------------

def villager_scaffold(seed: str) -> dict[str, Any]:
    """Generate a single-seed scaffold for upsert into percyphon_village."""
    return procedural_entity_generator([seed])


if __name__ == "__main__":
    result = procedural_entity_generator(
        ["claim-alpha", "claim-beta", "ontology-term-x"],
        psyche_wrath_velocity=0.8,
        psyche_forensic_shield_ratio=0.3,
    )
    print(json.dumps({
        "schema": result["schema"],
        "total_slot_count": result["total_slot_count"],
        "base_ternary_offset": result["base_ternary_offset"],
        "slot_0": result["slots"][0],
        "slot_27": result["slots"][27],
        "slot_28": result["slots"][28],
    }, indent=2))
