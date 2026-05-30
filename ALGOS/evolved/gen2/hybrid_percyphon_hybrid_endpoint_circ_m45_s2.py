# DARWIN HAMMER — match 45, survivor 2
# gen: 2
# parent_a: percyphon.py (gen0)
# parent_b: hybrid_endpoint_circuit_bre_serpentina_self_righ_m18_s0.py (gen1)
# born: 2026-05-29T23:23:40Z

"""Hybrid algorithm merging Percyphon procedural entity generation (Parent A) with
adaptive circuit‑breaker morphology analysis (Parent B).

Mathematical bridge:
- Each ProceduralSlot from Percyphon carries a deterministic UUID derived from a SHA‑256
  seed. The UUID is interpreted as four 64‑bit hex fragments and mapped to continuous
  values in [0,1].
- Those four scalars are transformed into a Morphology (length, width, height, mass).
- The morphology feeds the sphericity_index and flatness_index from Parent B.
- The adaptive circuit‑breaker threshold is then modulated by the ternary_offset
  (‑1, 0, +1) produced by Percyphon, yielding a unified threshold:
      threshold = base_failure_threshold * (1 + sphericity * flatness) * (1 + 0.1*ternary_offset)
- The hybrid functions expose generation, adaptive breaker creation and recovery
  simulation for each entity, fully integrating the governing equations of both parents.
"""

import hashlib
import json
import math
import random
import sys
import pathlib
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Sequence, List, Dict

import numpy as np

# ---------- Parent A structures (Percyphon) ----------
@dataclass(frozen=True)
class ProceduralSlot:
    slot_index: int
    name: str
    alias: str
    persona: str
    uuid: str
    ternary_offset: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _uuid_from_sha256(seed: str) -> str:
    h = hashlib.sha256(seed.encode("utf-8", errors="ignore")).hexdigest()
    return f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"


def _slot_name(seed: str, idx: int) -> tuple[str, str, str]:
    h = hashlib.sha256(f"{seed}:{idx}".encode()).hexdigest()
    name = f"Villager-{int(h[:6], 16) % 5000:04d}"
    alias = f"Alias-{h[6:10]}"
    persona = ["ledger", "runner", "witness", "archivist", "carrier", "scribe"][int(h[10:12], 16) % 6]
    return name, alias, persona


def procedural_entity_generator(
    villagers: Sequence[str] | None = None,
    *,
    psyche_wrath_velocity: float = 0.0,
    psyche_forensic_shield_ratio: float = 0.0,
    fluid_slots: int = 88,
) -> dict[str, Any]:
    villagers = list(villagers or [])
    seed = "|".join(villagers[:5000]) or "lucidota-villager-baseline"
    ternary_offset = 0
    if psyche_wrath_velocity > psyche_forensic_shield_ratio:
        ternary_offset = 1
    elif psyche_forensic_shield_ratio > psyche_wrath_velocity:
        ternary_offset = -1

    slots: List[ProceduralSlot] = []
    for idx in range(12):
        name, alias, persona = _slot_name(seed, idx)
        slots.append(
            ProceduralSlot(
                slot_index=idx,
                name=name,
                alias=alias,
                persona=persona,
                uuid=_uuid_from_sha256(f"{seed}:{idx}"),
                ternary_offset=ternary_offset,
            )
        )

    fluid: List[dict[str, Any]] = []
    for idx in range(int(fluid_slots)):
        fluid.append(
            {
                "fluid_slot": idx,
                "villager_ref": villagers[idx % len(villagers)] if villagers else f"baseline-{idx:04d}",
                "slot_uuid": _uuid_from_sha256(f"{seed}:fluid:{idx}"),
                "offset": ternary_offset,
            }
        )

    return {
        "schema": "lucidota.percyphon.procedural_entity_generator.v1",
        "source_count": min(5000, len(villagers) or 5000),
        "slot_count": 12,
        "fluid_slot_count": int(fluid_slots),
        "psyche_wrath_velocity": float(psyche_wrath_velocity),
        "psyche_forensic_shield_ratio": float(psyche_forensic_shield_ratio),
        "ternary_offset": ternary_offset,
        "slots": [s.as_dict() for s in slots],
        "fluid_slots": fluid,
        "zero_vram": True,
        "note": "integer arithmetic only; identity masks are procedural and not model-generated",
    }

# ---------- Parent B structures (Morphology & CircuitBreaker) ----------
@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float


class EndpointCircuitBreaker:
    def __init__(self, failure_threshold: int = 3):
        self.failure_threshold = failure_threshold
        self.failures = 0
        self.open = False
        self.last_event_at = ""

    def record_success(self) -> None:
        self.failures = 0
        self.open = False
        self.last_event_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    def record_failure(self) -> None:
        self.failures += 1
        self.open = self.failures >= self.failure_threshold
        self.last_event_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    def allow(self) -> bool:
        return not self.open

    def as_dict(self) -> dict[str, Any]:
        return {
            "failure_threshold": self.failure_threshold,
            "failures": self.failures,
            "open": self.open,
            "last_event_at": self.last_event_at,
        }


def sphericity_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length * width * height) ** (1.0 / 3.0) / length


def flatness_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length + width) / (2.0 * height)


def righting_time_index(m: Morphology, b: float = 1.0 / 3.0, k: float = 0.35, neck_lever: float = 1.0) -> float:
    if m.mass <= 0 or neck_lever <= 0:
        raise ValueError("mass and neck_lever must be positive")
    fi = flatness_index(m.length, m.width, m.height)
    return (m.mass ** b) * math.exp(k * fi) / neck_lever


def recovery_priority(m: Morphology, max_index: float = 10.0) -> float:
    return max(0.0, min(1.0, righting_time_index(m) / max_index))

# ---------- Hybrid core -------------------------------------------------
def _hex_fragment_to_float(fragment: str) -> float:
    """Map a hex fragment (up to 16 chars) to a float in [0, 1)."""
    int_val = int(fragment, 16)
    max_val = 16 ** len(fragment) - 1
    return int_val / max_val if max_val else 0.0


def slot_uuid_to_morphology(slot: ProceduralSlot) -> Morphology:
    """
    Deterministically convert a slot's UUID into a Morphology.
    The UUID format is 8-4-4-4-12 hex characters; we concatenate them and split
    into four equal fragments to drive length, width, height, mass.
    """
    clean = slot.uuid.replace("-", "")
    # Ensure at least 16 chars per fragment; pad if necessary.
    fragment_len = max(1, len(clean) // 4)
    fragments = [clean[i * fragment_len : (i + 1) * fragment_len] for i in range(4)]
    # Convert to floats in [0,1)
    scalars = [_hex_fragment_to_float(f) for f in fragments]
    # Map to physically plausible ranges
    length = 0.5 + 2.0 * scalars[0]   # 0.5 – 2.5
    width = 0.5 + 2.0 * scalars[1]    # 0.5 – 2.5
    height = 0.5 + 2.0 * scalars[2]   # 0.5 – 2.5
    mass = 1.0 + 9.0 * scalars[3]     # 1 – 10
    return Morphology(length=length, width=width, height=height, mass=mass)


def adaptive_circuit_breaker_hybrid(
    slot: ProceduralSlot,
    morphology: Morphology,
    base_failure_threshold: int = 3,
) -> EndpointCircuitBreaker:
    """
    Create an EndpointCircuitBreaker whose threshold is influenced by:
    - Parent B's sphericity * flatness term,
    - Parent A's ternary_offset (‑1, 0, +1) as a 10 % scaling factor.
    """
    si = sphericity_index(morphology.length, morphology.width, morphology.height)
    fi = flatness_index(morphology.length, morphology.width, morphology.height)
    # Base adaptation from Parent B
    adapted = base_failure_threshold * (1.0 + si * fi)
    # Incorporate ternary_offset from Parent A
    adapted *= (1.0 + 0.1 * slot.ternary_offset)
    # Ensure integer threshold >= 1
    final_threshold = max(1, int(round(adapted)))
    return EndpointCircuitBreaker(final_threshold)


def hybrid_operation(slot: ProceduralSlot, morphology: Morphology) -> bool:
    """
    Perform the hybrid operation: instantiate the adaptive breaker and query its state.
    """
    breaker = adaptive_circuit_breaker_hybrid(slot, morphology)
    return breaker.allow()


def simulate_recovery(slot: ProceduralSlot, morphology: Morphology) -> float:
    """
    Simulate recovery time for an entity.
    If the adaptive breaker permits operation, recovery is accelerated (×0.5);
    otherwise it is penalised (×2.0). The base recovery time derives from Parent B.
    """
    breaker = adaptive_circuit_breaker_hybrid(slot, morphology)
    base_time = righting_time_index(morphology)
    return base_time * (0.5 if breaker.allow() else 2.0)


def hybrid_entity_generator(
    villagers: Sequence[str] | None = None,
    *,
    psyche_wrath_velocity: float = 0.0,
    psyche_forensic_shield_ratio: float = 0.0,
    fluid_slots: int = 88,
    base_failure_threshold: int = 3,
) -> dict[str, Any]:
    """
    Full pipeline:
    1. Generate procedural slots (Parent A).
    2. Map each slot to a Morphology.
    3. Build an adaptive circuit breaker per slot.
    4. Compute hybrid operation flag and simulated recovery.
    Returns a rich JSON‑compatible structure.
    """
    base = procedural_entity_generator(
        villagers,
        psyche_wrath_velocity=psyche_wrath_velocity,
        psyche_forensic_shield_ratio=psyche_forensic_shield_ratio,
        fluid_slots=fluid_slots,
    )

    enriched_slots = []
    for slot_dict in base["slots"]:
        slot = ProceduralSlot(**slot_dict)
        morph = slot_uuid_to_morphology(slot)
        breaker = adaptive_circuit_breaker_hybrid(slot, morph, base_failure_threshold)
        enriched = {
            "slot": slot.as_dict(),
            "morphology": asdict(morph),
            "circuit_breaker": breaker.as_dict(),
            "operation_allowed": breaker.allow(),
            "simulated_recovery_time": simulate_recovery(slot, morph),
            "recovery_priority": recovery_priority(morph),
        }
        enriched_slots.append(enriched)

    result = {
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "base_failure_threshold": base_failure_threshold,
        },
        "entity_summary": {
            "total_slots": len(enriched_slots),
            "average_recovery_time": float(np.mean([e["simulated_recovery_time"] for e in enriched_slots])),
        },
        "slots": enriched_slots,
        "fluid_slots": base["fluid_slots"],
    }
    return result

# ---------- Smoke test -------------------------------------------------
if __name__ == "__main__":
    # Example villager list (could be empty)
    sample_villagers = [f"villager_{i}" for i in range(7)]
    hybrid_output = hybrid_entity_generator(
        villagers=sample_villagers,
        psyche_wrath_velocity=0.4,
        psyche_forensic_shield_ratio=0.2,
        fluid_slots=5,
        base_failure_threshold=4,
    )
    # Print a concise summary
    print(json.dumps({
        "metadata": hybrid_output["metadata"],
        "entity_summary": hybrid_output["entity_summary"],
        "first_slot": hybrid_output["slots"][0],
    }, indent=2))
    # Ensure no exception on repeated calls
    _ = hybrid_entity_generator()
    sys.exit(0)