# DARWIN HAMMER — match 1057, survivor 1
# gen: 4
# parent_a: hybrid_hybrid_hybrid_privac_hybrid_hybrid_endpoi_m16_s1.py (gen3)
# parent_b: hybrid_percyphon_hybrid_endpoint_circ_m45_s0.py (gen2)
# born: 2026-05-29T23:32:35Z

"""hybrid_model_morphology_fusion.py
Integrates:
- Parent Algorithm A (privacy health scoring and VRAM workshare allocation)
- Parent Algorithm B (procedural slot generation informed by morphological indices)

Mathematical Bridge:
The health score `h_i` of each model (derived from reconstruction risk) is used to build a normalized work‑share vector **W**.
Morphology provides two shape descriptors: sphericity `σ` and flatness `φ`. Their mean `μ = (σ+φ)/2` acts as a global scaling factor.
The final allocation of procedural slots to a model is
    n_i = round( W_i * B * μ )
where `B` is a base slot budget. This fuses the linear algebra of the work‑share matrix from A with the geometric indices from B, and the resulting `n_i` drives the ternary offset of each generated `ProceduralSlot`.
"""

from __future__ import annotations

import sys
import random
import pathlib
import hashlib
import math
from dataclasses import dataclass, asdict
from typing import Any, Iterable, List, Dict
import numpy as np
from datetime import datetime, timezone

# ----------------------------------------------------------------------
# Shared Data Structures
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class ModelTier:
    """Simple descriptor for an ML model tier."""
    name: str
    ram_mb: int
    tier: str
    vram_mb: int


TIER_T1_QWEN_0_5B = ModelTier("qwen-0.5b", 512, "T1", 1024)
TIER_T2_REASONING = ModelTier("reasoning-t2", 3000, "T2", 2048)
TIER_T2_TOOL = ModelTier("tool-t2", 2600, "T2", 2048)
TIER_T3_QWEN_7B = ModelTier("qwen-7b", 7000, "T3", 4096)


@dataclass(frozen=True)
class Morphology:
    """Geometric description of an entity."""
    length: float
    width: float
    height: float
    mass: float


@dataclass(frozen=True)
class ProceduralSlot:
    """One slot generated for a procedural entity."""
    slot_index: int
    name: str
    alias: str
    persona: str
    uuid: str
    ternary_offset: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


# ----------------------------------------------------------------------
# Parent A – Privacy / Health Scoring
# ----------------------------------------------------------------------


def reconstruction_risk_score(unique_quasi_identifiers: int, total_records: int) -> float:
    """Risk = proportion of quasi‑identifiers; clamped to [0,1]."""
    if total_records <= 0:
        return 0.0
    return max(0.0, min(1.0, unique_quasi_identifiers / total_records))


def health_score(reconstruction_risk: float, recovery_priority: float) -> float:
    """
    Core health equation from Parent A.
    Both risk and priority are in [0,1]; higher values lower health.
    """
    return (1.0 - reconstruction_risk) * (1.0 - recovery_priority)


def recovery_priority_from_risk(risk: float) -> float:
    """
    In Parent A the recovery priority is *inversely* proportional to health.
    For a simple deterministic bridge we map priority directly to risk.
    """
    return risk  # linear mapping for demonstration


def model_health_vector(
    models: List[ModelTier],
    unique_quasi_identifiers: int,
    total_records: int,
) -> np.ndarray:
    """
    Compute a health vector **H** (shape (M,)) for the supplied models.
    The same risk is applied to every model – in a real system each model
    would have its own statistics.
    """
    risk = reconstruction_risk_score(unique_quasi_identifiers, total_records)
    priority = recovery_priority_from_risk(risk)
    h = health_score(risk, priority)
    # Broadcast same health to all models – can be varied later.
    return np.full(len(models), h, dtype=float)


def normalized_workshare(health_vec: np.ndarray) -> np.ndarray:
    """
    Convert health vector **H** into a normalized work‑share vector **W**
    (sums to 1). Uses simple proportional allocation.
    """
    if health_vec.size == 0:
        raise ValueError("health vector must contain at least one element")
    total = health_vec.sum()
    if total == 0.0:
        # Avoid division by zero – give equal share.
        return np.full_like(health_vec, 1.0 / health_vec.size)
    return health_vec / total


# ----------------------------------------------------------------------
# Parent B – Morphology & Procedural Generation
# ----------------------------------------------------------------------


def sphericity_index(length: float, width: float, height: float) -> float:
    """Geometric sphericity (dimensionless)."""
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length * width * height) ** (1.0 / 3.0) / length


def flatness_index(length: float, width: float, height: float) -> float:
    """Geometric flatness (dimensionless)."""
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length + width) / (2.0 * height)


def morphology_descriptor(morph: Morphology) -> tuple[float, float]:
    """Return (sphericity, flatness) for a morphology."""
    sigma = sphericity_index(morph.length, morph.width, morph.height)
    phi = flatness_index(morph.length, morph.width, morph.height)
    return sigma, phi


def _uuid_from_sha256(seed: str) -> str:
    h = hashlib.sha256(seed.encode("utf-8", errors="ignore")).hexdigest()
    return f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"


def _slot_name(seed: str, idx: int) -> tuple[str, str, str]:
    h = hashlib.sha256(f"{seed}:{idx}".encode()).hexdigest()
    name = f"Villager-{int(h[:6], 16) % 5000:04d}"
    alias = f"Alias-{h[6:10]}"
    persona = ["ledger", "runner", "witness", "archivist", "carrier", "scribe"][int(h[10:12], 16) % 6]
    return name, alias, persona


def generate_procedural_slots(
    seed: str,
    count: int,
    priority_factor: float,
) -> List[ProceduralSlot]:
    """
    Generate `count` procedural slots.
    The `priority_factor` (derived from health/workshare) linearly scales the
    ternary offset, embedding the privacy‑driven priority into the entity.
    """
    slots: List[ProceduralSlot] = []
    for i in range(count):
        name, alias, persona = _slot_name(seed, i)
        uuid = _uuid_from_sha256(f"{seed}:{i}")
        # ternary offset is an integer in [-3,3]; we map priority_factor∈[0,1] to that range.
        offset = int(round((priority_factor * 6) - 3))
        slots.append(
            ProceduralSlot(
                slot_index=i,
                name=name,
                alias=alias,
                persona=persona,
                uuid=uuid,
                ternary_offset=offset,
            )
        )
    return slots


# ----------------------------------------------------------------------
# Hybrid Core – Fusion of Workshare & Morphology
# ----------------------------------------------------------------------


def hybrid_slot_allocation(
    models: List[ModelTier],
    morph: Morphology,
    unique_quasi_identifiers: int,
    total_records: int,
    base_slot_budget: int = 30,
) -> Dict[str, List[ProceduralSlot]]:
    """
    Allocate procedural slots to each model using a fused metric.

    Steps:
    1. Compute health vector **H** for all models (Parent A).
    2. Normalise to work‑share vector **W**.
    3. Compute morphology scaling μ = (σ + φ) / 2 (Parent B).
    4. For each model i, determine slot count
           n_i = round( W_i * base_slot_budget * μ )
       and generate that many slots with a ternary offset proportional to W_i.
    Returns a mapping model_name → list[ProceduralSlot].
    """
    # 1‑2: Health & workshare
    H = model_health_vector(models, unique_quasi_identifiers, total_records)
    W = normalized_workshare(H)

    # 3: Morphology descriptor
    sigma, phi = morphology_descriptor(morph)
    mu = (sigma + phi) / 2.0  # global scaling factor

    allocation: Dict[str, List[ProceduralSlot]] = {}
    for idx, model in enumerate(models):
        # Step 4 – slot count
        slot_cnt = int(round(W[idx] * base_slot_budget * mu))
        slot_cnt = max(0, slot_cnt)  # guard against negative rounding
        # The priority factor for ternary offset is simply the model's workshare.
        slots = generate_procedural_slots(
            seed=f"{model.name}-{datetime.now(timezone.utc).isoformat()}",
            count=slot_cnt,
            priority_factor=W[idx],
        )
        allocation[model.name] = slots
    return allocation


def vram_loading_order(models: List[ModelTier], workshare: np.ndarray) -> List[ModelTier]:
    """
    Produce a VRAM loading order based on workshare: higher share → earlier load.
    Returns a new list sorted descending by workshare.
    """
    paired = list(zip(models, workshare))
    paired.sort(key=lambda pair: pair[1], reverse=True)
    return [model for model, _ in paired]


def summarize_allocation(allocation: Dict[str, List[ProceduralSlot]]) -> str:
    """Human‑readable summary of the hybrid allocation."""
    lines = ["Hybrid Allocation Summary:"]
    total = 0
    for model_name, slots in allocation.items():
        lines.append(f"  Model '{model_name}': {len(slots)} slots")
        total += len(slots)
    lines.append(f"Total slots generated: {total}")
    return "\n".join(lines)


# ----------------------------------------------------------------------
# Smoke Test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Define a small set of models
    models = [TIER_T1_QWEN_0_5B, TIER_T2_REASONING, TIER_T3_QWEN_7B]

    # Example morphology (a roughly cuboid object)
    morph = Morphology(length=1.2, width=0.9, height=0.8, mass=3.5)

    # Privacy parameters
    uqis = 150   # number of quasi‑identifiers observed
    total = 1000  # total records

    # Compute hybrid allocation
    allocation = hybrid_slot_allocation(
        models=models,
        morph=morph,
        unique_quasi_identifiers=uqis,
        total_records=total,
        base_slot_budget=30,
    )

    # Print summary
    print(summarize_allocation(allocation))

    # Determine VRAM loading order using the same workshare vector
    H = model_health_vector(models, uqis, total)
    W = normalized_workshare(H)
    load_order = vram_loading_order(models, W)
    print("\nSuggested VRAM loading order (high → low priority):")
    for m in load_order:
        print(f"  {m.name} (VRAM {m.vram_mb} MB)")

    # Ensure at least one slot was generated for each model
    for name, slots in allocation.items():
        assert isinstance(slots, list)
        for s in slots:
            assert isinstance(s, ProceduralSlot)
    sys.exit(0)