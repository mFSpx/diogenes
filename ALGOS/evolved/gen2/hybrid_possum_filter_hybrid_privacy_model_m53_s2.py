# DARWIN HAMMER — match 53, survivor 2
# gen: 2
# parent_a: possum_filter.py (gen0)
# parent_b: hybrid_privacy_model_pool_m7_s2.py (gen1)
# born: 2026-05-29T23:23:51Z

"""Hybrid filter‑selection algorithm merging Possum‑style spatial‑signature filtering
(PARENT ALGORITHM A) with the privacy‑aware model‑resource linear formulation
(PARENT ALGORITHM B).

Mathematical bridge
-------------------
* For each **Entity** we define a 2‑dimensional resource vector  
  **eᵢ** = [ dᵢ , pᵢ ] where  

  • dᵢ = haversine distance (in metres) from a reference location – this mirrors
    the distance‑threshold logic of *keep_candidate* in algorithm A.  

  • pᵢ = β·σᵢ, σᵢ = 1 if the entity’s *signature* collides with any other entity,
    otherwise 0.  This treats signature duplication as a “privacy‑load” analogue
    to the privacy‑load term of algorithm B.

* For each **ModelTier** we reuse the resource vector defined in algorithm B:  

  **mⱼ** = [ RAMⱼ , α·τⱼ·μ ] where  

  • RAMⱼ is the model’s RAM consumption,  

  • τⱼ is the tier factor (T1→1, T2→2, T3→3),  

  • μ = mean(privacy_risk) over the provided records,  

  • α is a scaling constant.

Stacking all vectors yields a combined resource matrix **A** (rows = entities∪models,
columns = [spatial/RAM‑load , privacy‑load]).  Selecting a subset corresponds to a
binary indicator **x** and must satisfy the linear constraints  

    Aᵀ·x ≤ [ spatial_budget , privacy_budget ]

where *spatial_budget* is the total allowed distance (or 0 for pure model
selection) and *privacy_budget* is the privacy‑budget from algorithm B.
The greedy algorithm respects both topologies in a single unified decision
process.
"""

from __future__ import annotations

import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set

import numpy as np

# ---------- Parent A: spatial‑signature utilities ----------
@dataclass(frozen=True)
class Entity:
    id: str
    lat: float
    lon: float
    category: str
    score: float = 0.0
    address_signature: str = ""


def haversine_m(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Great‑circle distance in metres between two (lat,lon) points."""
    lat1, lon1 = map(math.radians, a)
    lat2, lon2 = map(math.radians, b)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    return 2 * 6_371_000.0 * math.asin(min(1.0, math.sqrt(h)))


def signature(e: Entity) -> str:
    """Canonical signature used for duplicate detection."""
    return (e.address_signature or e.category).strip().lower()


# ---------- Parent B: privacy & model utilities ----------
def reconstruction_risk_score(unique_quasi_identifiers: int, total_records: int) -> float:
    """Normalized reconstruction risk in [0,1]."""
    return 0.0 if total_records <= 0 else max(0.0, min(1.0, unique_quasi_identifiers / total_records))


def privacy_risk_vector(
    records: List[Dict[str, Any]], quasi_id_key: str = "quasi_id"
) -> np.ndarray:
    """Vector r where r_i is the reconstruction risk for record i."""
    total = len(records)
    risks = [
        reconstruction_risk_score(rec.get(quasi_id_key, 0), total) for rec in records
    ]
    return np.array(risks, dtype=float)


@dataclass(frozen=True)
class ModelTier:
    name: str
    ram_mb: int
    tier: str  # "T1", "T2", "T3"


# ---------- Hybrid resource constructions ----------
def entity_resource_matrix(
    entities: List[Entity],
    reference: tuple[float, float],
    beta: float = 0.1,
) -> np.ndarray:
    """
    Build resource matrix for entities (rows = entities, cols = [spatial, privacy]).
    - spatial column: haversine distance to *reference*.
    - privacy column: β if the entity shares a signature with any other entity,
      otherwise 0.
    """
    # Pre‑compute signature frequencies
    sig_counts: Dict[str, int] = {}
    for e in entities:
        sig = signature(e)
        sig_counts[sig] = sig_counts.get(sig, 0) + 1

    rows = []
    for e in entities:
        dist = haversine_m((e.lat, e.lon), reference)
        dup = 1 if sig_counts[signature(e)] > 1 else 0
        privacy_load = beta * dup
        rows.append([dist, privacy_load])
    return np.array(rows, dtype=float)


def model_resource_matrix(
    models: List[ModelTier],
    privacy_risk: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Build combined resource matrix for models (rows = models, cols = [RAM, privacy]).
    Privacy‑load = α·tier_factor·mean(privacy_risk).
    """
    mean_risk = float(privacy_risk.mean()) if privacy_risk.size else 0.0
    tier_factor = {"T1": 1.0, "T2": 2.0, "T3": 3.0}
    rows = []
    for m in models:
        ram = float(m.ram_mb)
        pf = tier_factor.get(m.tier, 1.0)
        privacy_load = alpha * pf * mean_risk
        rows.append([ram, privacy_load])
    return np.array(rows, dtype=float)


def hybrid_select(
    entities: List[Entity],
    models: List[ModelTier],
    records: List[Dict[str, Any]],
    *,
    spatial_budget: float = 5_000.0,  # metres
    ram_ceiling_mb: int = 6_000,
    privacy_budget: float = 1.0,
    reference_point: tuple[float, float] | None = None,
    beta: float = 0.1,
    alpha: float = 0.5,
) -> tuple[List[Entity], List[ModelTier]]:
    """
    Greedy joint selection respecting the linear constraints:

        Aᵀ·x ≤ [ spatial_budget , privacy_budget ]

    where A stacks entity and model resource rows.
    The function returns the selected entities and models.
    """
    if reference_point is None:
        # Use the first entity as reference; fall back to (0,0) if none.
        reference_point = (
            (entities[0].lat, entities[0].lon) if entities else (0.0, 0.0)
        )

    # 1️⃣ Compute privacy risk from the supplied records.
    risk_vec = privacy_risk_vector(records)

    # 2️⃣ Build resource matrices.
    E = entity_resource_matrix(entities, reference_point, beta=beta)  # (n_e,2)
    M = model_resource_matrix(models, risk_vec, alpha=alpha)          # (n_m,2)
    A = np.vstack([E, M])                                            # (n_e+n_m,2)

    # 3️⃣ Greedy ordering: prioritize rows with smallest combined load.
    # Combined load = (spatial / spatial_budget) + (privacy / privacy_budget)
    # For models the spatial term is their RAM column.
    spatial_budget = max(spatial_budget, 1e-9)
    privacy_budget = max(privacy_budget, 1e-9)
    load_scores = (
        A[:, 0] / spatial_budget + A[:, 1] / privacy_budget
    )
    order = np.argsort(load_scores)

    selected_entities: List[Entity] = []
    selected_models: List[ModelTier] = []
    cum_spatial = 0.0
    cum_privacy = 0.0

    for idx in order:
        row = A[idx]
        new_spatial = cum_spatial + row[0]
        new_privacy = cum_privacy + row[1]

        if new_spatial <= spatial_budget and new_privacy <= privacy_budget:
            cum_spatial = new_spatial
            cum_privacy = new_privacy
            if idx < len(entities):
                selected_entities.append(entities[idx])
            else:
                selected_models.append(models[idx - len(entities)])

    return selected_entities, selected_models


# ---------- Smoke test ----------
if __name__ == "__main__":
    # Sample entities
    ents = [
        Entity(id="e1", lat=40.7128, lon=-74.0060, category="restaurant", score=0.9),
        Entity(id="e2", lat=40.7130, lon=-74.0062, category="restaurant", score=0.8),
        Entity(id="e3", lat=40.7300, lon=-73.9950, category="cafe", score=0.85),
        Entity(id="e4", lat=40.7500, lon=-73.9900, category="cafe", score=0.7),
    ]

    # Sample models
    mods = [
        ModelTier(name="qwen-0.5b", ram_mb=512, tier="T1"),
        ModelTier(name="reasoning-t2", ram_mb=3000, tier="T2"),
        ModelTier(name="tool-t2", ram_mb=2600, tier="T2"),
        ModelTier(name="qwen-7b", ram_mb=7000, tier="T3"),
    ]

    # Dummy records for privacy risk computation
    recs = [
        {"quasi_id": 2},
        {"quasi_id": 3},
        {"quasi_id": 1},
        {"quasi_id": 0},
    ]

    sel_ents, sel_mods = hybrid_select(
        entities=ents,
        models=mods,
        records=recs,
        spatial_budget=10_000.0,  # 10 km total distance allowance
        ram_ceiling_mb=6_000,
        privacy_budget=0.5,
        reference_point=(40.7128, -74.0060),
        beta=0.2,
        alpha=0.4,
    )

    print("Selected Entities:")
    for e in sel_ents:
        print(f"  {e.id} ({e.category})")

    print("\nSelected Models:")
    for m in sel_mods:
        print(f"  {m.name} ({m.tier}, {m.ram_mb} MB)")

    print(
        f"\nTotal spatial load: {sum(haversine_m((e.lat, e.lon), (40.7128, -74.0060)) for e in sel_ents):.1f} m"
    )
    print(
        f"Total RAM load: {sum(m.ram_mb for m in sel_mods)} MB"
    )