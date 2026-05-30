# DARWIN HAMMER — match 528, survivor 4
# gen: 4
# parent_a: hybrid_hybrid_bayes_update__hybrid_possum_filter_m210_s1.py (gen3)
# parent_b: hybrid_hybrid_hybrid_privac_hybrid_hybrid_endpoi_m16_s4.py (gen3)
# born: 2026-05-29T23:29:28Z

"""Hybrid Bayesian‑Spatial‑Privacy‑VRAM Scheduler

Parents:
- hybrid_hybrid_bayes_update__hybrid_possum_filter_m210_s1.py
- hybrid_hybrid_hybrid_privac_hybrid_hybrid_endpoi_m16_s4.py

Mathematical bridge:
Each *entity* supplies a spatial‑aware privacy risk `p_i` (a prior probability).  
Each *model tier* supplies a health‑derived reliability `h_j` and a privacy
reconstruction risk `r_j`.  The joint likelihood of assigning entity *i* to model
*j* is taken as

    L_{ij} = h_j · (1 – r_j)

The Bayesian posterior over the model assignment matrix is

    Posterior_{ij} = (p_i · L_{ij}) / Σ_{k,l} (p_k · L_{kl})

The posterior matrix is then used to allocate VRAM (or work‑share) proportionally
to each model tier.  This single formulation fuses the spatial‑privacy vector
of Parent A with the health‑risk scoring and VRAM scheduling of Parent B.
"""

import math
import random
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Iterable, Tuple, Dict

import numpy as np

# ----------------------------------------------------------------------
# Core data structures (merged from both parents)
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class Entity:
    """Spatial entity with optional quasi‑identifier signature."""
    id: str
    lat: float
    lon: float
    category: str
    score: float = 0.0
    address_signature: str = ""


@dataclass(frozen=True)
class ModelTier:
    """VRAM‑aware model tier used for scheduling."""
    name: str
    ram_mb: int
    tier: str
    vram_mb: int


# ----------------------------------------------------------------------
# Primitive utilities (shared)
# ----------------------------------------------------------------------


def haversine_m(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Great‑circle distance in metres between two (lat, lon) points."""
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
    """Canonical quasi‑identifier for an entity."""
    return (e.address_signature or e.category).strip().lower()


def reconstruction_risk_score(unique_quasi_identifiers: int, total_records: int) -> float:
    """Privacy reconstruction risk ∈[0,1] (identical in both parents)."""
    return (
        0.0
        if total_records <= 0
        else max(0.0, min(1.0, unique_quasi_identifiers / total_records))
    )


# ----------------------------------------------------------------------
# 1️⃣ Spatial‑aware privacy risk vector (Parent A)
# ----------------------------------------------------------------------


def spatial_aware_privacy_risk_vector(
    entities: List[Entity], delta_m: float
) -> np.ndarray:
    """
    Compute a prior risk vector `p` where each entry corresponds to an entity.
    For entity *i* we collect all other entities whose signature matches and that
    lie within `delta_m` metres.  The risk is the reconstruction risk of the
    number of *distinct* signatures among that neighbourhood.

    Returns:
        np.ndarray of shape (n_entities,) that sums to 1 (probability simplex).
    """
    n = len(entities)
    raw_risks = np.empty(n, dtype=float)

    for i, ent in enumerate(entities):
        neighbours = [
            e
            for j, e in enumerate(entities)
            if i != j
            and signature(ent) == signature(e)
            and haversine_m((ent.lat, ent.lon), (e.lat, e.lon)) <= delta_m
        ]

        # Unique quasi‑identifiers are the distinct signatures in the neighbourhood
        unique_sig = {signature(e) for e in neighbours}
        uq = len(unique_sig)
        raw_risks[i] = reconstruction_risk_score(uq, n)

    # Normalise to obtain a proper prior distribution
    total = raw_risks.sum()
    if total == 0.0:
        # Uniform prior if no risk information is present
        return np.full(n, 1.0 / n, dtype=float)
    return raw_risks / total


# ----------------------------------------------------------------------
# 2️⃣ Health‑derived model score (Parent B)
# ----------------------------------------------------------------------


def health_score(failure_rate: float, recovery_priority: float) -> float:
    """
    Health of a model tier, 0 ≤ health ≤ 1.
    health = (1‑failure_rate) * (1‑recovery_priority)
    """
    return max(0.0, min(1.0, (1.0 - failure_rate) * (1.0 - recovery_priority)))


def combined_model_score(
    model: ModelTier,
    failure_rate: float,
    recovery_priority: float,
    reconstruction_risk: float,
) -> float:
    """
    Unified score used for VRAM scheduling and work‑share allocation.

        score = health * (1 - reconstruction_risk)

    The score lies in [0,1] and will serve as the *likelihood* component
    L_{ij} for every entity‑model pair.
    """
    h = health_score(failure_rate, recovery_priority)
    return h * (1.0 - reconstruction_risk)


# ----------------------------------------------------------------------
# 3️⃣ Bayesian fusion of prior (spatial risk) and likelihood (model score)
# ----------------------------------------------------------------------


def hybrid_bayes_update(prior: np.ndarray, likelihood: np.ndarray) -> np.ndarray:
    """
    Perform a Bayesian update over a joint entity‑model matrix.

    Args:
        prior:   shape (n_entities,) – probability simplex.
        likelihood: shape (n_entities, n_models) – non‑negative scores.

    Returns:
        posterior: shape (n_entities, n_models) – normalized joint distribution.
    """
    # Broadcast prior to match likelihood dimensions
    joint = prior[:, None] * likelihood
    total = joint.sum()
    if total == 0.0:
        # Avoid division by zero – fall back to uniform distribution
        n, m = likelihood.shape
        return np.full((n, m), 1.0 / (n * m), dtype=float)
    return joint / total


# ----------------------------------------------------------------------
# 4️⃣ Resource allocation using the posterior matrix
# ----------------------------------------------------------------------


def allocate_vram(
    entities: List[Entity],
    models: List[ModelTier],
    delta_m: float,
    model_failure_rates: Dict[str, float],
    model_recovery_priorities: Dict[str, float],
) -> Dict[str, float]:
    """
    Compute VRAM allocation (in MB) for each model tier.

    Steps:
    1. Compute spatial prior `p` from entities.
    2. Compute likelihood matrix `L_{ij}` = combined_model_score for model *j*.
    3. Bayesian posterior `P_{ij}` = hybrid_bayes_update(p, L).
    4. Sum posterior over entities to obtain a weight per model.
    5. Allocate VRAM proportionally to those weights.

    Returns:
        Mapping from model name to allocated VRAM (float MB).
    """
    if not entities or not models:
        return {}

    # 1️⃣ Prior
    prior = spatial_aware_privacy_risk_vector(entities, delta_m)  # (n,)

    # 2️⃣ Likelihood matrix
    n = len(entities)
    m = len(models)
    likelihood = np.empty((n, m), dtype=float)

    # Reconstruction risk for a model tier is derived from its own
    # quasi‑identifier count (here we reuse the generic risk function with a
    # placeholder of 1 unique identifier per model).
    for j, model in enumerate(models):
        recon_risk = reconstruction_risk_score(1, len(models))  # simple proxy
        score = combined_model_score(
            model,
            model_failure_rates.get(model.name, 0.0),
            model_recovery_priorities.get(model.name, 0.0),
            recon_risk,
        )
        likelihood[:, j] = score  # same score for every entity (could be refined)

    # 3️⃣ Posterior
    posterior = hybrid_bayes_update(prior, likelihood)  # (n, m)

    # 4️⃣ Model weights (sum over entities)
    model_weights = posterior.sum(axis=0)  # (m,)

    # 5️⃣ Allocate VRAM proportionally
    total_vram = sum(mdl.vram_mb for mdl in models)
    allocations = {}
    for mdl, w in zip(models, model_weights):
        allocations[mdl.name] = float(w * total_vram)

    return allocations


# ----------------------------------------------------------------------
# 5️⃣ Demonstration entry‑point
# ----------------------------------------------------------------------


if __name__ == "__main__":
    # Minimal smoke test – deterministic seed for reproducibility
    random.seed(0)
    np.random.seed(0)

    # Create a handful of synthetic entities
    ents = [
        Entity(id="e1", lat=37.7749, lon=-122.4194, category="A"),
        Entity(id="e2", lat=37.7750, lon=-122.4180, category="A"),
        Entity(id="e3", lat=34.0522, lon=-118.2437, category="B"),
        Entity(id="e4", lat=40.7128, lon=-74.0060, category="C"),
    ]

    # Define model tiers (mirroring Parent B constants)
    models = [
        ModelTier("qwen-0.5b", 512, "T1", 1024),
        ModelTier("reasoning-t2", 3000, "T2", 2048),
        ModelTier("tool-t2", 2600, "T2", 2048),
        ModelTier("qwen-7b", 7000, "T3", 4096),
    ]

    # Random but reproducible failure / recovery parameters
    failure_rates = {m.name: random.random() * 0.2 for m in models}
    recovery_priorities = {m.name: random.random() * 0.3 for m in models}

    # Perform allocation
    alloc = allocate_vram(
        entities=ents,
        models=models,
        delta_m=500.0,  # 500 m neighbourhood for spatial risk
        model_failure_rates=failure_rates,
        model_recovery_priorities=recovery_priorities,
    )

    # Simple sanity output
    print("VRAM allocation (MB) per model tier:")
    for name, vram in alloc.items():
        print(f"  {name:12s}: {vram:8.2f}")

    # Verify that allocations sum to total VRAM (within floating error)
    total_alloc = sum(alloc.values())
    expected_total = sum(m.vram_mb for m in models)
    assert math.isclose(total_alloc, expected_total, rel_tol=1e-6), "Allocation mismatch"
    print("✅ Allocation sanity check passed.")