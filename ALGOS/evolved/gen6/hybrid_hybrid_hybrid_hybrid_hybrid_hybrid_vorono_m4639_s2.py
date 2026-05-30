# DARWIN HAMMER — match 4639, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_hybrid_bayes__hybrid_hybrid_hybrid_m528_s4.py (gen4)
# parent_b: hybrid_hybrid_voronoi_parti_hybrid_liquid_time_c_m1036_s0.py (gen5)
# born: 2026-05-29T23:57:06Z

"""Hybrid Bayesian‑Voronoi‑HDC‑LTC Scheduler

Parents:
- hybrid_hybrid_hybrid_bayes__hybrid_hybrid_hybrid_m528_s4.py (Bayesian spatial‑privacy + VRAM scheduling)
- hybrid_hybrid_voronoi_parti_hybrid_liquid_time_c_m1036_s0.py (Voronoi partitioning + Hyperdimensional Computing + Liquid‑Time‑Constant gating)

Mathematical bridge:
1. The Bayesian posterior `Posterior_{ij}` (entity i ↔ model j) supplies a
   *probabilistic weight* that is spatially aware.
2. Voronoi partitioning groups entities into cells `C_k` based on Euclidean
   proximity to a set of seed points.
3. For each cell we aggregate the posterior weights of its members, obtaining a
   scalar `w_k`.  This scalar drives a Liquid‑Time‑Constant (LTC) gating
   function `g_k = σ(w_k – τ)` (σ = sigmoid, τ = learnable time‑constant).
4. Hyperdimensional vectors `v_model_j` (bipolar) are bound to a cell vector
   `v_cell_k` via element‑wise XOR.  The gated binding `g_k·(v_model_j ⊕ v_cell_k)`
   is then bundled (majority vote) across all models to obtain a single
   representation per cell, which finally drives VRAM allocation.

The module implements this fused pipeline with three core functions:
`compute_posterior`, `voronoi_assign`, and `hybrid_representation`.  An
`EndpointCircuitBreaker` from the Voronoi parent guards the assignment step.
"""

import math
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Any

import numpy as np

# ----------------------------------------------------------------------
# Core data structures (merged)
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
    # health‑derived reliability and reconstruction risk (parent A)
    reliability: float  # h_j
    recon_risk: float   # r_j


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


def euclidean_distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Plain Euclidean distance (used for Voronoi assignment)."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def now_z() -> str:
    """Current UTC timestamp in ISO‑8601 Z format."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


# ----------------------------------------------------------------------
# Endpoint circuit breaker (from Parent B)
# ----------------------------------------------------------------------


class EndpointCircuitBreaker:
    def __init__(self, failure_threshold: int = 3):
        if failure_threshold <= 0:
            raise ValueError("failure_threshold must be positive")
        self.failure_threshold = failure_threshold
        self.failures = 0
        self.open = False
        self.last_event_at = ""

    def record_success(self) -> None:
        self.failures = 0
        self.open = False
        self.last_event_at = now_z()

    def record_failure(self) -> None:
        self.failures += 1
        self.open = self.failures >= self.failure_threshold
        self.last_event_at = now_z()

    def allow(self) -> bool:
        return not self.open

    def as_dict(self) -> dict[str, Any]:
        return {
            "failure_threshold": self.failure_threshold,
            "failures": self.failures,
            "open": self.open,
            "last_event_at": self.last_event_at,
        }


# ----------------------------------------------------------------------
# Hyperdimensional utilities (from Parent B)
# ----------------------------------------------------------------------


Vector = np.ndarray  # bipolar hypervector of dtype int8 (values -1 or +1)


def random_vector(dim: int = 10_000, seed: int | None = None) -> Vector:
    """Generate a random bipolar hypervector."""
    rng = np.random.default_rng(seed)
    vec = rng.integers(0, 2, size=dim, dtype=np.int8)
    return np.where(vec == 0, -1, 1)


def bind(v1: Vector, v2: Vector) -> Vector:
    """Element‑wise XOR for bipolar vectors (equivalent to multiplication)."""
    return v1 * v2


def bundle(vectors: List[Vector]) -> Vector:
    """Majority‑vote bundling (sign of sum)."""
    if not vectors:
        raise ValueError("bundle requires at least one vector")
    sum_vec = np.sum(vectors, axis=0)
    return np.where(sum_vec >= 0, 1, -1).astype(np.int8)


# ----------------------------------------------------------------------
# 1️⃣ Bayesian posterior (Parent A)
# ----------------------------------------------------------------------


def compute_posterior(entities: List[Entity],
                      model_tiers: List[ModelTier],
                      priors: Dict[str, float]) -> np.ndarray:
    """
    Compute the Bayesian posterior matrix Posterior_{ij}.

    - `priors` maps entity.id → spatial‑privacy prior p_i.
    - L_{ij} = h_j * (1 – r_j)  (health reliability times complement of recon risk)
    - Posterior_{ij} = (p_i * L_{ij}) / Σ_{k,l} (p_k * L_{kl})

    Returns:
        posterior: shape (len(entities), len(model_tiers))
    """
    if not entities or not model_tiers:
        raise ValueError("entities and model_tiers must be non‑empty")

    p_vec = np.array([priors.get(e.id, 0.0) for e in entities], dtype=np.float64)
    L_mat = np.empty((len(entities), len(model_tiers)), dtype=np.float64)

    for j, mt in enumerate(model_tiers):
        L = mt.reliability * (1.0 - mt.recon_risk)
        L_mat[:, j] = L

    numer = p_vec[:, None] * L_mat
    denom = numer.sum()
    if denom == 0.0:
        raise ZeroDivisionError("Denominator of posterior is zero")
    posterior = numer / denom
    return posterior


# ----------------------------------------------------------------------
# 2️⃣ Voronoi assignment (Parent B)
# ----------------------------------------------------------------------


def voronoi_assign(entities: List[Entity],
                   seeds: List[Tuple[float, float]]) -> Dict[int, List[int]]:
    """
    Assign each entity to the nearest seed (Voronoi cell).

    Returns:
        cell_to_entity_idxs: dict mapping cell index → list of entity indices.
    """
    if not seeds:
        raise ValueError("At least one seed point is required")
    cell_to_entity_idxs: Dict[int, List[int]] = {i: [] for i in range(len(seeds))}
    for idx, ent in enumerate(entities):
        point = (ent.lat, ent.lon)
        distances = [euclidean_distance(point, s) for s in seeds]
        nearest = int(np.argmin(distances))
        cell_to_entity_idxs[nearest].append(idx)
    return cell_to_entity_idxs


# ----------------------------------------------------------------------
# 3️⃣ Liquid‑Time‑Constant gating (Parent B)
# ----------------------------------------------------------------------


def ltc_gate(weight: float, tau: float) -> float:
    """
    Simple LTC‑style gating: sigmoid centered at τ.

    g = 1 / (1 + exp(-(weight - τ)))
    """
    return 1.0 / (1.0 + math.exp(-(weight - tau)))


# ----------------------------------------------------------------------
# Hybrid operation – combines all three pillars
# ----------------------------------------------------------------------


def hybrid_representation(entities: List[Entity],
                          model_tiers: List[ModelTier],
                          priors: Dict[str, float],
                          seeds: List[Tuple[float, float]],
                          tau: float = 0.5,
                          hv_dim: int = 10_000,
                          breaker: EndpointCircuitBreaker | None = None) -> Dict[int, Vector]:
    """
    Produce a hyperdimensional representation per Voronoi cell that encodes:
      * Bayesian posterior‑derived weight per cell,
      * LTC gating of those weights,
      * Binding of model hypervectors with a cell‑specific random vector,
      * Bundling across all models.

    Returns:
        cell_id → bundled hypervector (np.ndarray of int8)
    """
    if breaker is None:
        breaker = EndpointCircuitBreaker()

    # Step 1 – Bayesian posterior
    posterior = compute_posterior(entities, model_tiers, priors)

    # Step 2 – Voronoi partition
    try:
        cell_map = voronoi_assign(entities, seeds)
        breaker.record_success()
    except Exception:
        breaker.record_failure()
        raise

    # Pre‑generate hypervectors for models and cells
    model_vectors = [random_vector(hv_dim, seed=i) for i in range(len(model_tiers))]
    cell_vectors = [random_vector(hv_dim, seed=1000 + i) for i in range(len(seeds))]

    cell_representations: Dict[int, Vector] = {}

    for cell_id, ent_idxs in cell_map.items():
        if not ent_idxs:
            # Empty cell → zero vector (all -1)
            cell_representations[cell_id] = np.full(hv_dim, -1, dtype=np.int8)
            continue

        # Aggregate posterior weight for this cell across all models
        cell_weights = posterior[ent_idxs, :]  # shape (n_entities_in_cell, n_models)
        # Sum over entities then over models to obtain a scalar weight
        agg_weight = float(cell_weights.sum())

        # LTC gating
        g = ltc_gate(agg_weight, tau)

        # Bind each model vector with the cell vector, scale by gating,
        # then bundle.
        bound_vectors = [
            bind(model_vectors[m_idx], cell_vectors[cell_id]) * (1 if random.random() < g else -1)
            for m_idx in range(len(model_tiers))
        ]
        bundled = bundle(bound_vectors)
        cell_representations[cell_id] = bundled

    return cell_representations


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------


if __name__ == "__main__":
    # Minimal synthetic data
    ents = [
        Entity(id="e1", lat=37.7749, lon=-122.4194, category="A"),
        Entity(id="e2", lat=34.0522, lon=-118.2437, category="B"),
        Entity(id="e3", lat=40.7128, lon=-74.0060, category="A"),
    ]

    models = [
        ModelTier(name="tiny", ram_mb=512, tier="low", vram_mb=1024,
                  reliability=0.9, recon_risk=0.1),
        ModelTier(name="small", ram_mb=1024, tier="mid", vram_mb=2048,
                  reliability=0.8, recon_risk=0.2),
    ]

    priors = {"e1": 0.6, "e2": 0.3, "e3": 0.1}
    seed_points = [(37.0, -122.0), (40.0, -75.0)]  # two Voronoi seeds

    breaker = EndpointCircuitBreaker(failure_threshold=2)

    reps = hybrid_representation(
        entities=ents,
        model_tiers=models,
        priors=priors,
        seeds=seed_points,
        tau=0.5,
        hv_dim=2000,
        breaker=breaker,
    )

    for cid, vec in reps.items():
        print(f"Cell {cid} representation (first 10 bits): {vec[:10].tolist()}")

    print("Circuit breaker state:", breaker.as_dict())