# DARWIN HAMMER — match 3578, survivor 1
# gen: 7
# parent_a: hybrid_hybrid_hybrid_bayes__hybrid_hybrid_hybrid_m528_s1.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_semantic_neig_m1539_s1.py (gen6)
# born: 2026-05-29T23:50:52Z

"""Hybrid algorithm merging Bayesian spatial‑privacy risk (Parent A) with
Ollivier‑Ricci curvature‑adjusted pheromone similarity (Parent B).

Mathematical bridge:
- Parent A supplies a spatial‑aware privacy risk vector **r**∈ℝⁿ for n
  entities, derived from quasi‑identifier reconstruction risk.
- Parent B supplies a curvature matrix **C**∈[−1,1]^{n×n} that rescales a
  distance matrix **D** (here the haversine distances between entities) to an
  adjusted distance **D′ = D ⊙ (1−C)**.
- Inverting **D′** yields a raw similarity matrix **S_raw = (D′+ε)^{-1}**.
- Row‑wise normalisation of **S_raw** produces a probability distribution
  **P** whose Shannon entropy **H(P)** can be used for filtering (omitted
  here for brevity).
- The hybrid neighbour score combines the curvature‑aware similarity with
  the Bayesian risk:  

        score_{ij} = S_{ij} · P_{j} · (1 − r_i)

  Thus a high privacy risk for entity *i* down‑weights its outgoing
  connections, while curvature‑adjusted similarity and pheromone‑like
  probabilities weight the neighbour *j*.

The implementation below provides the core mathematical operations and a
small smoke‑test.
"""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Tuple

import numpy as np


# ----------------------------------------------------------------------
# Data structures (shared between the two parents)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Entity:
    """Geospatial entity with optional categorical signature."""
    id: str
    lat: float
    lon: float
    category: str
    score: float = 0.0
    address_signature: str = ""


@dataclass(frozen=True)
class PheromoneEntry:
    """Light‑weight pheromone record used by the semantic‑neighbor side."""
    uuid: str
    surface_key: str
    signal_kind: str
    signal_value: float
    half_life_seconds: int
    created_at: float
    last_decay: float

    def __init__(self, surface_key: str, signal_kind: str,
                 signal_value: float, half_life_seconds: int):
        self.uuid = f"{random.getrandbits(128):032x}"
        self.surface_key = surface_key
        self.signal_kind = signal_kind
        self.signal_value = signal_value
        self.half_life_seconds = half_life_seconds
        now = random.random()  # placeholder for timestamp
        self.created_at = now
        self.last_decay = now


# ----------------------------------------------------------------------
# Helper functions from Parent A
# ----------------------------------------------------------------------
def haversine_m(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Great‑circle distance in metres between two (lat, lon) points."""
    lat1, lon1 = map(math.radians, a)
    lat2, lon2 = map(math.radians, b)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = (math.sin(dlat / 2) ** 2 +
         math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2)
    return 2 * 6_371_000.0 * math.asin(min(1.0, math.sqrt(h)))


def signature(e: Entity) -> str:
    """Canonical signature used for quasi‑identifier grouping."""
    return (e.address_signature or e.category).strip().lower()


def reconstruction_risk_score(unique_quasi_identifiers: int,
                              total_records: int) -> float:
    """Bayesian marginal reconstruction risk in [0,1]."""
    if total_records <= 0:
        return 0.0
    return max(0.0, min(1.0, unique_quasi_identifiers / total_records))


def spatial_aware_privacy_risk_vector(entities: List[Entity],
                                      delta_m: float) -> np.ndarray:
    """
    Compute a risk vector **r** where each entry is the reconstruction risk
    for an entity based on spatially proximate peers sharing the same signature.
    """
    n = len(entities)
    risks = np.zeros(n, dtype=float)
    for i, ent in enumerate(entities):
        # peers with identical signature within delta_m metres
        similar = [
            e for j, e in enumerate(entities)
            if i != j and signature(ent) == signature(e) and
            haversine_m((ent.lat, ent.lon), (e.lat, e.lon)) <= delta_m
        ]
        uq = len(similar)
        total = len(entities) - 1  # all other records
        risks[i] = reconstruction_risk_score(uq, total)
    return risks


# ----------------------------------------------------------------------
# Functions from Parent B (curvature & pheromone side)
# ----------------------------------------------------------------------
def curvature_adjusted_distance_matrix(entities: List[Entity],
                                        curvature: np.ndarray) -> np.ndarray:
    """
    Build the haversine distance matrix D and apply Ollivier‑Ricci curvature
    C∈[−1,1] to obtain D′ = D ⊙ (1−C).  The function expects C to be square
    with shape (n,n) where n = len(entities).
    """
    n = len(entities)
    D = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            d = haversine_m((entities[i].lat, entities[i].lon),
                            (entities[j].lat, entities[j].lon))
            D[i, j] = D[j, i] = d
    # Guard against malformed curvature input
    if curvature.shape != (n, n):
        raise ValueError("Curvature matrix must be of shape (n, n)")
    # Clip curvature to [−1,1] for safety
    C = np.clip(curvature, -1.0, 1.0)
    D_prime = D * (1.0 - C)  # element‑wise scaling
    return D_prime


def pheromone_similarity_matrix(D_prime: np.ndarray,
                                epsilon: float = 1e-6) -> np.ndarray:
    """
    Invert the adjusted distance matrix to obtain raw similarity,
    then row‑normalise to produce a probability matrix P.
    """
    # Avoid division by zero
    S_raw = 1.0 / (D_prime + epsilon)
    # Row‑wise normalisation
    row_sums = S_raw.sum(axis=1, keepdims=True)
    # Prevent division by zero for isolated rows
    row_sums[row_sums == 0] = 1.0
    P = S_raw / row_sums
    return P


def shannon_entropy(prob_matrix: np.ndarray) -> float:
    """Global Shannon entropy of the probability matrix (flattened)."""
    flat = prob_matrix.ravel()
    # Filter zero probabilities to avoid log(0)
    flat = flat[flat > 0]
    return -np.sum(flat * np.log2(flat))


# ----------------------------------------------------------------------
# Hybrid core: combine risk, curvature‑aware similarity and pheromone probability
# ----------------------------------------------------------------------
def hybrid_neighbor_scores(entities: List[Entity],
                           curvature: np.ndarray,
                           delta_m: float,
                           epsilon: float = 1e-6) -> np.ndarray:
    """
    Compute the hybrid neighbour score matrix **S_hybrid** of shape (n,n)
    where

        S_hybrid[i, j] = S[i, j] * P[i, j] * (1 - r_i)

    with
        - r = spatial privacy risk vector (Parent A)
        - D′ = curvature‑adjusted distance matrix (Parent B)
        - S = (D′ + ε)^{-1}
        - P = row‑normalised S (pheromone probability)

    The result can be interpreted as a weighted adjacency matrix ready for
    downstream scheduling or tier‑health updates.
    """
    n = len(entities)
    if n == 0:
        return np.empty((0, 0), dtype=float)

    # 1️⃣ Privacy risk from Parent A
    r = spatial_aware_privacy_risk_vector(entities, delta_m)  # shape (n,)

    # 2️⃣ Curvature‑adjusted distances from Parent B
    D_prime = curvature_adjusted_distance_matrix(entities, curvature)

    # 3️⃣ Raw similarity and pheromone probability
    S_raw = 1.0 / (D_prime + epsilon)          # similarity before normalisation
    row_sums = S_raw.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    P = S_raw / row_sums                       # pheromone‑like probability matrix

    # 4️⃣ Hybrid scoring
    risk_factor = (1.0 - r).reshape(-1, 1)     # broadcast over columns
    hybrid = S_raw * P * risk_factor

    return hybrid


# ----------------------------------------------------------------------
# Additional utility: health‑aware tier scheduling (illustrative)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class ModelTier:
    name: str
    ram_mb: int
    tier: str
    vram_mb: int


def update_tier_health(scores: np.ndarray,
                       tiers: List[ModelTier],
                       threshold: float = 0.5) -> List[Tuple[ModelTier, str]]:
    """
    Very simple health evaluator: if the mean outgoing hybrid score for a tier
    falls below *threshold* the tier is marked 'degraded', otherwise 'healthy'.
    This demonstrates how the hybrid matrix can drive tier‑level decisions.
    """
    if scores.shape[0] != len(tiers):
        raise ValueError("Score matrix size must match number of tiers")

    health_report = []
    for i, tier in enumerate(tiers):
        mean_score = float(scores[i].mean())
        status = "healthy" if mean_score >= threshold else "degraded"
        health_report.append((tier, status))
    return health_report


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create a tiny synthetic world
    entities = [
        Entity(id="A", lat=37.7749, lon=-122.4194, category="shop"),
        Entity(id="B", lat=37.7750, lon=-122.4180, category="shop"),
        Entity(id="C", lat=37.8044, lon=-122.2711, category="office"),
        Entity(id="D", lat=37.7600, lon=-122.4477, category="office"),
    ]

    n = len(entities)

    # Synthetic curvature matrix: small random values centred around 0
    rng = np.random.default_rng(42)
    curvature = rng.uniform(-0.2, 0.2, size=(n, n))
    np.fill_diagonal(curvature, 0.0)  # no self‑curvature

    delta_m = 200.0  # 200 m spatial window for privacy risk

    hybrid_scores = hybrid_neighbor_scores(entities, curvature, delta_m)

    print("Hybrid neighbour score matrix:")
    print(hybrid_scores)

    # Define dummy model tiers aligned with entities for demonstration
    tiers = [
        ModelTier(name="TierA", ram_mb=2048, tier="edge", vram_mb=512),
        ModelTier(name="TierB", ram_mb=4096, tier="cloud", vram_mb=1024),
        ModelTier(name="TierC", ram_mb=2048, tier="edge", vram_mb=512),
        ModelTier(name="TierD", ram_mb=4096, tier="cloud", vram_mb=1024),
    ]

    health = update_tier_health(hybrid_scores, tiers, threshold=0.001)
    for tier, status in health:
        print(f"{tier.name} ({tier.tier}) health: {status}")

    # Verify entropy computation does not crash
    entropy = shannon_entropy(hybrid_scores)
    print(f"Global Shannon entropy of hybrid scores: {entropy:.4f}")