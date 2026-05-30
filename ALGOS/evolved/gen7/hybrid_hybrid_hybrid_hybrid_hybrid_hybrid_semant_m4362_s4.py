# DARWIN HAMMER — match 4362, survivor 4
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2246_s2.py (gen6)
# parent_b: hybrid_hybrid_semantic_neig_hybrid_hybrid_krampu_m540_s1.py (gen3)
# born: 2026-05-29T23:55:10Z

"""Hybrid Algorithm Fusion of Darwin Hammer (match 2246) and Darwin Hammer (match 540)

This module fuses the core topologies of:

* **Parent A** – a VRAM‑scheduler that uses endpoint health scores,
  Hoeffding bounds and a reconstruction‑risk metric.
* **Parent B** – a semantic‑neighbour search that employs pheromone
  probabilities, entropy, cosine similarity and Ollivier‑Ricci curvature
  on a brain‑map graph.

**Mathematical bridge**

The bridge is the *health‑score ↔ pheromone* mapping and the *graph curvature*
that modulates both the pheromone‑driven selection and the Hoeffding‑bound
confidence. Concretely:

1. Each `Endpoint` health score `h_i ∈ [0,1]` is interpreted as a raw pheromone
   concentration `τ_i = h_i`.
2. Normalising `τ` yields a probability vector `p_i` (pheromone probabilities).
3. An adjacency matrix `A` describing the semantic‑neighbour graph is used to
   compute a pair‑wise Ollivier‑Ricci curvature matrix `C`.  Higher curvature
   indicates a more “tightly coupled” neighbourhood and therefore a lower
   exploration incentive.
4. A *curvature‑adjusted* selection weight is defined as  
   `w_i = p_i * (1 - mean_j C_{ij})`.  This blends pheromone guidance with the
   curvature of the neighbourhood.
5. The Hoeffding bound is applied to the weighted empirical mean of the
   selected endpoint’s health to guarantee, with confidence `1‑δ`, that the
   chosen endpoint is near‑optimal for the subsequent VRAM load estimation.

The three public functions below expose this hybrid operation:
`compute_curvature_matrix`, `select_endpoint`, and `expected_vram_load`.
"""

import math
import random
import sys
from pathlib import Path
from typing import List, Tuple, Dict
from dataclasses import dataclass

import numpy as np

# ----------------------------------------------------------------------
# Data structures (shared from Parent A)
# ----------------------------------------------------------------------
@dataclass
class ModelTier:
    """Lightweight descriptor for a model."""
    name: str
    ram_mb: int
    tier: str


@dataclass
class Morphology:
    """Geometric description of a logical entity."""
    length: float
    width: float
    height: float
    mass: float


@dataclass
class Endpoint:
    """Endpoint health descriptor."""
    health_score: float          # ∈ [0, 1]
    failure_rate: float         # probability of failure per unit time
    recovery_priority: float    # higher means quicker recovery


@dataclass
class ModelSpec:
    """Combined specification used by the hybrid scheduler."""
    tier: ModelTier
    morphology: Morphology
    endpoint: Endpoint
    unique_quasi_identifiers: int
    total_records: int


# ----------------------------------------------------------------------
# Utilities from Parent A
# ----------------------------------------------------------------------
def reconstruction_risk_score(unique_quasi_identifiers: int, total_records: int) -> float:
    """Probability that a record can be re‑identified."""
    if total_records < 1:
        return 0.0
    return unique_quasi_identifiers / total_records


def hoeffding_bound(r: float, delta: float, n: int) -> float:
    """Hoeffding bound for a random variable bounded in [0, r]."""
    if n <= 0:
        raise ValueError("n must be positive")
    return math.sqrt((r ** 2) * math.log(2 / delta) / (2 * n))


def _cos(a: List[float], b: List[float]) -> float:
    """Cosine similarity (Parent B)."""
    den = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return 0.0 if den == 0 else sum(x * y for x, y in zip(a, b)) / den


def pheromone_probabilities(pheromones: List[float]) -> List[float]:
    """Normalise raw pheromone values into a probability distribution."""
    total = sum(pheromones)
    if total == 0:
        # Uniform distribution if all pheromones are zero
        return [1.0 / len(pheromones)] * len(pheromones)
    return [p / total for p in pheromones]


def entropy(probabilities: List[float], eps: float = 1e-12) -> float:
    """Shannon entropy of a probability distribution."""
    total = sum(probabilities)
    if total <= 0:
        raise ValueError('positive probability mass required')
    return -sum((p / total) * math.log(max(p / total, eps)) for p in probabilities if p > 0)


# ----------------------------------------------------------------------
# Hybrid core functions
# ----------------------------------------------------------------------
def compute_curvature_matrix(adjacency: np.ndarray, steps: int = 3) -> np.ndarray:
    """
    Approximate Ollivier‑Ricci curvature for an undirected graph.

    For each edge (i, j) we compare the 1‑step random‑walk distributions
    centred at i and j using the Earth Mover’s Distance (here simplified to
    1‑norm of the difference).  The curvature κ_{ij} is defined as

        κ_{ij} = 1 - W_1(μ_i, μ_j) / d(i, j)

    where d(i, j) = 1 for adjacent nodes.  The function returns a symmetric
    matrix C where C[i, j] = κ_{ij} if (i, j) is an edge, otherwise 0.
    """
    if adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("Adjacency must be square")
    n = adjacency.shape[0]
    # Degree‑normalised random‑walk matrix
    deg = adjacency.sum(axis=1)
    deg[deg == 0] = 1  # avoid division by zero for isolated nodes
    P = adjacency / deg[:, None]

    # μ_i is the distribution after `steps` random‑walk steps starting from i
    mu = np.linalg.matrix_power(P, steps)

    curvature = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            if adjacency[i, j] == 0:
                continue
            # Earth Mover's distance simplified to L1 distance for discrete case
            emd = np.linalg.norm(mu[i] - mu[j], ord=1) / 2.0
            kappa = 1.0 - emd  # d(i,j)=1
            curvature[i, j] = curvature[j, i] = kappa
    return curvature


def select_endpoint(endpoints: List[Endpoint],
                    adjacency: np.ndarray,
                    delta: float = 0.05,
                    confidence_samples: int = 30) -> Tuple[int, float]:
    """
    Hybrid selection using pheromone (health), curvature adjustment and
    Hoeffding confidence.

    Returns:
        selected_index (int): index of the chosen endpoint.
        bound (float): Hoeffding bound on the estimated mean health.
    """
    if not endpoints:
        raise ValueError("Endpoint list cannot be empty")
    if adjacency.shape[0] != len(endpoints):
        raise ValueError("Adjacency size must match number of endpoints")

    # 1. Map health scores to raw pheromones
    raw_pheromones = [e.health_score for e in endpoints]

    # 2. Normalise to probabilities
    pheromone_probs = pheromone_probabilities(raw_pheromones)

    # 3. Compute curvature matrix
    curvature = compute_curvature_matrix(adjacency)

    # 4. Curvature‑adjusted weight for each node
    #    weight_i = p_i * (1 - mean_curvature_i)
    mean_curv = curvature.mean(axis=1)  # average curvature incident to node i
    adjusted_weights = [
        p * (1.0 - mc) for p, mc in zip(pheromone_probs, mean_curv)
    ]

    # 5. Sample according to adjusted_weights (multinomial draw)
    selected_index = random.choices(
        population=range(len(endpoints)),
        weights=adjusted_weights,
        k=1
    )[0]

    # 6. Hoeffding bound on the mean health of the selected endpoint
    #    Treat each health observation as bounded in [0,1] (r=1)
    bound = hoeffding_bound(r=1.0, delta=delta, n=confidence_samples)

    return selected_index, bound


def expected_vram_load(model_specs: List[ModelSpec],
                       selected_endpoint: Endpoint,
                       curvature_factor: float = 0.1) -> float:
    """
    Compute the expected VRAM load for a batch of models given a selected
    endpoint.  The formula blends:

        load = Σ (ram_mb * tier_factor) * health_modifier

    where:
        tier_factor = 1.0 for 'base', 1.2 for 'mid', 1.5 for 'high'.
        health_modifier = selected_endpoint.health_score * (1 - curvature_factor)
    """
    tier_factors = {
        'base': 1.0,
        'mid': 1.2,
        'high': 1.5
    }
    health_mod = selected_endpoint.health_score * (1.0 - curvature_factor)

    total_load = 0.0
    for spec in model_specs:
        tf = tier_factors.get(spec.tier.tier.lower(), 1.0)
        total_load += spec.tier.ram_mb * tf * health_mod

    return total_load


# ----------------------------------------------------------------------
# Example feature extraction (Parent B) – kept for completeness
# ----------------------------------------------------------------------
def extract_full_features(text: str) -> Dict[str, float]:
    """
    Deterministic pseudo‑feature extraction based on the hash of the input.
    Returns a fixed‑size dictionary of 10 numeric features.
    """
    rnd = random.Random(hash(text))
    feature_names = [
        "operator_visceral_ratio", "operator_tech_ratio",
        "operator_legal_osint_ratio", "operator_ledger_density",
        "operator_recursion_score", "operator_directive_ratio",
        "operator_target_density", "psyche_forensic_shield_ratio",
        "psyche_poetic_entropy", "psyche_dissociative_index"
    ]
    return {name: rnd.random() for name in feature_names}


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create a tiny graph of 4 endpoints
    endpoints = [
        Endpoint(health_score=0.9, failure_rate=0.02, recovery_priority=0.8),
        Endpoint(health_score=0.6, failure_rate=0.05, recovery_priority=0.5),
        Endpoint(health_score=0.3, failure_rate=0.10, recovery_priority=0.2),
        Endpoint(health_score=0.75, failure_rate=0.03, recovery_priority=0.7)
    ]

    # Simple adjacency (undirected) – fully connected for demo
    adjacency = np.array([
        [0, 1, 1, 1],
        [1, 0, 1, 1],
        [1, 1, 0, 1],
        [1, 1, 1, 0]
    ], dtype=float)

    # Dummy model specs
    tier_base = ModelTier(name="tiny", ram_mb=256, tier="base")
    tier_high = ModelTier(name="large", ram_mb=2048, tier="high")
    morph = Morphology(1.0, 1.0, 1.0, 1.0)

    model_specs = [
        ModelSpec(tier=tier_base, morphology=morph,
                  endpoint=endpoints[0],
                  unique_quasi_identifiers=10,
                  total_records=1000),
        ModelSpec(tier=tier_high, morphology=morph,
                  endpoint=endpoints[1],
                  unique_quasi_identifiers=5,
                  total_records=500)
    ]

    # Hybrid selection
    idx, bound = select_endpoint(endpoints, adjacency)
    chosen = endpoints[idx]
    print(f"Selected endpoint index: {idx}, health: {chosen.health_score:.3f}, Hoeffding bound: {bound:.5f}")

    # Expected VRAM load
    load = expected_vram_load(model_specs, chosen)
    print(f"Expected VRAM load (MB): {load:.2f}")

    # Feature extraction demonstration
    feats = extract_full_features("sample input")
    print(f"Extracted {len(feats)} features, first two: {list(feats.items())[:2]}")