# DARWIN HAMMER — match 3048, survivor 2
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_privacy_sketc_m2009_s1.py (gen6)
# parent_b: hybrid_hybrid_honeybee_stor_hybrid_hybrid_bandit_m2089_s0.py (gen6)
# born: 2026-05-29T23:47:27Z

"""Hybrid Algorithm integrating:
- hybrid_hybrid_hybrid_hybrid_hybrid_privacy_sketc_m2009_s1.py (node‑wise curvature proxy, Count‑Min Sketch)
- hybrid_hybrid_honeybee_stor_hybrid_hybrid_bandit_m2089_s0.py (honeybee store dynamics, SSIM‑weighted bandit router)

Mathematical bridge:
1. The cardinality estimated from the Count‑Min Sketch (parent A) is used as a
   temperature‑dependent scaling factor for the honeybee store update (parent B).
2. The node‑wise curvature vector (parent A) is compared with a reference curvature
   via the SSIM score (parent B); this SSIM value modulates the propensities of
   bandit actions, effectively linking the two topologies.
"""

import sys
import math
import random
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
import numpy as np

# ----------------------------------------------------------------------
# Parent A components
# ----------------------------------------------------------------------
def compute_curvature(adj_matrix: np.ndarray) -> np.ndarray:
    """Node‑wise curvature proxy."""
    n = adj_matrix.shape[0]
    curvature = np.zeros(n, dtype=np.float64)
    row_sums = adj_matrix.sum(axis=1)
    col_sums = adj_matrix.sum(axis=0)
    for i in range(n):
        for j in range(n):
            w = adj_matrix[i, j]
            if w > 0:
                curvature[i] += w * np.log(w / (row_sums[i] * col_sums[j] + 1e-12) + 1e-12)
    return curvature

def count_min_sketch(items: List[str], width: int = 64, depth: int = 4) -> np.ndarray:
    """Standard Count‑Min Sketch matrix."""
    cms = np.zeros((depth, width), dtype=np.int64)
    for item in items:
        for d in range(depth):
            col = (hash(item + str(d)) % width)
            cms[d, col] += 1
    return cms

def estimate_cardinality_from_cms(cms: np.ndarray) -> int:
    """Very rough cardinality estimator: average of column minima."""
    mins = cms.min(axis=0)
    return int(max(1, mins.mean()))

# ----------------------------------------------------------------------
# Parent B components
# ----------------------------------------------------------------------
ALPHA = 0.6          # store inflow coefficient
BETA = 0.4           # store outflow coefficient
DT = 1.0             # time step for store dynamics

K1 = 0.01
K2 = 0.03
L = 255.0

_POSITIVE_WEIGHTS = np.array([1600, 1200, 1400, 1000, 1200, 800,   0,   0,   0], dtype=np.float64)
_NEGATIVE_WEIGHTS = np.array([   0,    0,    0,    0,    0,   0, 1500, 700,1200], dtype=np.float64)

def calculate_ssim_score(mu1: float, sigma1: float, mu2: float, sigma2: float) -> float:
    """Simplified SSIM score (1‑minus similarity) used as a weighting factor."""
    c1 = (K1 * L) ** 2
    c2 = (K2 * L) ** 2
    numerator = (2 * mu1 * mu2 + c1)
    denominator = (mu1 ** 2 + mu2 ** 2 + c1)
    return 1.0 - (numerator / denominator)

def honeybee_store_update(store: float, inflow: float, outflow: float, dt: float = DT) -> float:
    """Euler update for the honeybee store."""
    return store + dt * (ALPHA * inflow - BETA * outflow)

# ----------------------------------------------------------------------
# Hybrid structures
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class BanditAction:
    action_id: str
    expected_reward: float
    confidence_bound: float
    algorithm: str = "hybrid"

def ssim_between_vectors(v1: np.ndarray, v2: np.ndarray) -> float:
    """Compute an SSIM‑like score between two 1‑D vectors."""
    mu1, mu2 = v1.mean(), v2.mean()
    sigma1, sigma2 = v1.var(), v2.var()
    return calculate_ssim_score(mu1, sigma1, mu2, sigma2)

def hybrid_curvature_cms(
    adj_matrix: np.ndarray,
    items: List[str]
) -> Tuple[np.ndarray, int]:
    """Compute curvature and a cardinality estimate from a CMS."""
    curvature = compute_curvature(adj_matrix)
    cms = count_min_sketch(items)
    cardinality = estimate_cardinality_from_cms(cms)
    return curvature, cardinality

def hybrid_store_step(
    store: float,
    curvature: np.ndarray,
    cardinality: int,
    dt: float = DT
) -> float:
    """
    Update the honeybee store where:
    - inflow is proportional to the L2 norm of curvature,
    - outflow is scaled by a temperature factor derived from cardinality.
    """
    inflow = np.linalg.norm(curvature, 2)
    # Temperature factor: higher cardinality → lower temperature → larger outflow
    temperature = math.exp(-0.001 * cardinality)  # simple Schoolfield‑like decay
    outflow = temperature * np.mean(curvature) if curvature.size else 0.0
    return honeybee_store_update(store, inflow, outflow, dt)

def hybrid_bandit_select(
    actions: List[BanditAction],
    curvature: np.ndarray,
    reference_curvature: np.ndarray,
    cardinality: int,
    temperature: float = 1.0
) -> BanditAction:
    """
    Select an action using a softmax over temperature‑scaled rewards.
    The reward for each action is boosted by an SSIM weight that measures
    similarity between the current curvature and a reference curvature.
    """
    ssim_weight = ssim_between_vectors(curvature, reference_curvature)
    # Convert cardinality into an additional temperature modifier
    temp_mod = temperature * (1.0 + 0.01 * cardinality)
    scores = []
    for a in actions:
        # Base score: expected reward + confidence bound
        base = a.expected_reward + a.confidence_bound
        # SSIM weight pushes actions whose curvature matches the reference
        weighted = base * (1.0 + ssim_weight)
        # Softmax denominator uses the temperature modifier
        scores.append(weighted / temp_mod)

    # Softmax to obtain propensities
    max_score = max(scores)
    exp_scores = [math.exp(s - max_score) for s in scores]
    total = sum(exp_scores)
    propensities = [e / total for e in exp_scores]

    # Attach propensities to actions (non‑mutating, return new instance)
    chosen_idx = random.choices(range(len(actions)), weights=propensities, k=1)[0]
    chosen = actions[chosen_idx]
    return BanditAction(
        action_id=chosen.action_id,
        expected_reward=chosen.expected_reward,
        confidence_bound=chosen.confidence_bound,
        algorithm=chosen.algorithm,
    )

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # 1. Build a small random adjacency matrix
    rng = np.random.default_rng(42)
    adj = rng.integers(0, 5, size=(5, 5)).astype(np.float64)
    np.fill_diagonal(adj, 0)  # no self‑loops

    # 2. Generate dummy items for the sketch
    items = [f"node_{i}" for i in range(5)] + [f"edge_{i}_{j}" for i in range(5) for j in range(5) if adj[i, j] > 0]

    # 3. Hybrid curvature + CMS
    curvature_vec, card_est = hybrid_curvature_cms(adj, items)
    print("Curvature:", curvature_vec)
    print("Cardinality estimate:", card_est)

    # 4. Store dynamics
    store = 100.0
    store = hybrid_store_step(store, curvature_vec, card_est)
    print("Updated store:", store)

    # 5. Bandit actions
    actions = [
        BanditAction(action_id="a1", expected_reward=1.2, confidence_bound=0.3),
        BanditAction(action_id="a2", expected_reward=0.8, confidence_bound=0.5),
        BanditAction(action_id="a3", expected_reward=1.0, confidence_bound=0.2),
    ]

    # Reference curvature (could be from a previous timestep or a target)
    reference = np.full_like(curvature_vec, curvature_vec.mean())

    chosen = hybrid_bandit_select(actions, curvature_vec, reference, card_est, temperature=1.0)
    print("Chosen action:", chosen.action_id, "propensity derived implicitly via softmax")