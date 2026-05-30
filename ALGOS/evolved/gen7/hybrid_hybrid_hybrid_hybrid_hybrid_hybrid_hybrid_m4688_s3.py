# DARWIN HAMMER — match 4688, survivor 3
# gen: 7
# parent_a: hybrid_hybrid_hybrid_bayes__hybrid_hybrid_hybrid_m1978_s1.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2517_s1.py (gen6)
# born: 2026-05-29T23:57:28Z

"""Hybrid Multivector Bayesian‑Regret Bandit (HMBB)

Parents:
- hybrid_hybrid_hybrid_bayes__hybrid_hybrid_hybrid_m1978_s1.py (Algorithm A)
- hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2517_s1.py (Algorithm B)

Mathematical bridge:
Algorithm A provides an Ollivier‑Ricci curvature matrix **C** that encodes
pairwise geometric affinity of points. Algorithm B builds ternary signatures
for actions and a similarity matrix **S** (dot‑product of high‑dimensional
ternary vectors).  The fusion treats **C** as a geometric prior that
modulates the purely combinatorial similarity **S** via an element‑wise
(Hadamard) product **F = C ⊙ S**.  The fused matrix **F** is then passed
through a Gaussian intensity → Fisher‑information pipeline to obtain a
confidence weight **w** for each arm.  Finally the Bayesian prior derived
from **C** is updated with **w** in a classic multi‑armed bandit soft‑max
update.  The resulting policy simultaneously respects curvature‑driven
geometry and regret‑weighted ternary decision structure.
"""

import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Shared type aliases
# ----------------------------------------------------------------------
Point = Tuple[float, float]

# ----------------------------------------------------------------------
# Parent A – Ollivier‑Ricci curvature utilities
# ----------------------------------------------------------------------


def distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def gaussian(r: float, epsilon: float = 1.0) -> float:
    """Gaussian kernel used both for curvature and for confidence mapping."""
    return math.exp(-((epsilon * r) ** 2))


def compute_ollivier_ricci_curvature(points: List[Point]) -> np.ndarray:
    """Symmetric curvature matrix C where C[i,j] = exp(-|p_i-p_j|^2)."""
    n = len(points)
    C = np.empty((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i, n):
            val = gaussian(distance(points[i], points[j]))
            C[i, j] = val
            C[j, i] = val
    return C


# ----------------------------------------------------------------------
# Parent B – Regret‑weighted ternary lens utilities
# ----------------------------------------------------------------------


class MathAction:
    """Immutable description of a bandit arm."""
    __slots__ = ("id", "expected_value", "cost", "risk")

    def __init__(self, id: str, expected_value: float,
                 cost: float = 0.0, risk: float = 0.0):
        self.id = id
        self.expected_value = expected_value
        self.cost = cost
        self.risk = risk

    def __repr__(self) -> str:
        return (f"MathAction(id={self.id!r}, ev={self.expected_value}, "
                f"cost={self.cost}, risk={self.risk})")


def _seed_from_id(action_id: str) -> int:
    """Deterministic integer seed from a string identifier."""
    return abs(hash(action_id)) % (2 ** 31 - 1)


def ternary_vector(dim: int, seed: int) -> np.ndarray:
    """Generate a deterministic ternary vector (-1,0,1) of length *dim*."""
    rng = random.Random(seed)
    return np.array([rng.choice((-1, 0, 1)) for _ in range(dim)], dtype=np.int8)


def build_ternary_signatures(actions: List[MathAction],
                             dim: int = 8) -> Tuple[np.ndarray, List[str]]:
    """
    For each action produce a ternary vector and stack them into a matrix
    V (shape = n_actions × dim).  Also return the list of ids for reference.
    """
    signatures = []
    ids = []
    for act in actions:
        seed = _seed_from_id(act.id)
        vec = ternary_vector(dim, seed)
        signatures.append(vec)
        ids.append(act.id)
    return np.stack(signatures), ids


def signature_similarity_matrix(V: np.ndarray) -> np.ndarray:
    """
    Cosine‑like similarity between ternary signatures.
    Normalized dot product in [-1,1]; shifted to [0,1] for later weighting.
    """
    dot = V @ V.T  # shape (n,n)
    norms = np.linalg.norm(V, axis=1, keepdims=True)
    norm_matrix = norms @ norms.T
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        sim = np.where(norm_matrix > 0, dot / norm_matrix, 0.0)
    # Shift to [0,1]
    return (sim + 1.0) / 2.0


# ----------------------------------------------------------------------
# Fusion core
# ----------------------------------------------------------------------


def fuse_curvature_and_similarity(C: np.ndarray,
                                  S: np.ndarray) -> np.ndarray:
    """
    Element‑wise (Hadamard) product of curvature matrix C and signature similarity S.
    Both matrices must be square and of the same size.
    """
    if C.shape != S.shape:
        raise ValueError("Curvature and similarity matrices must share shape")
    return C * S


def gaussian_fisher_confidence(F: np.ndarray,
                               epsilon: float = 1.0) -> np.ndarray:
    """
    Map fused matrix F to a confidence weight vector w (size = n_arms).

    1. Apply Gaussian intensity element‑wise.
    2. For each arm i compute a Fisher‑information‑like scalar:
           I_i = 1 / (Var_i + 1e-8)
       where Var_i is the variance of the i‑th row of the Gaussian‑filtered matrix.
    3. Return normalized confidence weights w_i = I_i / sum(I).
    """
    G = np.exp(- (epsilon * F) ** 2)          # Gaussian intensity
    variances = np.var(G, axis=1)            # row‑wise variance
    fisher_info = 1.0 / (variances + 1e-8)   # avoid division by zero
    w = fisher_info / np.sum(fisher_info)
    return w


def bayesian_prior_from_curvature(C: np.ndarray) -> np.ndarray:
    """
    Prior probability for each arm derived from curvature:
        π_i = sum_j C[i,j] / sum_{i,j} C[i,j]
    """
    row_sums = np.sum(C, axis=1)
    total = np.sum(row_sums)
    if total == 0:
        raise ValueError("Curvature matrix sum is zero")
    return row_sums / total


def bandit_update_with_confidence(prior: np.ndarray,
                                  confidence: np.ndarray,
                                  temperature: float = 0.1) -> np.ndarray:
    """
    Soft‑max style update: posterior ∝ prior^α * confidence^β.
    Here α = β = 1 for simplicity; temperature smooths the distribution.
    """
    unnorm = prior * confidence
    # Apply temperature via exponentiation
    with np.errstate(divide='ignore'):
        tempered = np.exp(np.log(unnorm + 1e-12) / temperature)
    posterior = tempered / np.sum(tempered)
    return posterior


# ----------------------------------------------------------------------
# High‑level hybrid policy
# ----------------------------------------------------------------------


def hybrid_policy(points: List[Point],
                  actions: List[MathAction],
                  ternary_dim: int = 8) -> Tuple[MathAction, np.ndarray]:
    """
    Execute the full hybrid pipeline and return the selected action together
    with the final posterior probability vector.
    """
    if len(points) != len(actions):
        raise ValueError("Number of points must equal number of actions")

    # 1. Geometry from Algorithm A
    C = compute_ollivier_ricci_curvature(points)

    # 2. Ternary signatures from Algorithm B
    V, _ = build_ternary_signatures(actions, dim=ternary_dim)
    S = signature_similarity_matrix(V)

    # 3. Fuse the two topologies
    F = fuse_curvature_and_similarity(C, S)

    # 4. Confidence via Gaussian‑Fisher mapping
    conf = gaussian_fisher_confidence(F)

    # 5. Prior from curvature, then Bayesian‑bandit update
    prior = bayesian_prior_from_curvature(C)
    posterior = bandit_update_with_confidence(prior, conf)

    # 6. Select action with highest posterior probability
    selected_idx = int(np.argmax(posterior))
    return actions[selected_idx], posterior


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Simple deterministic points (e.g., vertices of a unit square)
    pts = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]

    # Corresponding actions – ids must be unique
    acts = [
        MathAction(id="a0", expected_value=1.2, cost=0.1),
        MathAction(id="a1", expected_value=0.8, cost=0.2),
        MathAction(id="a2", expected_value=1.5, cost=0.15),
        MathAction(id="a3", expected_value=0.9, cost=0.05),
    ]

    chosen, probs = hybrid_policy(pts, acts, ternary_dim=8)
    print(f"Chosen action: {chosen}")
    print(f"Posterior probabilities: {probs}")
    # Verify that probabilities sum to 1
    assert np.isclose(probs.sum(), 1.0), "Probabilities do not sum to 1"
    # Ensure selected index is within range
    assert 0 <= acts.index(chosen) < len(acts), "Selected action out of range"
    print("Smoke test passed.")