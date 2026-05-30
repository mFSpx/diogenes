# DARWIN HAMMER — match 2535, survivor 1
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_infota_m1160_s0.py (gen5)
# parent_b: hybrid_hybrid_hybrid_nlms_o_hybrid_hybrid_hard_t_m845_s2.py (gen4)
# born: 2026-05-29T23:42:49Z

"""Hybrid Fusion Algorithm
Parents:
- hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_infota_m1160_s0.py (uses Shannon entropy to weight edge priors)
- hybrid_hybrid_hybrid_nlms_o_hybrid_hybrid_hard_t_m845_s2.py (NLMS weight update scaled by a compatibility score)

Mathematical Bridge:
The Shannon entropy **H** computed from categorical evidence is used to
scale the NLMS learning rate μ via μ′ = μ·exp(‑H).  The same entropy
produces a normalized edge‑prior distribution πₑ = exp(‑H)/∑ₑ′exp(‑H).
A compatibility score **s** (dot product of two vectors through a
positive‑definite matrix **P**) further scales the NLMS weight increment.
Thus the weight update becomes  

    w ← w + μ′·s· (error·x) / (‖x‖²+ε)

which fuses the probabilistic edge weighting of Parent A with the
gradient‑descent learning of Parent B.
"""

import math
import random
import sys
from pathlib import Path
from collections import Counter, deque
import numpy as np

# ----------------------------------------------------------------------
# 1. Entropy computation (Parent A)
# ----------------------------------------------------------------------
def compute_shannon_entropy(evidence: list[str]) -> float:
    """Return Shannon entropy H of a list of categorical tokens."""
    if not evidence:
        return 0.0
    counter = Counter(evidence)
    total = len(evidence)
    entropy = 0.0
    for cnt in counter.values():
        p = cnt / total
        entropy -= p * math.log2(p)
    return entropy

# ----------------------------------------------------------------------
# 2. Edge prior distribution weighted by entropy (Parent A)
# ----------------------------------------------------------------------
def compute_edge_priors(edges: list[tuple[int, int]], evidence: list[str]) -> dict[tuple[int, int], float]:
    """
    Compute a prior probability πₑ for each edge using the same entropy H.
    All edges receive the same weight exp(-H) and are normalized.
    """
    H = compute_shannon_entropy(evidence)
    if not edges:
        return {}
    weight = math.exp(-H)
    total_weight = weight * len(edges)
    prior = weight / total_weight
    return {e: prior for e in edges}

# ----------------------------------------------------------------------
# 3. Compatibility score (Parent B)
# ----------------------------------------------------------------------
def compatibility_score(v: np.ndarray, m: np.ndarray) -> float:
    """
    Compute s = vᵀ·P·m where P is a fixed positive‑definite 2×2 matrix.
    The vectors are projected onto the first two dimensions.
    """
    P = np.array([[1.0, 0.0],
                  [0.0, 1.0]])  # identity serves as a simple PD matrix
    v2 = v[:2]
    m2 = m[:2]
    return float(v2.T @ P @ m2)

# ----------------------------------------------------------------------
# 4. NLMS weight update scaled by entropy‑adjusted learning rate and compatibility (Parent B)
# ----------------------------------------------------------------------
def nlms_update(
    w: np.ndarray,
    x: np.ndarray,
    target: float,
    base_mu: float,
    eps: float,
    compat: float,
    entropy: float,
) -> np.ndarray:
    """
    Perform one NLMS step:
        μ′ = μ·exp(-H)
        w ← w + μ′·s· (error·x) / (‖x‖²+ε)
    """
    mu_prime = base_mu * math.exp(-entropy)
    y = float(w @ x)
    error = target - y
    power = float(x @ x) + eps
    increment = mu_prime * compat * error * x / power
    return w + increment

# ----------------------------------------------------------------------
# 5. High‑level hybrid operation combining the three pieces
# ----------------------------------------------------------------------
def hybrid_step(
    evidence: list[str],
    edges: list[tuple[int, int]],
    x: np.ndarray,
    target: float,
    w: np.ndarray,
    base_mu: float = 0.5,
    eps: float = 1e-9,
) -> tuple[np.ndarray, dict[tuple[int, int], float]]:
    """
    Execute a single hybrid iteration:
      * Compute entropy H from evidence.
      * Derive edge priors πₑ.
      * Compute a compatibility score using the current weight vector.
      * Update the weight vector with the entropy‑adjusted NLMS rule.
    Returns the updated weight vector and the edge‑prior map.
    """
    # 1. Entropy
    H = compute_shannon_entropy(evidence)

    # 2. Edge priors (may be used downstream by other components)
    priors = compute_edge_priors(edges, evidence)

    # 3. Compatibility score – use a dummy manifest vector m
    #    For demonstration, m is a random vector of the same size as w.
    m = np.random.rand(*w.shape)
    s = compatibility_score(w, m)

    # 4. NLMS weight update
    w_new = nlms_update(w, x, target, base_mu, eps, s, H)

    return w_new, priors

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Synthetic evidence
    evidence = ["alpha", "beta", "alpha", "gamma", "beta", "beta", "delta"]

    # Simple graph edges (undirected, represented as tuples)
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]

    # Random input vector and target scalar
    rng = np.random.default_rng(42)
    x = rng.random(10)
    target = 0.75

    # Initial weight vector
    w = rng.random(10)

    # Run the hybrid step
    w_updated, edge_priors = hybrid_step(evidence, edges, x, target, w)

    # Display results
    print("Entropy H:", compute_shannon_entropy(evidence))
    print("Edge priors:", edge_priors)
    print("Old weights (first 5):", w[:5])
    print("Updated weights (first 5):", w_updated[:5])
    print("Weight change norm:", np.linalg.norm(w_updated - w))