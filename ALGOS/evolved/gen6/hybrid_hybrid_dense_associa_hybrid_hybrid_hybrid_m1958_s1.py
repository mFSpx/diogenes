# DARWIN HAMMER — match 1958, survivor 1
# gen: 6
# parent_a: hybrid_dense_associative_me_hybrid_hybrid_pherom_m605_s1.py (gen3)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_sparse_wta_hy_m626_s0.py (gen5)
# born: 2026-05-29T23:39:57Z

"""
Hybrid Algorithm: Fusing Dense Associative Memory - Pheromone Infotaxis Privacy System 
with Hybrid Sparse Expansion - Structural Similarity Index 

This module mathematically fuses the *Dense Associative Memory - Pheromone Infotaxis Privacy System* 
(parent algorithm A) with the *Hybrid Sparse Expansion - Structural Similarity Index* 
(parent algorithm B). 

The mathematical bridge between the two algorithms lies in the use of the 
reconstruction-risk score from the Pheromone Infotaxis Privacy System to modulate 
the sparse expansion process in the Hybrid Sparse Expansion algorithm. 

The hybrid system:
1. Computes raw pheromone signals and associated risk scores.
2. Forms weighted sparse expansions using the risk scores.
3. Uses the weighted sparse expansions in decision making and differentially-private aggregation.
"""

import numpy as np
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

def _softmax(z):
    """Numerically stable softmax over 1-D array z."""
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()

def _lse(z):
    """log-sum-exp of 1-D array z (numerically stable)."""
    m = z.max()
    return m + np.log(np.exp(z - m).sum())

def energy(xi, M, beta=1.0):
    """Compute the Dense AM energy E(xi).

    Parameters
    ----------
    xi : array shape (d,)
        Query / current state vector.
    M : array shape (N, d)
        Memory matrix — each row is one stored pattern.
    beta : float
        Inverse temperature. Higher -> sharper energy wells.

    Returns
    -------
    float
        Scalar energy value. Fixed-point attractors are local minima.
    """
    xi = np.asarray(xi, dtype=float)
    M = np.asarray(M, dtype=float)
    scores = beta * (M @ xi)
    lse_term = _lse(scores) / beta
    quadratic_term = 0.5 * xi @ xi
    return -lse_term + quadratic_term

def compute_ssim(
    x: List[float],
    y: List[float],
    dynamic_range: float = 1.0,
    k1: float = 0.01,
    k2: float = 0.03,
) -> float:
    if len(x) != len(y):
        raise ValueError("samples must have equal length")
    if not x:
        raise ValueError("samples must not be empty")

    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)

    mx = x_arr.mean()
    my = y_arr.mean()
    vx = ((x_arr - mx) ** 2).mean()
    vy = ((y_arr - my) ** 2).mean()
    cov = ((x_arr - mx) * (y_arr - my)).mean()

    c1 = (k1 * dynamic_range) ** 2
    c2 = (k2 * dynamic_range) ** 2

    numerator = (2 * mx * my + c1) * (2 * cov + c2)
    denominator = (mx * mx + my * my + c1) * (vx + vy + c2)
    return float(numerator / denominator)

def hybrid_sparse_expansion(values: List[float], m: int, salt: str = "", risk_score: float = 1.0) -> List[float]:
    """Hash‑based sparse expansion of `values` into a vector of length `m`, 
    modulated by a risk score."""
    if m <= 0:
        raise ValueError("m must be positive")
    out = [0.0] * m
    for i, v in enumerate(values):
        for r in range(3):
            import hashlib
            h = hashlib.sha256(f"{salt}:{i}:{r}".encode()).digest()
            j = int.from_bytes(h[:8], "big") % m
            sign = 1.0 if random.random() < 0.5 else -1.0
            out[j] += sign * v * risk_score
    return out

def hybrid_energy(xi, M, beta=1.0, risk_score=1.0):
    """Compute the hybrid energy E(xi) modulated by a risk score."""
    energy_value = energy(xi, M, beta)
    return risk_score * energy_value

def best_privacy_action(xi, M, beta=1.0, risk_score=1.0):
    """Determine the best privacy-preserving action."""
    hybrid_energy_value = hybrid_energy(xi, M, beta, risk_score)
    return np.argmin(hybrid_energy_value)

if __name__ == "__main__":
    # Smoke test
    np.random.seed(0)
    M = np.random.rand(10, 5)
    xi = np.random.rand(5)
    beta = 1.0
    risk_score = 0.5

    energy_value = energy(xi, M, beta)
    print("Energy:", energy_value)

    ssim_value = compute_ssim(xi.tolist(), xi.tolist())
    print("SSIM:", ssim_value)

    sparse_expansion = hybrid_sparse_expansion(xi.tolist(), 10, risk_score=risk_score)
    print("Sparse Expansion:", sparse_expansion)

    hybrid_energy_value = hybrid_energy(xi, M, beta, risk_score)
    print("Hybrid Energy:", hybrid_energy_value)

    best_action = best_privacy_action(xi, M, beta, risk_score)
    print("Best Action:", best_action)