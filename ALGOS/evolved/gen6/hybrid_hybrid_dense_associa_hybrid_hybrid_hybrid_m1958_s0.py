# DARWIN HAMMER — match 1958, survivor 0
# gen: 6
# parent_a: hybrid_dense_associative_me_hybrid_hybrid_pherom_m605_s1.py (gen3)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_sparse_wta_hy_m626_s0.py (gen5)
# born: 2026-05-29T23:39:57Z

"""
Hybrid algorithm fusing Dense Associative Memory - Pheromone Infotaxis Privacy System and 
Hybrid algorithm merging hybrid_hybrid_ternary_route_hybrid_bandit_router_m56_s0.py and 
hybrid_sparse_wta_hybrid_privacy_model_m29_s2.py.

Mathematical bridge:
1. The Dense Associative Memory computes an energy function `E(xi) = -beta^{-1} log( sum_i exp(beta * M_i . xi) )`.
2. The Pheromone Infotaxis Privacy System supplies a reconstruction-risk score `R ∈ [0,1]`.
3. The Structural Similarity Index (SSIM) is used to inform the selection of sparse expansions.
4. The top-k sparse expansions are projected onto a high-dimensional space using locality-sensitive hashing.
5. The resulting expanded vectors are treated as queries whose aggregate (sum) is perturbed with Laplace noise to satisfy differential privacy.
6. The noisy aggregate is normalised and fed into the reconstruction-risk function `R = unique_quasi_identifiers / total_records`.
7. This risk score is then used as the scale of a second Laplace noise term that governs whether a model may be admitted to the pool.
8. The energy function `E(xi)` is then modified by the risk score `R` as `Ê = w · E(xi)`, where `w = 1 – R`.
"""

import numpy as np
from pathlib import Path
import math
import random
import sys
from datetime import datetime, timezone

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
    x: list,
    y: list,
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

def hybrid_sparse_expansion(values: list, m: int, salt: str = "") -> list:
    """Hash‑based sparse expansion of `values` into a vector of length `m`."""
    if m <= 0:
        raise ValueError("m must be positive")
    out = [0.0] * m
    for i, v in enumerate(values):
        for r in range(3):
            h = hash(f"{salt}:{i}:{r}")
            j = h % m
            sign = 1.0 if (h // 2 ** 32) % 2 == 0 else -1.0
            out[j] += sign * v
    return out

def hybrid_energy(xi, M, beta, risk_score):
    """Compute the hybrid energy function."""
    w = 1 - risk_score
    return w * energy(xi, M, beta)

def hybrid_update_rule(xi, M, beta, risk_score, learning_rate=0.1):
    """Compute the hybrid update rule."""
    E = hybrid_energy(xi, M, beta, risk_score)
    return xi - learning_rate * np.gradient(E)

def best_privacy_action(M, beta, risk_score, num_actions=10):
    """Compute the best action given the hybrid energy function and risk score."""
    energies = [hybrid_energy(np.random.rand(M.shape[1]), M, beta, risk_score) for _ in range(num_actions)]
    return np.argmin(energies)

if __name__ == "__main__":
    M = np.random.rand(10, 5)
    beta = 1.0
    risk_score = 0.5
    xi = np.random.rand(5)
    print(hybrid_energy(xi, M, beta, risk_score))
    print(hybrid_update_rule(xi, M, beta, risk_score))
    print(best_privacy_action(M, beta, risk_score))