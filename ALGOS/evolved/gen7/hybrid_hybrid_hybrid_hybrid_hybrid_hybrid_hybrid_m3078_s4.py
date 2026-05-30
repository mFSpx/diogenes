# DARWIN HAMMER — match 3078, survivor 4
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_sparse_wta_hy_m626_s2.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1204_s2.py (gen6)
# born: 2026-05-29T23:47:39Z

"""Hybrid algorithm merging:
- Parent A: SSIM‑based similarity, sparse hash expansion, Laplace‑DP noisy regret matching.
- Parent B: Sheaf Laplacian influencing tropical max‑plus algebra.

Mathematical bridge:
The SSIM similarity score `s` (a scalar utility) is lifted to a vector `u`
by broadcasting over the action space.  This utility vector participates in a
tropical max‑plus regret update

R_{t+1} = t_add( R_t , u , L )

where `t_add` and `t_mul` are the tropical addition/multiplication defined in
Parent B and `L` is the Sheaf Laplacian from Parent B.  After the tropical update
the regret vector is perturbed with Laplace noise calibrated to a privacy risk
term derived from the sparse expansion (Parent A).  The noisy regrets are then
converted to a mixed strategy via a softmax, yielding the final action
selection.

The module therefore fuses the similarity‑driven utility computation of Parent A
with the sheaf‑aware tropical algebra of Parent B, while preserving differential
privacy through Laplace perturbation.
"""

import hashlib
import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np

# ----------------------------------------------------------------------
# Shared constants
# ----------------------------------------------------------------------
PROTOTYPE_VECTOR = np.array([0.2, 0.5, 0.3, 0.7, 0.1], dtype=np.float64)
DEFAULT_SPARSE_DIM = 128          # dimension after sparse expansion
DEFAULT_HASH_SEED = 42
DEFAULT_EPSILON = 1.0            # privacy budget for Laplace mechanism

# ----------------------------------------------------------------------
# Parent‑A components
# ----------------------------------------------------------------------
def _hash_expand(value: float, dim: int = DEFAULT_SPARSE_DIM, seed: int = DEFAULT_HASH_SEED) -> np.ndarray:
    """Sparse hash‑based expansion of a scalar into a high‑dimensional binary vector.

    The function hashes the string representation of `value` repeatedly to obtain
    `dim` pseudo‑random bits, then casts them to {0,1}.  The resulting vector is
    sparse because only a fraction of bits are set to 1 (≈50 % on average).
    """
    random.seed(seed + int(value * 1e6))
    bits = [random.getrandbits(1) for _ in range(dim)]
    return np.array(bits, dtype=np.float64)


def expand_vector(v: np.ndarray) -> np.ndarray:
    """Apply the sparse hash expansion element‑wise and concatenate the results."""
    expanded_parts = [_hash_expand(x) for x in v]
    return np.concatenate(expanded_parts)


def compute_ssim(x: List[float], y: List[float],
                 dynamic_range: float = 1.0,
                 k1: float = 0.01,
                 k2: float = 0.03) -> float:
    """Structural Similarity Index (SSIM) for two equal‑length 1‑D signals."""
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
    denominator = (mx ** 2 + my ** 2 + c1) * (vx + vy + c2)
    return numerator / denominator


def add_laplace_noise(x: np.ndarray, scale: float) -> np.ndarray:
    """Add i.i.d. Laplace(0, scale) noise to each component of `x`."""
    noise = np.random.laplace(loc=0.0, scale=scale, size=x.shape)
    return x + noise


# ----------------------------------------------------------------------
# Parent‑B components
# ----------------------------------------------------------------------
class Sheaf:
    """Simple sheaf structure: nodes have scalar dimensions, edges are undirected."""
    def __init__(self, node_dims: Dict[int, int], edge_list: List[Tuple[int, int]]):
        self.node_dims = dict(node_dims)
        self.edges = list(edge_list)

    def compute_laplacian(self) -> np.ndarray:
        """Return the (signed) incidence‑derived Laplacian matrix."""
        n = len(self.node_dims)
        L = np.zeros((n, n), dtype=np.float64)
        for u, v in self.edges:
            L[u, v] -= 1
            L[v, u] -= 1
            L[u, u] += 1
            L[v, v] += 1
        return L


def t_add(x: np.ndarray, y: np.ndarray, L: np.ndarray) -> np.ndarray:
    """Tropical addition with Laplacian correction."""
    # element‑wise max + trace(L)
    trace_corr = np.trace(L)
    return np.maximum(x, y) + trace_corr


def t_mul(x: np.ndarray, y: np.ndarray, L: np.ndarray) -> np.ndarray:
    """Tropical multiplication with Laplacian correction."""
    # element‑wise sum + trace(L)
    trace_corr = np.trace(L)
    return x + y + trace_corr


def t_matmul(A: np.ndarray, B: np.ndarray, L: np.ndarray) -> np.ndarray:
    """Tropical matrix multiplication with Laplacian correction."""
    # (A ⊗ B)_{ij} = max_k (A_{ik} + B_{kj}) + trace(L)
    trace_corr = np.trace(L)
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    n, m = A.shape[0], B.shape[1]
    result = np.full((n, m), -np.inf, dtype=np.float64)
    for k in range(A.shape[1]):
        result = np.maximum(result, A[:, k][:, np.newaxis] + B[k, :][np.newaxis, :])
    return result + trace_corr


# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------
def hybrid_expand_ssim(input_vec: np.ndarray,
                       prototype_vec: np.ndarray = PROTOTYPE_VECTOR) -> float:
    """Sparse expansion of `input_vec`, expansion of prototype, then SSIM similarity."""
    e = expand_vector(input_vec)
    e_p = expand_vector(prototype_vec)
    # SSIM expects 1‑D real‑valued signals; we normalise to [0,1]
    s = compute_ssim(e.tolist(), e_p.tolist(), dynamic_range=1.0)
    return float(s)


def tropical_regret_update(regret_vec: np.ndarray,
                           utility_vec: np.ndarray,
                           sheaf: Sheaf) -> np.ndarray:
    """Perform a tropical max‑plus regret update using the sheaf Laplacian."""
    L = sheaf.compute_laplacian()
    updated = t_add(regret_vec, utility_vec, L)
    # Ensure non‑negative regrets (tropical semiring often works with ℝ∪{-∞})
    return np.maximum(updated, 0.0)


def noisy_regret_match_step(input_vec: np.ndarray,
                            regret_vec: np.ndarray,
                            sheaf: Sheaf,
                            epsilon: float = DEFAULT_EPSILON) -> Tuple[np.ndarray, np.ndarray]:
    """
    One hybrid iteration:
    1. Compute SSIM similarity → scalar utility `s`.
    2. Broadcast `s` to a utility vector of the same shape as `regret_vec`.
    3. Update regrets tropically.
    4. Add Laplace noise calibrated to a privacy risk derived from the sparse expansion.
    5. Convert noisy regrets to a probability distribution via softmax.
    Returns (mixed_strategy, new_regret_vec).
    """
    # 1. Similarity as utility
    s = hybrid_expand_ssim(input_vec)

    # 2. Broadcast utility
    utility_vec = np.full_like(regret_vec, fill_value=s, dtype=np.float64)

    # 3. Tropical regret update
    updated_regret = tropical_regret_update(regret_vec, utility_vec, sheaf)

    # 4. Privacy risk: ratio of number of ones in expansion to total bits
    expanded = expand_vector(input_vec)
    risk = expanded.sum() / expanded.size  # ∈ (0,1]
    laplace_scale = risk / epsilon

    noisy_regret = add_laplace_noise(updated_regret, scale=laplace_scale)

    # 5. Mixed strategy via softmax (ensures a proper probability vector)
    max_val = np.max(noisy_regret)
    exp_vals = np.exp(noisy_regret - max_val)  # numerical stability
    mixed_strategy = exp_vals / exp_vals.sum()

    return mixed_strategy, noisy_regret


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # deterministic seed for reproducibility
    np.random.seed(0)
    random.seed(0)

    # Dummy input vector (5‑dimensional like the prototype)
    input_vector = np.array([0.15, 0.45, 0.35, 0.65, 0.05], dtype=np.float64)

    # Initialise regrets (zero regret for all actions)
    num_actions = 6
    regrets = np.zeros(num_actions, dtype=np.float64)

    # Simple sheaf: three nodes with a chain topology
    node_dimensions = {0: 1, 1: 1, 2: 1}
    edges = [(0, 1), (1, 2)]
    sheaf = Sheaf(node_dimensions, edges)

    # Run a single hybrid step
    strategy, new_regret = noisy_regret_match_step(
        input_vec=input_vector,
        regret_vec=regrets,
        sheaf=sheaf,
        epsilon=DEFAULT_EPSILON,
    )

    print("Mixed strategy (probabilities):", strategy)
    print("Updated regret vector (noisy):", new_regret)