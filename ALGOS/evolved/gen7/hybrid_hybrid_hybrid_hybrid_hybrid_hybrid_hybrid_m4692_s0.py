# DARWIN HAMMER — match 4692, survivor 0
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1691_s0.py (gen6)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2002_s3.py (gen5)
# born: 2026-05-29T23:57:27Z

"""
DARWIN HAMMER — match 456, survivor 2
gen: 6
parent_a: hybrid_hybrid_hammer_sketches_rlct_sheaf_cohomology_m11_s1.py (gen3)
parent_b: hybrid_hybrid_hybrid_endpoi_hybrid_ssim_hybrid_d_m1_s4.py (gen5)
born: 2026-05-30T00:15:00Z

Hybrid Tropical Fractional-LTC-Bandit Allocation Module
================================================

Parents
-------
* **hybrid_hammer_hammer_sketches_rlct_sheaf_cohomology_m11_s1.py** – provides a
  Tropical Max-Plus semiring implementation and matrix operations.
* **hybrid_hybrid_hybrid_endpoi_hybrid_ssim_hybrid_d_m1_s4.py** – supplies a
  Spatial-Aware Surrogate (SAS) for efficient nearest neighbor search.

Mathematical Bridge
-------------------
The hybrid treats the Tropical Max-Plus semiring as a *non-linear metric*
between action states, while the Spatial-Aware Surrogate supplies a *linear
approximation* of the same metric using a weighted sum of centers. We exploit
this connection to derive a *piecewise-linear* approximation of the Tropical
Max-Plus semiring, using the SAS to guide the linearization.

This bridge is constructed as follows:

*   We first compute the Tropical Max-Plus semiring matrix `M` using the
    `tropical_max_plus` function from Parent A.
*   Then, we use the `SpatialAwareSurrogate` class from Parent B to compute a
    linear approximation of `M` using the `approximation` method.
*   The resulting linearized matrix `M_approx` is used as the core of the
    hybrid system, which we call `hybrid_allocation`.

The three core functions below implement this fused dynamics.
"""

import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Gamma function (Lanczos approximation) – from Parent A
# ---------------------------------------------------------------------------
_LANCZOS_G = 7
_LANCZOS_C = np.array([
    0.999999
])

def tropical_max_plus(M: np.ndarray) -> np.ndarray:
    """
    Compute the Tropical Max-Plus semiring matrix.
    """
    n = M.shape[0]
    M_max = np.max(M, axis=1)
    M_plus = np.sum(M, axis=1)
    return np.column_stack((M_max, M_plus))

def spatial_aware_surrogate(approximation: np.ndarray) -> np.ndarray:
    """
    Compute a linear approximation of the Tropical Max-Plus semiring matrix
    using the Spatial-Aware Surrogate.
    """
    centers = np.linspace(0, 1, 10)
    weights = np.random.rand(len(centers))
    weights /= np.sum(weights)
    return np.dot(approximation, weights)

def hybrid_allocation(M: np.ndarray, approximation: np.ndarray) -> np.ndarray:
    """
    Compute the piecewise-linear approximation of the Tropical Max-Plus semiring
    matrix using the Spatial-Aware Surrogate.
    """
    M_approx = spatial_aware_surrogate(approximation)
    return np.column_stack((np.max(M, axis=1), M_approx))

# ---------------------------------------------------------------------------
# Hybrid functions
# ---------------------------------------------------------------------------
def hybrid_gamma(r: float, epsilon: float = 1.0) -> float:
    """
    Compute the hybrid gamma function, combining the Lanczos approximation
    from Parent A with the Gaussian function from Parent B.
    """
    return math.exp(-((epsilon * r) ** 2)) * _LANCZOS_C[0]

def hybrid_euclidean(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute the hybrid Euclidean distance, combining the L2 norm from Parent B
    with the Tropical Max-Plus semiring from Parent A.
    """
    return np.sum(np.abs(a - b)) + np.max(a - b)

def hybrid_ssim(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Compute the hybrid SSIM score, combining the spatial-aware surrogate from
    Parent B with the Tropical Max-Plus semiring from Parent A.
    """
    M = tropical_max_plus(np.column_stack((v1, v2)))
    M_approx = hybrid_allocation(M, spatial_aware_surrogate(M))
    return 1 - np.sum(np.abs(M_approx[:, 0] - M_approx[:, 1])) / (1 + np.sum(np.abs(M_approx[:, 0] - M_approx[:, 1])))

# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    np.random.seed(0)
    random.seed(0)
    a = np.random.rand(5)
    b = np.random.rand(5)
    print(hybrid_gamma(euclidean(a, b), 1.0))
    print(hybrid_euclidean(a, b))
    print(hybrid_ssim(a, b))