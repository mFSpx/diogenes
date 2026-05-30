# DARWIN HAMMER — match 2859, survivor 0
# gen: 6
# parent_a: hybrid_hybrid_dense_associa_hybrid_hybrid_hybrid_m1726_s1.py (gen5)
# parent_b: hybrid_hybrid_hybrid_path_s_hybrid_hybrid_model__m1308_s0.py (gen4)
# born: 2026-05-29T23:46:28Z

"""
This module fuses the Dense Associative Memory (Modern Hopfield Networks) from 
hybrid_hybrid_dense_associa_hybrid_hybrid_hybrid_m1726_s1 with the path signature 
representation from hybrid_hybrid_hybrid_path_s_hybrid_hybrid_model__m1308_s0.
The mathematical bridge between these structures lies in the use of the 
softmax function (Boltzmann distribution) in the Dense Associative Memory and 
the representation of the path signature as a sequence of iterated integrals.
The fusion integrates the governing equations of both parents by using the 
Dense Associative Memory to store and retrieve patterns, and the path signature 
representation to compute the Fisher information of the retrieval process 
and to use this information to modulate the hypervectors.
"""

import numpy as np
import math
import random
import sys
from pathlib import Path

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
    return -beta**-1 * np.log(np.sum(np.exp(beta * M @ xi))) + lse_term + quadratic_term

def gaussian_beam(theta: float, center: float, width: float) -> float:
    if width <= 0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)

def fisher_score(theta: float, center: float = 0.0, width: float = 1.0, eps: float = 1e-12) -> float:
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / width**2)
    return derivative

def lead_lag_transform(path):
    """
    Lead-lag transform: interleave (lead, lag) channels for causality encoding.
    """
    path = np.asarray(path, dtype=float)
    T, d = path.shape
    out = np.empty((2 * T - 1, 2 * d), dtype=float)
    for t in range(T - 1):
        out[2 * t]     = np.concatenate([path[t],     path[t]])
        out[2 * t + 1] = np.concatenate([path[t + 1], path[t]])
    out[2 * (T - 1)] = np.concatenate([path[T - 1], path[T - 1]])
    return out

def bspline_basis(x, grid, k=3):
    """
    Evaluate B-spline basis functions of order k at positions x.
    """
    x = np.asarray(x, dtype=np.float64)
    grid = np.asarray(grid, dtype=np.float64)

    t = np.concatenate([
        np.full(k - 1, grid[0]),
        grid,
        np.full(k - 1, grid[-1]),
    ])

    n_basis = len(grid) + k - 2
    N = len(x)

    B = np.zeros((N, len(t) - 1), dtype=np.float64)
    for i in range(len(t) - 1):
        B[:, i] = np.where((x >= t[i]) & (x < t[i + 1]), 1.0, 0.0)
    B[:, -1] = np.where(x == t[-1], 1.0, B[:, -1])

    for order in range(2, k + 1):
        B_new = np.zeros((N, len(t) - order), dtype=np.float64)
        for i in range(len(t) - order):
            denom_l = t[i + order - 1] - t[i]
            denom_r = t[i + order] - t[i + 1]
            term_l = (
                (x - t[i]) / denom_l * B[:, i]
                if denom_l > 0 else np.zeros(N)
            )
            term_r = (
                (t[i + order] - x) / denom_r * B[:, i + 1]
                if denom_r > 0 else np.zeros(N)
            )
            B_new[:, i] = term_l + term_r
        B = B_new

    return B

def hybrid_energy(xi, M, path, beta=1.0):
    """Compute the hybrid energy E(xi).

    Parameters
    ----------
    xi : array shape (d,)
        Query / current state vector.
    M : array shape (N, d)
        Memory matrix — each row is one stored pattern.
    path : array shape (T, d)
        Path signature representation.
    beta : float
        Inverse temperature. Higher -> sharper energy wells.

    Returns
    -------
    float
        Scalar energy value. Fixed-point attractors are local minima.
    """
    lead_lag_path = lead_lag_transform(path)
    energy_term = energy(xi, M, beta)
    fisher_term = np.sum([fisher_score(x) for x in lead_lag_path.flatten()])
    return energy_term + fisher_term

def test_hybrid_energy():
    xi = np.array([1.0, 2.0, 3.0])
    M = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    path = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    print(hybrid_energy(xi, M, path))

if __name__ == "__main__":
    test_hybrid_energy()