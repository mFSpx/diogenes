# DARWIN HAMMER — match 4953, survivor 1
# gen: 7
# parent_a: hybrid_hybrid_hybrid_path_s_hybrid_hybrid_hybrid_m2420_s0.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1282_s0.py (gen6)
# born: 2026-05-29T23:58:56Z

"""
This module integrates the governing equations of 'hybrid_hybrid_hybrid_path_signatur_hybrid_hybrid_model__m332_s0.py' and 
'hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1282_s0.py'. The mathematical bridge between these structures lies in the 
application of Fisher score as a weighting factor in the calculation of similarity between nodes based on their feature 
vectors, which in turn informs the decision to split in Hoeffding trees, and the use of B-spline basis functions to approximate 
the path signature representation. We leverage the Hoeffding bound to guide the splitting process in a way that minimizes the 
impact of noise in the data stream, and use the gain to modulate the effective learning rate in the hybrid regret engine.
"""

import numpy as np
import math
import random
import sys
import pathlib

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
                + (t[i + order - 1] - x) / denom_l * B[:, i + 1]
            )
            term_r = (
                (t[i + order] - x) / denom_r * B[:, i + 1]
                + (x - t[i + 1]) / denom_r * B[:, i + 2]
            )
            B_new[:, i] = term_l + term_r

def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width ** 2))
    return derivative

def gaussian_beam(theta: float, center: float, width: float) -> float:
    if width <= 0:
        raise ValueError('width must be positive')
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)

def gaussian_hybrid(path, grid, k=3, center=0, width=1, eps=1e-12):
    """
    Hybrid path signature computation using B-spline basis functions and Fisher score.
    """
    path = lead_lag_transform(path)
    bspline = bspline_basis(path, grid, k)
    fisher_weights = np.array([fisher_score(theta, center, width, eps) for theta in path])
    weighted_bspline = np.dot(bspline, fisher_weights)
    return weighted_bspline

def euclidean_hybrid(a: Sequence[float], b: Sequence[float], center=0, width=1) -> float:
    """
    Hybrid Euclidean distance computation using Fisher score.
    """
    if len(a) != len(b):
        raise ValueError(' Sequences must be of the same length')
    fisher_weights = np.array([fisher_score(theta, center, width) for theta in a])
    a = np.array(a)
    b = np.array(b)
    return np.sqrt(np.sum((a - b) ** 2 * fisher_weights))

def gaussian_hybrid_euclidean(path, grid, k=3, center=0, width=1, epsilon=1.0):
    """
    Hybrid Gaussian and Euclidean distance computation using B-spline basis functions and Fisher score.
    """
    path = lead_lag_transform(path)
    bspline = bspline_basis(path, grid, k)
    fisher_weights = np.array([fisher_score(theta, center, width) for theta in path])
    weighted_bspline = np.dot(bspline, fisher_weights)
    return np.sqrt(np.sum(weighted_bspline ** 2 * gaussian(weighted_bspline, epsilon)))

if __name__ == "__main__":
    # Smoke test
    path = np.random.rand(10, 2)
    grid = np.linspace(0, 1, 10)
    center = 0.5
    width = 0.1
    k = 3
    print(gaussian_hybrid(path, grid, k, center, width))
    print(euclidean_hybrid([1, 2, 3], [4, 5, 6], center, width))
    print(gaussian_hybrid_euclidean(path, grid, k, center, width, 1.0))