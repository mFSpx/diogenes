# DARWIN HAMMER — match 4953, survivor 0
# gen: 7
# parent_a: hybrid_hybrid_hybrid_path_s_hybrid_hybrid_hybrid_m2420_s0.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1282_s0.py (gen6)
# born: 2026-05-29T23:58:56Z

"""
This module integrates the governing equations of 
'hybrid_hybrid_hybrid_path_s_hybrid_hybrid_hybrid_m2420_s0.py' and 
'hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1282_s0.py'. 
The mathematical bridge between these structures lies in the application 
of the B-spline basis functions from the first parent to enhance the 
representation of the Fisher score in the second parent, which is used 
as a weighting factor in the calculation of similarity between nodes 
based on their feature vectors.
"""

import numpy as np
import math
import random
import sys
import pathlib

def bspline_basis(x, grid, k=3):
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
            )
            term_r = (
                (t[i + order] - x) / denom_r * B[:, i + 1]
            )
            B_new[:, i] = (term_l + term_r) / (denom_l + denom_r)
        B = B_new
    return B

def gaussian_beam(theta: float, center: float, width: float) -> float:
    if width <= 0:
        raise ValueError('width must be positive')
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)

def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width ** 2))
    return derivative

def lead_lag_transform[path](path):
    path = np.asarray(path, dtype=float)
    T, d = path.shape
    out = np.empty((2 * T - 1, 2 * d), dtype=float)
    for t in range(T - 1):
        out[2 * t]     = np.concatenate([path[t],     path[t]])
        out[2 * t + 1] = np.concatenate([path[t + 1], path[t]])
    out[2 * (T - 1)] = np.concatenate([path[T - 1], path[T - 1]])
    return out

def enhanced_fisher_score(theta: float, center: float, width: float, grid: np.ndarray) -> float:
    B = bspline_basis(np.array([theta]), grid)
    return np.sum(B * fisher_score(theta, center, width))

def weighted_euclidean(a: np.ndarray, b: np.ndarray, weights: np.ndarray) -> float:
    if len(a) != len(b) or len(a) != len(weights):
        raise ValueError("all inputs must have the same length")
    return np.sum(weights * (a - b) ** 2)

if __name__ == "__main__":
    grid = np.linspace(0, 1, 10)
    print(enhanced_fisher_score(0.5, 0.5, 0.1, grid))
    print(gaussian_beam(0.5, 0.5, 0.1))
    path = np.random.rand(10, 3)
    print(lead_lag_transform(path).shape)