# DARWIN HAMMER — match 851, survivor 1
# gen: 4
# parent_a: hybrid_hybrid_workshare_all_hybrid_liquid_time_c_m39_s4.py (gen2)
# parent_b: hybrid_hybrid_path_signatur_hybrid_hybrid_model__m332_s0.py (gen3)
# born: 2026-05-29T23:31:17Z

"""
This module implements a novel hybrid algorithm that fuses the governing equations 
of the hybrid workshare allocator (hybrid_hybrid_workshare_all_hybrid_liquid_time_c_m39_s4.py) 
and the hybrid path signature (hybrid_hybrid_path_signatur_hybrid_hybrid_model__m332_s0.py) algorithms.
The mathematical bridge between these two structures lies in the use of the weekday weight vector 
from the workshare allocator to modulate the B-spline basis functions employed in the path signature computation.
By integrating the weekday weight vector into the B-spline basis evaluation, 
we can leverage the temporal information from the workshare allocator to improve the accuracy 
of the path signature representation.
"""

import numpy as np
import math
import random
import sys
import pathlib
from datetime import date

def doomsday(year: int, month: int, day: int) -> int:
    return (date(year, month, day).weekday() + 1) % 7

def weekday_weight_vector(groups: list[str], dow: int) -> np.ndarray:
    n = len(groups)
    if n == 0:
        raise ValueError("groups must contain at least one element")
    base_angles = np.arange(n) * (2.0 * math.pi) / n
    phase = (2.0 * math.pi) * (dow % 7) / 7.0
    amplitude = 0.2
    raw = 1.0 + amplitude * np.sin(base_angles + phase)
    weight_vec = raw / raw.sum()
    return weight_vec.astype(np.float64)

def bspline_basis(x, grid, k=3, weights=None):
    """
    Evaluate B-spline basis functions of order k at positions x, 
    with optional modulation by weights.
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

    if weights is not None:
        B *= weights

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
            B_new[:, i] = term_l + term_r
        B = B_new

    return B

def lead_lag_transform(path, weights):
    """
    Lead-lag transform: interleave (lead, lag) channels for causality encoding, 
    with modulation by weights.
    """
    path = np.asarray(path, dtype=float)
    T, d = path.shape
    out = np.empty((2 * T - 1, 2 * d), dtype=float)
    for t in range(T - 1):
        out[2 * t]     = np.concatenate([path[t] * weights,     path[t] * weights])
        out[2 * t + 1] = np.concatenate([path[t + 1] * weights, path[t] * weights])
    out[2 * (T - 1)] = np.concatenate([path[T - 1] * weights, path[T - 1] * weights])
    return out

def hybrid_operation(groups, path, date):
    dow = doomsday(date.year, date.month, date.day)
    weights = weekday_weight_vector(groups, dow)
    transformed_path = lead_lag_transform(path, weights)
    grid = np.linspace(0, 1, 10)
    bspline = bspline_basis(transformed_path[:, 0], grid, weights=weights)
    return bspline

if __name__ == "__main__":
    groups = ["codex", "groq", "cohere", "local_models"]
    path = np.random.rand(10, 5)
    date = date(2024, 1, 1)
    result = hybrid_operation(groups, path, date)
    print(result)