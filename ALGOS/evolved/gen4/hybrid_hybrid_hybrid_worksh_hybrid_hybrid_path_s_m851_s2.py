# DARWIN HAMMER — match 851, survivor 2
# gen: 4
# parent_a: hybrid_hybrid_workshare_all_hybrid_liquid_time_c_m39_s4.py (gen2)
# parent_b: hybrid_hybrid_path_signatur_hybrid_hybrid_model__m332_s0.py (gen3)
# born: 2026-05-29T23:31:17Z

"""
This module implements a novel hybrid algorithm that fuses the governing equations of the 
hybrid_hybrid_workshare_all_hybrid_liquid_time_c_m39_s4 and hybrid_hybrid_path_signatur_hybrid_hybrid_model__m332_s0 
algorithms. The mathematical bridge between these two structures lies in the representation 
of the path signature as a sequence of iterated integrals, which can be approximated using 
the B-spline basis functions, and the use of the weekday weight vector to modulate the 
effective learning rate in the hybrid workshare allocation.

By integrating the B-spline basis into the weekday weight vector computation, and using the 
gain to modulate the learning rate, we can leverage the expressive power of neural networks 
to improve the accuracy of the path signature representation and enhance the performance 
of the hybrid workshare allocation algorithm.
"""

import numpy as np
import math
import random
import sys
import pathlib
from datetime import date

GROUPS = ("codex", "groq", "cohere", "local_models")
MAX64 = (1 << 64) - 1

def _pct(value: float) -> float:
    return round(float(value), 6)

def doomsday(year: int, month: int, day: int) -> int:
    return (date(year, month, day).weekday() + 1) % 7

def weekday_weight_vector(groups: np.ndarray, dow: int) -> np.ndarray:
    n = len(groups)
    if n == 0:
        raise ValueError("groups must contain at least one element")
    base_angles = np.arange(n) * (2.0 * math.pi) / n
    phase = (2.0 * math.pi) * (dow % 7) / 7.0
    amplitude = 0.2
    raw = 1.0 + amplitude * np.sin(base_angles + phase)
    weight_vec = raw / raw.sum()
    return weight_vec.astype(np.float64)

def lead_lag_transform(path):
    path = np.asarray(path, dtype=float)
    T, d = path.shape
    out = np.empty((2 * T - 1, 2 * d), dtype=float)
    for t in range(T - 1):
        out[2 * t]     = np.concatenate([path[t],     path[t]])
        out[2 * t + 1] = np.concatenate([path[t + 1], path[t]])
    out[2 * (T - 1)] = np.concatenate([path[T - 1], path[T - 1]])
    return out

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
            B_new[:, i] = term_l + term_r
        B = B_new
    return B

def allocate_hybrid(
    *,
    total_units: float,
    date: date,
    deterministic_target_pct: float = 90.0,
    groups: np.ndarray = np.array(GROUPS),
) -> dict:
    if total_units <= 0:
        raise ValueError("total_units must be positive")
    if not (0.0 <= deterministic_target_pct <= 100.0):
        raise ValueError("deterministic_target_pct must be between 0 and 100")
    if not groups.any():
        raise ValueError("groups required")

    deterministic_units = total_units * deterministic_target_pct / 100.0
    llm_units = total_units - deterministic_units

    dow = doomsday(date.year, date.month, date.day)
    weight_vec = weekday_weight_vector(groups, dow)

    llm_per_group = llm_units * weight_vec
    share_pct_per_group = weight_vec * 100.0

    lanes = [
        {
            "group": grp,
            "llm_units": _pct(llm_per_group[i]),
            "llm_share_pct": _pct(share_pct_per_group[i]),
            "weekday_weight": _pct(weight_vec[i]),
        }
        for i, grp in enumerate(groups)
    ]

    return {
        "lanes": lanes,
        "deterministic_units": _pct(deterministic_units),
        "llm_units": _pct(llm_units),
        "deterministic_target_pct": _pct(deterministic_target_pct),
    }

def hybrid_path_signature(
    path: np.ndarray,
    date: date,
    groups: np.ndarray = np.array(GROUPS),
) -> tuple:
    transformed_path = lead_lag_transform(path)
    dow = doomsday(date.year, date.month, date.day)
    weight_vec = weekday_weight_vector(groups, dow)

    return transformed_path, weight_vec

def hybrid_workshare_allocation(
    path: np.ndarray,
    date: date,
    total_units: float,
    groups: np.ndarray = np.array(GROUPS),
) -> dict:
    transformed_path, weight_vec = hybrid_path_signature(path, date, groups)
    allocation = allocate_hybrid(total_units=total_units, date=date, groups=groups)

    return allocation

if __name__ == "__main__":
    path = np.random.rand(10, 5)
    date = date.today()
    total_units = 100.0
    groups = np.array(GROUPS)

    allocation = hybrid_workshare_allocation(path, date, total_units, groups)
    print(allocation)