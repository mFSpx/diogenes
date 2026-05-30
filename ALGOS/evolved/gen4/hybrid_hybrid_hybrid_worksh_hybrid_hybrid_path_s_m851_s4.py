# DARWIN HAMMER — match 851, survivor 4
# gen: 4
# parent_a: hybrid_hybrid_workshare_all_hybrid_liquid_time_c_m39_s4.py (gen2)
# parent_b: hybrid_hybrid_path_signatur_hybrid_hybrid_model__m332_s0.py (gen3)
# born: 2026-05-29T23:31:17Z

import numpy as np
import math
import random
import sys
import pathlib
import datetime as dt
from typing import Any, Dict, Iterable, List, Sequence, Tuple

def lead_lag_transform(path: np.ndarray) -> np.ndarray:
    path = np.asarray(path, dtype=float)
    T, d = path.shape
    out = np.empty((2 * T - 1, 2 * d), dtype=float)
    for t in range(T - 1):
        out[2 * t]     = np.concatenate([path[t],     path[t]])
        out[2 * t + 1] = np.concatenate([path[t + 1], path[t]])
    out[2 * (T - 1)] = np.concatenate([path[T - 1], path[T - 1]])
    return out

def weekday_weight_vector(groups: Sequence[str], dow: int) -> np.ndarray:
    n = len(groups)
    if n == 0:
        raise ValueError("groups must contain at least one element")
    base_angles = np.arange(n) * (2.0 * math.pi) / n
    phase = (2.0 * math.pi) * (dow % 7) / 7.0
    amplitude = 0.2
    raw = 1.0 + amplitude * np.sin(base_angles + phase)
    weight_vec = raw / raw.sum()
    return weight_vec.astype(np.float64)

def bspline_basis(x: np.ndarray, grid: np.ndarray, k: int = 3, weights: np.ndarray = None) -> np.ndarray:
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
        B = B * weights[:, np.newaxis]

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

def fused_lead_lag_transform(path: np.ndarray, groups: Sequence[str], date: dt.date) -> np.ndarray:
    dow = (dt.date(date.year, date.month, date.day).weekday() + 1) % 7
    weights = weekday_weight_vector(groups, dow)
    lead_lag_path = lead_lag_transform(path)
    return lead_lag_path * weights[:, np.newaxis]

def fused_bspline_basis(x: np.ndarray, grid: np.ndarray, k: int = 3, groups: Sequence[str] = None, date: dt.date = None) -> np.ndarray:
    if groups is not None and date is not None:
        dow = (dt.date(date.year, date.month, date.day).weekday() + 1) % 7
        weights = weekday_weight_vector(groups, dow)
        return bspline_basis(x, grid, k, weights)
    else:
        return bspline_basis(x, grid, k)

def fused_allocate_hybrid(
    *,
    total_units: float,
    date: dt.date,
    deterministic_target_pct: float = 90.0,
    groups: Tuple[str, ...] = ("codex", "groq", "cohere", "local_models"),
) -> Dict[str, Any]:
    if total_units <= 0:
        raise ValueError("total_units must be positive")
    if not (0.0 <= deterministic_target_pct <= 100.0):
        raise ValueError("deterministic_target_pct must be between 0 and 100")
    if not groups:
        raise ValueError("groups required")

    deterministic_units = total_units * deterministic_target_pct / 100.0
    llm_units = total_units - deterministic_units

    dow = (dt.date(date.year, date.month, date.day).weekday() + 1) % 7
    weight_vec = weekday_weight_vector(groups, dow)

    llm_per_group = llm_units * weight_vec
    share_pct_per_group = weight_vec * 100.0

    lanes = [
        {
            "group": grp,
            "llm_units": round(float(llm_per_group[i]), 6),
            "llm_share_pct": round(float(share_pct_per_group[i]), 6),
            "weekday_weight": round(float(weight_vec[i]), 6),
        }
        for i, grp in enumerate(groups)
    ]

    return {
        "lanes": lanes,
        "deterministic_target_pct": deterministic_target_pct,
    }

class HybridAllocator:
    def __init__(self, groups: Tuple[str, ...], k: int = 3):
        self.groups = groups
        self.k = k

    def allocate(self, total_units: float, date: dt.date, deterministic_target_pct: float = 90.0) -> Dict[str, Any]:
        return fused_allocate_hybrid(
            total_units=total_units,
            date=date,
            deterministic_target_pct=deterministic_target_pct,
            groups=self.groups,
        )

    def lead_lag_transform(self, path: np.ndarray, date: dt.date) -> np.ndarray:
        return fused_lead_lag_transform(path, self.groups, date)

    def bspline_basis(self, x: np.ndarray, grid: np.ndarray, date: dt.date = None) -> np.ndarray:
        return fused_bspline_basis(x, grid, self.k, self.groups, date)

if __name__ == "__main__":
    date = dt.date(2024, 1, 1)
    groups = ("codex", "groq", "cohere", "local_models")
    total_units = 100.0

    allocator = HybridAllocator(groups)
    result = allocator.allocate(total_units, date)
    print(result)

    path = np.random.rand(10, 3)
    fused_path = allocator.lead_lag_transform(path, date)
    print(fused_path.shape)

    x = np.linspace(0, 10, 100)
    grid = np.linspace(0, 10, 10)
    fused_basis = allocator.bspline_basis(x, grid, date)
    print(fused_basis.shape)