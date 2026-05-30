# DARWIN HAMMER — match 5052, survivor 1
# gen: 5
# parent_a: hybrid_hybrid_hybrid_rlct_g_hybrid_hybrid_hybrid_m480_s3.py (gen4)
# parent_b: hybrid_hybrid_hybrid_worksh_hybrid_hybrid_path_s_m851_s1.py (gen4)
# born: 2026-05-29T23:59:29Z

"""
This module implements a novel hybrid algorithm that fuses the governing equations 
of the PheromoneRLCTSystem (hybrid_hybrid_hybrid_rlct_g_hybrid_hybrid_hybrid_m480_s3.py) 
and the hybrid workshare allocator with path signature (hybrid_hybrid_hybrid_worksh_hybrid_hybrid_path_s_m851_s1.py) algorithms.
The mathematical bridge between these two structures lies in the use of the weekday weight vector 
from the workshare allocator to modulate the righting time index computation in the PheromoneRLCTSystem.
By integrating the weekday weight vector into the recovery priority calculation, 
we can leverage the temporal information from the workshare allocator to improve the accuracy 
of the PheromoneRLCTSystem representation.
"""

import math
import random
import sys
import pathlib
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple
import numpy as np
from datetime import date

EPISTEMIC_FLAGS: Tuple[str, ...] = ("FACT", "PROBABLE", "POSSIBLE", "BULLSHIT", "SURE_MAYBE")


@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float


def sphericity_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length * width * height) ** (1.0 / 3.0) / length


def flatness_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length + width) / (2.0 * height)


def righting_time_index(m: Morphology, b: float = 1.0 / 3.0,
                        k: float = 0.35, neck_lever: float = 1.0) -> float:
    if m.mass <= 0 or neck_lever <= 0:
        raise ValueError("mass and neck_lever must be positive")
    fi = flatness_index(m.length, m.width, m.height)
    return (m.mass ** b) * math.exp(k * fi) / neck_lever


def recovery_priority(m: Morphology, max_index: float = 10.0, weight: float = 1.0) -> float:
    """Maps righting‑time index to a confidence weight in [0,1]."""
    return max(0.0, min(1.0, righting_time_index(m) * weight / max_index))


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
            term1 = (x - t[i]) / denom_l * B[:, i]
            term2 = (t[i + order] - x) / denom_r * B[:, i + 1]
            B_new[:, i] = term1 + term2
        B = B_new

    return B


def hybrid_pheromone_rlct(m: Morphology, groups: list[str], year: int, month: int, day: int) -> float:
    weight = weekday_weight_vector(groups, doomsday(year, month, day))[0]
    return recovery_priority(m, weight=weight)


def hybrid_bspline_pheromone(x, grid, m: Morphology, groups: list[str], year: int, month: int, day: int):
    weights = weekday_weight_vector(groups, doomsday(year, month, day))
    basis = bspline_basis(x, grid, weights=weights)
    return basis, hybrid_pheromone_rlct(m, groups, year, month, day)


if __name__ == "__main__":
    m = Morphology(length=10.0, width=5.0, height=2.0, mass=1.0)
    groups = ['A', 'B', 'C']
    year = 2026
    month = 5
    day = 29
    x = np.array([1.0, 2.0, 3.0])
    grid = np.array([0.0, 1.0, 2.0, 3.0])
    print(hybrid_pheromone_rlct(m, groups, year, month, day))
    print(hybrid_bspline_pheromone(x, grid, m, groups, year, month, day))