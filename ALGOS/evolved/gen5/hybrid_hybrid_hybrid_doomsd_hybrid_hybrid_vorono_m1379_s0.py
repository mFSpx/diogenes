# DARWIN HAMMER — match 1379, survivor 0
# gen: 5
# parent_a: hybrid_hybrid_doomsday_cale_hybrid_hybrid_endpoi_m709_s0.py (gen3)
# parent_b: hybrid_hybrid_voronoi_parti_hybrid_hybrid_hybrid_m104_s7.py (gen4)
# born: 2026-05-29T23:35:52Z

from __future__ import annotations

import datetime as dt
import math
import random
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np


# ----------------------------------------------------------------------
# Parent A – Hybrid Doomsday-SSM Gini Engine
# ----------------------------------------------------------------------
def weekday_sakamoto(
    years: np.ndarray,
    months: np.ndarray,
    days: np.ndarray,
) -> np.ndarray:
    """
    Compute weekday indices for vectorised (year, month, day) arrays using
    Tomohiko Sakamoto's algorithm.  Result: 0 = Sunday … 6 = Saturday.
    """
    y = years.astype(np.int64)
    m = months.astype(np.int64)
    d = days.astype(np.int64)

    m_adj = np.where(m < 3, m + 12, m)
    y_adj = np.where(m < 3, y - 1, y)

    t = np.array([0, 3, 2, 5, 0, 3, 5, 1, 4, 6, 2, 4], dtype=np.int64)

    w = (y_adj + y_adj // 4 - y_adj // 100 + y_adj // 400 + t[m_adj - 1] + d) % 7
    return w.astype(np.int8)


# ----------------------------------------------------------------------
# Parent A – Gini coefficient
# ----------------------------------------------------------------------
def gini_coefficient(scores: np.ndarray) -> float:
    """
    Compute the Gini coefficient of the scores.
    """
    n = len(scores)
    x = np.sort(scores)
    area = np.trapz(x[:-1], x[1:])
    return (2 * area / (n * np.mean(x))) - (1 / n)


# ----------------------------------------------------------------------
# Parent B – Voronoi partition
# ----------------------------------------------------------------------
@dataclass
class Point:
    x: float
    y: float


def geometric_product(mv_a: dict, mv_b: dict) -> dict:
    """
    Euclidean Clifford geometric product using bit‑mask blades.
    Returns a new multivector.
    """
    result: dict = {}
    for blade_a, coeff_a in mv_a.items():
        for blade_b, coeff_b in mv_b.items():
            blade_res = blade_a ^ blade_b
            sign = _blade_sign(blade_a, blade_b)
            coeff_res = coeff_a * coeff_b * sign
            result[blade_res] = result.get(blade_res, 0.0) + coeff_res
    return {b: c for b, c in result.items() if abs(c) > 1e-6}


def _grade(blade: int) -> int:
    """Number of set bits = grade of the blade."""
    return blade.bit_count()


def _blade_sign(a: int, b: int) -> int:
    """
    Sign of the geometric product of two basis blades a and b
    under a Euclidean metric (e_i^2 = +1).
    Implements the rule:
        e_i e_j = - e_j e_i  for i != j
    The sign is (-1)^{N} where N is the number of swaps needed to
    reorder the concatenated index list.
    """
    # Count swaps using the classic "grade involution" method:
    # For each set bit in a, count how many bits in b are lower index.
    sign = 1
    while a:
        lowest = a & -a               # isolate lowest set bit of a
        idx = (lowest.bit_length() - 1)
        # bits in b with index < idx cause a swap
        lower = b & ((1 << idx) - 1)
        if lower.bit_count() & 1:
            sign = -sign
        a ^= lowest
    return sign


# ----------------------------------------------------------------------
# Hybrid algorithm
# ----------------------------------------------------------------------
def hybrid_doomsday_voronoi(years: np.ndarray, months: np.ndarray, days: np.ndarray) -> Tuple[np.ndarray, dict]:
    """
    Hybrid algorithm that combines the Sakamoto calendar and Voronoi partition.
    """
    # Compute weekday indices
    weekday = weekday_sakamoto(years, months, days)

    # Create a Voronoi partition with 7 points (one for each day of the week)
    points = [Point(x, y) for x, y in zip(weekday, np.random.rand(len(weekday)))]
    voronoi = {point.x: point.y for point in points}

    # Create a multivector to represent the Voronoi partition
    mv = {k: 1.0 for k in range(7)}

    # Compute the geometric product of the Sakamoto calendar and the Voronoi partition
    result = geometric_product(mv, {weekday[i]: 1.0 for i in range(len(weekday))})

    return weekday, result


def hybrid_doomsday_voronoi_gini(years: np.ndarray, months: np.ndarray, days: np.ndarray) -> float:
    """
    Hybrid algorithm that combines the Sakamoto calendar, Voronoi partition, and Gini coefficient.
    """
    weekday, voronoi_partition = hybrid_doomsday_voronoi(years, months, days)
    scores = np.array([voronoi_partition[x] for x in weekday])
    return gini_coefficient(scores)


def hybrid_doomsday_voronoi_visualization(years: np.ndarray, months: np.ndarray, days: np.ndarray) -> None:
    """
    Hybrid algorithm that combines the Sakamoto calendar, Voronoi partition, and visualization.
    """
    weekday, voronoi_partition = hybrid_doomsday_voronoi(years, months, days)
    import matplotlib.pyplot as plt
    points = [Point(x, y) for x, y in zip(weekday, np.random.rand(len(weekday)))]
    plt.scatter([point.x for point in points], [point.y for point in points])
    plt.show()


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    years = np.array([2022, 2023, 2024])
    months = np.array([1, 2, 3])
    days = np.array([1, 15, 30])
    print(hybrid_doomsday_voronoi_gini(years, months, days))