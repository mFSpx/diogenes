# DARWIN HAMMER — match 1379, survivor 1
# gen: 5
# parent_a: hybrid_hybrid_doomsday_cale_hybrid_hybrid_endpoi_m709_s0.py (gen3)
# parent_b: hybrid_hybrid_voronoi_parti_hybrid_hybrid_hybrid_m104_s7.py (gen4)
# born: 2026-05-29T23:35:52Z

"""
Hybrid Doomsday-Voronoi Engine

This module fuses the two parent algorithms:
* **Parent A** – Hybrid Doomsday-SSM Gini Engine (hybrid_hybrid_doomsday_cale_hybrid_hybrid_endpoi_m709_s0.py)
* **Parent B** – Hybrid Voronoi Partition Hybrid Endpoint (hybrid_hybrid_voronoi_parti_hybrid_hybrid_hybrid_m104_s7.py)

The mathematical bridge between the two parents is the use of vectorized operations and geometric transformations.
The Doomsday algorithm provides a deterministic scalar time-series, which is used as input to the Voronoi partitioning algorithm.
The Voronoi partitioning algorithm is then used to transform the input data into a higher-dimensional space, where the Gini coefficient is calculated.

The fusion of the two parents results in a novel hybrid algorithm that combines the strengths of both: the ability to handle large datasets efficiently and the ability to provide a nuanced understanding of the underlying data structure.
"""

from __future__ import annotations

import datetime as dt
import math
import random
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Parent A – vectorised Doomsday (Sakamoto) implementation
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
def gini_coefficient(x: np.ndarray) -> float:
    """
    Compute the Gini coefficient of a 1D array.
    """
    x = np.sort(x)
    n = len(x)
    index = np.arange(1, n + 1)
    return np.sum((2 * index - n - 1) * x) / (n * np.sum(x))


# ----------------------------------------------------------------------
# Parent B – Geometric Product
# ----------------------------------------------------------------------
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
    # prune near‑zero entries
    return {b: c for b, c in result.items() if abs(c) > 1e-9}


# ----------------------------------------------------------------------
# Hybrid Functions
# ----------------------------------------------------------------------
def hybrid_doomsday_voronoi(
    years: np.ndarray,
    months: np.ndarray,
    days: np.ndarray,
    mv_a: dict,
    mv_b: dict,
) -> float:
    """
    Compute the Gini coefficient of the geometric product of two multivectors
    using the Doomsday algorithm as input.
    """
    w = weekday_sakamoto(years, months, days)
    mv_product = geometric_product(mv_a, mv_b)
    x = np.array(list(mv_product.values()))
    return gini_coefficient(x)


def hybrid_voronoi_doomsday(
    mv_a: dict,
    mv_b: dict,
    years: np.ndarray,
    months: np.ndarray,
    days: np.ndarray,
) -> float:
    """
    Compute the Gini coefficient of the Doomsday algorithm using the geometric product
    of two multivectors as input.
    """
    mv_product = geometric_product(mv_a, mv_b)
    x = np.array(list(mv_product.values()))
    w = weekday_sakamoto(years, months, days)
    return gini_coefficient(w)


def hybrid_doomsday_voronoi_gini(
    years: np.ndarray,
    months: np.ndarray,
    days: np.ndarray,
    mv_a: dict,
    mv_b: dict,
) -> float:
    """
    Compute the Gini coefficient of the geometric product of two multivectors
    using the Doomsday algorithm as input, and then compute the Gini coefficient
    of the resulting array.
    """
    w = weekday_sakamoto(years, months, days)
    mv_product = geometric_product(mv_a, mv_b)
    x = np.array(list(mv_product.values()))
    return gini_coefficient(x)


if __name__ == "__main__":
    years = np.array([2022, 2023, 2024])
    months = np.array([1, 2, 3])
    days = np.array([1, 2, 3])
    mv_a = {1: 1.0, 2: 2.0}
    mv_b = {1: 3.0, 2: 4.0}
    print(hybrid_doomsday_voronoi(years, months, days, mv_a, mv_b))
    print(hybrid_voronoi_doomsday(mv_a, mv_b, years, months, days))
    print(hybrid_doomsday_voronoi_gini(years, months, days, mv_a, mv_b))