# DARWIN HAMMER — match 1379, survivor 4
# gen: 5
# parent_a: hybrid_hybrid_doomsday_cale_hybrid_hybrid_endpoi_m709_s0.py (gen3)
# parent_b: hybrid_hybrid_voronoi_parti_hybrid_hybrid_hybrid_m104_s7.py (gen4)
# born: 2026-05-29T23:35:52Z

"""Hybrid Doomsday‑Gini–Geometric Algebra Engine

This module fuses the two parent algorithms:

* **Parent A** – a vectorised Sakamoto calendar (``weekday_sakamoto``) that
  produces a deterministic scalar time‑series ``xₜ`` and a Gini‑coefficient
  routine that measures inequality of a scalar array.

* **Parent B** – a Euclidean Clifford geometric‑algebra core (blade handling,
  ``geometric_product``) that enables algebraic manipulation of multivectors.

**Mathematical bridge**

The scalar series ``xₜ`` is interpreted as a *scalar blade* (blade ``0``) of a
multivector.  A linear‑state‑space model is rewritten in the geometric‑algebra
domain:


hₜ   = 𝔸 ⊗ hₜ₋₁  ⊕  ℬ ⊗ xₜ          (state update)
yₜ   = 𝔠 ⊗ hₜ                     (output, scalar part)


where ``⊗`` is the geometric product, ``⊕`` is multivector addition, and
``𝔸, ℬ, 𝔠`` are multivectors that play the role of the usual matrices ``A,
B, C``.  After the whole request sequence we obtain the scalar output
vector ``Y = (y₁,…,y_T)`` and evaluate its Gini coefficient.  The three
public functions below demonstrate the end‑to‑end hybrid workflow.

"""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Parent A – vectorised Doomsday (Sakamoto) implementation
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
# Parent B – Euclidean Clifford geometric‑algebra core
# ----------------------------------------------------------------------
Blade = int                     # bitmask representation, e.g. e1^e3 -> 0b1010
Multivector = Dict[Blade, float]   # mapping blade → coefficient


def _grade(blade: Blade) -> int:
    """Number of set bits = grade of the blade."""
    return blade.bit_count()


def _blade_sign(a: Blade, b: Blade) -> int:
    """
    Sign of the geometric product of two basis blades a and b
    under a Euclidean metric (e_i^2 = +1).
    Implements the rule:
        e_i e_j = - e_j e_i  for i != j
    The sign is (-1)^{N} where N is the number of swaps needed to
    reorder the concatenated index list.
    """
    sign = 1
    while a:
        lowest = a & -a               # isolate lowest set bit of a
        idx = lowest.bit_length() - 1
        lower = b & ((1 << idx) - 1)
        if lower.bit_count() & 1:
            sign = -sign
        a ^= lowest
    return sign


def geometric_product(mv_a: Multivector, mv_b: Multivector) -> Multivector:
    """
    Euclidean Clifford geometric product using bit‑mask blades.
    Returns a new multivector.
    """
    result: Multivector = {}
    for blade_a, coeff_a in mv_a.items():
        for blade_b, coeff_b in mv_b.items():
            blade_res = blade_a ^ blade_b
            sign = _blade_sign(blade_a, blade_b)
            coeff_res = coeff_a * coeff_b * sign
            result[blade_res] = result.get(blade_res, 0.0) + coeff_res
    # prune near‑zero entries
    return {b: c for b, c in result.items() if abs(c) > 1e-14}


def scalar_part(mv: Multivector) -> float:
    """Return the scalar (grade‑0) component of a multivector."""
    return mv.get(0, 0.0)


def multivector_from_scalar(s: float) -> Multivector:
    """Embed a real scalar as a grade‑0 multivector."""
    return {0: float(s)}


def add_multivectors(a: Multivector, b: Multivector) -> Multivector:
    """Component‑wise addition of two multivectors."""
    result = a.copy()
    for blade, coeff in b.items():
        result[blade] = result.get(blade, 0.0) + coeff
    # prune near‑zero entries
    return {bl: c for bl, c in result.items() if abs(c) > 1e-14}


# ----------------------------------------------------------------------
# Gini coefficient (Parent A)
# ----------------------------------------------------------------------
def gini_coefficient(values: np.ndarray) -> float:
    """
    Compute the Gini coefficient of a 1‑D numeric array.
    Returns a value in [0, 1]; 0 means perfect equality.
    """
    if values.ndim != 1:
        raise ValueError("Gini coefficient requires a 1‑D array")
    n = len(values)
    if n == 0:
        return 0.0
    sorted_vals = np.sort(values.astype(np.float64))
    cumulative = np.cumsum(sorted_vals)
    sum_vals = cumulative[-1]
    if sum_vals == 0:
        return 0.0
    # Gini formula based on the Lorenz curve
    gini = (n + 1 - 2 * np.sum(cumulative) / sum_vals) / n
    return float(gini)


# ----------------------------------------------------------------------
# Hybrid engine – combines the calendar, geometric‑algebra SSM and Gini
# ----------------------------------------------------------------------
def evolve_state_geometric(
    x_series: np.ndarray,
    A_mv: Multivector,
    B_mv: Multivector,
    C_mv: Multivector,
    init_state: Multivector,
) -> np.ndarray:
    """
    Evolve a multivector state using a geometric‑algebra analogue of a linear
    state‑space model and return the scalar output series.

    Parameters
    ----------
    x_series : np.ndarray
        Input scalar series (e.g. weekday indices).
    A_mv, B_mv, C_mv : Multivector
        Multivector “matrices” that replace the usual A, B, C matrices.
    init_state : Multivector
        Initial health state (multivector).

    Returns
    -------
    y_series : np.ndarray
        Scalar outputs extracted from the grade‑0 part of C_mv ⊗ hₜ.
    """
    h = init_state
    y_list = []
    for x in x_series:
        x_mv = multivector_from_scalar(float(x))
        term1 = geometric_product(A_mv, h)
        term2 = geometric_product(B_mv, x_mv)
        h = add_multivectors(term1, term2)
        y_mv = geometric_product(C_mv, h)
        y = scalar_part(y_mv)
        y_list.append(y)
    return np.array(y_list, dtype=np.float64)


def hybrid_doomsday_gini(
    years: np.ndarray,
    months: np.ndarray,
    days: np.ndarray,
    A_mv: Multivector,
    B_mv: Multivector,
    C_mv: Multivector,
    init_state: Multivector,
) -> Tuple[np.ndarray, float]:
    """
    Complete hybrid pipeline:

    1. Compute weekday indices from calendar dates.
    2. Feed the indices as scalar blades into a geometric‑algebra state‑space
       model.
    3. Extract the scalar output series and evaluate its Gini coefficient.

    Returns the raw output series and the Gini value.
    """
    x_series = weekday_sakamoto(years, months, days).astype(np.float64)
    y_series = evolve_state_geometric(x_series, A_mv, B_mv, C_mv, init_state)
    gini = gini_coefficient(y_series)
    return y_series, gini


# ----------------------------------------------------------------------
# Helper – random multivector generator (for demonstration)
# ----------------------------------------------------------------------
def random_multivector(max_blade: int = 0b1111, density: float = 0.3) -> Multivector:
    """
    Generate a random multivector.

    Parameters
    ----------
    max_blade : int
        Upper bound (exclusive) for possible blade bit‑patterns.
    density : float
        Approximate fraction of blades that receive a non‑zero coefficient.

    Returns
    -------
    Multivector
    """
    mv: Multivector = {}
    for blade in range(max_blade):
        if random.random() < density:
            mv[blade] = random.uniform(-1.0, 1.0)
    # Ensure at least the scalar part exists
    if 0 not in mv:
        mv[0] = random.uniform(-1.0, 1.0)
    return mv


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Generate a simple date range (30 consecutive days starting today)
    today = np.datetime64("today", "D")
    dates = today + np.arange(30).astype("timedelta64[D]")
    years = dates.astype("datetime64[Y]").astype(int) + 1970
    months = dates.astype("datetime64[M]").astype(int) % 12 + 1
    days = dates.astype("datetime64[D]").astype(int) % 31 + 1

    # Random geometric‑algebra “matrices”
    A_mv = random_multivector()
    B_mv = random_multivector()
    C_mv = random_multivector()
    init_state = random_multivector()

    y, g = hybrid_doomsday_gini(
        years,
        months,
        days,
        A_mv,
        B_mv,
        C_mv,
        init_state,
    )

    print("Output series (first 10 values):", y[:10])
    print("Gini coefficient of output:", g)
    sys.exit(0)