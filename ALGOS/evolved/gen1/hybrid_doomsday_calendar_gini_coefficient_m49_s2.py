# DARWIN HAMMER — match 49, survivor 2
# gen: 1
# parent_a: doomsday_calendar.py (gen0)
# parent_b: gini_coefficient.py (gen0)
# born: 2026-05-29T23:23:58Z

"""Hybrid module combining Doomsday weekday calculation (Algorithm A) with Gini inequality
coefficient (Algorithm B).

Mathematical bridge:
- Algorithm A maps each calendar date to a numeric weekday w∈{0,…,6} via
  w = (weekday(date) + 1) mod 7.
- Algorithm B quantifies inequality of a non‑negative numeric distribution {x_i}
  with G = Σ (2·i – n – 1)·x_i / (n·Σ x_i), where the x_i are sorted.

The hybrid treats the weekday frequencies of a collection of dates as the numeric
distribution fed to the Gini formula.  Concretely, for a multiset D of dates we
compute the count c_w of each weekday w, form the vector C = (c_0,…,c_6) and
evaluate G(C).  This fuses the discrete mapping of A with the inequality metric of
B, yielding a “weekday inequality index” that measures how evenly a set of dates
covers the week.

All core operations are implemented with NumPy for vectorised efficiency.
"""

from __future__ import annotations

import datetime as dt
import sys
import random
from pathlib import Path
from typing import Iterable, Sequence, Tuple, List, Union

import numpy as np

# ----------------------------------------------------------------------
# Algorithm A – Doomsday (vectorised)
# ----------------------------------------------------------------------
def doomsday_numpy(
    years: np.ndarray,
    months: np.ndarray,
    days: np.ndarray,
) -> np.ndarray:
    """
    Vectorised Doomsday calculation.
    Returns an array of weekday indices where 0 = Sunday … 6 = Saturday.
    The implementation mirrors ``(date.weekday() + 1) % 7`` but works on
    NumPy integer arrays.
    """
    # Build a datetime64 array of shape matching the inputs
    dates = np.stack([years, months, days], axis=-1).astype('datetime64[D]')
    # NumPy's weekday: Monday=0 … Sunday=6
    np_weekday = dates.astype('datetime64[D]').astype('datetime64[ns]').astype('int64')
    # Convert to Python datetime for reliable weekday extraction
    # (NumPy lacks a direct weekday vectorised function, so we use a small loop)
    flat = dates.ravel()
    py_weekday = np.fromiter(
        (dt.datetime.utcfromtimestamp(int(d.astype('datetime64[s]'))).weekday() for d in flat),
        dtype=np.int8,
        count=flat.size,
    )
    py_weekday = py_weekday.reshape(dates.shape[:-1])
    # Apply the Doomsday offset
    return (py_weekday + 1) % 7


# ----------------------------------------------------------------------
# Algorithm B – Gini coefficient (vectorised)
# ----------------------------------------------------------------------
def gini_coefficient_numpy(values: np.ndarray) -> float:
    """
    Compute the Gini coefficient for a 1‑D NumPy array of non‑negative numbers.
    Mirrors the pure‑Python implementation but leverages sorting and vectorised
    arithmetic.
    """
    if values.ndim != 1:
        raise ValueError("values must be a 1‑D array")
    xs = np.sort(values.astype(float))
    if xs.size == 0 or xs.sum() == 0.0:
        return 0.0
    if xs[0] < 0:
        raise ValueError("values must be non‑negative")
    n = xs.size
    i = np.arange(1, n + 1)  # 1‑based index
    numerator = np.sum((2 * i - n - 1) * xs)
    denominator = n * xs.sum()
    return numerator / denominator


# ----------------------------------------------------------------------
# Hybrid functions
# ----------------------------------------------------------------------
def weekday_counts(dates: Iterable[Union[dt.date, Tuple[int, int, int]]]) -> np.ndarray:
    """
    Given an iterable of dates (either ``datetime.date`` objects or ``(y,m,d)`` tuples),
    return a length‑7 NumPy array C where C[w] is the count of dates falling on weekday w
    (0=Sunday,…,6=Saturday) using the Doomsday mapping.
    """
    # Normalise input to three parallel NumPy integer arrays
    years, months, days = [], [], []
    for d in dates:
        if isinstance(d, dt.date):
            y, m, day = d.year, d.month, d.day
        else:
            y, m, day = d
        years.append(y)
        months.append(m)
        days.append(day)
    years_np = np.array(years, dtype=np.int32)
    months_np = np.array(months, dtype=np.int32)
    days_np = np.array(days, dtype=np.int32)

    weekdays = doomsday_numpy(years_np, months_np, days_np)
    counts = np.bincount(weekdays, minlength=7)
    return counts.astype(int)


def weekday_gini(dates: Iterable[Union[dt.date, Tuple[int, int, int]]]) -> float:
    """
    Hybrid metric: Gini coefficient of the weekday distribution of *dates*.
    It quantifies how evenly the dates cover the seven days of the week.
    """
    counts = weekday_counts(dates)
    return gini_coefficient_numpy(counts)


def weekday_gini_matrix(dates: Iterable[Union[dt.date, Tuple[int, int, int]]]) -> np.ndarray:
    """
    Construct a 7×7 matrix M where M[i,j] = |i - j| weighted by the product of
    weekday frequencies.  The overall inequality is then the Gini of the flattened
    weighted differences.  This demonstrates a deeper algebraic fusion of the two
    parent topologies (weekday mapping + Gini on a derived matrix).
    """
    counts = weekday_counts(dates).astype(float)
    # Outer product gives weight for each pair of weekdays
    weight = np.outer(counts, counts)
    # Absolute difference matrix of weekday indices
    idx = np.arange(7)
    diff = np.abs(idx[:, None] - idx[None, :])
    weighted_diff = weight * diff
    # Flatten and compute Gini on the non‑zero weighted differences
    flat = weighted_diff.ravel()
    # Exclude zero entries to avoid artificial inflation of equality
    nonzero = flat[flat > 0]
    if nonzero.size == 0:
        return np.zeros((7, 7))
    gini_val = gini_coefficient_numpy(nonzero)
    # Return a matrix where each element is the Gini of the distribution scaled
    # by the corresponding weighted difference (purely illustrative)
    return weighted_diff * gini_val


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Generate 100 random dates between 2000‑01‑01 and 2025‑12‑31
    start = dt.date(2000, 1, 1).toordinal()
    end = dt.date(2025, 12, 31).toordinal()
    random_dates: List[dt.date] = [
        dt.date.fromordinal(random.randint(start, end)) for _ in range(100)
    ]

    # Hybrid metrics
    counts = weekday_counts(random_dates)
    print("Weekday counts (Sun→Sat):", counts.tolist())

    gini_val = weekday_gini(random_dates)
    print("Weekday Gini coefficient:", gini_val)

    matrix = weekday_gini_matrix(random_dates)
    print("Weighted difference matrix (sample):")
    print(matrix)

    # Ensure no exceptions and reasonable outputs
    assert isinstance(gini_val, float)
    assert matrix.shape == (7, 7)
    sys.exit(0)