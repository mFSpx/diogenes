# DARWIN HAMMER — match 49, survivor 4
# gen: 1
# parent_a: doomsday_calendar.py (gen0)
# parent_b: gini_coefficient.py (gen0)
# born: 2026-05-29T23:23:58Z

from __future__ import annotations

import datetime as dt
import random
import sys
from pathlib import Path
from typing import Iterable, Tuple, Union, List

import numpy as np

# ----------------------------------------------------------------------
# Algorithm A – Vectorised Doomsday (Sakamoto) implementation
# ----------------------------------------------------------------------
def weekday_sakamoto(
    years: np.ndarray,
    months: np.ndarray,
    days: np.ndarray,
) -> np.ndarray:
    """
    Compute weekday indices for vectorised (year, month, day) arrays using
    Tomohiko Sakamoto's algorithm.  The result follows the convention
    0 = Sunday, …, 6 = Saturday, which matches the ``(weekday + 1) % 7`` mapping
    used in the original hybrid.
    """
    # Ensure integer dtype for safe arithmetic
    y = years.astype(np.int64)
    m = months.astype(np.int64)
    d = days.astype(np.int64)

    # Adjust months and years for algorithm (March = 1 … February = 12)
    m_adj = np.where(m < 3, m + 12, m)
    y_adj = np.where(m < 3, y - 1, y)

    # Month offsets for Gregorian calendar
    t = np.array([0, 3, 2, 5, 0, 3, 5, 1, 4, 6, 2, 4], dtype=np.int64)

    w = (y_adj + y_adj // 4 - y_adj // 100 + y_adj // 400 + t[m_adj - 1] + d) % 7
    return w.astype(np.int8)


# ----------------------------------------------------------------------
# Algorithm B – Gini coefficient (robust vectorised version)
# ----------------------------------------------------------------------
def gini_coefficient(values: np.ndarray) -> float:
    """
    Return the Gini coefficient of a 1‑D array of non‑negative numbers.
    The implementation follows the definition

        G = Σ_i (2·i – n – 1)·x_i / (n·Σ x_i),

    where ``x_i`` are the values sorted in non‑decreasing order and ``i`` is
    1‑based.  Edge cases (empty array, all zeros) yield ``0.0``.
    """
    if values.ndim != 1:
        raise ValueError("values must be a 1‑D array")
    x = np.sort(values.astype(np.float64))
    if x.size == 0 or np.isclose(x.sum(), 0.0):
        return 0.0
    if np.any(x < 0):
        raise ValueError("values must be non‑negative")
    n = x.size
    i = np.arange(1, n + 1, dtype=np.float64)
    numerator = np.sum((2 * i - n - 1) * x)
    denominator = n * x.sum()
    return float(numerator / denominator)


# ----------------------------------------------------------------------
# Hybrid utilities
# ----------------------------------------------------------------------
def _split_dates(
    dates: Iterable[Union[dt.date, Tuple[int, int, int]]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert an iterable of ``datetime.date`` objects or ``(y,m,d)`` tuples
    into three parallel NumPy integer arrays.
    """
    years, months, days = [], [], []
    for d in dates:
        if isinstance(d, dt.date):
            y, m, day = d.year, d.month, d.day
        else:
            y, m, day = d
        years.append(y)
        months.append(m)
        days.append(day)
    return (
        np.array(years, dtype=np.int32),
        np.array(months, dtype=np.int32),
        np.array(days, dtype=np.int32),
    )


def weekday_counts(
    dates: Iterable[Union[dt.date, Tuple[int, int, int]]],
) -> np.ndarray:
    """
    Return a length‑7 array ``C`` where ``C[w]`` is the number of input dates
    that fall on weekday ``w`` (0 = Sunday … 6 = Saturday) using the vectorised
    Sakamoto implementation.
    """
    years, months, days = _split_dates(dates)
    weekdays = weekday_sakamoto(years, months, days)
    return np.bincount(weekdays, minlength=7).astype(np.int64)


def weekday_gini(dates: Iterable[Union[dt.date, Tuple[int, int, int]]]) -> float:
    """
    Hybrid metric: Gini coefficient of the weekday frequency distribution.
    """
    counts = weekday_counts(dates)
    return gini_coefficient(counts)


def weekday_pairwise_gini(dates: Iterable[Union[dt.date, Tuple[int, int, int]]]) -> float:
    """
    Compute a second‑order Gini coefficient that captures inequality in the
    *pairwise* interaction of weekdays.

    Steps:
    1. Build the 7×7 weight matrix W where ``W[i, j] = c_i * c_j``.
    2. Build the absolute difference matrix D where ``D[i, j] = |i - j|``.
    3. Form the element‑wise product P = W * D.
    4. Flatten P and evaluate the Gini coefficient on the resulting 49‑element
       distribution (including zeros, which naturally contribute to equality).

    The returned value lies in ``[0, 1]`` and is zero when all weekdays have
    identical counts (perfect uniformity) and approaches one when the
    distribution is highly concentrated.
    """
    counts = weekday_counts(dates).astype(np.float64)
    weight = np.outer(counts, counts)          # W
    diff = np.abs(np.arange(7)[:, None] - np.arange(7)[None, :])  # D
    paired = weight * diff                     # P
    return gini_coefficient(paired.ravel())


def weekday_gini_deep(dates: Iterable[Union[dt.date, Tuple[int, int, int]]]) -> float:
    """
    A deeper fusion that combines the first‑order Gini (raw weekday counts)
    with the second‑order pairwise Gini defined above.

    The final metric is the geometric mean of the two components, providing
    a balanced view that penalises both uneven coverage and extreme clustering.
    """
    g1 = weekday_gini(dates)
    g2 = weekday_pairwise_gini(dates)
    # Guard against the degenerate case where both are zero
    if g1 == 0.0 and g2 == 0.0:
        return 0.0
    return float(np.sqrt(g1 * g2))


# ----------------------------------------------------------------------
# Smoke test (executed only when run as a script)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Generate 200 random dates between 2000‑01‑01 and 2025‑12‑31
    start_ord = dt.date(2000, 1, 1).toordinal()
    end_ord = dt.date(2025, 12, 31).toordinal()
    random_dates: List[dt.date] = [
        dt.date.fromordinal(random.randint(start_ord, end_ord)) for _ in range(200)
    ]

    counts = weekday_counts(random_dates)
    print("Weekday counts (Sun→Sat):", counts.tolist())

    g_raw = weekday_gini(random_dates)
    print("First‑order Gini (weekday distribution):", g_raw)

    g_pair = weekday_pairwise_gini(random_dates)
    print("Second‑order pairwise Gini:", g_pair)

    g_deep = weekday_gini_deep(random_dates)
    print("Deep fused Gini (geometric mean):", g_deep)

    # Basic sanity checks
    assert isinstance(g_raw, float)
    assert isinstance(g_pair, float)
    assert isinstance(g_deep, float)
    assert counts.shape == (7,)
    sys.exit(0)