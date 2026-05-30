# DARWIN HAMMER — match 49, survivor 3
# gen: 1
# parent_a: doomsday_calendar.py (gen0)
# parent_b: gini_coefficient.py (gen0)
# born: 2026-05-29T23:23:58Z

from __future__ import annotations

import datetime as dt
import sys
import random
from pathlib import Path
from typing import Iterable, Sequence, Tuple, List, Union
import numpy as np

def doomsday_numpy(
    years: np.ndarray,
    months: np.ndarray,
    days: np.ndarray,
) -> np.ndarray:
    dates = np.stack([years, months, days], axis=-1).astype('datetime64[D]')
    np_weekday = dates.astype('datetime64[D]').astype('datetime64[ns]').astype('int64')
    flat = dates.ravel()
    py_weekday = np.fromiter(
        (dt.datetime.utcfromtimestamp(int(d.astype('datetime64[s]'))).weekday() for d in flat),
        dtype=np.int8,
        count=flat.size,
    )
    py_weekday = py_weekday.reshape(dates.shape[:-1])
    return (py_weekday + 1) % 7


def gini_coefficient_numpy(values: np.ndarray) -> float:
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


def weekday_counts(dates: Iterable[Union[dt.date, Tuple[int, int, int]]]) -> np.ndarray:
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
    counts = weekday_counts(dates)
    return gini_coefficient_numpy(counts)


def weekday_gini_matrix(dates: Iterable[Union[dt.date, Tuple[int, int, int]]]) -> np.ndarray:
    counts = weekday_counts(dates).astype(float)
    weight = np.outer(counts, counts)
    idx = np.arange(7)
    diff = np.abs(idx[:, None] - idx[None, :])
    weighted_diff = weight * diff
    flat = weighted_diff.ravel()
    nonzero = flat[flat > 0]
    if nonzero.size == 0:
        return np.zeros((7, 7))
    gini_val = gini_coefficient_numpy(nonzero)
    return weighted_diff * gini_val


def weekday_gini_entropy(dates: Iterable[Union[dt.date, Tuple[int, int, int]]]) -> float:
    counts = weekday_counts(dates).astype(float)
    probabilities = counts / counts.sum()
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy


def weekday_gini_entropy_matrix(dates: Iterable[Union[dt.date, Tuple[int, int, int]]]) -> np.ndarray:
    counts = weekday_counts(dates).astype(float)
    probabilities = counts / counts.sum()
    weight = np.outer(probabilities, probabilities)
    idx = np.arange(7)
    diff = np.abs(idx[:, None] - idx[None, :])
    weighted_diff = weight * diff
    flat = weighted_diff.ravel()
    nonzero = flat[flat > 0]
    if nonzero.size == 0:
        return np.zeros((7, 7))
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return weighted_diff * entropy


if __name__ == "__main__":
    start = dt.date(2000, 1, 1).toordinal()
    end = dt.date(2025, 12, 31).toordinal()
    random_dates: List[dt.date] = [
        dt.date.fromordinal(random.randint(start, end)) for _ in range(100)
    ]

    counts = weekday_counts(random_dates)
    print("Weekday counts (Sun→Sat):", counts.tolist())

    gini_val = weekday_gini(random_dates)
    print("Weekday Gini coefficient:", gini_val)

    matrix = weekday_gini_matrix(random_dates)
    print("Weighted difference matrix (sample):")
    print(matrix)

    entropy = weekday_gini_entropy(random_dates)
    print("Weekday Gini entropy:", entropy)

    entropy_matrix = weekday_gini_entropy_matrix(random_dates)
    print("Weighted difference matrix with entropy (sample):")
    print(entropy_matrix)

    assert isinstance(gini_val, float)
    assert matrix.shape == (7, 7)
    sys.exit(0)