# DARWIN HAMMER — match 1230, survivor 1
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_sparse_wta_hy_m626_s2.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m612_s0.py (gen5)
# born: 2026-05-29T23:35:58Z

import numpy as np
import hashlib
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple
from collections import Counter
import datetime as dt

def doomsday_numpy(years: np.ndarray, months: np.ndarray, days: np.ndarray) -> np.ndarray:
    """Vectorised weekday calculation (Sun=0 … Sat=6)."""
    dates = np.stack([years, months, days], axis=-1).astype("datetime64[D]")
    flat = dates.ravel()
    py_weekday = np.fromiter(
        (
            dt.datetime.utcfromtimestamp(int(d.astype("datetime64[s]"))).weekday()
            for d in flat
        ),
        dtype=np.int8,
        count=flat.size,
    )
    py_weekday = py_weekday.reshape(dates.shape[:-1])
    # shift: Monday=0 → Sunday=0
    return (py_weekday + 1) % 7


def weekday_counts(dates: List[dt.date]) -> np.ndarray:
    """Return a length‑7 array with counts of each weekday (Sun=0 … Sat=6)."""
    counts = np.zeros(7, dtype=int)
    for d in dates:
        if isinstance(d, dt.datetime):
            d = d.date()
        counts[d.weekday() % 7] += 1
    # shift to match doomsday_numpy convention (Sun=0)
    return np.roll(counts, 1)


def gini_coefficient(values: np.ndarray) -> float:
    """Gini coefficient for a 1‑D non‑negative array."""
    if values.size == 0:
        return 0.0
    sorted_vals = np.sort(values.astype(float))
    n = values.size
    cumulative = np.cumsum(sorted_vals)
    return ((np.arange(1, n+1) * sorted_vals).sum() * 2.0 - (n + 1) * cumulative.sum()) / (n * cumulative.sum())


def compute_ssim(
    x: List[float],
    y: List[float],
    dynamic_range: float = 1.0,
    k1: float = 0.01,
    k2: float = 0.03,
) -> float:
    """Structural Similarity Index between two equal‑length vectors."""
    if len(x) != len(y):
        raise ValueError("samples must have equal length")
    if not x:
        raise ValueError("samples must not be empty")

    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)

    mx = x_arr.mean()
    my = y_arr.mean()
    vx = ((x_arr - mx) ** 2).mean()
    vy = ((y_arr - my) ** 2).mean()
    cov = ((x_arr - mx) * (y_arr - my)).mean()

    c1 = (k1 * dynamic_range) ** 2
    c2 = (k2 * dynamic_range) ** 2

    ssim = ((2 * mx * my + c1) * (2 * cov + c2)) / ((mx ** 2 + my ** 2 + c1) * (vx + vy + c2))

    return ssim


def expand(v: List[float], seed: int = 42) -> List[float]:
    """Hash-based expansion of a vector."""
    hash_object = hashlib.md5(str(seed).encode())
    for val in v:
        hash_object.update(str(val).encode())
    hex_dig = hash_object.hexdigest()
    e = [int(d, 16) / 15 for d in hex_dig]
    return e


def hybrid_operation(
    input_vector: List[float],
    prototype_vector: List[float],
    dates: List[dt.date],
) -> Tuple[float, float]:
    """Hybrid operation integrating both parent algorithms."""
    e = expand(input_vector)
    e_p = expand(prototype_vector)

    ssim_score = compute_ssim(e, e_p)

    weekday_counts_array = weekday_counts(dates)
    gini_coef = gini_coefficient(weekday_counts_array)

    health_score = ssim_score * (1 - gini_coef)

    return health_score, gini_coef


def add_laplace_noise(value: float, sensitivity: float, epsilon: float) -> float:
    """Add Laplace noise to a value."""
    laplace_noise = np.random.laplace(loc=0, scale=sensitivity / epsilon)
    return value + laplace_noise


if __name__ == "__main__":
    input_vector = [0.2, 0.5, 0.3, 0.7, 0.1]
    prototype_vector = [0.2, 0.5, 0.3, 0.7, 0.1]
    dates = [dt.date(2022, 1, 1) + dt.timedelta(days=i) for i in range(7)]

    health_score, gini_coef = hybrid_operation(input_vector, prototype_vector, dates)
    print("Health Score:", health_score)
    print("Gini Coefficient:", gini_coef)

    epsilon = 1.0
    sensitivity = 1.0
    noisy_health_score = add_laplace_noise(health_score, sensitivity, epsilon)
    print("Noisy Health Score:", noisy_health_score)