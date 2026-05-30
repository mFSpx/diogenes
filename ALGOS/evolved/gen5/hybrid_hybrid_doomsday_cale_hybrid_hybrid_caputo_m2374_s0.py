# DARWIN HAMMER — match 2374, survivor 0
# gen: 5
# parent_a: hybrid_doomsday_calendar_gini_coefficient_m49_s2.py (gen1)
# parent_b: hybrid_hybrid_caputo_fracti_hybrid_hybrid_regret_m880_s0.py (gen4)
# born: 2026-05-29T23:41:57Z

"""
Hybrid module combining the Doomsday weekday calculation with the Gini inequality coefficient and the fractional-memory regret-weighted strategy.

The mathematical bridge between these structures lies in the application of the Caputo fractional derivative to the regret-weighted strategy's decision-making process, 
where the input to the regret-weighted strategy is the weekday frequency distribution of a collection of dates, 
and the output is a score that measures the inequality of the weekday distribution, 
taking into account the past contributions with a slowly decaying algebraic factor.

This fusion integrates the governing equations of both parents, 
using the Doomsday weekday calculation to map each calendar date to a numeric weekday, 
and then using the Gini inequality coefficient to quantify the inequality of the weekday frequency distribution, 
and finally applying the Caputo fractional derivative to the regret-weighted strategy to obtain a fractional-memory variant.

"""

import numpy as np
import math
import random
import sys
from pathlib import Path
from typing import Iterable, Sequence, Tuple, List, Union
from collections import deque
from dataclasses import dataclass

_LANCZOS_G = 7
_LANCZOS_C = np.array([
    0.99999999999980993,
    676.5203681218851,
    -1259.1392167224028,
    771.32342877765313,
    -176.61502916214059,
    12.507343278686905,
    -0.13857109526572012,
    9.9843695780195716e-6,
    1.5056327351493116e-7,
])

def gamma_lanczos(z: float) -> float:
    if z < 0.5:
        return math.pi / (math.sin(math.pi * z) * gamma_lanczos(1 - z))
    zz = z - 1
    x = _LANCZOS_C / (zz + np.arange(_LANCZOS_G) + 1)
    return math.sqrt(2 * math.pi) * (zz + _LANCZOS_G + 0.5) ** (zz + 0.5) * np.exp(-(zz + _LANCZOS_G + 0.5)) * np.prod(x)

def caputo_weight(alpha: float, T: int, k: int) -> float:
    return ((T - 1 - k) ** (alpha - 1)) / gamma_lanczos(alpha)

def doomsday_numpy(
    years: np.ndarray,
    months: np.ndarray,
    days: np.ndarray,
) -> np.ndarray:
    dates = np.stack([years, months, days], axis=-1).astype('datetime64[D]')
    np_weekday = dates.astype('datetime64[D]').astype('datetime64[ns]').astype('int64')
    flat = dates.ravel()
    py_weekday = np.fromiter(
        (np.datetime64(int(d)).item().weekday() for d in flat),
        dtype=np.int8,
        count=flat.size,
    )
    py_weekday = py_weekday.reshape(dates.shape[:-1])
    return (py_weekday + 1) % 7

def compute_weekday_frequencies(dates: np.ndarray) -> np.ndarray:
    weekday_counts = np.zeros(7)
    weekdays = doomsday_numpy(dates[:, 0], dates[:, 1], dates[:, 2])
    for i in range(7):
        weekday_counts[i] = np.sum(weekdays == i)
    return weekday_counts

def gini_coefficient(weekday_frequencies: np.ndarray) -> float:
    sorted_frequencies = np.sort(weekday_frequencies)
    n = len(sorted_frequencies)
    gini = 0
    for i in range(n):
        gini += (2 * i - n + 1) * sorted_frequencies[i]
    gini /= n * np.sum(sorted_frequencies)
    return gini

def compute_hybrid_score(alpha: float, dates: np.ndarray) -> float:
    weekday_frequencies = compute_weekday_frequencies(dates)
    gini = gini_coefficient(weekday_frequencies)
    T = len(dates)
    regret_weighted_score = 0
    for k in range(T):
        weight = caputo_weight(alpha, T, k)
        regret_weighted_score += weight * gini
    return regret_weighted_score

if __name__ == "__main__":
    dates = np.random.randint(0, 100, size=(100, 3))
    alpha = 0.5
    score = compute_hybrid_score(alpha, dates)
    print(score)