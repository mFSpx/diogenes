# DARWIN HAMMER — match 709, survivor 0
# gen: 3
# parent_a: hybrid_doomsday_calendar_gini_coefficient_m49_s4.py (gen1)
# parent_b: hybrid_hybrid_endpoint_circ_state_space_duality_m1_s2.py (gen2)
# born: 2026-05-29T23:30:31Z

"""Hybrid Doomsday‑SSM Gini Engine

This module fuses the two parent algorithms:

* **Parent A** – ``weekday_sakamoto`` (vectorised Sakamoto calendar) and
  ``gini_coefficient`` (inequality measure on a 1‑D array).
* **Parent B** – endpoint health/state‑space representation where each
  endpoint is a state dimension of a linear SSM.  The SSM is defined by
  per‑step matrices **Aₜ**, **Bₜ**, **Cₜ** and admits a semiseparable
  lower‑triangular matrix **M** such that ``Y = M·X``.

**Mathematical bridge**

The calendar provides a deterministic scalar time‑series ``xₜ`` (the
weekday index of each request).  This series is fed as the input
``X`` of the endpoint SSM.  The SSM evolves the health vector
``hₜ ∈ ℝᴺ`` (``N`` = number of endpoints) and produces a scalar score
``yₜ`` per request:


hₜ = A·hₜ₋₁ + B·xₜ               (A is diagonal, B is column)
yₜ = C·hₜ                         (C is row, the health_score)


After processing the whole request sequence we obtain ``Y = (y₁,…,y_T)``.
The Gini coefficient of ``Y`` quantifies the inequality of the generated
scores.  The three public functions below demonstrate the end‑to‑end
hybrid workflow.

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
# Parent A – Gini coefficient (robust vectorised version)
# ----------------------------------------------------------------------
def gini_coefficient(values: np.ndarray) -> float:
    """
    Return the Gini coefficient of a 1‑D array of non‑negative numbers.
    Edge cases (empty array, all zeros) yield ``0.0``.
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
# Helper – build static endpoint matrices (Parent B)
# ----------------------------------------------------------------------
def build_endpoint_matrices(
    failure_fractions: np.ndarray,
    morphology: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Given per‑endpoint failure fractions (0‒1) and a morphology vector,
    construct the static SSM matrices:

    * ``A`` – diagonal matrix diag(failure_fractions)
    * ``B`` – column vector (N×1) derived from morphology
    * ``C`` – row vector (1×N) = health_score = (1‑fr)*(1‑m_norm)
    * ``health_score`` – the same as ``C`` for external use
    """
    if failure_fractions.shape != morphology.shape:
        raise ValueError("failure_fractions and morphology must have the same shape")
    N = failure_fractions.size

    # Diagonal decay matrix
    A = np.diag(failure_fractions.astype(np.float64))

    # Normalise morphology to [0,1] (avoid division by zero)
    if np.allclose(morphology, 0):
        m_norm = np.zeros_like(morphology, dtype=np.float64)
    else:
        m_norm = (morphology - morphology.min()) / (morphology.max() - morphology.min())

    B = m_norm.reshape(N, 1).astype(np.float64)          # column
    C = ((1.0 - failure_fractions) * (1.0 - m_norm)).reshape(1, N).astype(np.float64)  # row

    health_score = C.ravel()
    return A, B, C, health_score


# ----------------------------------------------------------------------
# Core hybrid – SSM evaluation driven by weekday inputs
# ----------------------------------------------------------------------
def ssmt_scores_from_dates(
    dates: Iterable[dt.date],
    failure_fractions: np.ndarray,
    morphology: np.ndarray,
) -> np.ndarray:
    """
    Convert a sequence of ``datetime.date`` objects into weekday indices,
    feed them as the input series ``xₜ`` to the endpoint SSM and return the
    resulting scalar scores ``yₜ`` (shape ``(T,)``).

    The SSM uses the static matrices built from ``failure_fractions`` and
    ``morphology``.  The initial state ``h₀`` is zero.
    """
    # ------------------------------------------------------------------
    # 1️⃣  Calendar → weekday vector (Parent A)
    # ------------------------------------------------------------------
    dates = list(dates)
    if not dates:
        return np.array([], dtype=np.float64)

    years = np.array([d.year for d in dates], dtype=np.int64)
    months = np.array([d.month for d in dates], dtype=np.int64)
    days = np.array([d.day for d in dates], dtype=np.int64)

    weekdays = weekday_sakamoto(years, months, days).astype(np.float64)  # 0‑6

    # Normalise weekdays to a small load factor (e.g., 0.1 … 0.7)
    x_series = (weekdays + 1) / 10.0  # now in [0.1, 0.7]

    # ------------------------------------------------------------------
    # 2️⃣  Build static endpoint matrices (Parent B)
    # ------------------------------------------------------------------
    A, B, C, _ = build_endpoint_matrices(failure_fractions, morphology)

    # ------------------------------------------------------------------
    # 3️⃣  Run the SSM (recursive formulation, equivalent to semiseparable M·X)
    # ------------------------------------------------------------------
    N = failure_fractions.size
    T = x_series.size
    h = np.zeros((N, ), dtype=np.float64)          # state vector
    y = np.empty((T, ), dtype=np.float64)          # output series

    for t in range(T):
        # hₜ = A·hₜ₋₁ + B·xₜ   (B is N×1, xₜ scalar)
        h = A @ h + (B.ravel() * x_series[t])
        # yₜ = C·hₜ
        y[t] = C @ h

    return y


# ----------------------------------------------------------------------
# Hybrid metric – combine SSM scores with Gini (full fusion)
# ----------------------------------------------------------------------
def hybrid_doomsday_gini_metric(
    dates: Iterable[dt.date],
    failure_fractions: np.ndarray,
    morphology: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """
    Execute the full hybrid pipeline:

    1. Compute weekday‑driven SSM scores ``yₜ``.
    2. Evaluate the Gini coefficient of the score distribution.

    Returns ``(scores, gini)`` where ``scores`` is the ``y`` vector and
    ``gini`` is a scalar in ``[0,1]``.
    """
    scores = ssmt_scores_from_dates(dates, failure_fractions, morphology)
    gini = gini_coefficient(scores)
    return scores, gini


# ----------------------------------------------------------------------
# Demonstration / smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # 1️⃣  Generate a month of dates (April 2026)
    start = dt.date(2026, 4, 1)
    dates = [start + dt.timedelta(days=i) for i in range(30)]

    # 2️⃣  Random but reproducible endpoint characteristics (N=5)
    random.seed(0)
    N = 5
    failure_fractions = np.array([random.random() for _ in range(N)], dtype=np.float64)
    morphology = np.array([random.random() for _ in range(N)], dtype=np.float64)

    # 3️⃣  Run the hybrid pipeline
    scores, gini = hybrid_doomsday_gini_metric(dates, failure_fractions, morphology)

    # 4️⃣  Simple sanity output
    print("Weekday‑driven SSM scores (first 10):", scores[:10])
    print("Gini coefficient of scores:", gini)
    # Ensure no NaNs and gini in [0,1]
    assert not np.isnan(scores).any()
    assert 0.0 <= gini <= 1.0
    print("Smoke test passed.")