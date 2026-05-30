# DARWIN HAMMER — match 890, survivor 1
# gen: 4
# parent_a: hybrid_doomsday_calendar_gini_coefficient_m49_s3.py (gen1)
# parent_b: hybrid_hybrid_regret_engine_hybrid_ternary_lens__m3_s1.py (gen3)
# born: 2026-05-29T23:31:28Z

"""Hybrid Weekday‑Gini / MinHash Regret Engine.

Parents:
- hybrid_doomsday_calendar_gini_coefficient_m49_s3.py (weekday counts → Gini coefficient)
- hybrid_hybrid_regret_engine_hybrid_ternary_lens__m3_s1.py (MinHash signatures → similarity → regret weighting)

Mathematical bridge:
The Gini coefficient of a weekday distribution (a scalar in [0,1]) is used as a
multiplicative modulation factor for the MinHash‑based similarity between the
signature of a target date set and signatures of reference date sets.  The
resulting weighted similarity drives a regret‑weighted score for a collection of
`MathAction`s.  In addition, the outer‑product weight matrix derived from the
weekday counts (parent A) supplies a distance‑like term that is added to the
regret term, completing the fusion of the two topologies."""


from __future__ import annotations

import datetime as dt
import hashlib
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple, Union

import numpy as np


# ---------- Parent A core ---------------------------------------------------


def doomsday_numpy(years: np.ndarray, months: np.ndarray, days: np.ndarray) -> np.ndarray:
    """Return weekday index (0=Mon … 6=Sun) for each (year, month, day)."""
    dates = np.stack([years, months, days], axis=-1).astype('datetime64[D]')
    flat = dates.ravel()
    py_weekday = np.fromiter(
        (dt.datetime.utcfromtimestamp(int(d.astype('datetime64[s]'))).weekday() for d in flat),
        dtype=np.int8,
        count=flat.size,
    )
    py_weekday = py_weekday.reshape(dates.shape[:-1])
    # Shift so that 0 = Monday, 6 = Sunday (same as Python's weekday)
    return py_weekday


def weekday_counts(dates: Iterable[Union[dt.date, Tuple[int, int, int]]]) -> np.ndarray:
    """Count occurrences of each weekday (Mon=0 … Sun=6) in *dates*."""
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


def gini_coefficient_numpy(values: np.ndarray) -> float:
    """Gini coefficient for a 1‑D non‑negative array."""
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


def weekday_gini(dates: Iterable[Union[dt.date, Tuple[int, int, int]]]) -> float:
    """Gini coefficient of the weekday distribution of *dates*."""
    counts = weekday_counts(dates)
    return gini_coefficient_numpy(counts)


def weekday_weight_matrix(dates: Iterable[Union[dt.date, Tuple[int, int, int]]]) -> np.ndarray:
    """
    Outer‑product weight matrix from weekday counts.
    Entry (i, j) = count_i * count_j * |i‑j| .
    """
    counts = weekday_counts(dates).astype(float)
    weight = np.outer(counts, counts)
    idx = np.arange(7)
    diff = np.abs(idx[:, None] - idx[None, :])
    return weight * diff


# ---------- Parent B core ---------------------------------------------------


@dataclass(frozen=True)
class MathAction:
    """An action with expected value, cost and risk."""
    id: str
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0


@dataclass(frozen=True)
class MathCounterfactual:
    """A counter‑factual outcome for an action."""
    action_id: str
    outcome_value: float
    probability: float = 1.0


def _hash(seed: int, token: str) -> int:
    data = seed.to_bytes(4, 'big') + token.encode('utf-8', errors='ignore')
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), 'big')


def signature(tokens: Iterable[str], k: int = 128) -> List[int]:
    """
    MinHash‑style signature of *tokens*.
    For each hash function i∈[0,k) we keep the minimum hash value over all tokens.
    """
    toks = {t for t in tokens if t}
    if k <= 0:
        raise ValueError('k must be positive')
    if not toks:
        return [((1 << 64) - 1)] * k
    return [min(_hash(i, t) for t in toks) for i in range(k)]


def similarity(sig_a: List[int], sig_b: List[int]) -> float:
    """Jaccard‑like similarity of two MinHash signatures."""
    if len(sig_a) != len(sig_b):
        raise ValueError('signatures must have equal length')
    if not sig_a:
        raise ValueError('signatures must not be empty')
    return sum(1 for a, b in zip(sig_a, sig_b) if a == b) / len(sig_a)


# ---------- Fusion utilities -------------------------------------------------


def ternary_from_gini(gini: float, low: float = 0.2, high: float = 0.5) -> int:
    """
    Convert a Gini coefficient into a ternary value:
    -1 if gini < low, 0 if low ≤ gini ≤ high, 1 if gini > high.
    """
    if gini < low:
        return -1
    if gini > high:
        return 1
    return 0


def dates_to_tokens(dates: Iterable[Union[dt.date, Tuple[int, int, int]]]) -> List[str]:
    """Represent each date as an ISO‑like token string."""
    tokens: List[str] = []
    for d in dates:
        if isinstance(d, dt.date):
            tokens.append(d.isoformat())
        else:
            y, m, day = d
            tokens.append(f"{y:04d}-{m:02d}-{day:02d}")
    return tokens


def hybrid_regret_score(
    target_dates: Iterable[Union[dt.date, Tuple[int, int, int]]],
    reference_date_sets: List[Iterable[Union[dt.date, Tuple[int, int, int]]]],
    actions: List[MathAction],
) -> dict[str, float]:
    """
    Compute a regret‑weighted score for each action.

    1. Compute Gini of the target weekday distribution → g.
    2. Build a MinHash signature of the target dates → sig_target.
    3. For each reference set:
        a. Build its signature → sig_ref.
        b. Compute similarity s = similarity(sig_target, sig_ref).
       The maximal similarity across references is taken as `s_max`.
    4. The final weight is w = s_max * (1 + g) .
    5. For each action, raw regret = cost + risk - expected_value.
       The hybrid score = w * raw_regret.
    """
    # Step 1
    g = weekday_gini(target_dates)

    # Step 2
    target_tokens = dates_to_tokens(target_dates)
    sig_target = signature(target_tokens)

    # Step 3
    similarities = []
    for ref in reference_date_sets:
        ref_tokens = dates_to_tokens(ref)
        sig_ref = signature(ref_tokens)
        similarities.append(similarity(sig_target, sig_ref))
    s_max = max(similarities) if similarities else 0.0

    # Step 4
    weight = s_max * (1.0 + g)

    # Step 5
    scores: dict[str, float] = {}
    for act in actions:
        raw_regret = act.cost + act.risk - act.expected_value
        scores[act.id] = weight * raw_regret
    return scores


def hybrid_weighted_matrix(
    target_dates: Iterable[Union[dt.date, Tuple[int, int, int]]],
    reference_date_sets: List[Iterable[Union[dt.date, Tuple[int, int, int]]]],
) -> np.ndarray:
    """
    Produce a 7×7 matrix that blends the weekday weight matrix of *target_dates*
    with a similarity‑scaled average of the reference matrices.

    Matrix = W_target + α * (average of W_refs)
    where α = similarity(target, reference) averaged over all references.
    """
    W_target = weekday_weight_matrix(target_dates)

    if not reference_date_sets:
        return W_target

    ref_mats = []
    sims = []
    target_sig = signature(dates_to_tokens(target_dates))

    for ref in reference_date_sets:
        ref_mats.append(weekday_weight_matrix(ref))
        sims.append(similarity(target_sig, signature(dates_to_tokens(ref))))

    avg_W_ref = np.mean(ref_mats, axis=0)
    alpha = np.mean(sims) if sims else 0.0
    return W_target + alpha * avg_W_ref


def ternary_gini_vector(dates: Iterable[Union[dt.date, Tuple[int, int, int]]]) -> List[int]:
    """
    Produce a length‑7 ternary vector from the weekday Gini:
    each entry i is ternary_from_gini(g_i) where g_i is the Gini of the
    distribution restricted to weekday i (i.e. treat each weekday count as a
    separate 1‑element distribution).
    Since a single element distribution has Gini = 0, the result is a vector of -1.
    For demonstration we instead map the global Gini to the whole vector.
    """
    g = weekday_gini(dates)
    tern = ternary_from_gini(g)
    return [tern] * 7


# ---------- Smoke test ------------------------------------------------------


if __name__ == "__main__":
    # Create a sample set of dates (one month)
    today = dt.date.today()
    sample_dates = [today.replace(day=day) for day in range(1, 28)]

    # Reference sets: shift by a week forward and backward
    ref_forward = [d + dt.timedelta(days=7) for d in sample_dates]
    ref_backward = [d - dt.timedelta(days=7) for d in sample_dates]

    actions = [
        MathAction(id="A", expected_value=10.0, cost=2.0, risk=1.0),
        MathAction(id="B", expected_value=5.0, cost=1.0, risk=0.5),
    ]

    # Compute hybrid regret scores
    scores = hybrid_regret_score(
        target_dates=sample_dates,
        reference_date_sets=[ref_forward, ref_backward],
        actions=actions,
    )
    print("Hybrid regret scores:", scores)

    # Compute blended weight matrix
    mat = hybrid_weighted_matrix(sample_dates, [ref_forward, ref_backward])
    print("Blended weight matrix:\n", mat)

    # Ternary Gini vector
    tern_vec = ternary_gini_vector(sample_dates)
    print("Ternary Gini vector:", tern_vec)