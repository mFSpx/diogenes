# DARWIN HAMMER — match 2277, survivor 1
# gen: 5
# parent_a: hybrid_gini_coefficient_hybrid_hybrid_rbf_su_m344_s0.py (gen4)
# parent_b: hybrid_hybrid_workshare_all_hybrid_liquid_time_c_m39_s5.py (gen2)
# born: 2026-05-29T23:41:43Z

"""Hybrid Gini‑Weight‑Similarity Allocation
This module fuses two parent algorithms:

* **Parent A** – provides a Gini‑coefficient based inequality measure and a
  similarity matrix derived from perceptual hashing of feature vectors.
* **Parent B** – supplies a weekday‑dependent weight vector for allocating a
  total resource across a set of groups.

The mathematical bridge is the use of the Gini coefficient to modulate the
weekday‑derived weight vector with the similarity information.  The workflow
is:

1. Compute a similarity matrix **S** for the groups using perceptual hashes.
2. Derive a per‑group average similarity from **S**.
3. Measure the inequality of these averages with the Gini coefficient **G**.
4. Form a bias factor **b = 1 + G** and multiply the weekday weight vector
   **w** by the average similarity and the bias.
5. Normalise the resulting vector to the requested total amount.

The result is a single allocation that respects temporal (weekday) patterns,
structural similarity between groups, and the inequality of those similarities.
"""

import math
import random
import sys
import pathlib
import datetime as dt
from typing import Hashable, Sequence, List, Dict, Set, Tuple, Iterable, Any

import numpy as np

# ----------------------------------------------------------------------
# Types
# ----------------------------------------------------------------------
Node = Hashable
Graph = Dict[Node, Set[Node]]
FeatureVec = Sequence[float]

# ----------------------------------------------------------------------
# Parent A – Gini and similarity utilities
# ----------------------------------------------------------------------
def gini_coefficient(values: Iterable[float]) -> float:
    """Return the Gini coefficient of a non‑negative value collection."""
    xs = sorted(float(x) for x in values)
    if not xs or sum(xs) == 0.0:
        return 0.0
    if xs[0] < 0:
        raise ValueError("values must be non‑negative")
    n = len(xs)
    return sum((2 * i - n - 1) * x for i, x in enumerate(xs, 1)) / (n * sum(xs))


def compute_phash(values: List[float]) -> int:
    """Simple perceptual hash: 1 bit per value (up to 64 bits)."""
    if not values:
        return 0
    avg = sum(values) / len(values)
    bits = 0
    for v in values[:64]:
        bits = (bits << 1) | int(v >= avg)
    return bits


def hamming_distance(a: int, b: int) -> int:
    """Hamming distance between two 64‑bit integers."""
    return (a ^ b).bit_count()


def similarity_matrix(features: Dict[Node, FeatureVec]) -> Tuple[np.ndarray, List[Node]]:
    """
    Build an N×N similarity matrix where entry (i,j) = 1 – Hamming/64
    between perceptual hashes of the feature vectors of nodes i and j.
    Returns the matrix and the ordered list of nodes.
    """
    nodes = list(features.keys())
    n = len(nodes)
    S = np.empty((n, n), dtype=np.float64)

    # Pre‑compute hashes
    hashes = [compute_phash(list(features[node])) for node in nodes]

    for i in range(n):
        for j in range(i, n):
            if i == j:
                S[i, j] = 1.0
            else:
                d = hamming_distance(hashes[i], hashes[j])
                sim = 1.0 - d / 64.0
                S[i, j] = sim
                S[j, i] = sim
    return S, nodes


# ----------------------------------------------------------------------
# Parent B – Weekday weight utilities
# ----------------------------------------------------------------------
def _pct(value: float) -> float:
    """Round a float to six decimal places."""
    return round(float(value), 6)


def doomsday(year: int, month: int, day: int) -> int:
    """Return weekday index where 0 = Sunday … 6 = Saturday."""
    return (dt.date(year, month, day).weekday() + 1) % 7


def weekday_weight_vector(groups: Sequence[str], dow: int) -> np.ndarray:
    """
    Normalised weight vector for *groups* based on weekday ``dow``.
    Sinusoidal rotation yields a row‑stochastic vector.
    """
    n = len(groups)
    if n == 0:
        raise ValueError("groups must contain at least one element")
    base_angles = np.arange(n) * (2.0 * math.pi) / n
    phase = (2.0 * math.pi) * (dow % 7) / 7.0
    amplitude = 0.2
    raw = 1.0 + amplitude * np.sin(base_angles + phase)
    weight_vec = raw / raw.sum()
    return weight_vec.astype(np.float64)


# ----------------------------------------------------------------------
# Hybrid core – combine similarity, Gini, and weekday weights
# ----------------------------------------------------------------------
def hybrid_allocation(
    *,
    total_units: float,
    date: dt.date,
    features: Dict[str, FeatureVec],
    deterministic_target_pct: float = 90.0,
) -> Dict[str, float]:
    """
    Allocate ``total_units`` across the keys of ``features``.
    Steps:
      1. Split deterministic vs. residual (LLM) part.
      2. Compute weekday‑based weight vector for the groups.
      3. Compute average similarity per group from the similarity matrix.
      4. Compute Gini coefficient of the similarity distribution.
      5. Bias the weekday weights by (1 + Gini) and by the similarity averages.
      6. Normalise to the residual amount and add back deterministic part.

    Returns a mapping from group name to allocated amount.
    """
    if total_units <= 0.0:
        raise ValueError("total_units must be positive")
    if not (0.0 <= deterministic_target_pct <= 100.0):
        raise ValueError("deterministic_target_pct must be between 0 and 100")
    if not features:
        raise ValueError("features dictionary must not be empty")

    # 1. Deterministic split
    deterministic_units = total_units * deterministic_target_pct / 100.0
    residual_units = total_units - deterministic_units

    # 2. Weekday weight vector
    dow = doomsday(date.year, date.month, date.day)
    groups = list(features.keys())
    w_vec = weekday_weight_vector(groups, dow)          # shape (G,)

    # 3. Similarity matrix & average similarity per group (excluding self)
    S, ordered_nodes = similarity_matrix(features)
    # Exclude diagonal (self‑similarity = 1) when averaging
    mask = ~np.eye(S.shape[0], dtype=bool)
    avg_sim = np.where(mask, S, np.nan).mean(axis=1)     # shape (G,)

    # 4. Gini coefficient of the average similarity distribution
    gini = gini_coefficient(avg_sim)

    # 5. Bias computation
    bias_factor = 1.0 + gini
    biased = w_vec * avg_sim * bias_factor

    # Guard against all‑zero vector (possible if avg_sim contains NaNs)
    if np.allclose(biased, 0):
        biased = w_vec.copy()

    # 6. Normalise to residual_units and add deterministic part proportionally
    residual_alloc = biased / biased.sum() * residual_units
    final_alloc = deterministic_units / len(groups) + residual_alloc

    # Build result dictionary preserving original group order
    allocation = {group: _pct(amount) for group, amount in zip(ordered_nodes, final_alloc)}
    return allocation


def allocate_hybrid(
    *,
    total_units: float,
    date: dt.date,
    deterministic_target_pct: float = 90.0,
    groups: Tuple[str, ...] = ("codex", "groq", "cohere", "local_models"),
) -> Dict[str, Any]:
    """
    Legacy allocator from Parent B, retained for API compatibility.
    It distributes ``total_units`` between a deterministic chunk and a
    residual chunk that is split across ``groups`` using the weekday weight
    vector.
    """
    if total_units <= 0:
        raise ValueError("total_units must be positive")
    if not (0.0 <= deterministic_target_pct <= 100.0):
        raise ValueError("deterministic_target_pct must be between 0 and 100")
    if not groups:
        raise ValueError("groups required")

    deterministic_units = total_units * deterministic_target_pct / 100.0
    residual_units = total_units - deterministic_units
    dow = doomsday(date.year, date.month, date.day)
    w_vec = weekday_weight_vector(groups, dow)
    residual_alloc = w_vec * residual_units

    allocation = {
        "date": date.isoformat(),
        "deterministic_units": _pct(deterministic_units),
        "residual_units": _pct(residual_units),
        "group_allocation": {g: _pct(a) for g, a in zip(groups, residual_alloc)},
    }
    return allocation


def gaussian(r: float, epsilon: float = 1.0) -> float:
    """Radial basis function used in some similarity contexts."""
    return math.exp(-((epsilon * r) ** 2))


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Dummy feature vectors for the four default groups
    dummy_features = {
        "codex": [random.random() for _ in range(10)],
        "groq": [random.random() for _ in range(10)],
        "cohere": [random.random() for _ in range(10)],
        "local_models": [random.random() for _ in range(10)],
    }

    today = dt.date.today()
    total = 1000.0

    # Run the hybrid allocation that fuses both parents
    alloc = hybrid_allocation(
        total_units=total,
        date=today,
        features=dummy_features,
        deterministic_target_pct=85.0,
    )
    print("Hybrid allocation:", alloc)

    # Run the legacy allocator for comparison
    legacy = allocate_hybrid(
        total_units=total,
        date=today,
        deterministic_target_pct=85.0,
    )
    print("Legacy allocation:", legacy)