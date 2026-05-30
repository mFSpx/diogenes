# DARWIN HAMMER — match 313, survivor 4
# gen: 2
# parent_a: hybrid_doomsday_calendar_gini_coefficient_m49_s1.py (gen1)
# parent_b: hybrid_hdc_serpentina_self_righ_m50_s2.py (gen1)
# born: 2026-05-29T23:28:14Z

"""Hybrid Doomsday‑Gini & Hyperdimensional Morphology

This module fuses the mathematical core of two parent algorithms:

* **Parent A** – computes a Gini coefficient on an arbitrary value set and
  combines it with the Doomsday weekday of a given date.
* **Parent B** – builds bipolar hypervectors for morphological scalars,
  binds them with symbolic vectors and bundles them into a single
  morphology hypervector that can be compared by similarity.

**Mathematical bridge**

1. The Doomsday‑Gini result is a scalar `s ∈ [0,1]`.  
2. We encode `s` into a bipolar hypervector `S` by fixing the proportion
   of `+1` entries to `s` (the remaining entries are `‑1`).  
3. The weekday (0‑6) is turned into a symbolic hypervector `W` via
   `symbol_vector('weekday_i')` and bound to `S` → temporal hypervector `T`.
4. Morphological scalars are encoded exactly as in Parent B, producing
   a morphology hypervector `M`.
5. The unified hybrid hypervector is `H = bind(T, M)`.  
6. Similarity of `H` with a reference hypervector `R` (representing a
   “critical” state) yields a proxy value that is linearly mapped back
   onto the original Doomsday‑Gini scale, giving a final hybrid score.

The three public functions below demonstrate this integration.
"""

from __future__ import annotations

import hashlib
import math
import random
import sys
from pathlib import Path
from typing import Iterable, List, Dict

import numpy as np

# ----------------------------------------------------------------------
# Hyperdimensional primitives (from Parent B)
# ----------------------------------------------------------------------
Vector = List[int]


def random_vector(dim: int = 10000, seed: str | int | None = None) -> Vector:
    rng = random.Random(seed)
    return [1 if rng.getrandbits(1) else -1 for _ in range(dim)]


def symbol_vector(symbol: str, dim: int = 10000) -> Vector:
    seed = int.from_bytes(hashlib.sha256(symbol.encode("utf-8")).digest()[:8], "big")
    return random_vector(dim, seed)


def bind(a: Vector, b: Vector) -> Vector:
    if len(a) != len(b):
        raise ValueError("vectors must have equal length")
    return [x * y for x, y in zip(a, b)]


def bundle(vectors: Iterable[Vector]) -> Vector:
    """Element‑wise majority vote (bipolar bundling)."""
    vecs = list(vectors)
    if not vecs:
        raise ValueError("no vectors to bundle")
    dim = len(vecs[0])
    if any(len(v) != dim for v in vecs):
        raise ValueError("all vectors must have the same dimension")
    summed = [sum(v[i] for v in vecs) for i in range(dim)]
    return [1 if s >= 0 else -1 for s in summed]


def similarity(a: Vector, b: Vector) -> float:
    """Cosine‑like similarity for bipolar vectors (range [-1, 1])."""
    if len(a) != len(b):
        raise ValueError("vectors must have equal length")
    return float(np.dot(a, b) / len(a))


# ----------------------------------------------------------------------
# Parent A core – Gini coefficient and Doomsday weekday
# ----------------------------------------------------------------------
def gini_coefficient(values: Iterable[float]) -> float:
    xs = sorted(float(x) for x in values)
    if not xs or sum(xs) == 0.0:
        return 0.0
    if xs[0] < 0:
        raise ValueError("values must be non‑negative")
    n = len(xs)
    return sum((2 * i - n - 1) * x for i, x in enumerate(xs, 1)) / (n * sum(xs))


def doomsday_weekday(year: int, month: int, day: int) -> int:
    """Return Doomsday weekday as an integer 0‑6 (Monday=0)."""
    # Python's weekday(): Monday=0 … Sunday=6
    # Parent A used (weekday+1)%7 to shift; we keep 0‑6 for simplicity.
    return (dt.date(year, month, day).weekday() + 1) % 7


# ----------------------------------------------------------------------
# Hybrid encoding utilities
# ----------------------------------------------------------------------
def scalar_to_hv(value: float, reference: float = 0.5, dim: int = 10000) -> Vector:
    """
    Encode a scalar in [0,1] as a bipolar hypervector where the proportion
    of +1 entries equals the normalized distance from the reference.
    """
    if not 0.0 <= value <= 1.0:
        raise ValueError("scalar must be in [0,1]")
    # Proportion of +1 entries:
    p_pos = abs(value - reference)  # already in [0,0.5] for reference=0.5
    # Scale to [0,1]
    p_pos = min(1.0, max(0.0, p_pos * 2))
    rng = random.Random(hash((value, reference)))
    return [1 if rng.random() < p_pos else -1 for _ in range(dim)]


def encode_temporal_hv(year: int, month: int, day: int, values: Iterable[float],
                       dim: int = 10000) -> Vector:
    """Encode the Doomsday‑Gini pair into a temporal hypervector."""
    gini = gini_coefficient(values)  # ∈ [0,1]
    weekday = doomsday_weekday(year, month, day)  # 0‑6
    # Encode Gini as a scalar hypervector
    gini_hv = scalar_to_hv(gini, reference=0.5, dim=dim)
    # Symbolic vector for the specific weekday
    weekday_sym = symbol_vector(f"weekday_{weekday}", dim=dim)
    # Bind Gini encoding with weekday symbol → temporal HV
    return bind(gini_hv, weekday_sym)


def encode_morphology_hv(attributes: Dict[str, float],
                         references: Dict[str, float] | None = None,
                         dim: int = 10000) -> Vector:
    """
    Encode morphological scalars (e.g., length, width, mass) into a single
    hypervector. Each attribute is bound to its symbolic vector after
    being turned into a bipolar scalar HV.
    """
    if references is None:
        references = {k: 1.0 for k in attributes}  # default reference = 1.0
    bound_vectors: List[Vector] = []
    for name, val in attributes.items():
        ref = references.get(name, 1.0)
        # Normalise to [0,1] assuming positive values; clip for safety
        norm = max(0.0, min(1.0, val / (ref * 2)))  # simple linear scaling
        scalar_hv = scalar_to_hv(norm, reference=0.5, dim=dim)
        symbol_hv = symbol_vector(name, dim=dim)
        bound_vectors.append(bind(symbol_hv, scalar_hv))
    return bundle(bound_vectors)


def hybrid_score(year: int, month: int, day: int, values: Iterable[float],
                 attributes: Dict[str, float],
                 dim: int = 10000) -> float:
    """
    Compute a unified hybrid score.

    1. Build temporal HV T from Doomsday‑Gini.
    2. Build morphology HV M from physical attributes.
    3. Bind them → H.
    4. Similarity(H, R) where R is a critical reference HV.
    5. Linearly map similarity ([-1,1]) back to the original Doomsday‑Gini
       range, yielding the final score.
    """
    T = encode_temporal_hv(year, month, day, values, dim=dim)
    M = encode_morphology_hv(attributes, dim=dim)
    H = bind(T, M)

    # Reference hypervector representing a “critical” state
    R = symbol_vector("critical_state", dim=dim)

    sim = similarity(H, R)  # ∈ [-1, 1]

    # Map similarity to [0,1] (same range as Gini)
    mapped = (sim + 1.0) / 2.0

    # Optionally blend with the raw Gini for interpretability
    raw_gini = gini_coefficient(values)
    return 0.6 * mapped + 0.4 * raw_gini


# ----------------------------------------------------------------------
# Additional demonstration functions
# ----------------------------------------------------------------------
def temporal_inequality_score(year: int, month: int, day: int,
                              num_days: int, dim: int = 10000) -> float:
    """
    Simulate a weekday distribution over `num_days`, compute its Gini,
    encode temporally and return the similarity to the critical reference.
    """
    weekdays = [(dt.date(year, month, day) + dt.timedelta(days=i)).weekday()
                for i in range(num_days)]
    # Count occurrences of each weekday (0‑6)
    counts = [weekdays.count(w) for w in range(7)]
    gini = gini_coefficient(counts)
    T = encode_temporal_hv(year, month, day, counts, dim=dim)
    R = symbol_vector("critical_state", dim=dim)
    return similarity(T, R) * gini  # hybrid proxy


def morphology_priority(attributes: Dict[str, float],
                       dim: int = 10000) -> float:
    """
    Compute a priority index purely from morphology using the same
    hyperdimensional pipeline, then map similarity to a 0‑1 scale.
    """
    M = encode_morphology_hv(attributes, dim=dim)
    R = symbol_vector("critical_state", dim=dim)
    return (similarity(M, R) + 1.0) / 2.0


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import datetime as dt

    # Sample data for temporal part
    year, month, day = 2024, 3, 15
    sample_values = [5, 12, 7, 3, 9, 11, 4]

    # Sample morphological attributes (arbitrary units)
    morph_attrs = {
        "length": 0.8,
        "width": 0.6,
        "height": 0.4,
        "mass": 1.2,
    }
    # Reference values for normalisation (could be species‑specific)
    morph_refs = {
        "length": 1.0,
        "width": 1.0,
        "height": 1.0,
        "mass": 1.0,
    }

    # Compute hybrid score
    score = hybrid_score(year, month, day, sample_values, morph_attrs, dim=8000)
    print(f"Hybrid Doomsday‑Gini / Morphology score: {score:.4f}")

    # Additional diagnostics
    temp_score = temporal_inequality_score(year, month, day, 30, dim=8000)
    morph_score = morphology_priority(morph_attrs, dim=8000)
    print(f"Temporal similarity proxy: {temp_score:.4f}")
    print(f"Morphology similarity proxy: {morph_score:.4f}")