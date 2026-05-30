# DARWIN HAMMER — match 5123, survivor 4
# gen: 7
# parent_a: hybrid_sparse_wta_hybrid_hybrid_hybrid_m1937_s5.py (gen6)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2509_s1.py (gen6)
# born: 2026-05-29T23:59:59Z

"""Hybrid Sparse‑WTA & Pheromone‑Bayesian Multivector Fusion

Parents
-------
* Parent A – ``sparse_wta.py`` (hash‑based sparse expansion + top‑k winner‑take‑all)
* Parent B – ``..._pheromone_decay`` + Bayesian update with entropy‑driven decision

Mathematical Bridge
-------------------
The sparse expansion ``expand(values, m)`` maps a low‑dimensional real vector
``values`` to a high‑dimensional *component list* ``c ∈ ℝ^m``.  Treat this list
as the coefficient vector of a geometric‑algebra multivector:

    M = Σ_i c_i e_i

where ``e_i`` are orthogonal basis blades.  A pheromone entry supplies a
*prior* ``π_i`` for each blade (decayed signal value).  The expanded coefficients
serve as a *likelihood* ``ℓ_i``.  A Bayesian update yields a posterior

    p_i = π_i · ℓ_i / (π_i·ℓ_i + (1-π_i)·ε)

with a small false‑positive rate ``ε``.  The posterior multivector
``P = Σ_i p_i e_i`` is then reduced to a scalar field via the simple geometric
product ``⟨P·P⟩ = Σ_i p_i²``.  Selecting the top‑k blades of ``p`` (or the blade
that minimises the expected post‑selection entropy) provides a sparse tag that
can be compared with Hamming distance, while the scalar field can be used for
regularised‑log‑canonical‑type estimates.

The three public functions below demonstrate this fusion:
1. ``hybrid_pheromone_multivector`` – expand → pheromone scaling → posterior →
   top‑k binary mask.
2. ``allocate_adaptive_workshare_with_pheromone`` – Count‑Min sketch of past
   allocations, pheromone‑biased weighting, and adaptive work‑share allocation.
3. ``hybrid_rlct_estimate_with_multivector`` – posterior multivector → geometric
   product → regularised scalar estimate.
"""

from __future__ import annotations

import hashlib
import math
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Parent A – Sparse Winner‑Take‑All utilities
# ----------------------------------------------------------------------


def _hash_index(key: str, salt: str, m: int) -> int:
    """Deterministic hash → index in [0, m)."""
    h = hashlib.blake2b((key + salt).encode(), digest_size=8).digest()
    return int.from_bytes(h, "little") % m


def expand(values: List[float], m: int, salt: str = "") -> List[float]:
    """
    Hash‑based sparse expansion.

    Each input value is assigned a pseudo‑random index in the high‑dimensional
    space of size ``m``; the value is added to that slot (collision handling is
    simple additive accumulation).  The result is a dense ``list`` of length
    ``m`` with many zeros – the “sparse” multivector coefficients.
    """
    out = [0.0] * m
    for idx, val in enumerate(values):
        key = f"{idx}:{val}"
        pos = _hash_index(key, salt, m)
        out[pos] += val
    return out


def top_k_mask(vector: List[float], k: int) -> List[int]:
    """
    Return a binary mask (list of 0/1) where the ``k`` largest absolute entries
    are 1 and the rest are 0.
    """
    if k <= 0:
        return [0] * len(vector)
    if k >= len(vector):
        return [1] * len(vector)

    # indices of top‑k absolute values
    abs_vals = np.abs(vector)
    threshold = np.partition(abs_vals, -k)[-k]
    return [int(abs(v) >= threshold) for v in vector]


# ----------------------------------------------------------------------
# Parent B – Pheromone container and Bayesian / Entropy utilities
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class Span:
    """Placeholder from the original parent – retained for compatibility."""
    start: int
    end: int
    text: str
    label: str
    score: float
    backend: str = "literal_fallback"


class PheromoneEntry:
    """Decay‑aware pheromone container."""

    __slots__ = (
        "uuid",
        "surface_key",
        "signal_kind",
        "signal_value",
        "half_life_seconds",
        "created_at",
        "last_decay",
    )

    def __init__(self, surface_key: str, signal_kind: str, signal_value: float, half_life_seconds: int):
        import uuid
        from datetime import datetime, timezone

        self.uuid = str(uuid.uuid4())
        self.surface_key = surface_key
        self.signal_kind = signal_kind
        self.signal_value = signal_value
        self.half_life_seconds = half_life_seconds
        now = datetime.now(timezone.utc)
        self.created_at = now
        self.last_decay = now

    def age_seconds(self) -> float:
        from datetime import datetime, timezone

        return (datetime.now(timezone.utc) - self.last_decay).total_seconds()

    def decay_factor(self) -> float:
        """Exponential decay based on half‑life."""
        age = self.age_seconds()
        if self.half_life_seconds <= 0:
            return 1.0
        return 0.5 ** (age / self.half_life_seconds)

    def current_value(self) -> float:
        """Signal value after decay."""
        return self.signal_value * self.decay_factor()


# Simple Count‑Min sketch implementation (hash‑based frequency estimator)
class CountMinSketch:
    def __init__(self, depth: int = 4, width: int = 1024, seed: int = 0):
        self.depth = depth
        self.width = width
        self.tables = [np.zeros(width, dtype=np.int64) for _ in range(depth)]
        self.seeds = [seed + i * 31 for i in range(depth)]

    def _hash(self, item: str, i: int) -> int:
        h = hashlib.blake2b((item + str(self.seeds[i])).encode(), digest_size=8).digest()
        return int.from_bytes(h, "little") % self.width

    def add(self, item: str, count: int = 1) -> None:
        for i in range(self.depth):
            idx = self._hash(item, i)
            self.tables[i][idx] += count

    def estimate(self, item: str) -> int:
        return min(self.tables[i][self._hash(item, i)] for i in range(self.depth))


# ----------------------------------------------------------------------
# Hybrid core – mathematical bridge
# ----------------------------------------------------------------------


def _posterior_vector(
    likelihood: List[float],
    priors: List[float],
    eps: float = 1e-6,
) -> List[float]:
    """
    Bayesian posterior for each blade.

    p_i = π_i * ℓ_i / (π_i*ℓ_i + (1-π_i)*ε)
    """
    post = []
    for ℓ, π in zip(likelihood, priors):
        num = π * ℓ
        den = π * ℓ + (1.0 - π) * eps
        post.append(num / den if den != 0 else 0.0)
    return post


def _extract_priors_from_pheromones(
    pheromones: List[PheromoneEntry],
    dim: int,
) -> List[float]:
    """
    Map pheromone entries to a prior vector of length ``dim``.
    If fewer entries than ``dim`` are supplied, the remaining priors are set to a
    small uniform baseline (1e-3).  Excess entries are ignored.
    """
    baseline = 1e-3
    priors = [baseline] * dim
    for i, entry in enumerate(pheromones):
        if i >= dim:
            break
        priors[i] = max(min(entry.current_value(), 1.0), 0.0)  # clamp to [0,1]
    return priors


# ----------------------------------------------------------------------
# Public hybrid operations
# ----------------------------------------------------------------------


def hybrid_pheromone_multivector(
    values: List[float],
    m: int,
    k: int,
    pheromones: List[PheromoneEntry],
    salt: str = "",
) -> List[int]:
    """
    1. Expand ``values`` to a high‑dimensional coefficient list ``c`` (size ``m``).
    2. Obtain priors ``π`` from ``pheromones`` (length ``m``).
    3. Compute posterior ``p`` via Bayesian update.
    4. Return a top‑k binary mask of the posterior coefficients.
    """
    expanded = expand(values, m, salt)
    priors = _extract_priors_from_pheromones(pheromones, m)
    posterior = _posterior_vector(expanded, priors)
    mask = top_k_mask(posterior, k)
    return mask


def allocate_adaptive_workshare_with_pheromone(
    past_allocations: Iterable[int],
    total_work: int,
    pheromones: List[PheromoneEntry],
    sketch_depth: int = 4,
    sketch_width: int = 1024,
) -> List[int]:
    """
    Adaptive work‑share allocation.

    * Build a Count‑Min sketch of historical allocation frequencies.
    * Derive a weight for each possible unit (0 … total_work‑1) from the sketch.
    * Modulate those weights with decayed pheromone values (if provided for the
      same index) to bias the allocation toward “hot” or “fresh” units.
    * Return a list of allocated unit indices (length = total_work) sampled
      proportionally to the final weights.
    """
    sketch = CountMinSketch(depth=sketch_depth, width=sketch_width)
    for alloc in past_allocations:
        sketch.add(str(alloc))

    # Base weights from sketch (add‑one smoothing)
    base_weights = np.array([sketch.estimate(str(i)) + 1 for i in range(total_work)], dtype=np.float64)

    # Pheromone modulation (align length if possible)
    pheromone_weights = np.ones(total_work, dtype=np.float64)
    for i, entry in enumerate(pheromones):
        if i >= total_work:
            break
        pheromone_weights[i] = entry.current_value()

    combined = base_weights * pheromone_weights
    if combined.sum() == 0:
        prob = np.full(total_work, 1.0 / total_work)
    else:
        prob = combined / combined.sum()

    # Sample without replacement according to the probability distribution.
    allocated = np.random.choice(total_work, size=total_work, replace=False, p=prob)
    return allocated.tolist()


def hybrid_rlct_estimate_with_multivector(
    values: List[float],
    m: int,
    pheromones: List[PheromoneEntry],
    eps: float = 1e-6,
    salt: str = "",
) -> float:
    """
    Regularised Log‑Canonical‑Type (RLCT) estimate.

    * Expand ``values`` → ``c``.
    * Compute posterior coefficients ``p``.
    * Perform a simple geometric product of the posterior multivector with itself:
      ⟨P·P⟩ = Σ_i p_i².
    * Return ``-log(⟨P·P⟩ + ε)`` as a scalar regularised measure.
    """
    expanded = expand(values, m, salt)
    priors = _extract_priors_from_pheromones(pheromones, m)
    posterior = _posterior_vector(expanded, priors, eps=eps)
    geom_scalar = float(np.sum(np.square(posterior)))
    return -math.log(geom_scalar + eps)


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    # Example low‑dimensional input
    low_vec = [random.uniform(-1, 1) for _ in range(5)]

    # Create a few pheromone entries (half‑life of 60 seconds)
    pheros = [
        PheromoneEntry(surface_key=f"dim{i}", signal_kind="test", signal_value=random.random(), half_life_seconds=60)
        for i in range(10)
    ]

    # 1. Hybrid WTA mask
    mask = hybrid_pheromone_multivector(values=low_vec, m=256, k=8, pheromones=pheros, salt="demo")
    print("Hybrid top‑k mask (sum = {}):".format(sum(mask)), mask[:32], "...")

    # 2. Adaptive allocation
    past = [random.randint(0, 31) for _ in range(200)]
    allocation = allocate_adaptive_workshare_with_pheromone(past_allocations=past, total_work=32, pheromones=pheros[:32])
    print("Adaptive allocation (first 10):", allocation[:10])

    # 3. RLCT estimate
    rlct = hybrid_rlct_estimate_with_multivector(values=low_vec, m=256, pheromones=pheros, salt="demo")
    print("RLCT estimate:", rlct)