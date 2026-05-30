# DARWIN HAMMER — match 1937, survivor 5
# gen: 6
# parent_a: sparse_wta.py (gen0)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m871_s2.py (gen5)
# born: 2026-05-29T23:39:53Z

"""Hybrid Sparse-WTA & Pheromone-Driven Multivector Algorithm

This module fuses the core topologies of **sparse_wta.py** (Algorithm A) and the
“hybrid_hybrid…” family (Algorithm B).  

*Algorithm A* provides a *hash‑based sparse expansion* (`expand`) that maps a low‑dimensional
real vector into a high‑dimensional sparse representation, together with a
winner‑take‑all mask (`top_k_mask`).  

*Algorithm B* works with *multivector‑style geometric products* that are modulated by
*pheromone signals* and adapts its behaviour using a *Count‑Min sketch* (log‑count
statistics).

The mathematical bridge is the **expanded sparse vector**: it serves as the
component list of a multivector.  By scaling these components with a pheromone
vector we obtain a *pheromone‑modulated multivector*.  The geometric product of
this multivector is then reduced to a scalar field whose top‑k entries are
selected, yielding a sparse tag that can be compared with Hamming distance.
Thus the two algorithms are fused at the level of the expanded representation
and the pheromone‑scaled geometric product.

The public API offers three representative hybrid operations:

1. `hybrid_pheromone_multivector` – expands an input, applies pheromone scaling,
   computes a simple geometric product, and returns a top‑k binary mask.
2. `allocate_adaptive_workshare_with_pheromone` – allocates work units per day,
   adapts the allocation using a Count‑Min sketch of past allocations and
   biases it with pheromone signals.
3. `hybrid_rlct_estimate_with_multivector` – estimates a regularized
   log‑canonical‑type (RLCT) quantity from the pheromone‑modulated multivector
   and the sketch‑based log‑likelihood.

All functions rely only on the Python standard library and NumPy.
"""

from __future__ import annotations

import hashlib
import math
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Set, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Algorithm A – sparse winner‑take‑all utilities
# ----------------------------------------------------------------------


def expand(values: List[float], m: int, salt: str = "") -> List[float]:
    """Hash‑based sparse expansion.

    Each input value contributes to three randomly chosen positions in a
    length‑`m` vector, with a random sign.  The result is a dense list of length
    `m` that is typically very sparse because `m` ≫ `len(values)`.
    """
    if m <= 0:
        raise ValueError("m must be positive")
    out = [0.0] * m
    for i, v in enumerate(values):
        for r in range(3):
            h = hashlib.sha256(f"{salt}:{i}:{r}".encode()).digest()
            j = int.from_bytes(h[:8], "big") % m
            sign = 1.0 if h[8] & 1 else -1.0
            out[j] += sign * v
    return out


def top_k_mask(values: List[float], k: int) -> List[int]:
    """Return a binary mask with ones at the indices of the top‑k values."""
    k = max(0, min(k, len(values)))
    winners = {
        i
        for i, _ in sorted(enumerate(values), key=lambda x: (-x[1], x[0]))[:k]
    }
    return [1 if i in winners else 0 for i in range(len(values))]


def hamming(a: List[int], b: List[int]) -> int:
    """Hamming distance between two equal‑length binary vectors."""
    if len(a) != len(b):
        raise ValueError("vectors must be same length")
    return sum(x != y for x, y in zip(a, b))


# ----------------------------------------------------------------------
# Algorithm B – pheromone‑modulated multivector & Count‑Min sketch
# ----------------------------------------------------------------------


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x)),
    )


class CountMinSketch:
    """Simple Count‑Min sketch with logarithmic updates."""

    def __init__(self, width: int = 1024, depth: int = 4, seed: int = 0):
        self.width = width
        self.depth = depth
        self.tables = np.zeros((depth, width), dtype=np.int64)
        rng = random.Random(seed)
        self.hashes = [
            (rng.randint(1, 2**31 - 1), rng.randint(0, 2**31 - 1))
            for _ in range(depth)
        ]

    def _hash(self, item: int, i: int) -> int:
        a, b = self.hashes[i]
        return (a * item + b) % self.width

    def add(self, item: int, count: int = 1) -> None:
        for i in range(self.depth):
            idx = self._hash(item, i)
            self.tables[i, idx] += count

    def estimate(self, item: int) -> int:
        """Return the minimum count across hash rows (upper bound)."""
        return min(self.tables[i, self._hash(item, i)] for i in range(self.depth))

    def total(self) -> int:
        """Total count of all items (approximate)."""
        return int(self.tables.sum() / self.depth)


def _blade_sign(indices: Tuple[int, ...]) -> int:
    """Sign of a basis blade given its ordered index tuple.

    The sign follows the rule of swapping indices to achieve ascending order.
    """
    sign = 1
    lst = list(indices)
    n = len(lst)
    for i in range(n):
        for j in range(i + 1, n):
            if lst[i] > lst[j]:
                sign = -sign
    return sign


def _geometric_product(vec: np.ndarray) -> np.ndarray:
    """A simplified geometric product.

    For a vector `v` we compute all pairwise products v_i * v_j multiplied by
    the blade sign of (i, j).  The result is a vector of length `len(v)`.
    """
    n = vec.size
    result = np.zeros(n, dtype=vec.dtype)
    for i in range(n):
        acc = 0.0
        for j in range(n):
            if i == j:
                continue
            sign = _blade_sign((i, j))
            acc += sign * vec[i] * vec[j]
        result[i] = acc
    return result


# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------


def hybrid_pheromone_multivector(
    values: List[float],
    m: int,
    k: int,
    pheromone: List[float],
    salt: str = "",
) -> List[int]:
    """
    1. Expand `values` into a sparse high‑dimensional vector of length `m`.
    2. Modulate the expansion with a pheromone vector (element‑wise scaling).
    3. Apply a simplified geometric product to obtain a new representation.
    4. Return a binary top‑k mask of the resulting magnitudes.

    Parameters
    ----------
    values: Input real vector.
    m: Length of the expanded space (must be ≥ len(values)).
    k: Number of winners for the final mask.
    pheromone: Pheromone signal of length at least `m`.  If shorter, it is tiled.
    salt: Optional salt for reproducible hashing.

    Returns
    -------
    Binary mask (list of 0/1) of length `m`.
    """
    # 1. Sparse expansion
    expanded = np.array(expand(values, m, salt=salt), dtype=np.float64)

    # 2. Pheromone scaling (tile if necessary)
    pher = np.array(pheromone, dtype=np.float64)
    if pher.size < m:
        repeats = (m + pher.size - 1) // pher.size
        pher = np.tile(pher, repeats)[:m]
    scaled = expanded * pher

    # 3. Geometric product (pairwise interaction)
    geo = _geometric_product(scaled)

    # 4. Top‑k binary mask
    mask = top_k_mask(geo.tolist(), k)
    return mask


def allocate_adaptive_workshare_with_pheromone(
    day_of_week: int,
    total_units: int,
    pheromone: List[float],
    sketch: CountMinSketch,
) -> List[int]:
    """
    Allocate `total_units` work units across 7 days, adapting the allocation
    using a Count‑Min sketch of past allocations and biasing it with pheromone
    signals.

    Parameters
    ----------
    day_of_week: Integer 0‑6 (Monday=0 … Sunday=6) indicating the current day.
    total_units: Total number of indistinguishable work units to distribute.
    pheromone: Pheromone signal (length ≥ 7) that biases the day‑wise share.
    sketch: Count‑Min sketch storing historical allocations per day.

    Returns
    -------
    List of length 7 with integer allocations for each day; the entry at
    `day_of_week` reflects the current allocation.
    """
    # Base proportional allocation (simple linear ramp)
    base_weights = np.linspace(1.0, 2.0, 7)  # Monday gets 1, Sunday gets 2
    base_weights /= base_weights.sum()

    # Pheromone bias (tile if needed)
    pher = np.array(pheromone[:7], dtype=np.float64)
    pher = pher / pher.sum()  # normalize

    # Sketch‑based feedback: recent allocations per day (log‑scaled)
    feedback = np.array(
        [math.log1p(sketch.estimate(d) + 1) for d in range(7)], dtype=np.float64
    )
    feedback /= feedback.sum() if feedback.sum() != 0 else 1.0

    # Combine the three influences multiplicatively then renormalize
    combined = base_weights * pher * feedback
    combined /= combined.sum()

    # Allocate integer units (largest‑remainder method)
    raw = combined * total_units
    int_part = np.floor(raw).astype(int)
    remainder = total_units - int_part.sum()
    # Distribute the remaining units to the days with largest fractional parts
    fractions = raw - int_part
    extra_indices = np.argsort(-fractions)[:remainder]
    int_part[extra_indices] += 1

    # Record the allocation of the current day into the sketch
    sketch.add(day_of_week, int_part[day_of_week])

    return int_part.tolist()


def hybrid_rlct_estimate_with_multivector(
    values: List[float],
    m: int,
    pheromone: List[float],
    sketch: CountMinSketch,
    salt: str = "",
) -> float:
    """
    Estimate a Regularized Log‑Canonical‑Type (RLCT) quantity by:

    1. Expanding and pheromone‑modulating the input vector.
    2. Computing a scalar loss (squared L2 norm of the geometric product).
    3. Adding a log‑likelihood term approximated via the Count‑Min sketch.
    4. Normalising by the effective dimensionality.

    Returns a single float representing the RLCT estimate.
    """
    # 1. Expand and scale
    expanded = np.array(expand(values, m, salt=salt), dtype=np.float64)
    pher = np.array(pheromone, dtype=np.float64)
    if pher.size < m:
        repeats = (m + pher.size - 1) // pher.size
        pher = np.tile(pher, repeats)[:m]
    scaled = expanded * pher

    # 2. Geometric product and loss
    geo = _geometric_product(scaled)
    loss = np.sum(geo ** 2)  # L2‑squared loss

    # 3. Sketch‑based log‑likelihood (using total count as proxy)
    total_counts = sketch.total() + 1  # avoid log(0)
    log_likelihood = math.log(total_counts)

    # 4. Normalisation
    rlct = (loss + log_likelihood) / max(1, m)
    return rlct


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Simple deterministic seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    # Example input
    vec = [0.5, -1.2, 3.3]
    m_dim = 128
    top_k = 10
    pheromone_signal = [random.random() for _ in range(m_dim)]

    # Hybrid multivector mask
    mask = hybrid_pheromone_multivector(vec, m_dim, top_k, pheromone_signal, salt="test")
    print("Hybrid mask (sum of ones):", sum(mask))

    # Count‑Min sketch for allocation feedback
    cms = CountMinSketch(width=256, depth=5, seed=123)

    # Adaptive allocation for Wednesday (day 2) with 100 units
    allocation = allocate_adaptive_workshare_with_pheromone(
        day_of_week=2,
        total_units=100,
        pheromone=pheromone_signal,
        sketch=cms,
    )
    print("Allocation per day:", allocation)

    # RLCT estimate
    rlct = hybrid_rlct_estimate_with_multivector(
        values=vec,
        m=m_dim,
        pheromone=pheromone_signal,
        sketch=cms,
        salt="rlct",
    )
    print("RLCT estimate:", rlct)

    # Verify Hamming distance between two masks generated from slightly different inputs
    mask2 = hybrid_pheromone_multivector(
        [v + 0.01 for v in vec], m_dim, top_k, pheromone_signal, salt="test2"
    )
    print("Hamming distance between masks:", hamming(mask, mask2))