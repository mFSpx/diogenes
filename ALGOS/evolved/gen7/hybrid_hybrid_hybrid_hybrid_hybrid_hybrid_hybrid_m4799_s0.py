# DARWIN HAMMER — match 4799, survivor 0
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hoeffding_tre_m296_s1.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_vorono_m2690_s1.py (gen6)
# born: 2026-05-29T23:58:07Z

"""Hybrid Hoeffding‑Voronoi‑RBF Associative Memory

Parents
-------
* **hybrid_hybrid_hybrid_hybrid_hybrid_hoeffding_tre_m296_s1.py** – provides a
  geometric‑algebra `Multivector` type and a Hoeffding‑bound function that
  scales a scalar (grade‑0) component of a multivector.
* **hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_vorono_m2690_s1.py** – implements a
  Voronoi‑partition / Radial‑Basis‑Function (RBF) associative memory where a
  query point is weighted by Gaussian similarities to seed centroids and the
  result is read from a per‑seed memory matrix (a “sheaf”).

Mathematical Bridge
-------------------
Both parents rely on a *distance* between a query and a set of seeds:

* In the Voronoi/RBF side the Euclidean distance `d(q, s)` drives a Gaussian
  similarity `g = exp(-(ε·d)²)`.
* In the Hoeffding side the bound `h = sqrt((r²·ln(1/δ))/(2·n))` uses a scalar
  `r` that can be interpreted as a “radius” attached to a seed.

The hybrid therefore treats the scalar component of each seed’s `Multivector`
as the radius `r`.  For every seed we compute


h_i = HoeffdingBound(r_i, δ, n)          # confidence penalty
g_i = Gaussian( Euclidean(q, s_i), ε )   # distance‑aware similarity
w_i = g_i * exp(-h_i)                    # fused weight


The fused weights `w_i` are applied to the linear readout stored in the
sheaf (one memory matrix per seed).  The final output is a distance‑aware,
confidence‑aware associative recall.

The module below implements this hybrid pipeline with three public functions
demonstrating the core operations.
"""

import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple, FrozenSet

import numpy as np

# ----------------------------------------------------------------------
# Parent A – Geometric Algebra & Hoeffding bound
# ----------------------------------------------------------------------


def _blade_sign(indices: FrozenSet[int]) -> Tuple[List[int], int]:
    """Return a sorted list of basis indices and the sign of the permutation."""
    lst = list(indices)
    sign = 1
    n = len(lst)
    for i in range(n):
        for j in range(n - 1 - i):
            if lst[j] > lst[j + 1]:
                lst[j], lst[j + 1] = lst[j + 1], lst[j]
                sign *= -1
            elif lst[j] == lst[j + 1]:
                # duplicate basis vectors cancel (grade reduction)
                lst.pop(j)
                lst.pop(j)
                return lst, sign
    return lst, sign


def _multiply_blades(blade_a: FrozenSet[int], blade_b: FrozenSet[int]) -> Tuple[FrozenSet[int], int]:
    combined = list(blade_a) + list(blade_b)
    result, sign = _blade_sign(frozenset(combined))
    return frozenset(result), sign


class Multivector:
    """A minimal multivector supporting outer product and scalar extraction."""

    def __init__(self, components: Dict[FrozenSet[int], float], n: int):
        # keep only non‑zero components
        self.components = {k: float(v) for k, v in components.items() if v != 0.0}
        self.n = int(n)

    def grade(self, k: int) -> "Multivector":
        """Return the grade‑k part of the multivector."""
        return Multivector(
            {bl: self.components[bl] for bl in self.components if len(bl) == k}, self.n
        )

    def __mul__(self, other):
        if isinstance(other, Multivector):
            result: Dict[FrozenSet[int], float] = {}
            for k, v in self.components.items():
                for k2, v2 in other.components.items():
                    new_k, sign = _multiply_blades(k, k2)
                    result[new_k] = result.get(new_k, 0.0) + sign * v * v2
            return Multivector(result, self.n)
        elif isinstance(other, (int, float)):
            return Multivector({k: v * other for k, v in self.components.items()}, self.n)
        else:
            raise TypeError("Unsupported operand type for Multivector multiplication")

    __rmul__ = __mul__


def hoeffding_bound(r: float, delta: float, n: int, mv: Multivector) -> float:
    """
    Hoeffding bound scaled by the scalar (grade‑0) component of `mv`.

    Parameters
    ----------
    r : float
        Base radius (must be > 0).
    delta : float
        Failure probability (0 < delta < 1).
    n : int
        Sample size (> 0).
    mv : Multivector
        Multivector whose scalar component scales the bound.

    Returns
    -------
    float
        The Hoeffding bound.
    """
    if r <= 0 or not (0 < delta < 1) or n <= 0:
        raise ValueError("r > 0, 0 < delta < 1, n > 0 required")
    # scalar component is the coefficient of the empty blade `frozenset()`
    scalar = mv.components.get(frozenset(), 1.0)
    scaled_r = r * scalar
    return math.sqrt((scaled_r * scaled_r * math.log(1.0 / delta)) / (2.0 * n))


# ----------------------------------------------------------------------
# Parent B – Voronoi / RBF associative memory utilities
# ----------------------------------------------------------------------


def gaussian(r: float, epsilon: float = 1.0) -> float:
    """Gaussian RBF kernel."""
    return math.exp(-((epsilon * r) ** 2))


def euclidean(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance between two vectors."""
    return np.linalg.norm(a - b)


def nearest(point: np.ndarray, seeds: np.ndarray) -> int:
    """Index of the seed closest to `point`."""
    return int(np.argmin(np.linalg.norm(seeds - point, axis=1)))


def assign(points: np.ndarray, seeds: np.ndarray) -> np.ndarray:
    """
    Voronoi assignment matrix.
    `regions[i, j] == 1` iff point j belongs to seed i.
    """
    n_seeds = seeds.shape[0]
    n_pts = points.shape[0]
    regions = np.zeros((n_seeds, n_pts), dtype=int)
    for j, p in enumerate(points):
        i = nearest(p, seeds)
        regions[i, j] = 1
    return regions


# ----------------------------------------------------------------------
# Hybrid Functions
# ----------------------------------------------------------------------


def compute_hoeffding_weights(
    seed_vectors: np.ndarray,
    seed_multivectors: List[Multivector],
    delta: float,
    n_samples: int,
) -> np.ndarray:
    """
    Compute a confidence‑penalty weight for each seed using the Hoeffding bound.

    The scalar radius `r` is taken as the Euclidean norm of the seed vector.
    The result is a 1‑D array `h_i = exp(-HoeffdingBound(...))` suitable for
    multiplicative fusion with similarity scores.
    """
    weights = np.empty(len(seed_vectors), dtype=float)
    for i, (vec, mv) in enumerate(zip(seed_vectors, seed_multivectors)):
        r = euclidean(vec, np.zeros_like(vec))  # norm of the seed
        h = hoeffding_bound(r, delta, n_samples, mv)
        weights[i] = math.exp(-h)
    return weights


def compute_rbf_weights(
    query: np.ndarray,
    seed_vectors: np.ndarray,
    epsilon: float = 1.0,
) -> np.ndarray:
    """
    Gaussian RBF similarity of `query` to each seed.
    Returns a 1‑D array of raw similarities (not normalized).
    """
    dists = np.linalg.norm(seed_vectors - query, axis=1)
    return np.vectorize(lambda d: gaussian(d, epsilon))(dists)


def hybrid_associative_memory(
    query: np.ndarray,
    seed_vectors: np.ndarray,
    seed_multivectors: List[Multivector],
    sheaf: Dict[int, np.ndarray],
    delta: float,
    n_samples: int,
    epsilon: float = 1.0,
) -> np.ndarray:
    """
    Perform a distance‑aware, confidence‑aware readout from a distributed
    associative memory.

    Steps
    -----
    1. Compute RBF similarities `g_i`.
    2. Compute Hoeffding‑derived confidence factors `c_i = exp(-h_i)`.
    3. Fuse them: `w_i = g_i * c_i`.
    4. For each seed i, read the linear memory `M_i @ query`.
    5. Return the weighted sum `∑ w_i * (M_i @ query)`.

    Parameters
    ----------
    query : np.ndarray
        Query vector (shape: (d,)).
    seed_vectors : np.ndarray
        Array of seed centroids (shape: (k, d)).
    seed_multivectors : List[Multivector]
        Multivector attached to each seed (length k).
    sheaf : Dict[int, np.ndarray]
        Mapping seed index → memory matrix (shape: (d, d) or compatible).
    delta, n_samples, epsilon : float
        Hyper‑parameters for Hoeffding and RBF.

    Returns
    -------
    np.ndarray
        The retrieved vector (shape: (d,)).
    """
    if len(seed_vectors) != len(seed_multivectors):
        raise ValueError("seed_vectors and seed_multivectors must have the same length")
    if any(i not in sheaf for i in range(len(seed_vectors))):
        raise KeyError("Sheaf must contain a memory matrix for every seed index")

    g = compute_rbf_weights(query, seed_vectors, epsilon)          # (k,)
    c = compute_hoeffding_weights(seed_vectors, seed_multivectors, delta, n_samples)  # (k,)
    w = g * c                                                       # (k,)

    # Normalize weights to avoid exploding magnitudes (optional but stable)
    if w.sum() != 0:
        w = w / w.sum()

    result = np.zeros_like(query, dtype=float)
    for i, weight in enumerate(w):
        M = sheaf[i]                      # (d, d) or (d, m)
        readout = M @ query               # linear readout
        result += weight * readout
    return result


# ----------------------------------------------------------------------
# Simple smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Dimensionality
    d = 8
    k = 5  # number of seeds / memory partitions

    # Random seed vectors
    rng = np.random.default_rng(42)
    seeds = rng.normal(size=(k, d))

    # Random multivectors: we only need a scalar component for the bound.
    seed_mvs = []
    for _ in range(k):
        scalar = rng.uniform(0.5, 1.5)
        mv = Multivector({frozenset(): scalar}, n=d)
        seed_mvs.append(mv)

    # Construct a sheaf: each seed owns a random linear memory matrix.
    sheaf = {i: rng.normal(size=(d, d)) for i in range(k)}

    # Random query
    q = rng.normal(size=d)

    # Hyper‑parameters
    delta = 0.05
    n_samples = 100
    epsilon = 0.8

    # Run the hybrid memory retrieval
    out = hybrid_associative_memory(
        query=q,
        seed_vectors=seeds,
        seed_multivectors=seed_mvs,
        sheaf=sheaf,
        delta=delta,
        n_samples=n_samples,
        epsilon=epsilon,
    )

    # Simple sanity check: output shape matches query shape
    assert out.shape == q.shape
    print("Hybrid associative memory output (first 5 entries):", out[:5])