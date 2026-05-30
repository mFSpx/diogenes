# DARWIN HAMMER — match 409, survivor 5
# gen: 3
# parent_a: hybrid_hybrid_geometric_pro_hybrid_krampus_brain_m6_s3.py (gen2)
# parent_b: hybrid_rbf_surrogate_hybrid_distributed_l_m58_s0.py (gen2)
# born: 2026-05-29T23:28:56Z

"""Hybrid Voronoi‑RBF Multivector Algorithm
================================================

This module fuses the core mathematics of two parent algorithms:

* **Parent A** – geometric Voronoi partitioning with a full geometric algebra
  (multivector) implementation.
* **Parent B** – radial‑basis‑function (RBF) similarity modelling used to weight
  interactions in a graph‑based algorithm.

The mathematical bridge is the Euclidean distance `d(p, q)`.  
Parent A uses it to assign points to the nearest seed, while Parent B feeds the
same distance into a Gaussian RBF `exp(-(ε·d)²)` to obtain a similarity weight.
The hybrid algorithm therefore:

1. Computes the Voronoi region of each point (nearest‑seed assignment).
2. Evaluates the Gaussian RBF weight of the point‑seed pair.
3. Encodes each weighted contribution as a **multivector** whose blade is the
   seed index.  Region‑wise multivectors are obtained by summing these weighted
   blades.

The result is a set of region multivectors that carry both geometric (region
membership) and functional (RBF‑derived magnitude) information.

The public API consists of three demonstration functions:

* `compute_rbf_weights(seeds, points, epsilon)` – builds the RBF weight matrix.
* `voronoi_assign(points, seeds)` – classic nearest‑seed assignment.
* `region_multivectors(points, seeds, epsilon)` – returns a dictionary mapping
  each seed index to its aggregated multivector.

A small smoke test is provided under ``if __name__ == "__main__"``.
"""

import math
import random
import sys
import pathlib
from typing import List, Tuple, Dict, Set, FrozenSet

import numpy as np

# ----------------------------------------------------------------------
# Basic geometric utilities (from Parent A)
# ----------------------------------------------------------------------
Point = Tuple[float, float]

def distance(a: Point, b: Point) -> float:
    """Euclidean distance between two 2‑D points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])

def nearest(point: Point, seeds: List[Point]) -> int:
    """Index of the seed closest to ``point`` (break ties by smallest index)."""
    if not seeds:
        raise ValueError('seeds required')
    return min(range(len(seeds)), key=lambda i: (distance(point, seeds[i]), i))

# ----------------------------------------------------------------------
# Gaussian RBF (from Parent B)
# ----------------------------------------------------------------------
def gaussian(r: float, epsilon: float = 1.0) -> float:
    """Gaussian radial basis function."""
    return math.exp(-((epsilon * r) ** 2))

def compute_rbf_weights(
    seeds: List[Point],
    points: List[Point],
    epsilon: float = 1.0
) -> np.ndarray:
    """
    Return an ``(len(points), len(seeds))`` matrix ``W`` where
    ``W[i, j] = gaussian(distance(points[i], seeds[j]), epsilon)``.
    """
    n_pts = len(points)
    n_seeds = len(seeds)
    W = np.empty((n_pts, n_seeds), dtype=np.float64)
    for i, p in enumerate(points):
        for j, s in enumerate(seeds):
            W[i, j] = gaussian(distance(p, s), epsilon)
    return W

# ----------------------------------------------------------------------
# Multivector implementation (core of Parent A)
# ----------------------------------------------------------------------
def _blade_sign(indices: List[int]) -> Tuple[List[int], int]:
    """
    Sort ``indices`` by bubble‑sort while tracking the sign of the permutation.
    Identical consecutive indices cancel each other (Grassmann algebra rule).
    Returns the reduced sorted list and the accumulated sign (+1 or -1).
    """
    lst = list(indices)
    sign = 1
    n = len(lst)
    i = 0
    while i < n:
        j = 0
        while j < n - 1 - i:
            if lst[j] > lst[j + 1]:
                lst[j], lst[j + 1] = lst[j + 1], lst[j]
                sign *= -1
            elif lst[j] == lst[j + 1]:
                # cancel a pair of equal indices
                lst.pop(j)
                lst.pop(j)
                n -= 2
                i -= 1  # stay at current i because length changed
                continue
            j += 1
        i += 1
    return lst, sign

def _multiply_blades(blade_a: FrozenSet[int], blade_b: FrozenSet[int]) -> Tuple[FrozenSet[int], int]:
    combined = list(blade_a) + list(blade_b)
    result, sign = _blade_sign(combined)
    return frozenset(result), sign

class Multivector:
    """
    Sparse representation of a multivector in an n‑dimensional geometric algebra.
    ``components`` maps a blade (frozenset of basis indices) to a scalar coefficient.
    """
    def __init__(self, components: Dict[FrozenSet[int], float], n: int):
        self.n = int(n)
        # discard zero coefficients
        self.components = {k: float(v) for k, v in components.items() if abs(v) > 1e-15}

    def __add__(self, other: "Multivector") -> "Multivector":
        result = dict(self.components)
        for blade, coef in other.components.items():
            result[blade] = result.get(blade, 0.0) + coef
        return Multivector(result, self.n)

    def __repr__(self) -> str:
        if not self.components:
            return "Multivector(0)"
        terms = []
        for blade, coef in sorted(self.components.items(),
                                  key=lambda x: (len(x[0]), sorted(x[0]))):
            if blade:
                label = "e" + "".join(str(i) for i in sorted(blade))
            else:
                label = "1"
            terms.append(f"{coef:+.6g}*{label}")
        return "Multivector(" + " ".join(terms) + ")"

    def scalar_part(self) -> float:
        """Return the coefficient of the scalar (grade‑0) blade."""
        return self.components.get(frozenset(), 0.0)

    def grade(self, k: int) -> "Multivector":
        """Extract the grade‑k part of the multivector."""
        return Multivector(
            {blade: coef for blade, coef in self.components.items() if len(blade) == k},
            self.n,
        )

# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------
def voronoi_assign(points: List[Point], seeds: List[Point]) -> Dict[int, List[Point]]:
    """
    Classic Voronoi assignment: each point is placed in the region of its nearest seed.
    Returns a mapping ``seed_index -> list_of_points``.
    """
    regions: Dict[int, List[Point]] = {i: [] for i in range(len(seeds))}
    for p in points:
        idx = nearest(p, seeds)
        regions[idx].append(p)
    return regions

def point_to_multivector(seed_idx: int, weight: float, n_seeds: int) -> Multivector:
    """
    Encode a weighted contribution of a point as a multivector.
    The blade is the singleton ``{seed_idx}`` and the coefficient is ``weight``.
    """
    blade = frozenset({seed_idx})
    return Multivector({blade: weight}, n_seeds)

def region_multivectors(
    points: List[Point],
    seeds: List[Point],
    epsilon: float = 1.0
) -> Dict[int, Multivector]:
    """
    Compute a multivector for each Voronoi region.
    For every point ``p`` we:
        1. Find its nearest seed ``s`` (index ``i``).
        2. Evaluate the Gaussian RBF weight ``w = gaussian(distance(p, s), epsilon)``.
        3. Create a multivector ``w * e_i``.
    The region multivector is the sum of all such contributions belonging to that region.
    """
    n_seeds = len(seeds)
    # Initialise empty multivectors for each region
    region_mv: Dict[int, Multivector] = {i: Multivector({}, n_seeds) for i in range(n_seeds)}

    for p in points:
        i = nearest(p, seeds)
        w = gaussian(distance(p, seeds[i]), epsilon)
        mv = point_to_multivector(i, w, n_seeds)
        region_mv[i] = region_mv[i] + mv

    return region_mv

def hybrid_voronoi_rbf(
    points: List[Point],
    seeds: List[Point],
    epsilon: float = 1.0
) -> Tuple[Dict[int, List[Point]], np.ndarray, Dict[int, Multivector]]:
    """
    Full hybrid pipeline returning:
        * Voronoi region assignment (dict).
        * RBF weight matrix ``W`` of shape (len(points), len(seeds)).
        * Region multivectors that aggregate weighted blades.
    """
    regions = voronoi_assign(points, seeds)
    W = compute_rbf_weights(seeds, points, epsilon)
    region_mv = region_multivectors(points, seeds, epsilon)
    return regions, W, region_mv

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    random.seed(42)
    # generate 30 random points in the unit square
    points = [(random.random(), random.random()) for _ in range(30)]
    # pick 4 random seeds from the same distribution
    seeds = [(random.random(), random.random()) for _ in range(4)]

    eps = 2.0  # RBF spread parameter

    regions, W, region_mv = hybrid_voronoi_rbf(points, seeds, epsilon=eps)

    print("Voronoi region sizes:")
    for idx, pts in regions.items():
        print(f"  Seed {idx}: {len(pts)} points")

    print("\nRBF weight matrix (first 5 points):")
    print(W[:5])

    print("\nRegion multivectors:")
    for idx, mv in region_mv.items():
        print(f"  Seed {idx}: {mv}")