# DARWIN HAMMER — match 4088, survivor 3
# gen: 6
# parent_a: hybrid_voronoi_partition_hybrid_hybrid_rbf_su_m1815_s0.py (gen5)
# parent_b: hybrid_hybrid_tropical_maxp_hybrid_hybrid_hybrid_m2163_s1.py (gen5)
# born: 2026-05-29T23:53:28Z

"""Hybrid Voronoi–Tropical RBF Surrogate

This module fuses the two parent algorithms:

* **Parent A** – Voronoi partitioning with a radial‑basis‑function (RBF) surrogate that
  predicts a scalar “energy” for any point in Euclidean space.
* **Parent B** – Tropical max‑plus algebra used to propagate a belief (maximum‑log‑probability)
  through a matrix derived from pairwise distances, together with an SSIM‑style weighting
  concept.

**Mathematical bridge**

The Euclidean distance matrix `D` of the Voronoi seeds is turned into a tropical
matrix `L = -D`.  In max‑plus algebra the “multiplication’’ is ordinary addition and
the “addition’’ is the maximum, i.e.


C[i, j] = max_k (A[i, k] + B[k, j])


We propagate a belief vector from a chosen root seed using repeated tropical
matrix multiplication.  The resulting belief for each seed is a log‑probability
that reflects its proximity (shorter paths → larger belief).  The belief of the
seed that owns a point is then used as a weighting factor for the RBF surrogate
prediction, yielding a single hybrid score that couples spatial partitioning,
probabilistic propagation and surrogate modelling.
"""

import math
import random
import sys
import pathlib
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence, List, Tuple, Dict

import numpy as np

# ----------------------------------------------------------------------
# Basic geometric utilities (from Parent A)
# ----------------------------------------------------------------------
Vector = Sequence[float]
Point = Tuple[float, float]

def euclidean(a: Vector, b: Vector) -> float:
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def distance(p1: Point, p2: Point) -> float:
    """2‑D Euclidean distance (convenient wrapper)."""
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def nearest(point: Point, seeds: List[Point]) -> int:
    """Index of the closest seed (ties broken by index)."""
    if not seeds:
        raise ValueError("seed list is empty")
    return min(range(len(seeds)), key=lambda i: (distance(point, seeds[i]), i))

def assign(points: List[Point], seeds: List[Point]) -> Dict[int, List[Point]]:
    """Voronoi assignment of each point to its nearest seed."""
    regions: Dict[int, List[Point]] = {i: [] for i in range(len(seeds))}
    for p in points:
        regions[nearest(p, seeds)].append(p)
    return regions

# ----------------------------------------------------------------------
# RBF surrogate (from Parent A)
# ----------------------------------------------------------------------
def gaussian(r: float, epsilon: float = 1.0) -> float:
    """Isotropic Gaussian RBF."""
    return math.exp(-((epsilon * r) ** 2))

@dataclass(frozen=True)
class RBFSurrogate:
    centers: List[Tuple[float, ...]]
    weights: List[float]
    epsilon: float = 1.0

    def predict(self, x: Vector) -> float:
        """Linear combination of Gaussian RBFs."""
        return sum(
            w * gaussian(euclidean(x, c), self.epsilon)
            for w, c in zip(self.weights, self.centers)
        )

# ----------------------------------------------------------------------
# Tropical max‑plus utilities (from Parent B)
# ----------------------------------------------------------------------
def t_add(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Tropical addition (⊕) = element‑wise max."""
    return np.maximum(x, y)

def t_mul(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Tropical multiplication (⊗) = element‑wise addition."""
    return x + y

def t_matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Tropical matrix multiplication C = A ⊗ B.
    C[i, j] = max_k (A[i, k] + B[k, j])
    """
    if A.shape[1] != B.shape[0]:
        raise ValueError("Incompatible shapes for tropical matmul")
    # Use broadcasting: (i,k) + (k,j) -> (i,k,1) + (1,k,j) then max over k
    A_exp = A[:, :, np.newaxis]          # shape (i, k, 1)
    B_exp = B[np.newaxis, :, :]          # shape (1, k, j)
    sum_ = A_exp + B_exp                 # shape (i, k, j)
    return np.max(sum_, axis=1)          # max over k -> shape (i, j)

def tropical_propagate(L: np.ndarray, root_idx: int, steps: int = 3) -> np.ndarray:
    """
    Propagate a belief vector from ``root_idx`` through the tropical matrix ``L``.
    The belief after ``steps`` tropical multiplications is returned.
    """
    n = L.shape[0]
    # Initialise belief as a row vector: 0 for root (log‑prob 0), -inf elsewhere
    belief = np.full((1, n), -np.inf)
    belief[0, root_idx] = 0.0
    for _ in range(steps):
        belief = t_matmul(belief, L)
    return belief.flatten()   # shape (n,)

# ----------------------------------------------------------------------
# Hybrid operation (core of the new algorithm)
# ----------------------------------------------------------------------
def hybrid_predict(
    point: Point,
    seeds: List[Point],
    surrogate: RBFSurrogate,
    belief: np.ndarray,
) -> float:
    """
    Compute a hybrid score for ``point``:

    1. Use the RBF surrogate to predict an energy ``e``.
    2. Find the seed that owns the point (Voronoi region) and fetch its
       tropical belief ``b`` (log‑probability).
    3. Combine them multiplicatively after converting the belief to a linear
       weight: w = exp(b).  The final hybrid score is ``w * e``.
    """
    e = surrogate.predict(point)
    seed_idx = nearest(point, seeds)
    b = belief[seed_idx]            # log‑probability (may be -inf)
    w = math.exp(b) if b > -np.inf else 0.0
    return w * e

def hybrid_region_scores(
    points: List[Point],
    seeds: List[Point],
    surrogate: RBFSurrogate,
    belief: np.ndarray,
) -> Dict[int, List[float]]:
    """
    Return, for each Voronoi region, the list of hybrid scores of the points
    that belong to that region.
    """
    regions = assign(points, seeds)
    scores: Dict[int, List[float]] = {i: [] for i in range(len(seeds))}
    for idx, pts in regions.items():
        for p in pts:
            scores[idx].append(hybrid_predict(p, seeds, surrogate, belief))
    return scores

def distance_matrix(seeds: List[Point]) -> np.ndarray:
    """Pairwise Euclidean distance matrix of the seed points."""
    n = len(seeds)
    D = np.empty((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            D[i, j] = distance(seeds[i], seeds[j])
    return D

# ----------------------------------------------------------------------
# Simple smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    # Generate synthetic seeds and points
    num_seeds = 5
    seeds = [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(num_seeds)]

    num_points = 30
    points = [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(num_points)]

    # Build a trivial RBF surrogate (centers = seeds, random weights)
    centers = [tuple(s) for s in seeds]                     # use seed coordinates as centers
    weights = list(np.random.randn(len(centers)))           # random weights
    surrogate = RBFSurrogate(centers=centers, weights=weights, epsilon=0.05)

    # Tropical belief propagation
    D = distance_matrix(seeds)
    L = -D                                 # convert distances to log‑probabilities
    root = 0                               # choose first seed as root
    belief = tropical_propagate(L, root_idx=root, steps=4)

    # Compute hybrid scores per region
    region_scores = hybrid_region_scores(points, seeds, surrogate, belief)

    # Print a concise summary
    for idx, scores in region_scores.items():
        avg = sum(scores) / len(scores) if scores else 0.0
        print(f"Region {idx}: {len(scores)} points, average hybrid score = {avg:.4f}")

    print("\nBelief vector (log‑probabilities):")
    print(belief)
    sys.exit(0)