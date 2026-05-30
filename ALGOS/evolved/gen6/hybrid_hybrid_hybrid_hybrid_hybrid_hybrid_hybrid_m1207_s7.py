# DARWIN HAMMER — match 1207, survivor 7
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_decrea_m527_s0.py (gen5)
# parent_b: hybrid_hybrid_hybrid_ternar_hybrid_hybrid_worksh_m786_s0.py (gen4)
# born: 2026-05-29T23:34:26Z

"""Hybrid Algorithm: DarwInHammer‑Regret‑Voronoi‑Curvature Fusion
=================================================================

This module fuses the core mathematics of the two parent algorithms
*PARENT A* (``hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_decrea_m527_s0.py``) and
*PARENT B* (``hybrid_hybrid_hybrid_ternar_hybrid_hybrid_worksh_m786_s0.py``).

The bridge is built on the following observations:

1. **Regret‑weighted edge cost** – Parent A defines a *regret* that modulates the
   Euclidean distance between two points (``length`` + ``bayes_marginal``‑style
   weighting).  This provides a scalar *edge weight* ``w(a,b)``.

2. **Voronoi partitioning** – Parent B splits a set of points into regions
   defined by a set of seed points.  Each region can be processed independently.

3. **Curvature matrix** – Parent B builds a (symmetric) curvature matrix from a
   feature vector and uses it to weight a one‑hot group encoding.

The fusion therefore proceeds as:

* Partition the input points into Voronoi regions (Parent B).
* Inside each region construct a **minimum‑cost spanning tree** where the cost of
  an edge ``(i,j)`` is the *regret‑weighted* Euclidean distance defined by Parent A.
* The curvature matrix (Parent B) is projected onto the set of groups; the
  resulting scalar per‑group weight is used to distribute a *work‑share*
  proportional to the total tree weight of the region.

The result is a single unified system that simultaneously respects spatial
partitioning, regret‑aware distance metrics, and group‑wise curvature‑driven
allocation.

The public API consists of three high‑level functions:

* ``voronoi_partition(points, seeds)`` – Voronoi assignment (Parent B).
* ``region_mst(points, regret_func)`` – Regret‑weighted MST per region (Parent A).
* ``allocate_workshare(regions, groups, curvature_matrix)`` – Curvature‑driven
  allocation (Parent B).

A tiny smoke‑test is provided under ``if __name__ == "__main__":``.  The
implementation uses only the standard library and NumPy, respecting the
project constraints.
"""

import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Shared Types
# ----------------------------------------------------------------------
Point = Tuple[float, float]
Edge = Tuple[int, int, float]  # (index_i, index_j, weight)

# ----------------------------------------------------------------------
# Parent A – Regret‑Weighted Distance Utilities
# ----------------------------------------------------------------------
def euclidean(a: Point, b: Point) -> float:
    """Euclidean distance between two points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def regret_weighted_length(
    a: Point,
    b: Point,
    regret: float,
    prior: float = 0.5,
    likelihood: float = 0.9,
    false_positive: float = 0.1,
) -> float:
    """
    Combine Euclidean distance with a Bayesian regret term.

    The Bayesian term follows the spirit of ``bayes_marginal`` from Parent A:
    ``posterior = (likelihood * prior) / marginal`` where the marginal includes
    false positives.  The resulting scalar modulates the distance.

    Parameters
    ----------
    a, b : Point
        End points of the edge.
    regret : float
        A non‑negative scalar representing the *regret* associated with traversing
        the edge.
    prior, likelihood, false_positive : float
        Probabilities in [0, 1] used for the Bayesian term.

    Returns
    -------
    float
        Regret‑weighted edge cost.
    """
    if regret < 0:
        raise ValueError("regret must be non‑negative")
    # Bayesian marginal (simple version)
    marginal = likelihood * prior + false_positive * (1 - prior)
    posterior = (likelihood * prior) / marginal if marginal != 0 else 0.0
    # The regret term inflates the distance; posterior acts as a discount.
    distance = euclidean(a, b)
    return distance * (1.0 + regret) * (1.0 - posterior)


# ----------------------------------------------------------------------
# Parent B – Voronoi Partitioning
# ----------------------------------------------------------------------
def nearest(point: Point, seeds: List[Point]) -> int:
    """Return the index of the seed closest to *point* (ties broken by index)."""
    if not seeds:
        raise ValueError("seed list cannot be empty")
    return min(range(len(seeds)), key=lambda i: (euclidean(point, seeds[i]), i))


def assign_voronoi(points: List[Point], seeds: List[Point]) -> Dict[int, List[int]]:
    """
    Assign each point to the index of its nearest seed.

    Returns
    -------
    dict[int, list[int]]
        Mapping ``seed_index -> list_of_point_indices``.
    """
    regions: Dict[int, List[int]] = {i: [] for i in range(len(seeds))}
    for idx, p in enumerate(points):
        regions[nearest(p, seeds)].append(idx)
    return regions


# ----------------------------------------------------------------------
# Curvature Matrix (Parent B)
# ----------------------------------------------------------------------
def curvature_matrix(feature: np.ndarray) -> np.ndarray:
    """
    Produce a symmetric curvature matrix from a 1‑D feature vector.

    The construction mimics a simple outer‑product with a small regularisation
    term to guarantee positive‑definiteness:

        C = f fᵀ + ε I

    Parameters
    ----------
    feature : np.ndarray
        1‑D array of floats.

    Returns
    -------
    np.ndarray
        (n, n) symmetric positive‑definite matrix.
    """
    if feature.ndim != 1:
        raise ValueError("feature must be a 1‑D vector")
    eps = 1e-6
    C = np.outer(feature, feature) + eps * np.eye(feature.size)
    return C


# ----------------------------------------------------------------------
# Union‑Find Helper for Kruskal's MST
# ----------------------------------------------------------------------
class _UF:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x: int, y: int) -> bool:
        xr, yr = self.find(x), self.find(y)
        if xr == yr:
            return False
        if self.rank[xr] < self.rank[yr]:
            self.parent[xr] = yr
        elif self.rank[xr] > self.rank[yr]:
            self.parent[yr] = xr
        else:
            self.parent[yr] = xr
            self.rank[xr] += 1
        return True


# ----------------------------------------------------------------------
# Region‑wise Regret‑Weighted Minimum Spanning Tree
# ----------------------------------------------------------------------
def region_mst(
    points: List[Point],
    point_indices: List[int],
    regret_lookup: Dict[Tuple[int, int], float],
) -> List[Edge]:
    """
    Compute a regret‑weighted MST for a subset of points.

    Parameters
    ----------
    points : list[Point]
        Complete list of points (global coordinates).
    point_indices : list[int]
        Indices of points belonging to the region.
    regret_lookup : dict[tuple[int, int], float]
        Mapping ``(i, j) -> regret`` where ``i`` and ``j`` are global indices.

    Returns
    -------
    list[Edge]
        Edges of the MST expressed with global indices and their weight.
    """
    if len(point_indices) < 2:
        return []

    # Build all possible edges inside the region
    edges: List[Edge] = []
    for i in range(len(point_indices)):
        for j in range(i + 1, len(point_indices)):
            gi, gj = point_indices[i], point_indices[j]
            regret = regret_lookup.get((gi, gj), regret_lookup.get((gj, gi), 0.0))
            w = regret_weighted_length(
                points[gi], points[gj], regret, prior=0.5, likelihood=0.9, false_positive=0.1
            )
            edges.append((gi, gj, w))

    # Kruskal's algorithm
    edges.sort(key=lambda e: e[2])  # sort by weight
    uf = _UF(len(points))
    mst: List[Edge] = []
    for u, v, w in edges:
        if uf.union(u, v):
            mst.append((u, v, w))
            if len(mst) == len(point_indices) - 1:
                break
    return mst


# ----------------------------------------------------------------------
# Work‑share Allocation using Curvature Matrix
# ----------------------------------------------------------------------
def allocate_workshare(
    region_msts: Dict[int, List[Edge]],
    groups: Tuple[str, ...],
    curvature: np.ndarray,
) -> Dict[str, float]:
    """
    Allocate a scalar work‑share to each group.

    The total weight of each region's MST is projected onto the group space
    via the curvature matrix.  Concretely:

        w_g = Σ_region (region_weight * (c_g · v))

    where ``c_g`` is the one‑hot encoding of group *g* and ``v`` is the
    eigenvector corresponding to the largest eigenvalue of ``curvature``.
    The result is finally normalised to sum to 1.

    Parameters
    ----------
    region_msts : dict[int, list[Edge]]
        Mapping ``region_index -> list of MST edges``.
    groups : tuple[str, ...]
        Available group identifiers.
    curvature : np.ndarray
        Symmetric matrix from ``curvature_matrix``.

    Returns
    -------
    dict[str, float]
        Normalised work‑share per group.
    """
    if curvature.shape[0] != len(groups):
        raise ValueError("curvature matrix size must match number of groups")

    # Principal eigenvector (largest magnitude)
    eigvals, eigvecs = np.linalg.eig(curvature)
    principal = eigvecs[:, np.argmax(eigvals)].real
    principal = principal / np.linalg.norm(principal)  # unit length

    # One‑hot encodings as rows of the identity matrix
    one_hot = np.eye(len(groups))

    # Compute raw scores
    raw_scores = np.zeros(len(groups))
    for region_edges in region_msts.values():
        region_weight = sum(w for _, _, w in region_edges)
        # Projection of principal onto each group’s one‑hot vector is just the
        # corresponding component of the eigenvector.
        raw_scores += region_weight * principal

    # Normalise to a probability distribution
    total = raw_scores.sum()
    if total == 0:
        # fallback: uniform distribution
        return {g: 1.0 / len(groups) for g in groups}
    normalized = raw_scores / total
    return {g: float(normalized[i]) for i, g in enumerate(groups)}


# ----------------------------------------------------------------------
# High‑Level Hybrid Orchestration
# ----------------------------------------------------------------------
def hybrid_process(
    points: List[Point],
    seeds: List[Point],
    regret_lookup: Dict[Tuple[int, int], float],
    feature: np.ndarray,
    groups: Tuple[str, ...] = ("codex", "groq", "cohere", "local_models"),
) -> Dict[str, float]:
    """
    Full hybrid pipeline:

    1. Voronoi partition the points using *seeds*.
    2. Build a regret‑weighted MST inside each region.
    3. Compute a curvature matrix from *feature*.
    4. Allocate work‑share to *groups*.

    Returns
    -------
    dict[str, float]
        Normalised work‑share per group.
    """
    # Step 1 – Voronoi regions (indices of points per region)
    regions = assign_voronoi(points, seeds)

    # Step 2 – MST per region
    region_msts: Dict[int, List[Edge]] = {}
    for reg_idx, pt_idxs in regions.items():
        region_msts[reg_idx] = region_mst(points, pt_idxs, regret_lookup)

    # Step 3 – Curvature matrix (size must match number of groups)
    curv = curvature_matrix(feature)
    if curv.shape != (len(groups), len(groups)):
        # Simple projection / truncation to match required size
        curv = np.resize(curv, (len(groups), len(groups)))

    # Step 4 – Allocation
    allocation = allocate_workshare(region_msts, groups, curv)
    return allocation


# ----------------------------------------------------------------------
# Smoke Test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Synthetic data
    rng = random.Random(42)

    # Generate 30 random 2‑D points
    points = [(rng.uniform(0, 100), rng.uniform(0, 100)) for _ in range(30)]

    # Choose 4 seed points (could be a subset of points)
    seeds = points[:4]

    # Regret lookup: random non‑negative values for each unordered pair
    regret_lookup: Dict[Tuple[int, int], float] = {}
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            regret_lookup[(i, j)] = rng.random()  # 0 ≤ regret < 1

    # Feature vector length must equal number of groups (4)
    feature_vec = np.array([0.8, 0.1, 0.5, 0.3])

    # Run the hybrid pipeline
    workshare = hybrid_process(points, seeds, regret_lookup, feature_vec)

    # Simple sanity check: values sum to (approximately) 1
    total = sum(workshare.values())
    print("Work‑share allocation per group:")
    for g, w in workshare.items():
        print(f"  {g}: {w:.4f}")
    print(f"Total (should be 1.0): {total:.6f}")
    assert abs(total - 1.0) < 1e-6, "Allocation does not sum to 1"
    print("Smoke test completed successfully.")