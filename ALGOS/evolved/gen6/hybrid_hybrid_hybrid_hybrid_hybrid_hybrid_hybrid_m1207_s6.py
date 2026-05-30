# DARWIN HAMMER — match 1207, survivor 6
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_decrea_m527_s0.py (gen5)
# parent_b: hybrid_hybrid_hybrid_ternar_hybrid_hybrid_worksh_m786_s0.py (gen4)
# born: 2026-05-29T23:34:26Z

"""
Hybrid Algorithm: hybrid_hybrid_hybrid_decrea_m527_s0 + hybrid_hybrid_hybrid_ternar_hybrid_hybrid_worksh_m786_s0

This module fuses the *regret‑weighted minimum‑cost tree* from Parent A with the
*Voronoi‑based region partitioning* and *curvature‑matrix workshare* from Parent B.

Mathematical bridge
-------------------
1. Points are first partitioned into Voronoi regions using a set of seed points
   (Parent B).  
2. Within each region a graph is built on the points; edge weights are the
   Euclidean distance multiplied by a regret factor (Parent A).  
3. A minimum‑spanning tree (MST) is extracted from each regional graph using the
   regret‑weighted weights, then optionally pruned with the time‑dependent
   probability from Parent A.  
4. A feature vector derived from an arbitrary text seed is turned into a
   curvature matrix **C** (diagonal of the vector).  The matrix projects a
   one‑hot encoding of the group name, yielding a scalar *workshare weight* for
   each group.  The total workshare for a region is distributed among groups
   proportionally to these weights.

The three core functions below demonstrate this pipeline:
* `voronoi_partition`
* `region_mst_with_regret`
* `allocate_workshare`

All operations rely only on NumPy and the Python standard library.
"""

import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Constants from the original parents
# ----------------------------------------------------------------------
GROUPS = ("codex", "groq", "cohere", "local_models")
Point = Tuple[float, float]

# ----------------------------------------------------------------------
# Utility functions (shared)
# ----------------------------------------------------------------------
def euclidean(a: Point, b: Point) -> float:
    """Euclidean distance between two 2‑D points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def prune_probability(t: float, lam: float = 1.0, alpha: float = 0.2) -> float:
    """Time‑dependent probability of pruning an edge."""
    if t < 0 or lam < 0 or alpha < 0:
        raise ValueError("t, lam, and alpha must be non‑negative")
    return min(1.0, lam * math.exp(-alpha * t))


def prune_edges(edges: List[Tuple[int, int, float]],
                t: float,
                lam: float = 1.0,
                alpha: float = 0.2,
                seed: int | str | None = None) -> List[Tuple[int, int, float]]:
    """Randomly drop edges according to `prune_probability`."""
    rng = random.Random(seed)
    p = prune_probability(t, lam, alpha)
    return [e for e in edges if rng.random() >= p]


def _rng_from_text(text: str) -> random.Random:
    """Deterministic RNG seeded from an arbitrary text string."""
    import hashlib
    h = hashlib.sha256(text.encode("utf-8")).digest()
    seed = int.from_bytes(h[:8], "big")
    return random.Random(seed)


# ----------------------------------------------------------------------
# 1. Voronoi partitioning (Parent B)
# ----------------------------------------------------------------------
def nearest(point: Point, seeds: List[Point]) -> int:
    """Return the index of the seed nearest to `point` (ties broken by index)."""
    if not seeds:
        raise ValueError("seed list cannot be empty")
    return min(range(len(seeds)),
               key=lambda i: (euclidean(point, seeds[i]), i))


def voronoi_partition(points: List[Point],
                      seeds: List[Point]) -> Dict[int, List[int]]:
    """
    Assign each point to the index of its nearest seed.
    Returns a mapping ``region_id -> list of point indices``.
    """
    regions: Dict[int, List[int]] = {i: [] for i in range(len(seeds))}
    for idx, p in enumerate(points):
        region = nearest(p, seeds)
        regions[region].append(idx)
    return regions


# ----------------------------------------------------------------------
# 2. Regret‑weighted MST inside each region (Parent A)
# ----------------------------------------------------------------------
def regret_weighted_length(a: Point, b: Point, regret: float) -> float:
    """
    Edge weight = Euclidean distance * (1 + regret).
    `regret` is assumed to be non‑negative; larger regret inflates the cost.
    """
    return euclidean(a, b) * (1.0 + regret)


def region_mst_with_regret(points: List[Point],
                           region_point_ids: List[int],
                           regrets: Dict[Tuple[int, int], float],
                           prune_time: float = 0.0,
                           seed: int | str | None = None) -> List[Tuple[int, int, float]]:
    """
    Build a regret‑weighted Minimum Spanning Tree for a single region.

    Parameters
    ----------
    points
        Global list of all points.
    region_point_ids
        Indices of points belonging to the region.
    regrets
        Mapping ``(i, j) -> regret`` where ``i`` and ``j`` are global point indices.
    prune_time
        If >0, edges are probabilistically pruned after the MST is built.
    seed
        Random seed for pruning.

    Returns
    -------
    List of edges ``(i, j, weight)`` belonging to the MST (after optional pruning).
    """
    if len(region_point_ids) < 2:
        return []  # nothing to connect

    # Build full adjacency matrix for the region
    n = len(region_point_ids)
    idx_map = {pid: pos for pos, pid in enumerate(region_point_ids)}
    # Prim's algorithm
    in_tree = [False] * n
    min_edge = [math.inf] * n
    parent = [-1] * n

    # Start from the first vertex
    min_edge[0] = 0.0

    for _ in range(n):
        # Select the vertex with the smallest connecting edge not yet in the tree
        u = -1
        best = math.inf
        for v in range(n):
            if not in_tree[v] and min_edge[v] < best:
                best = min_edge[v]
                u = v
        if u == -1:
            break  # disconnected (should not happen)

        in_tree[u] = True
        # Update neighbours
        for v in range(n):
            if in_tree[v]:
                continue
            i_global = region_point_ids[u]
            j_global = region_point_ids[v]
            key = (i_global, j_global) if i_global < j_global else (j_global, i_global)
            regret = regrets.get(key, 0.0)
            w = regret_weighted_length(points[i_global], points[j_global], regret)
            if w < min_edge[v]:
                min_edge[v] = w
                parent[v] = u

    # Extract edges
    edges: List[Tuple[int, int, float]] = []
    for v in range(1, n):
        u = parent[v]
        i_global = region_point_ids[u]
        j_global = region_point_ids[v]
        key = (i_global, j_global) if i_global < j_global else (j_global, i_global)
        regret = regrets.get(key, 0.0)
        w = regret_weighted_length(points[i_global], points[j_global], regret)
        edges.append((i_global, j_global, w))

    # Optional pruning
    if prune_time > 0.0:
        edges = prune_edges(edges, prune_time, seed=seed)

    return edges


# ----------------------------------------------------------------------
# 3. Curvature matrix and workshare allocation (Parent B)
# ----------------------------------------------------------------------
def feature_vector_from_text(text: str, dim: int = len(GROUPS)) -> np.ndarray:
    """
    Produce a deterministic pseudo‑random feature vector in [0, 1]^dim
    from an arbitrary text seed.
    """
    rng = _rng_from_text(text)
    return np.array([rng.random() for _ in range(dim)], dtype=float)


def curvature_matrix(feature_vec: np.ndarray) -> np.ndarray:
    """
    Build a diagonal curvature matrix C = diag(feature_vec).
    """
    return np.diag(feature_vec)


def allocate_workshare(region_edges: List[Tuple[int, int, float]],
                       curvature: np.ndarray,
                       group_names: Tuple[str, ...] = GROUPS) -> Dict[str, float]:
    """
    Distribute a region's total edge weight among groups using the curvature matrix.

    The allocation weight for a group g is:
        w_g = (C * e_g)^T (C * e_g) = (feature_vec[g])^2
    where e_g is the one‑hot vector for group g.
    The region's total cost (sum of edge weights) is split proportionally to w_g.

    Returns a mapping ``group_name -> allocated cost``.
    """
    total_cost = sum(w for _, _, w in region_edges)
    if total_cost == 0.0:
        # Avoid division by zero – give each group an equal share
        equal_share = 0.0
        return {g: equal_share for g in group_names}

    # Compute squared feature values as raw weights
    raw_weights = np.square(np.diag(curvature))
    raw_weights_sum = raw_weights.sum()
    if raw_weights_sum == 0.0:
        # All features zero – fallback to uniform allocation
        raw_weights = np.ones_like(raw_weights)

    allocation: Dict[str, float] = {}
    for idx, name in enumerate(group_names):
        share = (raw_weights[idx] / raw_weights.sum()) * total_cost
        allocation[name] = share
    return allocation


# ----------------------------------------------------------------------
# High‑level hybrid pipeline
# ----------------------------------------------------------------------
def hybrid_process(points: List[Point],
                   seed_points: List[Point],
                   regret_dict: Dict[Tuple[int, int], float],
                   text_seed: str,
                   prune_time: float = 0.0,
                   rng_seed: int | str | None = None) -> Dict[int, Dict[str, float]]:
    """
    End‑to‑end hybrid algorithm.

    1. Voronoi partition the `points` using `seed_points`.
    2. For each region compute a regret‑weighted MST (pruned if `prune_time` > 0).
    3. Build a curvature matrix from `text_seed`.
    4. Allocate the region's total MST cost to the predefined groups.

    Returns a mapping ``region_id -> {group_name: allocated_cost}``.
    """
    # Step 1: partition
    regions = voronoi_partition(points, seed_points)

    # Step 2: per‑region MST
    region_msts: Dict[int, List[Tuple[int, int, float]]] = {}
    for rid, idxs in regions.items():
        mst = region_mst_with_regret(points,
                                    idxs,
                                    regret_dict,
                                    prune_time=prune_time,
                                    seed=rng_seed)
        region_msts[rid] = mst

    # Step 3: curvature matrix from text
    feat_vec = feature_vector_from_text(text_seed)
    C = curvature_matrix(feat_vec)

    # Step 4: allocate workshare
    allocation: Dict[int, Dict[str, float]] = {}
    for rid, edges in region_msts.items():
        allocation[rid] = allocate_workshare(edges, C)

    return allocation


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Generate a simple synthetic dataset
    rng = random.Random(42)
    num_points = 30
    points = [(rng.uniform(0, 100), rng.uniform(0, 100)) for _ in range(num_points)]

    # Choose 4 seed points (one per group)
    seeds = points[:4]

    # Create a regret dictionary with small random regrets
    regret_dict: Dict[Tuple[int, int], float] = {}
    for i in range(num_points):
        for j in range(i + 1, num_points):
            regret_dict[(i, j)] = rng.random() * 0.3  # regret in [0, 0.3)

    # Run the hybrid algorithm
    result = hybrid_process(points,
                            seeds,
                            regret_dict,
                            text_seed="Hybrid test seed",
                            prune_time=0.1,
                            rng_seed=12345)

    # Print a concise summary
    for region_id, alloc in sorted(result.items()):
        total = sum(alloc.values())
        print(f"Region {region_id}: total cost = {total:.3f}")
        for g, val in alloc.items():
            print(f"  {g}: {val:.3f}")
    sys.exit(0)