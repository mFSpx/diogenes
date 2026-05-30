# DARWIN HAMMER — match 3450, survivor 4
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1207_s7.py (gen6)
# parent_b: hybrid_hybrid_hybrid_liquid_hybrid_hybrid_hybrid_m2033_s1.py (gen5)
# born: 2026-05-29T23:50:18Z

import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import hashlib

# ----------------------------------------------------------------------
# Parent B – MinHash utilities
# ----------------------------------------------------------------------
MAX64 = (1 << 64) - 1


def _hash(seed: int, token: str) -> int:
    """Hash a token with a 4‑byte seed using Blake2b (64‑bit output)."""
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")


def signature(tokens: List[str], k: int = 128) -> List[int]:
    """Return a MinHash signature of length ``k`` for a token list."""
    toks = {t for t in tokens if t}
    if k <= 0:
        raise ValueError("k must be positive")
    if not toks:
        return [MAX64] * k
    return [min(_hash(i, t) for t in toks) for i in range(k)]


def similarity(sig_a: List[int], sig_b: List[int]) -> float:
    """Jaccard‑like similarity between two MinHash signatures."""
    if len(sig_a) != len(sig_b):
        raise ValueError("signatures must have equal length")
    if not sig_a:
        raise ValueError("signatures must not be empty")
    return sum(1 for a, b in zip(sig_a, sig_b) if a == b) / len(sig_a)


# ----------------------------------------------------------------------
# Parent A – Voronoi, regret‑weighted MST, curvature allocation
# ----------------------------------------------------------------------


def voronoi_partition(points: np.ndarray, seeds: np.ndarray) -> Dict[int, List[int]]:
    """
    Assign each point to the index of its nearest seed.

    Parameters
    ----------
    points : (N, D) array
        Input points.
    seeds : (S, D) array
        Seed points that define Voronoi cells.

    Returns
    -------
    dict[int, list[int]]
        Mapping ``seed_index -> list of point indices``.
    """
    if points.ndim != 2 or seeds.ndim != 2:
        raise ValueError("points and seeds must be 2‑dimensional")
    if points.shape[1] != seeds.shape[1]:
        raise ValueError("dimensionality mismatch between points and seeds")

    # Compute squared Euclidean distance matrix (N x S)
    diff = points[:, None, :] - seeds[None, :, :]  # shape (N, S, D)
    dists = np.einsum("nsd,nsd->ns", diff, diff)   # squared distances

    nearest = np.argmin(dists, axis=1)  # (N,)

    regions: Dict[int, List[int]] = {i: [] for i in range(seeds.shape[0])}
    for idx, seed_idx in enumerate(nearest):
        regions[int(seed_idx)].append(int(idx))
    return regions


def _regret_weighted_cost(
    i: int,
    j: int,
    points: np.ndarray,
    sigs: List[List[int]],
) -> float:
    """
    Regret‑weighted edge cost between points ``i`` and ``j``.

    w_ij = ||p_i - p_j||_2 * (1 + (1 - similarity(sig_i, sig_j)))

    The term ``1 - similarity`` is the regret factor.
    """
    # Euclidean distance
    dist = np.linalg.norm(points[i] - points[j])

    # MinHash similarity
    sim = similarity(sigs[i], sigs[j])

    regret = 1.0 - sim
    return dist * (1.0 + regret)


def region_mst(
    indices: List[int],
    points: np.ndarray,
    tokens: List[List[str]],
) -> Tuple[float, List[Tuple[int, int]]]:
    """
    Compute a regret‑weighted Minimum Spanning Tree (MST) for a region.

    Parameters
    ----------
    indices : list[int]
        Indices of points belonging to the region.
    points : (N, D) array
        Full point set.
    tokens : list[list[str]]
        Token list per point (used for MinHash signatures).

    Returns
    -------
    total_weight : float
        Sum of edge weights in the MST.
    edges : list[tuple[int, int]]
        List of edges (original point indices) forming the MST.
    """
    if len(indices) == 0:
        return 0.0, []

    # Pre‑compute MinHash signatures for the points in the region
    sigs = {idx: signature(tokens[idx]) for idx in indices}

    # Kruskal's algorithm with a priority queue (region sizes are
    # modest in typical evolutionary runs).
    edges = []
    for i in indices:
        for j in indices:
            if i < j:
                w = _regret_weighted_cost(i, j, points, sigs)
                edges.append((w, i, j))

    edges.sort()

    parent = {i: i for i in indices}
    rank = {i: 0 for i in indices}

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        root_x = find(x)
        root_y = find(y)

        if root_x != root_y:
            if rank[root_x] > rank[root_y]:
                parent[root_y] = root_x
            else:
                parent[root_x] = root_y
                if rank[root_x] == rank[root_y]:
                    rank[root_y] += 1

    total_weight = 0.0
    mst_edges: List[Tuple[int, int]] = []

    for w, frm, to in edges:
        if find(frm) != find(to):
            union(frm, to)
            total_weight += w
            mst_edges.append((frm, to))

    return total_weight, mst_edges


def allocate_workshare(
    regions: Dict[int, List[int]],
    groups: List[int],
    curvature_matrix: np.ndarray,
    mst_weights: Dict[int, float],
) -> Dict[int, Dict[int, float]]:
    """
    Allocate each region's MST weight to groups using a curvature matrix.

    Parameters
    ----------
    regions : dict[int, list[int]]
        Voronoi region mapping (seed index -> point indices).
    groups : list[int]
        Group identifier per point (length = number of points).
    curvature_matrix : (G, G) array
        Symmetric curvature matrix where G = number of distinct groups.
    mst_weights : dict[int, float]
        Total MST weight per region (seed index -> weight).

    Returns
    -------
    dict[int, dict[int, float]]
        Allocation of MST weights to groups per region.
    """
    num_groups = len(set(groups))
    if curvature_matrix.shape != (num_groups, num_groups):
        raise ValueError("curvature matrix has incorrect shape")

    group_counts = {i: {} for i in regions}
    for region, indices in regions.items():
        for idx in indices:
            group = groups[idx]
            if group not in group_counts[region]:
                group_counts[region][group] = 0
            group_counts[region][group] += 1

    for region in regions:
        counts = np.array([group_counts[region].get(g, 0) for g in range(num_groups)])
        counts = counts / counts.sum()
        allocation = np.dot(curvature_matrix, counts)
        total_weight = mst_weights[region]
        allocation = {g: total_weight * a for g, a in enumerate(allocation)}
        mst_weights[region] = {k: v for k, v in allocation.items()}

    return mst_weights