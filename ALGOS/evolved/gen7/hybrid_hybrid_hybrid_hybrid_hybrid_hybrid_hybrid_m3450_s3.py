# DARWIN HAMMER — match 3450, survivor 3
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1207_s7.py (gen6)
# parent_b: hybrid_hybrid_hybrid_liquid_hybrid_hybrid_hybrid_m2033_s1.py (gen5)
# born: 2026-05-29T23:50:18Z

"""Hybrid Voronoi‑Regret‑MinHash‑Curvature Fusion
================================================

This module merges the two parent algorithms:

* **Parent A** – Voronoi partitioning + regret‑weighted Euclidean edge cost
  (used to build a minimum‑cost spanning tree per region) + curvature‑matrix
  based work‑share allocation.

* **Parent B** – MinHash signature generation and Jaccard‑like similarity
  (used in the Regret‑Weighted Hoeffding Tree) together with a symmetric
  curvature matrix derived from group feature vectors.

**Mathematical bridge**

For every pair of points ``i`` and ``j`` we compute a *regret‑weighted* edge
cost


d_ij   = ||p_i - p_j||_2                                 # Euclidean distance
sim_ij = similarity(sig_i, sig_j)                        # MinHash similarity
r_ij   = 1.0 - sim_ij                                     # regret factor (0 ≤ r ≤ 1)
w_ij   = d_ij * (1.0 + r_ij)                              # final edge weight


Thus the MinHash similarity from Parent B directly modulates the regret term
in Parent A’s edge cost.  The curvature matrix ``C`` (symmetric, size
``G×G`` where ``G`` is the number of groups) is then projected onto the
group‑wise counts inside each Voronoi region to obtain a per‑group work‑share
scalar.  The final allocation for a region ``R`` and group ``g`` is


share(R,g) = (C·c_R)_g * total_MST_weight_R / Σ_h (C·c_R)_h


where ``c_R`` is the one‑hot count vector of groups inside region ``R``.

The public API consists of three high‑level functions:

* ``voronoi_partition(points, seeds)`` – assign each point to the nearest seed.
* ``region_mst(indices, points, tokens)`` – compute a regret‑weighted MST for a
  set of point indices.
* ``allocate_workshare(regions, groups, curvature_matrix, mst_weights)`` –
  curvature‑driven allocation of the MST weight to groups.

All code uses only the Python standard library and NumPy.
"""

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

    # Prim's algorithm with a simple O(N^2) implementation (region sizes are
    # modest in typical evolutionary runs).
    visited = set()
    start = indices[0]
    visited.add(start)

    # Candidate edges: (weight, from, to)
    candidates: List[Tuple[float, int, int]] = []
    for j in indices[1:]:
        w = _regret_weighted_cost(start, j, points, sigs)
        candidates.append((w, start, j))

    total_weight = 0.0
    mst_edges: List[Tuple[int, int]] = []

    while len(visited) < len(indices):
        # Pick the smallest candidate that connects to an unvisited node
        candidates.sort(key=lambda x: x[0])  # O(N log N) but fine for small N
        for idx, (w, frm, to) in enumerate(candidates):
            if to not in visited:
                # Accept this edge
                total_weight += w
                mst_edges.append((frm, to))
                visited.add(to)
                # Remove used candidate
                candidates.pop(idx)
                # Add new edges from the newly visited node
                for nxt in indices:
                    if nxt not in visited:
                        w_new = _regret_weighted_cost(to, nxt, points, sigs)
                        candidates.append((w_new, to, nxt))
                break
        else:
            # Should never happen if the graph is connected
            raise RuntimeError("MST construction failed: no connecting edge found")

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
        ``region -> {group -> allocated share}``.
    """
    unique_groups = sorted(set(groups))
    G = len(unique_groups)
    if curvature_matrix.shape != (G, G):
        raise ValueError("curvature_matrix shape must match number of groups")

    # Mapping from group id to matrix index
    grp_to_idx = {g: i for i, g in enumerate(unique_groups)}

    allocation: Dict[int, Dict[int, float]] = {}

    for region_id, point_idxs in regions.items():
        # One‑hot count vector c_R (size G)
        c_R = np.zeros(G, dtype=float)
        for idx in point_idxs:
            g = groups[idx]
            c_R[grp_to_idx[g]] += 1.0

        # Project curvature matrix onto counts
        proj = curvature_matrix @ c_R  # shape (G,)

        # Avoid division by zero
        if proj.sum() == 0.0:
            weights = np.full(G, 1.0 / G)
        else:
            weights = proj / proj.sum()

        region_share = mst_weights.get(region_id, 0.0)
        allocation[region_id] = {
            g: float(weights[grp_to_idx[g]] * region_share) for g in unique_groups
        }

    return allocation


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Seed reproducibility
    random.seed(42)
    np.random.seed(42)

    # Generate synthetic data
    N = 50                     # number of points
    D = 3                      # dimensionality
    S = 5                      # number of Voronoi seeds
    G = 4                      # number of groups

    points = np.random.randn(N, D).astype(float)

    # Random seeds (choose a subset of points)
    seed_indices = np.random.choice(N, S, replace=False)
    seeds = points[seed_indices]

    # Random token lists (simulating textual features)
    vocab = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta"]
    tokens = [
        random.sample(vocab, k=random.randint(1, len(vocab)))
        for _ in range(N)
    ]

    # Random group assignment
    groups = [random.randint(0, G - 1) for _ in range(N)]

    # Random symmetric curvature matrix (positive semi‑definite)
    A = np.random.randn(G, G)
    curvature = A @ A.T  # ensures symmetry and PSD

    # 1) Voronoi partition
    regions = voronoi_partition(points, seeds)

    # 2) Compute MST per region and collect weights
    mst_weights: Dict[int, float] = {}
    for region_id, idxs in regions.items():
        weight, _ = region_mst(idxs, points, tokens)
        mst_weights[region_id] = weight

    # 3) Allocate workshare using curvature matrix
    allocation = allocate_workshare(regions, groups, curvature, mst_weights)

    # Simple sanity printout
    print("Region MST weights:")
    for rid, w in mst_weights.items():
        print(f"  Region {rid}: {w:.4f}")

    print("\nWorkshare allocation (region -> group -> share):")
    for rid, grp_dict in allocation.items():
        print(f"Region {rid}:")
        for g, share in grp_dict.items():
            print(f"  Group {g}: {share:.4f}")

    # Verify that each region's allocated shares sum to its MST weight
    for rid, grp_dict in allocation.items():
        total = sum(grp_dict.values())
        expected = mst_weights[rid]
        assert math.isclose(total, expected, rel_tol=1e-6), (
            f"Allocation sum mismatch in region {rid}: {total} vs {expected}"
        )

    print("\nSmoke test completed successfully.")