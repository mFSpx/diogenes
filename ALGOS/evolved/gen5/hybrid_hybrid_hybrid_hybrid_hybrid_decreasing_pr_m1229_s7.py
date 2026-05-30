# DARWIN HAMMER — match 1229, survivor 7
# gen: 5
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_geomet_m1117_s0.py (gen4)
# parent_b: hybrid_decreasing_pruning_hybrid_hybrid_minimu_m63_s1.py (gen3)
# born: 2026-05-29T23:34:44Z

"""Hybrid Algorithm integrating:
- Parent A: fractional Caputo kernel modifying geometric distances (Voronoi/Euclidean).
- Parent B: epistemic‑certainty weighted minimum‑cost tree with decreasing‑rate pruning.

Mathematical bridge:
The Euclidean edge metric `d(i,j)` is first transformed by a Caputo kernel
`K_α(t) = t^{α-1} / Γ(α)` where `t` is a discrete time‑step (or any scalar
associated with the edge).  The transformed metric `d' = d * K_α(t)` is then
scaled by an epistemic certainty factor `c(flag) ∈ (0,1]` derived from the
parent‑B flag set.  The resulting weight `w = d' * c` feeds a classic
minimum‑spanning‑tree (MST) construction.  Finally the edge set is filtered
through the decreasing‑rate pruning schedule `p(t)=min(1, λ·exp(-α·t))`
from parent B.  This single pipeline fuses the fractional calculus core of
A with the epistemic‑aware pruning core of B."""

import math
import random
import sys
import pathlib
from typing import List, Tuple, Dict, Iterable, Hashable

import numpy as np

# ----------------------------------------------------------------------
# Parent A components (Caputo kernel & Euclidean geometry)
# ----------------------------------------------------------------------
_LANCZOS_G = 7
_LANCZOS_C = np.array([
    0.99999999999980993,
    676.5203681218851,
    -1259.1392167224028,
    771.32342877765313,
    -176.61502916214059,
    12.507343278686905,
    -0.13857
])

def _gamma(z: float) -> float:
    """Lanczos approximation of the Gamma function."""
    if z < 0.5:
        return math.pi / (math.sin(math.pi * z) * _gamma(1 - z))
    z -= 1
    x = _LANCZOS_C[0]
    for i in range(1, _LANCZOS_G + 2):
        x += _LANCZOS_C[i] / (z + i)
    t = z + _LANCZOS_G + 0.5
    return math.sqrt(2 * math.pi) * t ** (z + 0.5) * math.exp(-t) * x

def caputo_kernel(alpha: float, t: np.ndarray) -> np.ndarray:
    """Raw (unnormalized) Caputo kernel values for a vector of time indices."""
    if alpha <= 0:
        raise ValueError("Fractional order alpha must be positive.")
    t = np.where(t == 0, 1e-12, t)  # avoid division by zero
    return t ** (alpha - 1) / _gamma(alpha)

def euclidean_distance(a: Tuple[float, ...], b: Tuple[float, ...]) -> float:
    """Standard Euclidean distance."""
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def nearest_point(point: Tuple[float, ...], seeds: List[Tuple[float, ...]]) -> int:
    """Return index of the seed closest to `point` using Euclidean metric."""
    distances = [euclidean_distance(point, s) for s in seeds]
    return int(np.argmin(distances))

# ----------------------------------------------------------------------
# Parent B components (epistemic certainty & decreasing pruning)
# ----------------------------------------------------------------------
EPISTEMIC_FLAGS = ("FACT", "PROBABLE", "POSSIBLE", "BULLSHIT", "SURE_MAYBE")
# Mapping from flag to a numeric certainty factor (higher = more trustworthy)
_EPISTEMIC_FACTOR: Dict[str, float] = {
    "FACT": 0.9,
    "PROBABLE": 0.7,
    "POSSIBLE": 0.5,
    "BULLSHIT": 0.2,
    "SURE_MAYBE": 0.6,
}

def certainty_factor(flag: str) -> float:
    """Return the numeric factor associated with an epistemic flag."""
    if flag not in EPISTEMIC_FLAGS:
        raise ValueError(f"Unknown epistemic flag: {flag}")
    return _EPISTEMIC_FACTOR[flag]

def prune_probability(t: float, lam: float = 1.0, alpha: float = 0.2) -> float:
    """Decreasing‑rate pruning probability."""
    if t < 0 or lam < 0 or alpha < 0:
        raise ValueError("t, lam, and alpha must be non‑negative")
    return min(1.0, lam * math.exp(-alpha * t))

def prune_edges(edges: List[Hashable],
                t: float,
                lam: float = 1.0,
                alpha: float = 0.2,
                seed: int | str | None = None) -> List[Hashable]:
    """Randomly drop edges according to the decreasing‑rate schedule."""
    rng = random.Random(seed)
    p = prune_probability(t, lam, alpha)
    return [e for e in edges if rng.random() >= p]

# ----------------------------------------------------------------------
# Hybrid core (three public functions)
# ----------------------------------------------------------------------
def weighted_edge(i: int,
                 j: int,
                 points: List[Tuple[float, ...]],
                 time_idx: int,
                 alpha_caputo: float,
                 epistemic_flag: str) -> Tuple[int, int, float]:
    """
    Compute a hybrid weight for edge (i, j):
        w = d(i,j) * K_α(t) * c(flag)
    where
        d   – Euclidean distance,
        K_α – Caputo kernel evaluated at discrete time `t`,
        c   – epistemic certainty factor.
    """
    d = euclidean_distance(points[i], points[j])
    k = caputo_kernel(alpha_caputo, np.array([time_idx]))[0]
    c = certainty_factor(epistemic_flag)
    return (i, j, d * k * c)

def hybrid_minimum_spanning_tree(points: List[Tuple[float, ...]],
                                 seeds: List[Tuple[float, ...]],
                                 time_idx: int,
                                 alpha_caputo: float,
                                 epistemic_map: Dict[Tuple[int, int], str]) -> List[Tuple[int, int, float]]:
    """
    Build a minimum‑spanning tree (MST) on `points` where edge weights are the
    hybrid metric from `weighted_edge`.  The algorithm uses Prim's method
    (O(N²) – sufficient for the test harness) and respects the Voronoi
    partition induced by `seeds`: edges are only considered between points
    belonging to the same Voronoi cell, reducing graph size.
    """
    n = len(points)
    # Assign each point to a Voronoi cell
    cell_of = [nearest_point(p, seeds) for p in points]

    # Pre‑compute adjacency matrix with None for disallowed edges
    adj: List[List[float | None]] = [[None] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if cell_of[i] != cell_of[j]:
                continue  # forbid cross‑cell edges
            flag = epistemic_map.get((i, j), "POSSIBLE")
            _, _, w = weighted_edge(i, j, points, time_idx, alpha_caputo, flag)
            adj[i][j] = adj[j][i] = w

    # Prim's algorithm
    in_mst = [False] * n
    key = [math.inf] * n
    parent = [-1] * n
    key[0] = 0.0

    for _ in range(n):
        # pick the vertex with the smallest key not yet in MST
        u = min((idx for idx, used in enumerate(in_mst) if not used),
                key=lambda idx: key[idx],
                default=-1)
        if u == -1 or key[u] == math.inf:
            break  # disconnected component
        in_mst[u] = True
        for v, w in enumerate(adj[u]):
            if w is not None and not in_mst[v] and w < key[v]:
                key[v] = w
                parent[v] = u

    # Assemble edge list
    mst_edges: List[Tuple[int, int, float]] = []
    for v in range(1, n):
        u = parent[v]
        if u != -1:
            mst_edges.append((u, v, adj[u][v]))  # type: ignore[arg-type]

    return mst_edges

def hybrid_tree_with_pruning(points: List[Tuple[float, ...]],
                             seeds: List[Tuple[float, ...]],
                             time_idx: int,
                             alpha_caputo: float,
                             epistemic_map: Dict[Tuple[int, int], str],
                             prune_t: float,
                             lam: float = 1.0,
                             alpha_prune: float = 0.2,
                             seed: int | str | None = None) -> List[Tuple[int, int, float]]:
    """
    Full pipeline:
        1. Build hybrid MST using fractional‑weighted, epistemic‑aware edges.
        2. Apply decreasing‑rate pruning to the edge set.
    Returns the surviving edge list.
    """
    mst = hybrid_minimum_spanning_tree(points,
                                       seeds,
                                       time_idx,
                                       alpha_caputo,
                                       epistemic_map)
    pruned = prune_edges(mst, prune_t, lam, alpha_prune, seed)
    return pruned

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # deterministic seed for reproducibility
    random.seed(42)
    np.random.seed(0)

    # generate synthetic data
    NUM_POINTS = 30
    DIM = 2
    points = [tuple(coord) for coord in np.random.rand(NUM_POINTS, DIM)]

    # choose a few seed points for Voronoi partitioning
    seeds = [points[i] for i in random.sample(range(NUM_POINTS), k=4)]

    # time index and fractional order
    t_idx = 5
    alpha = 0.8  # Caputo order (0 < α < 1 yields long‑memory behavior)

    # create a random epistemic flag for each possible edge (symmetric)
    epistemic_map: Dict[Tuple[int, int], str] = {}
    for i in range(NUM_POINTS):
        for j in range(i + 1, NUM_POINTS):
            epistemic_map[(i, j)] = random.choice(EPISTEMIC_FLAGS)

    # run the hybrid pipeline
    result = hybrid_tree_with_pruning(points,
                                      seeds,
                                      time_idx=t_idx,
                                      alpha_caputo=alpha,
                                      epistemic_map=epistemic_map,
                                      prune_t=3.0,
                                      lam=0.9,
                                      alpha_prune=0.15,
                                      seed=123)

    print(f"Generated {len(result)} edges after pruning (out of {NUM_POINTS-1} possible MST edges).")
    for edge in result[:10]:  # show a few edges
        print(edge)