# DARWIN HAMMER — match 1229, survivor 6
# gen: 5
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_geomet_m1117_s0.py (gen4)
# parent_b: hybrid_decreasing_pruning_hybrid_hybrid_minimu_m63_s1.py (gen3)
# born: 2026-05-29T23:34:44Z

"""Hybrid Algorithm combining fractional Caputo kernel weighting with epistemic‑certainty‑aware
minimum‑cost tree construction and decreasing‑rate pruning.

Parents:
- hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_geomet_m1117_s0.py (Caputo kernel + Voronoi geometry)
- hybrid_decreasing_pruning_hybrid_hybrid_minimu_m63_s1.py (Epistemic certainty flags, edge‑weighted MST,
  decreasing‑rate pruning)

Mathematical bridge:
The Caputo kernel provides a time‑dependent scaling factor  κ(t;α)=t^{α‑1}/Γ(α) that can be
applied to any scalar cost.  In the parent B the edge cost is a product of Euclidean distance
and an epistemic‑certainty factor ϕ(flag)∈[0,1].  By multiplying the original cost with κ(t;α) we
obtain a unified, time‑varying edge weight

    w_{ij}(t) = d(p_i,p_j) · ϕ(flag_{ij}) · κ(t;α)

which simultaneously embeds fractional‑calculus dynamics (parent A) and epistemic‑certainty
weighting (parent B).  The resulting weighted graph is fed to a classic minimum‑spanning‑tree
(Kruskal) algorithm, after which a decreasing‑rate pruning schedule (parent B) removes edges
with probability p(t)=λ·e^{‑α·t}.  The three core functions below implement this fused pipeline.
"""

import math
import random
import sys
import pathlib
from typing import List, Tuple, Dict, Hashable

import numpy as np

# ----------------------------------------------------------------------
# Fractional calculus (Parent A)
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
        # Reflection formula
        return math.pi / (math.sin(math.pi * z) * _gamma(1 - z))
    z -= 1
    x = _LANCZOS_C[0]
    for i in range(1, _LANCZOS_G + 2):
        x += _LANCZOS_C[i] / (z + i)
    t = z + _LANCZOS_G + 0.5
    return math.sqrt(2 * math.pi) * t ** (z + 0.5) * math.exp(-t) * x

def caputo_kernel(alpha: float, t: np.ndarray) -> np.ndarray:
    """Unnormalised Caputo kernel κ(t;α)=t^{α‑1}/Γ(α) for a vector of time indices."""
    if alpha <= 0:
        raise ValueError("Fractional order alpha must be positive.")
    t = np.where(t == 0, 1e-12, t)          # avoid division by zero
    return t ** (alpha - 1) / _gamma(alpha)

# ----------------------------------------------------------------------
# Epistemic certainty & geometric utilities (Parent B)
# ----------------------------------------------------------------------
EPISTEMIC_FLAGS = ("FACT", "PROBABLE", "POSSIBLE", "BULLSHIT", "SURE_MAYBE")
# Mapping from flag to a certainty factor in (0,1]; higher means more trustworthy
_EPISTEMIC_FACTOR: Dict[str, float] = {
    "FACT": 0.9,
    "PROBABLE": 0.7,
    "POSSIBLE": 0.5,
    "BULLSHIT": 0.2,
    "SURE_MAYBE": 0.8,
}

def euclidean_distance(a: Tuple[float, ...], b: Tuple[float, ...]) -> float:
    """Standard Euclidean distance between two points of equal dimension."""
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def prune_probability(t: float, lam: float = 1.0, alpha: float = 0.2) -> float:
    """Decreasing‑rate pruning probability p(t)=min(1, λ·e^{‑α·t})."""
    if t < 0 or lam < 0 or alpha < 0:
        raise ValueError('t, lam, and alpha must be non‑negative')
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
# Hybrid core functions
# ----------------------------------------------------------------------
def compute_edge_weight(p1: Tuple[float, ...],
                        p2: Tuple[float, ...],
                        flag: str,
                        time: float,
                        alpha_frac: float) -> float:
    """
    Unified edge weight:
        w = d(p1,p2) * ϕ(flag) * κ(time;α_frac)

    where
        d   – Euclidean distance,
        ϕ   – epistemic certainty factor,
        κ   – Caputo kernel.
    """
    if flag not in _EPISTEMIC_FACTOR:
        raise ValueError(f"Unknown epistemic flag '{flag}'.")
    dist = euclidean_distance(p1, p2)
    phi = _EPISTEMIC_FACTOR[flag]
    kappa = caputo_kernel(alpha_frac, np.array([time]))[0]
    return dist * phi * kappa

class UnionFind:
    """Simple Union‑Find (Disjoint‑Set) data structure for Kruskal's algorithm."""
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

def minimum_spanning_tree(points: List[Tuple[float, ...]],
                          flags: List[str],
                          time: float,
                          alpha_frac: float) -> List[Tuple[int, int, float]]:
    """
    Build an MST over `points` where each edge (i,j) receives a weight from
    `compute_edge_weight`.  The `flags` list must have length equal to the number of
    possible undirected edges (n*(n‑1)//2) and is ordered lexicographically by vertex index.
    Returns a list of edges (i, j, weight) belonging to the MST.
    """
    n = len(points)
    if n < 2:
        return []
    # Generate all possible edges with their weights
    edges: List[Tuple[int, int, float]] = []
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            w = compute_edge_weight(points[i], points[j], flags[idx], time, alpha_frac)
            edges.append((i, j, w))
            idx += 1
    # Kruskal
    edges.sort(key=lambda e: e[2])          # sort by weight
    uf = UnionFind(n)
    mst: List[Tuple[int, int, float]] = []
    for i, j, w in edges:
        if uf.union(i, j):
            mst.append((i, j, w))
        if len(mst) == n - 1:
            break
    return mst

def prune_mst(mst: List[Tuple[int, int, float]],
              t: float,
              lam: float = 1.0,
              alpha: float = 0.2,
              seed: int | str | None = None) -> List[Tuple[int, int, float]]:
    """
    Apply decreasing‑rate pruning to the edges of an MST.
    The function returns the subset of edges that survive the pruning step.
    """
    # We only need the edge identifiers for pruning; keep the full tuple for later use.
    edge_ids = [(i, j) for i, j, _ in mst]
    kept_ids = set(prune_edges(edge_ids, t, lam, alpha, seed))
    return [(i, j, w) for i, j, w in mst if (i, j) in kept_ids]

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Generate a small random point set
    random.seed(42)
    np.random.seed(42)
    num_points = 6
    points = [tuple(np.random.rand(2)) for _ in range(num_points)]

    # Create a deterministic flag list for all possible edges
    possible_flags = list(EPISTEMIC_FLAGS)
    num_edges = num_points * (num_points - 1) // 2
    flags = [possible_flags[i % len(possible_flags)] for i in range(num_edges)]

    # Parameters
    time = 3.0          # current time for Caputo scaling
    alpha_frac = 0.7    # fractional order
    lam = 0.9
    alpha_prune = 0.15

    # Build MST with hybrid weighting
    mst = minimum_spanning_tree(points, flags, time, alpha_frac)
    print("MST edges (i, j, weight):")
    for e in mst:
        print(e)

    # Prune the MST
    pruned = prune_mst(mst, t=time, lam=lam, alpha=alpha_prune, seed=123)
    print("\nEdges after pruning:")
    for e in pruned:
        print(e)

    # Simple sanity check: pruned edges should be a subset of MST edges
    assert set((i, j) for i, j, _ in pruned).issubset(
        set((i, j) for i, j, _ in mst)
    ), "Pruned edges must be subset of original MST"

    print("\nSmoke test completed successfully.")