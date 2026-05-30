# DARWIN HAMMER — match 4725, survivor 3
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_decreasing_pr_m1229_s6.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_cockpi_m1673_s2.py (gen5)
# born: 2026-05-29T23:57:43Z

"""Hybrid Algorithm integrating fractional Caputo dynamics with Fisher‑information weighted graph
construction.

Parents:
- hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_decreasing_pr_m1229_s6.py
- hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_cockpi_m1673_s2.py

Mathematical bridge:
The Caputo kernel κ(t;α)=t^{α‑1}/Γ(α) supplies a time‑dependent scaling factor.
Fisher information I(θ) of a Gaussian model provides a data‑driven weight that depends on a
scalar θ (here taken as the Euclidean distance between two points).  By multiplying the
edge cost of a Euclidean graph with both the epistemic‑uncertainty flag ϕ(flag) and the
fractional kernel, and finally modulating it with I(d), we obtain a unified, time‑varying
edge weight

    w_{ij}(t) = d(p_i,p_j) · ϕ(flag_{ij}) · κ(t;α) · I(d(p_i,p_j)) .

The weighted complete graph is reduced to a minimum‑spanning‑tree (Kruskal).  A decreasing‑rate
pruning schedule p(t)=λ·e^{‑β·t} then stochastically removes edges, yielding the final
hybrid structure.
"""

import math
import random
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Hashable, Sequence

import numpy as np

# ----------------------------------------------------------------------
# Fractional calculus utilities (Parent A)
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
    """
    Unnormalised Caputo kernel κ(t;α)=t^{α‑1}/Γ(α) for a vector of time indices.
    """
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    t = np.asarray(t, dtype=float)
    # avoid division by zero at t=0 when α<1
    t = np.where(t == 0, np.finfo(float).tiny, t)
    return t ** (alpha - 1) / _gamma(alpha)

# ----------------------------------------------------------------------
# Gaussian‑Fisher utilities (Parent B)
# ----------------------------------------------------------------------
def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Gaussian intensity centred at *center* with standard deviation *width*."""
    if width <= 0.0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)

def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    """
    Fisher information for a single‑parameter Gaussian model.
    I(θ) = (∂ℓ/∂θ)² / ℓ, where ℓ = Gaussian intensity.
    """
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity

def gaussian_filter(data: np.ndarray, sigma: float) -> np.ndarray:
    """Apply a 1‑D Gaussian smoothing kernel (sigma) to *data*."""
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    radius = int(3 * sigma)
    xs = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (xs / sigma) ** 2)
    kernel /= kernel.sum()
    return np.convolve(data, kernel, mode="same")

# ----------------------------------------------------------------------
# Graph utilities
# ----------------------------------------------------------------------
class UnionFind:
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

def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))

# ----------------------------------------------------------------------
# Hybrid core functions (integration of both parents)
# ----------------------------------------------------------------------
def compute_edge_weights(
    points: np.ndarray,
    flags: np.ndarray,
    alpha: float,
    t: float,
    fisher_center: float,
    fisher_width: float,
    phi_func=lambda flag: 1.0 - flag  # simple epistemic flag mapping
) -> List[Tuple[int, int, float]]:
    """
    Compute time‑dependent edge weights w_{ij}(t) = d·ϕ·κ·I(d).

    Returns a list of (i, j, weight) tuples for the complete undirected graph.
    """
    n = len(points)
    kappa = caputo_kernel(alpha, np.array([t]))[0]
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            d = euclidean_distance(points[i], points[j])
            phi = phi_func(flags[i, j])
            fisher = fisher_score(d, fisher_center, fisher_width)
            w = d * phi * kappa * fisher
            edges.append((i, j, w))
    return edges

def kruskal_mst(edges: List[Tuple[int, int, float]], n_vertices: int) -> List[Tuple[int, int, float]]:
    """
    Classic Kruskal algorithm returning the minimum‑spanning‑tree as a list of edges.
    """
    # sort by weight
    sorted_edges = sorted(edges, key=lambda e: e[2])
    uf = UnionFind(n_vertices)
    mst = []
    for i, j, w in sorted_edges:
        if uf.union(i, j):
            mst.append((i, j, w))
            if len(mst) == n_vertices - 1:
                break
    return mst

def decreasing_prune(
    mst: List[Tuple[int, int, float]],
    t: float,
    lam: float,
    beta: float
) -> List[Tuple[int, int, float]]:
    """
    Stochastically prune edges of the MST with probability p(t)=λ·e^{‑β·t}.
    """
    p = lam * math.exp(-beta * t)
    pruned = [e for e in mst if random.random() > p]
    return pruned

def hybrid_process(
    points: np.ndarray,
    flags: np.ndarray,
    alpha: float,
    time_grid: np.ndarray,
    fisher_center: float,
    fisher_width: float,
    lam: float,
    beta: float
) -> List[Tuple[int, int, float]]:
    """
    Full pipeline:
    1. For each time step compute edge weights.
    2. Build MST.
    3. Apply decreasing‑rate pruning.
    4. Return the union of surviving edges across all time steps.
    """
    n = len(points)
    surviving_edges: Dict[Tuple[int, int], float] = {}

    for t in time_grid:
        edges = compute_edge_weights(
            points, flags, alpha, float(t), fisher_center, fisher_width
        )
        mst = kruskal_mst(edges, n)
        kept = decreasing_prune(mst, float(t), lam, beta)
        for i, j, w in kept:
            key = (min(i, j), max(i, j))
            # keep the smallest weight observed for robustness
            if key not in surviving_edges or w < surviving_edges[key]:
                surviving_edges[key] = w

    # convert dict back to list of tuples
    result = [(i, j, w) for (i, j), w in surviving_edges.items()]
    return result

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    random.seed(42)
    np.random.seed(0)

    # synthetic dataset: 10 points in 2‑D
    points = np.random.rand(10, 2)

    # epistemic flags matrix (symmetric, 0/1)
    flags = np.zeros((10, 10), dtype=int)
    for i in range(10):
        for j in range(i + 1, 10):
            flags[i, j] = flags[j, i] = random.choice([0, 1])

    # parameters
    alpha = 0.7                     # fractional order
    time_grid = np.linspace(1, 5, 5)  # five time stamps
    fisher_center = 0.0
    fisher_width = 0.3
    lam = 0.2                       # pruning intensity
    beta = 0.5                      # pruning decay

    edges = hybrid_process(
        points,
        flags,
        alpha,
        time_grid,
        fisher_center,
        fisher_width,
        lam,
        beta
    )

    print("Resulting hybrid edges (i, j, weight):")
    for e in edges:
        print(e)