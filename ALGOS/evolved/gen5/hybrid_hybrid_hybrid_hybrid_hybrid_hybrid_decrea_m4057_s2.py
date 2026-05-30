# DARWIN HAMMER — match 4057, survivor 2
# gen: 5
# parent_a: hybrid_hybrid_hybrid_percyp_hybrid_caputo_fracti_m610_s2.py (gen4)
# parent_b: hybrid_hybrid_decreasing_pr_capybara_optimizatio_m1003_s1.py (gen4)
# born: 2026-05-29T23:53:19Z

"""Hybrid algorithm merging:
- Parent A: fractional Caputo derivative and Lanczos gamma for tree cost evaluation.
- Parent B: decreasing-rate pruning via evasion delta schedule (prune_probability) and Bayesian utilities.

Mathematical bridge:
Edge weights of the minimum‑cost tree are modulated by the evasion‑delta schedule:
    w_ij(t) = d_ij * (1 - p_prune(t))
where d_ij is the Euclidean distance and p_prune(t) = lam * exp(-alpha*t).
The temporal evolution of the total tree cost C(t) is then described by a
Caputo fractional derivative of order α (from Parent A), giving the algorithm
memory of past pruning decisions. The hybrid optimizer iteratively
1) updates edge weights via the schedule,
2) builds a minimum‑cost spanning tree,
3) evaluates its cost,
4) applies the Caputo derivative to obtain a fractional change,
5) prunes edges with decreasing probability.
"""

import math
import random
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Any, Hashable

import numpy as np

# ---------- Parent A components -------------------------------------------------

def gamma_lanczos(z: float) -> float:
    """Lanczos approximation of the Gamma function (valid for real z)."""
    _LANCZOS_G = 7
    _LANCZOS_C = np.array([
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ])
    if z < 0.5:
        return math.pi / (math.sin(math.pi * z) * gamma_lanczos(1 - z))
    z_minus_one = z - 1
    x = _LANCZOS_C[0]
    for i in range(1, len(_LANCZOS_C)):
        x += _LANCZOS_C[i] / (z_minus_one + i)
    t = z_minus_one + _LANCZOS_G + 0.5
    return math.sqrt(2 * math.pi) * t ** (z_minus_one + 0.5) * math.exp(-t) * x


def caputo_derivative(alpha: float, times: List[int], values: List[float]) -> List[float]:
    """
    Compute the Caputo fractional derivative of order `alpha` for a discrete
    time series `values` evaluated at integer times `times`.
    Returns a list of the same length containing the derivative at each time.
    """
    if not (0 < alpha < 1):
        raise ValueError("alpha must be in (0,1) for the implemented formula")
    n = len(times)
    deriv = [0.0] * n
    gamma_term = gamma_lanczos(1 - alpha)
    for i in range(1, n):
        integral = 0.0
        for j in range(i):
            dt = times[i] - times[j]
            integral += (values[i] - values[j]) / (dt ** alpha)
        deriv[i] = integral / gamma_term
    return deriv

# ---------- Parent B components -------------------------------------------------

def prune_probability(t: float, lam: float = 1.0, alpha: float = 0.2) -> float:
    """Decreasing‑rate pruning probability (evasion delta schedule)."""
    if t < 0 or lam < 0 or alpha < 0:
        raise ValueError('t, lam, and alpha must be non‑negative')
    return min(1.0, lam * math.exp(-alpha * t))


def prune_edges(edges: List[Hashable],
                t: float,
                lam: float = 1.0,
                alpha: float = 0.2,
                seed: int | str | None = None) -> List[Hashable]:
    """Randomly drop edges according to the pruning probability at time `t`."""
    rng = random.Random(seed)
    p = prune_probability(t, lam, alpha)
    return [e for e in edges if rng.random() >= p]


def euclidean_distance(a: Tuple[float, float],
                       b: Tuple[float, float]) -> float:
    """Standard Euclidean distance between two 2‑D points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])

# ---------- Hybrid utilities ----------------------------------------------------

class UnionFind:
    """Simple Union‑Find (Disjoint Set) structure for Kruskal's algorithm."""
    def __init__(self, elements: List[int]):
        self.parent = {e: e for e in elements}
        self.rank = {e: 0 for e in elements}

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


def weighted_edge(a: int,
                  b: int,
                  nodes: Dict[int, Tuple[float, float]],
                  t: float,
                  lam: float,
                  alpha: float) -> float:
    """
    Compute the hybrid edge weight:
        w_ij(t) = d_ij * (1 - p_prune(t))
    where d_ij is Euclidean distance and p_prune is the evasion‑delta schedule.
    """
    d = euclidean_distance(nodes[a], nodes[b])
    p = prune_probability(t, lam, alpha)
    return d * (1.0 - p)


def minimum_cost_spanning_tree(nodes: Dict[int, Tuple[float, float]],
                               edges: List[Tuple[int, int]],
                               root: int,
                               t: float,
                               lam: float,
                               alpha: float) -> Tuple[List[Tuple[int, int]], float]:
    """
    Kruskal's algorithm on hybrid‑weighted edges.
    Returns the list of edges in the tree and its total cost.
    """
    weighted = [(weighted_edge(u, v, nodes, t, lam, alpha), u, v) for u, v in edges]
    weighted.sort(key=lambda x: x[0])
    uf = UnionFind(list(nodes.keys()))
    tree_edges: List[Tuple[int, int]] = []
    total_cost = 0.0
    for w, u, v in weighted:
        if uf.union(u, v):
            tree_edges.append((u, v))
            total_cost += w
        if len(tree_edges) == len(nodes) - 1:
            break
    # Ensure connectivity to root (trivial for a spanning tree)
    return tree_edges, total_cost


def hybrid_optimize(nodes: Dict[int, Tuple[float, float]],
                    edges: List[Tuple[int, int]],
                    root: int,
                    timesteps: int,
                    lam: float = 1.0,
                    alpha_prune: float = 0.2,
                    alpha_caputo: float = 0.5,
                    seed: int | str | None = None) -> Dict[str, Any]:
    """
    Perform the hybrid optimization:
    1. At each integer time `t` compute a minimum‑cost tree with evasion‑modulated
       edge weights.
    2. Record the total cost C(t).
    3. After the full horizon, compute the Caputo fractional derivative of the
       cost series to obtain a memory‑aware change metric.
    4. Use the derivative magnitude to adaptively prune edges for the next run.
    Returns a dictionary with the final tree, cost series, and derivative series.
    """
    rng = random.Random(seed)
    cost_series: List[float] = []
    tree_series: List[List[Tuple[int, int]]] = []
    current_edges = edges.copy()

    for t in range(timesteps):
        # 1) Modulate and prune edges
        pruned = prune_edges(current_edges, t, lam, alpha_prune, seed=rng.randint(0, 2**31 - 1))
        # 2) Build minimum‑cost spanning tree
        tree, cost = minimum_cost_spanning_tree(nodes, pruned, root, t, lam, alpha_prune)
        cost_series.append(cost)
        tree_series.append(tree)
        # 3) Keep the pruned set for the next iteration (allows memory of removals)
        current_edges = pruned

    # 4) Fractional derivative of the cost series (Caputo)
    times = list(range(timesteps))
    derivative_series = caputo_derivative(alpha_caputo, times, cost_series)

    # 5) Adaptive final pruning based on derivative magnitude (larger derivative → more aggressive pruning)
    final_edges = edges.copy()
    for i, der in enumerate(derivative_series):
        if abs(der) > np.mean(np.abs(derivative_series)):
            # increase pruning intensity for later times
            final_edges = prune_edges(final_edges, i, lam * 1.5, alpha_prune, seed=rng.randint(0, 2**31 - 1))

    final_tree, final_cost = minimum_cost_spanning_tree(nodes, final_edges, root,
                                                        timesteps - 1, lam, alpha_prune)

    return {
        "final_tree": final_tree,
        "final_cost": final_cost,
        "cost_series": cost_series,
        "derivative_series": derivative_series,
        "tree_series": tree_series,
    }

# ---------- Smoke test ---------------------------------------------------------

if __name__ == "__main__":
    # Simple planar graph with 5 nodes
    nodes_example = {
        0: (0.0, 0.0),
        1: (1.0, 0.0),
        2: (0.0, 1.0),
        3: (1.0, 1.0),
        4: (0.5, 1.5),
    }
    # Fully connected edges (undirected)
    edges_example = [(i, j) for i in nodes_example for j in nodes_example if i < j]

    result = hybrid_optimize(
        nodes=nodes_example,
        edges=edges_example,
        root=0,
        timesteps=8,
        lam=0.9,
        alpha_prune=0.3,
        alpha_caputo=0.4,
        seed=42,
    )

    print("Final tree edges:", result["final_tree"])
    print("Final cost:", result["final_cost"])
    print("Cost series:", result["cost_series"])
    print("Caputo derivative series:", result["derivative_series"])