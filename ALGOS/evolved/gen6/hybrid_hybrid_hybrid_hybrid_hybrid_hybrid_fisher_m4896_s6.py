# DARWIN HAMMER — match 4896, survivor 6
# gen: 6
# parent_a: hybrid_hybrid_hybrid_doomsd_hybrid_hybrid_minimu_m1187_s1.py (gen3)
# parent_b: hybrid_hybrid_fisher_locali_hybrid_hybrid_geomet_m412_s1.py (gen5)
# born: 2026-05-29T23:58:35Z

"""Hybrid Fusion of Geometric Multivector Algebra and Fisher‑Weighted Information

Parents
-------
* hybrid_hybrid_hybrid_doomsd_hybrid_hybrid_minimu_m1187_s1.py
  – Provides a Clifford (geometric) algebra implementation via blades,
    multivectors and Euclidean tree metrics.
* hybrid_hybrid_fisher_locali_hybrid_hybrid_geomet_m412_s1.py
  – Supplies Fisher information derived from a Gaussian beam model.

Mathematical Bridge
-------------------
The bridge is built on the observation that a *directed edge* of a
tree can be interpreted as a *direction* (an angle θ) in the plane.
The Fisher information `F(θ)` quantifies the sensitivity of a Gaussian
intensity to changes of that angle.  By weighting the geometric product
of the multivectors attached to the two incident nodes with `F(θ)` we
obtain a single scalar that simultaneously encodes:

1. the algebraic relationship of the two nodes (geometric product), and
2. the information‑theoretic relevance of their spatial orientation
   (Fisher weight).

The core routine `fisher_weighted_geometric_cost` traverses a rooted
tree, computes the geometric product of node‑pair multivectors, weights
each product by the Fisher score of the edge angle, and finally sums
the contributions.  This yields a unified cost metric that blends the
topologies of both parent algorithms.
"""

import math
import random
import sys
import pathlib
from collections import defaultdict
from typing import Dict, FrozenSet, Tuple, List

import numpy as np


def _blade_sign(indices: Tuple[int, ...]) -> Tuple[Tuple[int, ...], int]:
    """
    Sort a tuple of basis indices, cancel pairs of equal indices (e_i*e_i = 1),
    and return the sorted tuple together with the sign introduced by the swaps.
    """
    # Count occurrences to cancel even repeats
    counts: Dict[int, int] = {}
    for i in indices:
        counts[i] = counts.get(i, 0) + 1

    # Keep only odd occurrences (each contributes a single index)
    remaining = [i for i, c in counts.items() if c % 2 == 1]

    # Build a list preserving one copy of each odd‑count index in original order
    cleaned: List[int] = []
    seen = set()
    for i in indices:
        if i in remaining and i not in seen:
            cleaned.append(i)
            seen.add(i)

    # Bubble‑sort while tracking sign
    lst = list(cleaned)
    sign = 1
    n = len(lst)
    for i in range(n):
        for j in range(n - 1 - i):
            if lst[j] > lst[j + 1]:
                lst[j], lst[j + 1] = lst[j + 1], lst[j]
                sign *= -1
    return tuple(lst), sign


def _multiply_blades(blade_a: FrozenSet[int], blade_b: FrozenSet[int]) -> Tuple[FrozenSet[int], int]:
    """
    Geometric product of two basis blades represented as frozensets of indices.
    Returns (result_blade, sign).
    """
    combined = tuple(list(blade_a) + list(blade_b))
    sorted_inds, sign = _blade_sign(combined)
    return frozenset(sorted_inds), sign


class Multivector:
    """
    Simple multivector for Cl(n,0).  Internally stored as a dict
    mapping frozenset‑indexed blades to scalar coefficients.
    """

    def __init__(self, terms: Dict[FrozenSet[int], float] = None):
        self.terms: Dict[FrozenSet[int], float] = dict(terms) if terms else {}

    def __add__(self, other: "Multivector") -> "Multivector":
        result = Multivector(self.terms)
        for blade, coeff in other.terms.items():
            result.terms[blade] = result.terms.get(blade, 0.0) + coeff
        return result

    def __mul__(self, other: "Multivector") -> "Multivector":
        """Geometric product."""
        result_terms: Dict[FrozenSet[int], float] = defaultdict(float)
        for blade_a, coeff_a in self.terms.items():
            for blade_b, coeff_b in other.terms.items():
                blade_res, sign = _multiply_blades(blade_a, blade_b)
                result_terms[blade_res] += sign * coeff_a * coeff_b
        return Multivector(dict(result_terms))

    def scalar_part(self) -> float:
        """Return the coefficient of the scalar (empty) blade."""
        return self.terms.get(frozenset(), 0.0)

    def __repr__(self) -> str:
        if not self.terms:
            return "0"
        parts = []
        for blade, coeff in sorted(self.terms.items(), key=lambda x: (len(x[0]), x[0])):
            if blade:
                basis = "*".join(f"e{i}" for i in sorted(blade))
                parts.append(f"{coeff:.3g}*{basis}")
            else:
                parts.append(f"{coeff:.3g}")
        return " + ".join(parts)


def length(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Euclidean distance between two 2‑D points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def tree_metrics(
    nodes: List[int],
    edges: List[Tuple[int, int]],
    positions: Dict[int, Tuple[float, float]],
    root: int,
) -> Tuple[Dict[int, List[int]], Dict[Tuple[int, int], float], Dict[int, float]]:
    """
    Build adjacency, compute Euclidean edge lengths and root‑to‑node distances.

    Returns
    -------
    adj : dict mapping node → list of neighbours
    edge_len : dict mapping (u, v) ordered as supplied → Euclidean length
    dist : dict mapping node → distance from *root* (sum of edge lengths along the unique path)
    """
    adj = {n: [] for n in nodes}
    edge_len: Dict[Tuple[int, int], float] = {}
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        edge_len[(u, v)] = length(positions[u], positions[v])
        edge_len[(v, u)] = edge_len[(u, v)]

    # BFS to accumulate distances from root
    dist = {root: 0.0}
    visited = {root}
    queue = [root]
    while queue:
        cur = queue.pop(0)
        for nb in adj[cur]:
            if nb not in visited:
                visited.add(nb)
                dist[nb] = dist[cur] + edge_len[(cur, nb)]
                queue.append(nb)

    return adj, edge_len, dist


def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Gaussian intensity I(θ) of a beam centred at *center* with *width*."""
    if width <= 0.0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)


def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    """Fisher information for a single angle θ.

    F(θ) = (∂I/∂θ)² / I   where I = Gaussian beam intensity.
    """
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity


def angle_between(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Return the polar angle of the vector p2‑p1 in radians (0…2π)."""
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    return math.atan2(dy, dx) % (2 * math.pi)


def node_multivector(node_id: int, dim: int = 3) -> Multivector:
    """
    Construct a simple multivector for a node.
    For demonstration we embed the integer id into the scalar part and
    also create a single‑vector blade e_{id mod dim}.
    """
    scalar = float(node_id)
    blade = frozenset({node_id % dim})
    return Multivector({frozenset(): scalar, blade: 1.0})


def fisher_weighted_geometric_cost(
    nodes: List[int],
    edges: List[Tuple[int, int]],
    positions: Dict[int, Tuple[float, float]],
    root: int,
    fisher_center: float = 0.0,
    fisher_width: float = 0.5,
) -> float:
    """
    Compute a hybrid cost for a rooted tree.

    For each edge (u, v):
        * Compute the geometric product G_uv = M_u * M_v.
        * Extract its scalar part s_uv.
        * Determine the edge angle θ_uv.
        * Compute Fisher weight w_uv = F(θ_uv).
        * Accumulate w_uv * s_uv.

    The result mixes the algebraic interaction of the node multivectors
    with the information‑theoretic relevance of the spatial orientation.
    """
    _, edge_len, _ = tree_metrics(nodes, edges, positions, root)

    total = 0.0
    for u, v in edges:
        # Multivectors for the two incident nodes
        mu = node_multivector(u)
        mv = node_multivector(v)

        # Geometric product and scalar extraction
        gv = (mu * mv).scalar_part()

        # Edge orientation angle
        theta = angle_between(positions[u], positions[v])

        # Fisher weight
        w = fisher_score(theta, fisher_center, fisher_width)

        total += w * gv

    return total


def hybrid_tree_analysis(
    nodes: List[int],
    edges: List[Tuple[int, int]],
    positions: Dict[int, Tuple[float, float]],
    root: int,
) -> Tuple[Dict[int, float], float]:
    """
    Perform a combined analysis:
        1. Standard root‑to‑node distances (from parent A).
        2. Fisher‑weighted geometric cost (from parent B).

    Returns (distances, hybrid_cost).
    """
    _, _, dist = tree_metrics(nodes, edges, positions, root)
    cost = fisher_weighted_geometric_cost(nodes, edges, positions, root)
    return dist, cost


if __name__ == "__main__":
    # Simple smoke test with a triangle graph
    nodes = [0, 1, 2]
    edges = [(0, 1), (1, 2), (2, 0)]
    positions = {
        0: (0.0, 0.0),
        1: (1.0, 0.0),
        2: (0.5, math.sqrt(3) / 2),
    }
    root = 0

    distances, hybrid_cost = hybrid_tree_analysis(nodes, edges, positions, root)

    print("Root‑to‑node distances:", distances)
    print("Fisher‑weighted geometric cost:", hybrid_cost)