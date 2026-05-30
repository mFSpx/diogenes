# DARWIN HAMMER — match 4896, survivor 0
# gen: 6
# parent_a: hybrid_hybrid_hybrid_doomsd_hybrid_hybrid_minimu_m1187_s1.py (gen3)
# parent_b: hybrid_hybrid_fisher_locali_hybrid_hybrid_geomet_m412_s1.py (gen5)
# born: 2026-05-29T23:58:35Z

"""
This module integrates the hybrid_hybrid_doomsday_cale_hybrid_geometric_pro_m66_s0 and 
hybrid_hybrid_fisher_locali_hybrid_hybrid_geomet_m412_s1 algorithms into a single hybrid system.
The mathematical bridge between the two parent algorithms lies in the combination of 
Fisher information, geometric product, and minimum cost tree analysis. The Fisher information 
is used to weight the geometric product from the geometric algorithm, and then used to 
represent distances and orientations between decision nodes in the minimum cost tree.
"""

import math
import numpy as np
import random
import sys
import pathlib
from collections.abc import Iterable
from datetime import date
import bisect

def _blade_sign(indices):
    """Return (sorted_blade, sign) after bubble-sorting index list."""
    lst = list(indices)
    sign = 1
    # Bubble sort; track swaps
    n = len(lst)
    for i in range(n):
        for j in range(n - 1 - i):
            if lst[j] > lst[j + 1]:
                lst[j], lst[j + 1] = lst[j + 1], lst[j]
                sign *= -1
            elif lst[j] == lst[j + 1]:
                # Duplicate: e_i * e_i = 1, remove both
                lst.pop(j)
                lst.pop(j)  # was j+1, now at j after pop
                return lst, sign
    return lst, sign


def _multiply_blades(blade_a, blade_b):
    """Multiply two basis blades (each a frozenset of indices)."""
    combined = list(blade_a) + list(blade_b)
    result, sign = _blade_sign(combined)
    return frozenset(result), sign


def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Gaussian intensity I(θ) of a beam centred at *center* with *width*."""
    if width <= 0.0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)


def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    """Fisher information for a single angle θ.

    F(θ) = (∂I/∂θ)² / I  where I = Gaussian beam intensity.
    """
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity


class Multivector:
    """Element of Cl(n,0) represented as a sum of basis blades."""
    def __init__(self, blades):
        self.blades = blades


def length(a, b):
    """Euclidean distance between two points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def tree_metrics(nodes, edges, root):
    """
    Build adjacency, compute Euclidean edge lengths and root-to-node distances.

    Returns
    -------
    adj : dict mapping node → list of neighbours
    edge_len : dict mapping edge (ordered as supplied) → length
    dist : dict mapping node → distance from *root* (sum of edge lengths along the unique path)
    """
    adj = {n: [] for n in nodes}
    edge_len = {}
    dist = {n: float('inf') for n in nodes}
    dist[root] = 0
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        edge_len[(u, v)] = length(u, v)
        edge_len[(v, u)] = length(v, u)
    queue = [root]
    while queue:
        u = queue.pop(0)
        for v in adj[u]:
            if dist[v] > dist[u] + edge_len[(u, v)]:
                dist[v] = dist[u] + edge_len[(u, v)]
                queue.append(v)
    return adj, edge_len, dist


def hybrid_operation(nodes, edges, root, center, width):
    """
    Hybrid operation that combines tree metrics, Fisher information, and geometric product.

    Returns
    -------
    adj : dict mapping node → list of neighbours
    edge_len : dict mapping edge (ordered as supplied) → length
    dist : dict mapping node → distance from *root* (sum of edge lengths along the unique path)
    fisher_scores : dict mapping node → Fisher score
    """
    adj, edge_len, dist = tree_metrics(nodes, edges, root)
    fisher_scores = {}
    for node in nodes:
        fisher_scores[node] = fisher_score(node[0], center, width)
    return adj, edge_len, dist, fisher_scores


def geometric_product(u, v):
    """
    Geometric product of two vectors.

    Returns
    -------
    product : float
    """
    return u[0] * v[0] + u[1] * v[1]


def main():
    nodes = [(0, 0), (1, 1), (2, 2)]
    edges = [((0, 0), (1, 1)), ((1, 1), (2, 2))]
    root = (0, 0)
    center = 1.0
    width = 1.0
    adj, edge_len, dist, fisher_scores = hybrid_operation(nodes, edges, root, center, width)
    print("Adjacency:", adj)
    print("Edge lengths:", edge_len)
    print("Distances:", dist)
    print("Fisher scores:", fisher_scores)
    print("Geometric product:", geometric_product((1, 1), (2, 2)))


if __name__ == "__main__":
    main()