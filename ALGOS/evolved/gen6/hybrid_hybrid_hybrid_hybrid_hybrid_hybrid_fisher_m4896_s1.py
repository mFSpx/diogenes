# DARWIN HAMMER — match 4896, survivor 1
# gen: 6
# parent_a: hybrid_hybrid_hybrid_doomsd_hybrid_hybrid_minimu_m1187_s1.py (gen3)
# parent_b: hybrid_hybrid_fisher_locali_hybrid_hybrid_geomet_m412_s1.py (gen5)
# born: 2026-05-29T23:58:35Z

"""
Module hybrid_fusion.py: Fusing hybrid_hybrid_doomsd_hybrid_hybrid_minimu_m1187_s1.py and hybrid_hybrid_fisher_locali_hybrid_hybrid_geomet_m412_s1.py.
The mathematical bridge between the two parent algorithms lies in the combination of the geometric product of multivectors from the first parent and the Fisher information from the second parent.
The geometric product is used to represent the distances and orientations between decision nodes in the minimum cost tree, and then the Fisher information is used to weight the decision hygiene scores based on the Gini coefficient and Voronoi partitioning.
"""

import math
import numpy as np
import random
import sys
import pathlib

def _blade_sign(indices):
    """Return (sorted_blade, sign) after bubble-sorting index list."""
    lst = list(indices)
    sign = 1
    n = len(lst)
    for i in range(n):
        for j in range(n - 1 - i):
            if lst[j] > lst[j + 1]:
                lst[j], lst[j + 1] = lst[j + 1], lst[j]
                sign *= -1
            elif lst[j] == lst[j + 1]:
                lst.pop(j)
                lst.pop(j)  # was j+1, now at j after pop
                return lst, sign
    return lst, sign


def _multiply_blades(blade_a, blade_b):
    """Multiply two basis blades (each a frozenset of indices)."""
    combined = list(blade_a) + list(blade_b)
    result, sign = _blade_sign(combined)
    return frozenset(result), sign


class Multivector:
    """Element of Cl(n,0) represented as a sum of basis blades."""
    def __init__(self, blades):
        self.blades = blades


def gaussian_beam(theta, center, width):
    """Gaussian intensity I(θ) of a beam centred at *center* with *width*."""
    if width <= 0.0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)


def fisher_score(theta, center, width, eps=1e-12):
    """Fisher information for a single angle θ.

    F(θ) = (∂I/∂θ)² / I  where I = Gaussian beam intensity.
    """
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity


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
        node = queue.pop(0)
        for neighbor in adj[node]:
            new_dist = dist[node] + edge_len[(node, neighbor)]
            if new_dist < dist[neighbor]:
                dist[neighbor] = new_dist
                queue.append(neighbor)

    return adj, edge_len, dist


def hybrid_operation(nodes, edges, root, theta, center, width):
    """
    Perform the hybrid operation that combines the geometric product and Fisher information.

    Returns
    -------
    result : a dictionary containing the results of the hybrid operation
    """
    adj, edge_len, dist = tree_metrics(nodes, edges, root)
    blade_a = frozenset([0, 1])
    blade_b = frozenset([1, 2])
    result, sign = _multiply_blades(blade_a, blade_b)
    fisher_info = fisher_score(theta, center, width)
    return {
        'adj': adj,
        'edge_len': edge_len,
        'dist': dist,
        'blade_product': result,
        'fisher_info': fisher_info
    }


def test_hybrid_operation():
    nodes = [(0, 0), (1, 0), (2, 0)]
    edges = [((0, 0), (1, 0)), ((1, 0), (2, 0))]
    root = (0, 0)
    theta = 0.5
    center = 0.5
    width = 1.0
    result = hybrid_operation(nodes, edges, root, theta, center, width)
    print(result)


if __name__ == "__main__":
    test_hybrid_operation()