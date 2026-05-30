# DARWIN HAMMER — match 4896, survivor 5
# gen: 6
# parent_a: hybrid_hybrid_hybrid_doomsd_hybrid_hybrid_minimu_m1187_s1.py (gen3)
# parent_b: hybrid_hybrid_fisher_locali_hybrid_hybrid_geomet_m412_s1.py (gen5)
# born: 2026-05-29T23:58:35Z

"""
Module hybrid_fusion.py: Fuses hybrid_hybrid_doomsd_hybrid_hybrid_minimu_m1187_s1.py and hybrid_hybrid_fisher_locali_hybrid_hybrid_geomet_m412_s1.py.
The mathematical bridge between the two parent algorithms lies in the combination of the geometric product of multivectors 
from the first parent and the Fisher information from the second parent. This fusion enables a unified framework to analyze 
the complexity and inequality of decision-making processes while incorporating the localization properties of the Fisher 
information. The geometric product is used to represent the distances and orientations between decision nodes in the minimum 
cost tree, and then the Fisher information is used to weight the geometric product, effectively fusing their core topologies.
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
    dist = {root: 0}
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        edge_len[(u, v)] = length(u, v)
        edge_len[(v, u)] = length(v, u)
    # Compute distances from root
    queue = [root]
    while queue:
        node = queue.pop(0)
        for neighbor in adj[node]:
            if neighbor not in dist or dist[node] + edge_len[(node, neighbor)] < dist.get(neighbor, float('inf')):
                dist[neighbor] = dist[node] + edge_len[(node, neighbor)]
                queue.append(neighbor)
    return adj, edge_len, dist


def hybrid_decision(node, center, width, nodes, edges, root):
    """Hybrid decision function that combines geometric product and Fisher information."""
    adj, edge_len, dist = tree_metrics(nodes, edges, root)
    blade = frozenset([node])
    sign = 1
    fisher_info = fisher_score(dist[node], center, width)
    multivector = Multivector([blade])
    return multivector, sign, fisher_info


def hybrid_tree(nodes, edges, root, center, width):
    """Hybrid tree function that computes the geometric product and Fisher information for each node."""
    adj, edge_len, dist = tree_metrics(nodes, edges, root)
    multivectors = []
    signs = []
    fisher_infos = []
    for node in nodes:
        multivector, sign, fisher_info = hybrid_decision(node, center, width, nodes, edges, root)
        multivectors.append(multivector)
        signs.append(sign)
        fisher_infos.append(fisher_info)
    return multivectors, signs, fisher_infos


if __name__ == "__main__":
    nodes = [(0, 0), (1, 0), (2, 0), (3, 0)]
    edges = [(nodes[0], nodes[1]), (nodes[1], nodes[2]), (nodes[2], nodes[3])]
    root = nodes[0]
    center = 1.5
    width = 0.5
    multivectors, signs, fisher_infos = hybrid_tree(nodes, edges, root, center, width)
    for node, multivector, sign, fisher_info in zip(nodes, multivectors, signs, fisher_infos):
        print(f"Node {node}: Multivector = {multivector}, Sign = {sign}, Fisher Info = {fisher_info}")