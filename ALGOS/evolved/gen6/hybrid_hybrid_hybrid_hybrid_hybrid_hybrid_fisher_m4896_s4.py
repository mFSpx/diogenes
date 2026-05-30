# DARWIN HAMMER — match 4896, survivor 4
# gen: 6
# parent_a: hybrid_hybrid_hybrid_doomsd_hybrid_hybrid_minimu_m1187_s1.py (gen3)
# parent_b: hybrid_hybrid_fisher_locali_hybrid_hybrid_geomet_m412_s1.py (gen5)
# born: 2026-05-29T23:58:35Z

import math
import numpy as np
import random
import sys
import pathlib

class Multivector:
    """Element of Cl(n,0) represented as a sum of basis blades."""
    def __init__(self, blades):
        self.blades = blades

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


def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    """Fisher information for a single angle θ.

    F(θ) = (∂I/∂θ)² / I  where I = Gaussian beam intensity.
    """
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity


def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Gaussian intensity I(θ) of a beam centred at *center* with *width*."""
    if width <= 0.0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)


def geometric_product(multivector_a, multivector_b):
    """Compute the geometric product between two multivectors."""
    result = None
    for blade_a in multivector_a.blades:
        for blade_b in multivector_b.blades:
            product = _multiply_blades(blade_a, blade_b)
            if result is None:
                result = Multivector([product])
            else:
                result.blades.append(product)
    return result


def decision_hygiene_score(multivector, fisher_info):
    """Compute the decision hygiene score using the geometric product and Fisher information."""
    product = geometric_product(multivector, Multivector([fisher_info]))
    return np.sum([sign * np.prod([math.sqrt(len(blade)) for blade in blades]) for blades, sign in product.blades])


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
    dist = {n: 0 for n in nodes}
    for edge in edges:
        adj[edge[0]].append(edge[1])
        adj[edge[1]].append(edge[0])
        edge_len[tuple(edge)] = length(edge[0], edge[1])
    stack = [root]
    while stack:
        node = stack.pop()
        for neighbour in adj[node]:
            if dist[neighbour] == 0:
                dist[neighbour] = dist[node] + edge_len[(node, neighbour)]
                stack.append(neighbour)
    return adj, edge_len, dist


def length(a, b):
    """Euclidean distance between two points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def hybrid_operation(nodes, edges, root, multivector, fisher_info):
    """Perform the hybrid operation between the geometric product and Fisher information."""
    adj, edge_len, dist = tree_metrics(nodes, edges, root)
    hygiene_scores = []
    for node in nodes:
        multivector_node = Multivector([Multivector([frozenset([(node, i)])]) for i in range(len(edges))])
        hygiene_scores.append(decision_hygiene_score(multivector_node, fisher_info))
    return adj, edge_len, dist, hygiene_scores


if __name__ == "__main__":
    nodes = [0, 1, 2, 3]
    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    root = 0
    multivector = Multivector([frozenset([(0, 0)]), frozenset([(1, 1)])])
    fisher_info = fisher_score(0.5, 0.5, 1.0)
    adj, edge_len, dist, hygiene_scores = hybrid_operation(nodes, edges, root, multivector, fisher_info)
    print("Hybrid Operation Results:")
    print("Adjacency:", adj)
    print("Edge Lengths:", edge_len)
    print("Distances:", dist)
    print("Hygiene Scores:", hygiene_scores)