# DARWIN HAMMER — match 35, survivor 3
# gen: 4
# parent_a: hybrid_geometric_product_voronoi_partition_m4_s1.py (gen1)
# parent_b: hybrid_hybrid_ternary_route_hybrid_ternary_route_m98_s1.py (gen3)
# born: 2026-05-29T23:26:34Z

"""
This module integrates the geometric product from the Clifford algebra (Cl(n,0)) 
with the hybrid ternary route algorithm. The mathematical bridge between these two 
structures is formed by using the geometric product to compute distances and 
orientations between points in the ternary route, and then applying these 
computations to assign points to their nearest seeds.

The governing equations of the Clifford algebra are used to compute the 
geometric product of multivectors, which are then used to represent points and 
vectors in the ternary route. The hybrid ternary route algorithm is used to 
compute the shortest paths between points, and the geometric product is used to 
compute the distances and orientations between these points.

This module provides functions to compute the geometric product of multivectors, 
assign points to their nearest seeds using the ternary route, and 
visualize the resulting assignments.
"""

import math
import numpy as np
import random
import sys
import pathlib
from typing import Any, Dict, List, Tuple

# Core blade arithmetic helpers
def _blade_sign(indices):
    """Return (sorted_blade, sign) after bubble-sorting index list.

    Each transposition of adjacent indices that are out of order flips the
    sign (anti-commutativity).  Duplicate indices cancel (e_i^2 = 1 → they
    annihilate and contribute +1 to the sign, but the index disappears).
    """
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
    """Multiply two basis blades (each a frozenset of indices).

    Returns (result_blade_frozenset, sign).
    """
    combined = list(blade_a) + list(blade_b)
    result, sign = _blade_sign(combined)
    return frozenset(result), sign


# Multivector
class Multivector:
    """Element of Cl(n,0) represented as a sum of basis blades.

    components: dict mapping frozenset(basis_indices) -> float coefficient
    """

    def __init__(self, components):
        self.components = components

    def __mul__(self, other):
        result_components = {}
        for blade_a, coeff_a in self.components.items():
            for blade_b, coeff_b in other.components.items():
                result_blade, sign = _multiply_blades(blade_a, blade_b)
                result_components[result_blade] = result_components.get(result_blade, 0) + sign * coeff_a * coeff_b
        return Multivector(result_components)


def euclidean_length(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Straight‑line distance between two 2‑D points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def build_length_matrix(
    nodes: Dict[str, Tuple[float, float]], edges: List[Tuple[str, str]]
) -> Tuple[np.ndarray, List[Tuple[int, int]], List[Tuple[str, str]]]:
    """
    Return a symmetric matrix L where L[i, j] = Euclidean length if (i, j) is an edge,
    otherwise 0. Also return the ordered list of edge index pairs matching the
    non‑zero entries of L (used for vectorised prior updates) and edge list.
    """
    idx_map = {name: i for i, name in enumerate(sorted(nodes))}
    n = len(nodes)
    L = np.zeros((n, n), dtype=float)
    edge_idx: List[Tuple[int, int]] = []
    edge_list: List[Tuple[str, str]] = []

    for a, b in edges:
        i, j = idx_map[a], idx_map[b]
        length = euclidean_length(nodes[a], nodes[b])
        L[i, j] = L[j, i] = length
        edge_idx.append((i, j))
        edge_list.append((a, b))

    return L, edge_idx, edge_list


def hybrid_ternary_route_geometric_product(
    nodes: Dict[str, Tuple[float, float]], 
    edges: List[Tuple[str, str]], 
    start_node: str, 
    end_node: str
) -> Tuple[List[str], Multivector]:
    """
    Compute the shortest path between two nodes in a graph using the hybrid ternary route algorithm,
    and represent the path as a multivector using the geometric product.
    """
    L, edge_idx, edge_list = build_length_matrix(nodes, edges)
    idx_map = {name: i for i, name in enumerate(sorted(nodes))}
    n = len(nodes)

    # Initialize distances and previous nodes
    distances = np.full(n, np.inf)
    previous = np.full(n, -1, dtype=int)
    distances[idx_map[start_node]] = 0

    # Ternary route algorithm
    for _ in range(n):
        for i, j in edge_idx:
            if distances[i] + L[i, j] < distances[j]:
                distances[j] = distances[i] + L[i, j]
                previous[j] = i

    # Build path
    path = []
    current = idx_map[end_node]
    while current != -1:
        path.append(list(nodes.keys())[current])
        current = previous[current]
    path.reverse()

    # Represent path as multivector
    multivector_components = {}
    for i in range(len(path) - 1):
        node_a = path[i]
        node_b = path[i + 1]
        vector = (nodes[node_b][0] - nodes[node_a][0], nodes[node_b][1] - nodes[node_a][1])
        multivector_components[frozenset()]: Multivector({frozenset(): 1, frozenset((0,)): vector[0], frozenset((1,)): vector[1]})
    
    return path, Multivector(multivector_components)


def geometric_product_distance(multivector: Multivector) -> float:
    """
    Compute the Euclidean distance represented by a multivector.
    """
    distance = 0
    for blade, coeff in multivector.components.items():
        if len(blade) == 0:
            continue
        elif len(blade) == 1:
            distance += coeff ** 2
    return math.sqrt(distance)


if __name__ == "__main__":
    nodes = {
        'A': (0, 0),
        'B': (3, 0),
        'C': (3, 4),
        'D': (0, 4)
    }
    edges = [('A', 'B'), ('B', 'C'), ('C', 'D'), ('D', 'A')]
    start_node = 'A'
    end_node = 'C'

    path, multivector = hybrid_ternary_route_geometric_product(nodes, edges, start_node, end_node)
    print("Shortest path:", path)
    print("Geometric product distance:", geometric_product_distance(multivector))