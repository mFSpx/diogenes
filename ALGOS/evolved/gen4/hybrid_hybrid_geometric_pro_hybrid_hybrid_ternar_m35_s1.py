# DARWIN HAMMER — match 35, survivor 1
# gen: 4
# parent_a: hybrid_geometric_product_voronoi_partition_m4_s1.py (gen1)
# parent_b: hybrid_hybrid_ternary_route_hybrid_ternary_route_m98_s1.py (gen3)
# born: 2026-05-29T23:26:34Z

"""
This module integrates the geometric product from the Clifford algebra (Cl(n,0)) 
with the Voronoi partitioning of space and ternary routing. The mathematical bridge 
between these structures is formed by using the geometric product to compute 
distances and orientations between points in the Voronoi diagram, and then 
applying these computations to assign points to their nearest seeds. The 
ternary routing is used to find the shortest path between points. The governing 
equations of the Clifford algebra are used to compute the geometric product of 
multivectors, which are then used to represent points and vectors in the Voronoi 
diagram. The Voronoi partitioning is used to assign points to their nearest seeds, 
and the geometric product is used to compute the distances and orientations between 
these points and seeds. The ternary routing is used to find the shortest path 
between points.

Parent algorithms: 
- hybrid_geometric_product_voronoi_partition_m4_s1.py
- hybrid_hybrid_ternary_route_hybrid_ternary_route_m98_s1.py
"""

import math
import numpy as np
import random
import sys
import pathlib

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

    components: dict mapping frozenset(basis_indices) -> float coeffi
    """
    def __init__(self, components):
        self.components = components


# Geometry utilities
def euclidean_length(a, b):
    """Straight‑line distance between two 2‑D points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def build_length_matrix(nodes, edges):
    """
    Return a symmetric matrix L where L[i, j] = Euclidean length if (i, j) is an edge,
    otherwise 0. Also return the ordered list of edge index pairs matching the
    non‑zero entries of L (used for vectorised prior updates) and edge list.
    """
    idx_map = {name: i for i, name in enumerate(sorted(nodes))}
    n = len(nodes)
    L = np.zeros((n, n), dtype=float)
    edge_idx = []
    edge_list = []

    for a, b in edges:
        i, j = idx_map[a], idx_map[b]
        length = euclidean_length(nodes[a], nodes[b])
        L[i, j] = L[j, i] = length
        edge_idx.append((i, j))
        edge_list.append((a, b))

    return L, edge_idx, edge_list


# Hybrid function 1: Geometric Product Voronoi Partitioning
def geometric_product_voronoi(nodes, edges):
    """Compute the geometric product of multivectors representing points and vectors 
    in the Voronoi diagram, and then assign points to their nearest seeds.
    """
    L, edge_idx, edge_list = build_length_matrix(nodes, edges)
    multivector = Multivector({frozenset(): 1.0})
    for i in range(len(nodes)):
        for j in range(i+1, len(nodes)):
            distance = L[i, j]
            if distance != 0:
                blade = frozenset([i, j])
                sign = _blade_sign(blade)[1]
                multivector.components[blade] = sign * distance
    return multivector


# Hybrid function 2: Ternary Routing
def ternary_routing(nodes, edges):
    """Find the shortest path between points using ternary routing.
    """
    L, edge_idx, edge_list = build_length_matrix(nodes, edges)
    # Simple implementation of Dijkstra's algorithm
    def dijkstra(start):
        distances = {node: float('infinity') for node in nodes}
        distances[start] = 0
        unvisited_nodes = list(nodes.keys())
        while unvisited_nodes:
            current_node = min(unvisited_nodes, key=lambda node: distances[node])
            unvisited_nodes.remove(current_node)
            for neighbor, length in [(node, L[nodes.index(current_node), nodes.index(node)]) for node in nodes if L[nodes.index(current_node), nodes.index(node)] != 0]:
                distance = distances[current_node] + length
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
        return distances
    return {node: dijkstra(node) for node in nodes}


# Hybrid function 3: Geometric Product Ternary Routing
def geometric_product_ternary_routing(nodes, edges):
    """Compute the geometric product of multivectors representing points and vectors 
    in the ternary routing, and then find the shortest path between points.
    """
    L, edge_idx, edge_list = build_length_matrix(nodes, edges)
    multivector = Multivector({frozenset(): 1.0})
    for i in range(len(nodes)):
        for j in range(i+1, len(nodes)):
            distance = L[i, j]
            if distance != 0:
                blade = frozenset([i, j])
                sign = _blade_sign(blade)[1]
                multivector.components[blade] = sign * distance
    distances = ternary_routing(nodes, edges)
    return multivector, distances


if __name__ == "__main__":
    nodes = {'A': (0, 0), 'B': (3, 4), 'C': (6, 8)}
    edges = [('A', 'B'), ('B', 'C'), ('A', 'C')]
    print("Geometric Product Voronoi Partitioning:")
    print(geometric_product_voronoi(nodes, edges).components)
    print("Ternary Routing:")
    print(ternary_routing(nodes, edges))
    print("Geometric Product Ternary Routing:")
    multivector, distances = geometric_product_ternary_routing(nodes, edges)
    print("Multivector:", multivector.components)
    print("Distances:", distances)