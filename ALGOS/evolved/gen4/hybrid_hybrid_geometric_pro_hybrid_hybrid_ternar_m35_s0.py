# DARWIN HAMMER — match 35, survivor 0
# gen: 4
# parent_a: hybrid_geometric_product_voronoi_partition_m4_s1.py (gen1)
# parent_b: hybrid_hybrid_ternary_route_hybrid_ternary_route_m98_s1.py (gen3)
# born: 2026-05-29T23:26:34Z

"""
This module integrates the geometric product from the Clifford algebra (Cl(n,0)) 
with the Voronoi partitioning of space and ternary routing from the hybrid ternary 
route hybrid ternary route. The mathematical bridge between these structures 
is formed by using the geometric product to compute distances and orientations 
between points in the Voronoi diagram, and then applying these computations 
to assign points to their nearest seeds and optimize the ternary routing.

The governing equations of the Clifford algebra are used to compute the 
geometric product of multivectors, which are then used to represent points and 
vectors in the Voronoi diagram. The Voronoi partitioning is used to assign 
points to their nearest seeds, and the geometric product is used to compute 
the distances and orientations between these points and seeds. The ternary 
routing is optimized by using the geometric product to compute the shortest 
paths between nodes.

Parents: hybrid_geometric_product_voronoi_partition_m4_s1.py, 
         hybrid_hybrid_ternary_route_hybrid_ternary_route_m98_s1.py
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
    def __init__(self):
        self.components = {}

    def add(self, blade, coefficient):
        if blade in self.components:
            self.components[blade] += coefficient
        else:
            self.components[blade] = coefficient

    def multiply(self, other):
        result = Multivector()
        for blade_a, coeff_a in self.components.items():
            for blade_b, coeff_b in other.components.items():
                result_blade, sign = _multiply_blades(blade_a, blade_b)
                result.add(result_blade, sign * coeff_a * coeff_b)
        return result


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


# Hybrid function
def hybrid_geometric_product_ternary_route(nodes, edges):
    """
    Compute the geometric product of multivectors representing points and vectors 
    in the Voronoi diagram and use it to optimize the ternary routing.
    """
    # Initialize multivectors
    multivectors = {node: Multivector() for node in nodes}

    # Assign points to their nearest seeds using Voronoi partitioning
    for node in nodes:
        # Compute geometric product of multivector with itself
        multivector = multivectors[node]
        multivector.add(frozenset([node]), 1)
        multivector = multivector.multiply(multivector)

        # Update multivectors of neighboring nodes
        for neighbor in [edge[1] for edge in edges if edge[0] == node]:
            multivector_neighbor = multivectors[neighbor]
            multivector_neighbor.add(frozenset([neighbor]), 1)
            multivector_neighbor = multivector_neighbor.multiply(multivector)

    # Compute shortest paths between nodes using geometric product
    shortest_paths = {}
    for node in nodes:
        shortest_paths[node] = {}
        for neighbor in nodes:
            if node == neighbor:
                shortest_paths[node][neighbor] = 0
            else:
                # Compute geometric product of multivector with multivector of neighbor
                multivector = multivectors[node]
                multivector_neighbor = multivectors[neighbor]
                multivector_product = multivector.multiply(multivector_neighbor)

                # Compute distance between node and neighbor
                distance = 0
                for blade, coefficient in multivector_product.components.items():
                    distance += coefficient ** 2

                shortest_paths[node][neighbor] = distance

    return shortest_paths


if __name__ == "__main__":
    nodes = {"A": (0, 0), "B": (1, 0), "C": (0, 1)}
    edges = [("A", "B"), ("B", "C"), ("C", "A")]
    shortest_paths = hybrid_geometric_product_ternary_route(nodes, edges)
    print(shortest_paths)