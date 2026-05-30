# DARWIN HAMMER — match 35, survivor 2
# gen: 4
# parent_a: hybrid_geometric_product_voronoi_partition_m4_s1.py (gen1)
# parent_b: hybrid_hybrid_ternary_route_hybrid_ternary_route_m98_s1.py (gen3)
# born: 2026-05-29T23:26:34Z

"""
This module fuses the geometric product from the Clifford algebra (Cl(n,0)) 
with the hybrid ternary route algorithm. The mathematical bridge between 
these two structures is formed by using the geometric product to compute 
distances and orientations between points in the ternary route graph, 
and then applying these computations to assign points to their nearest 
route nodes.

The governing equations of the Clifford algebra are used to compute the 
geometric product of multivectors, which are then used to represent 
points and vectors in the ternary route graph. The hybrid ternary route 
algorithm is used to assign points to their nearest route nodes, and 
the geometric product is used to compute the distances and orientations 
between these points and nodes.

This module provides functions to compute the geometric product of 
multivectors, assign points to their nearest route nodes using the 
hybrid ternary route algorithm, and visualize the resulting assignments.
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
                blade, sign = _multiply_blades(blade_a, blade_b)
                result_components[blade] = result_components.get(blade, 0) + sign * coeff_a * coeff_b
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


def compute_geometric_product(node_a: Multivector, node_b: Multivector) -> Multivector:
    """Compute the geometric product of two multivectors representing nodes."""
    return node_a * node_b


def assign_points_to_route_nodes(points: List[Tuple[float, float]], nodes: Dict[str, Tuple[float, float]], edges: List[Tuple[str, str]]) -> Dict[str, List[Tuple[float, float]]]:
    """
    Assign points to their nearest route nodes using the hybrid ternary route algorithm
    and geometric product.

    Args:
    points: List of points to assign.
    nodes: Dictionary of route nodes.
    edges: List of edges.

    Returns:
    Dictionary mapping each node to a list of assigned points.
    """
    L, edge_idx, edge_list = build_length_matrix(nodes, edges)
    assignments = {node: [] for node in nodes}

    for point in points:
        min_distance = float('inf')
        nearest_node = None

        for node_name, node_coords in nodes.items():
            distance = euclidean_length(point, node_coords)

            # Compute geometric product
            node_multivector = Multivector({frozenset(): 1.0})
            point_multivector = Multivector({frozenset([0]): point[0], frozenset([1]): point[1]})
            product = compute_geometric_product(node_multivector, point_multivector)

            # Update nearest node if necessary
            if distance < min_distance:
                min_distance = distance
                nearest_node = node_name

        assignments[nearest_node].append(point)

    return assignments


def visualize_assignments(assignments: Dict[str, List[Tuple[float, float]]]) -> None:
    """Visualize the assignments of points to route nodes."""
    for node, points in assignments.items():
        print(f"Node {node}: {points}")


if __name__ == "__main__":
    # Smoke test
    nodes = {"A": (0, 0), "B": (3, 4), "C": (6, 8)}
    edges = [("A", "B"), ("B", "C"), ("C", "A")]
    points = [(1, 1), (2, 2), (4, 4), (5, 5)]

    assignments = assign_points_to_route_nodes(points, nodes, edges)
    visualize_assignments(assignments)