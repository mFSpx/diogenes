# DARWIN HAMMER — match 2828, survivor 1
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_geomet_m2707_s0.py (gen6)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1887_s4.py (gen6)
# born: 2026-05-29T23:46:09Z

"""
This module integrates the geometric product from the Clifford algebra (Cl(n,0)) 
with the cellular sheaf structure on a directed graph from the hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1887_s4 algorithm.
The mathematical bridge between these two structures is formed by using the geometric product 
to compute distances and orientations between points in the stylometry feature space, 
and then applying these computations to assign points to their nearest seeds in the sheaf.
The governing equations of the Clifford algebra are used to compute the geometric product of multivectors, 
which are then used to represent points and vectors in the stylometry feature space.
The sheaf structure provides a way to organize and relate these points and vectors, 
and to compute the shortest paths between them.
"""

import math
import numpy as np
import random
import sys
import pathlib
from collections import Counter
from dataclasses import dataclass, field
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
    return _blade_sign(combined)


class Sheaf:
    """
    Cellular sheaf on a directed graph.

    Each node ``n`` carries a vector space ℝ^{dim(n)}.
    Each directed edge ``(u, v)`` carries a pair of linear restriction maps

        src_map : ℝ^{dim(u)} → ℝ^{dim(e)}
        dst_map : ℝ^{dim(v)} → ℝ^{dim(e)}.

    A *section* is an assignment of a vector to every node that is compatible
    with all restriction maps, i.e. for every edge (u, v)

        src_map @ s[u]  ≈  dst_map @ s[v].

    The class provides utilities to set maps, set sections and to project
    a raw query vector onto the nearest compatible section.
    """

    def __init__(self, node_dims: Dict[Any, int], edges: List[Tuple[Any, Any]]):
        self.node_dims = dict(node_dims)
        self.edges = list(edges)
        self._restrictions: Dict[Tuple[Any, Any], Tuple[np.ndarray, np.ndarray]] = {}
        self._sections: Dict[Any, np.ndarray] = {}

    def set_restriction(
        self,
        edge: Tuple[Any, Any],
        src_map: np.ndarray,
        dst_map: np.ndarray,
    ) -> None:
        """Register the two restriction matrices for a directed edge."""
        u, v = edge
        if u not in self.node_dims or v not in self.node_dims:
            raise KeyError(f"Edge {edge} refers to undefined nodes.")
        if src_map.shape[1] != self.node_dims[u]:
            raise ValueError("src_map column dimension must match dim(node u)")
        if dst_map.shape[1] != self.node_dims[v]:
            raise ValueError("dst_map column dimension must match dim(node v)")
        if src_map.shape[0] != dst_map.shape[0]:
            raise ValueError("src_map and dst_map must have the same row dimension")
        self._restrictions[edge] = (np.asarray(src_map, dtype=float), np.asarray(dst_map, dtype=float))

    def geometric_product(self, blade_a, blade_b):
        """Compute the geometric product of two blades."""
        return _multiply_blades(blade_a, blade_b)

    def assign_points_to_seeds(self, points, seeds):
        """Assign points to their nearest seeds in the sheaf."""
        assignments = {}
        for point in points:
            min_distance = float('inf')
            nearest_seed = None
            for seed in seeds:
                distance = np.linalg.norm(np.array(point) - np.array(seed))
                if distance < min_distance:
                    min_distance = distance
                    nearest_seed = seed
            assignments[point] = nearest_seed
        return assignments

    def compute_shortest_paths(self, points):
        """Compute the shortest paths between points in the sheaf."""
        paths = {}
        for point1 in points:
            for point2 in points:
                if point1 != point2:
                    path = []
                    current_point = point1
                    while current_point != point2:
                        next_point = min(points, key=lambda x: np.linalg.norm(np.array(x) - np.array(current_point)))
                        path.append(next_point)
                        current_point = next_point
                    paths[(point1, point2)] = path
        return paths


def main():
    sheaf = Sheaf({1: 2, 2: 3, 3: 2}, [(1, 2), (2, 3), (3, 1)])
    sheaf.set_restriction((1, 2), np.array([[1, 0], [0, 1]]), np.array([[1, 0, 0], [0, 1, 0]]))
    sheaf.set_restriction((2, 3), np.array([[1, 0, 0], [0, 1, 0]]), np.array([[1, 0], [0, 1]]))
    sheaf.set_restriction((3, 1), np.array([[1, 0], [0, 1]]), np.array([[1, 0, 0], [0, 1, 0]]))
    points = [(1, 2), (3, 4), (5, 6)]
    seeds = [(1, 1), (3, 3), (5, 5)]
    assignments = sheaf.assign_points_to_seeds(points, seeds)
    paths = sheaf.compute_shortest_paths(points)
    print(assignments)
    print(paths)


if __name__ == "__main__":
    main()