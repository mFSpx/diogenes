# DARWIN HAMMER — match 2828, survivor 3
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_geomet_m2707_s0.py (gen6)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1887_s4.py (gen6)
# born: 2026-05-29T23:46:09Z

"""
This module integrates the geometric product from the Clifford algebra (Cl(n,0)) 
with the cellular sheaf structure from the hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1887_s4 algorithm.
The mathematical bridge between these two structures is formed by using the geometric product 
to compute distances and orientations between points in the sheaf's vector spaces, 
and then applying these computations to assign points to their nearest sections.

The governing equations of the Clifford algebra are used to compute the geometric product of multivectors, 
which are then used to represent points and vectors in the sheaf's vector spaces. 
The sheaf's restriction maps are used to compute the shortest paths between points, 
and the geometric product is used to compute the distances and orientations between these points.
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
    return frozenset(_blade_sign(combined)[0]), _blade_sign(combined)[1]


class HybridSheaf:
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

    def compute_geometric_product(self, multivector_a, multivector_b):
        """Compute the geometric product of two multivectors."""
        result = 0
        for blade_a in multivector_a:
            for blade_b in multivector_b:
                result_blade, sign = _multiply_blades(blade_a, blade_b)
                result += sign * result_blade
        return result

    def assign_point_to_section(self, point, section):
        """Assign a point to its nearest section using the geometric product."""
        # Compute the geometric product of the point and section
        gp = self.compute_geometric_product(point, section)
        # Compute the distance between the point and section
        distance = np.linalg.norm(gp)
        return distance

    def project_point_onto_section(self, point, section):
        """Project a point onto its nearest section."""
        # Compute the geometric product of the point and section
        gp = self.compute_geometric_product(point, section)
        # Compute the projection of the point onto the section
        projection = gp / np.linalg.norm(gp)
        return projection


if __name__ == "__main__":
    # Create a sample sheaf
    node_dims = {"A": 2, "B": 2, "C": 2}
    edges = [("A", "B"), ("B", "C"), ("A", "C")]
    sheaf = HybridSheaf(node_dims, edges)

    # Set restriction maps
    sheaf.set_restriction(("A", "B"), np.array([[1, 0], [0, 1]]), np.array([[1, 0], [0, 1]]))
    sheaf.set_restriction(("B", "C"), np.array([[1, 0], [0, 1]]), np.array([[1, 0], [0, 1]]))
    sheaf.set_restriction(("A", "C"), np.array([[1, 0], [0, 1]]), np.array([[1, 0], [0, 1]]))

    # Create sample multivectors
    multivector_a = [{1}, {2}]
    multivector_b = [{1, 2}]

    # Compute geometric product
    gp = sheaf.compute_geometric_product(multivector_a, multivector_b)
    print(gp)

    # Assign point to section
    point = np.array([1, 2])
    section = np.array([3, 4])
    distance = sheaf.assign_point_to_section(point, section)
    print(distance)

    # Project point onto section
    projection = sheaf.project_point_onto_section(point, section)
    print(projection)