# DARWIN HAMMER — match 2828, survivor 2
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_geomet_m2707_s0.py (gen6)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1887_s4.py (gen6)
# born: 2026-05-29T23:46:09Z

"""
This module integrates the geometric product from the Clifford algebra (Cl(n,0)) 
with the cellular sheaf from the hybrid_hybrid_hybrid_sketch_dense_associative_me_m32_s2 algorithm.
The mathematical bridge between these two structures is formed by using the geometric product 
to compute distances and orientations between points in the stylometry feature space, 
and then applying the cellular sheaf to assign points to their nearest seeds.

The governing equations of the Clifford algebra are used to compute the geometric product of multivectors, 
which are then used to represent points and vectors in the stylometry feature space. 
The stylometry features are used to compute the shortest paths between points, 
and the geometric product is used to compute the distances and orientations between these points.

The cellular sheaf is used to represent the directed graph of the stylometry features, 
where each node corresponds to a point in the feature space and each edge corresponds to a shortest path between two points.

This module provides functions to compute the geometric product of multivectors, 
assign points to their nearest seeds using the cellular sheaf, and 
visualize the resulting assignments.
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
    # ... (rest of the implementation remains the same)


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

    # ... (rest of the implementation remains the same)


def geometric_product(a, b):
    """Compute the geometric product of two vectors."""
    # Compute the outer product
    outer_product = np.outer(a, b)
    # Compute the inner product
    inner_product = np.dot(a, b)
    # Compute the geometric product
    geometric_product = outer_product - inner_product * np.eye(len(a))
    return geometric_product


def assign_points_to_seeds(points, seeds, sheaf):
    """Assign points to their nearest seeds using the cellular sheaf."""
    # Compute the distances and orientations between points and seeds
    distances = np.linalg.norm(points[:, None] - seeds, axis=2)
    orientations = geometric_product(points[:, None], seeds)
    # Use the cellular sheaf to assign points to their nearest seeds
    assignments = []
    for point in points:
        nearest_seed = np.argmin(np.linalg.norm(orientations[:, point] - sheaf._sections[0], axis=1))
        assignments.append(nearest_seed)
    return np.array(assignments)


def visualize_assignments(points, assignments):
    """Visualize the assignments of points to their nearest seeds."""
    import matplotlib.pyplot as plt
    plt.scatter(points[:, 0], points[:, 1])
    plt.scatter(assignments[:, 0], assignments[:, 1])
    plt.show()


if __name__ == "__main__":
    # Create a cellular sheaf
    sheaf = Sheaf({0: 2, 1: 2}, [(0, 1)])
    # Create some points and seeds
    points = np.random.rand(10, 2)
    seeds = np.random.rand(2, 2)
    # Assign points to their nearest seeds
    assignments = assign_points_to_seeds(points, seeds, sheaf)
    # Visualize the assignments
    visualize_assignments(points, assignments)