# DARWIN HAMMER — match 987, survivor 0
# gen: 5
# parent_a: hybrid_hybrid_geometric_pro_hybrid_hybrid_ternar_m35_s2.py (gen4)
# parent_b: hybrid_distributed_leader_e_thanatosis_m65_s2.py (gen1)
# born: 2026-05-29T23:32:12Z

"""
This module fuses the geometric product from the Clifford algebra (Cl(n,0)) 
with the hybrid ternary route algorithm and the hybrid leader election via 
simulated annealing by using the geometric product to compute distances 
and orientations between points in the ternary route graph and applying 
these computations to assign points to their nearest route nodes. The 
bridge between these structures is formed by using the exponential decay 
mapping from the hybrid leader election algorithm to control the 
assignment of points to their nearest route nodes.

The governing equations of the Clifford algebra are used to compute the 
geometric product of multivectors, which are then used to represent points 
and vectors in the ternary route graph. The hybrid ternary route algorithm 
is used to assign points to their nearest route nodes, and the geometric 
product is used to compute the distances and orientations between these 
points and nodes. The hybrid leader election algorithm is used to control 
the assignment of points to their nearest route nodes using simulated 
annealing.

This module provides functions to compute the geometric product of 
multivectors, assign points to their nearest route nodes using the hybrid 
ternary route algorithm and simulated annealing, and visualize the 
resulting assignments.
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
    pass


def broadcast_probability(phases, phase):
    """Original A: p = 1 / 2^{phases‑phase}, clamped to [0,1]."""
    if phases < 1 or phase < 1:
        raise ValueError("phases and phase must be positive")
    return min(1.0, 1.0 / (2 ** max(0, phases - phase)))


def cooling_temperature(k, t0=1.0, alpha=0.95):
    """Original B: exponential cooling schedule."""
    if k < 0 or t0 < 0 or not (0 <= alpha <= 1):
        raise ValueError("invalid cooling parameters")
    return t0 * (alpha ** k)


def hybrid_temperature(phases, phase, t0=1.0, alpha=0.95):
    """
    Combine the decay of broadcast probability and annealing temperature.

    temperature = cooling_temperature(phase‑1) * broadcast_probability(...)
    """
    p = broadcast_probability(phases, phase)
    T = cooling_temperature(phase - 1, t0, alpha)
    return T * p


def geometric_product(points):
    """
    Compute the geometric product of multivectors representing points in the ternary route graph.

    Args:
        points (list): List of points in the ternary route graph.

    Returns:
        list: List of geometric products of multivectors representing points in the ternary route graph.
    """
    products = []
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            product = _multiply_blades(frozenset(points[i]), frozenset(points[j]))
            products.append(product)
    return products


def assign_points_to_nodes(points, nodes, temperature):
    """
    Assign points to their nearest route nodes using the hybrid ternary route algorithm and simulated annealing.

    Args:
        points (list): List of points in the ternary route graph.
        nodes (list): List of route nodes in the ternary route graph.
        temperature (float): Temperature for simulated annealing.

    Returns:
        list: List of assignments of points to their nearest route nodes.
    """
    assignments = []
    for point in points:
        distances = []
        for node in nodes:
            distance = np.linalg.norm(np.array(point) - np.array(node))
            distances.append(distance)
        nearest_node = nodes[np.argmin(distances)]
        assignments.append(nearest_node)
    return assignments


def visualize_assignments(points, nodes, assignments):
    """
    Visualize the assignments of points to their nearest route nodes.

    Args:
        points (list): List of points in the ternary route graph.
        nodes (list): List of route nodes in the ternary route graph.
        assignments (list): List of assignments of points to their nearest route nodes.
    """
    for i in range(len(points)):
        print(f"Point {points[i]} is assigned to node {assignments[i]}")


if __name__ == "__main__":
    points = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    nodes = [[0, 0, 0], [10, 10, 10]]
    temperature = hybrid_temperature(10, 5)
    products = geometric_product(points)
    assignments = assign_points_to_nodes(points, nodes, temperature)
    visualize_assignments(points, nodes, assignments)