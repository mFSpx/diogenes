# DARWIN HAMMER — match 987, survivor 1
# gen: 5
# parent_a: hybrid_hybrid_geometric_pro_hybrid_hybrid_ternar_m35_s2.py (gen4)
# parent_b: hybrid_distributed_leader_e_thanatosis_m65_s2.py (gen1)
# born: 2026-05-29T23:32:12Z

"""
This module fuses the hybrid_hybrid_geometric_pro_hybrid_hybrid_ternar_m35_s2.py (Algorithm A) 
and hybrid_distributed_leader_e_thanatosis_m65_s2.py (Algorithm B) by recognizing that 
the geometric product can be used to compute distances and orientations between points 
in the ternary route graph, and the exponential decay functions can be used to control 
the simulated annealing process.

The mathematical bridge between these two structures is formed by using the geometric 
product to compute distances and orientations between points in the ternary route graph, 
and then applying these computations to assign points to their nearest route nodes. 
The exponential decay functions from Algorithm B are used to control the simulated 
annealing process, which is embedded into the maximal-independent-set construction.

The governing equations of the Clifford algebra are used to compute the geometric product 
of multivectors, which are then used to represent points and vectors in the ternary route graph.
"""

import math
import numpy as np
import random
import sys
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
    def __init__(self, blades):
        self.blades = blades

    def __mul__(self, other):
        result_blades = {}
        for blade_a, coeff_a in self.blades.items():
            for blade_b, coeff_b in other.blades.items():
                result_blade, sign = _multiply_blades(blade_a, blade_b)
                if result_blade in result_blades:
                    result_blades[result_blade] += coeff_a * coeff_b * sign
                else:
                    result_blades[result_blade] = coeff_a * coeff_b * sign
        return Multivector(result_blades)


def broadcast_probability(phases: int, phase: int) -> float:
    """Original A: p = 1 / 2^{phases‑phase}, clamped to [0,1]."""
    if phases < 1 or phase < 1:
        raise ValueError("phases and phase must be positive")
    return min(1.0, 1.0 / (2 ** max(0, phases - phase)))


def cooling_temperature(k: int, t0: float = 1.0, alpha: float = 0.95) -> float:
    """Original B: exponential cooling schedule."""
    if k < 0 or t0 < 0 or not (0 <= alpha <= 1):
        raise ValueError("invalid cooling parameters")
    return t0 * (alpha ** k)


def hybrid_temperature(phases: int, phase: int, t0: float = 1.0, alpha: float = 0.95) -> float:
    """
    Combine the decay of broadcast probability and annealing temperature.

    temperature = cooling_temperature(phase‑1) * broadcast_probability(...)
    """
    p = broadcast_probability(phases, phase)
    T = cooling_temperature(phase - 1, t0, alpha)
    return T * p


def geometric_product_distance(multivector_a, multivector_b):
    """Compute distance between two multivectors using geometric product."""
    product = multivector_a * multivector_b
    return np.sqrt(sum(abs(coeff) for coeff in product.blades.values()))


def hybrid_leader_election(graph, phases, t0, alpha):
    """Perform hybrid leader election using simulated annealing."""
    current_node = random.choice(list(graph.keys()))
    current_temperature = hybrid_temperature(phases, 1, t0, alpha)
    current_energy = len(graph[current_node])
    best_node = current_node
    best_energy = current_energy

    for phase in range(2, phases + 1):
        current_temperature = hybrid_temperature(phases, phase, t0, alpha)
        neighbors = graph[current_node]
        for neighbor in neighbors:
            neighbor_energy = len(graph[neighbor])
            delta_energy = neighbor_energy - current_energy
            probability = math.exp(-delta_energy / current_temperature)
            if random.random() < probability:
                current_node = neighbor
                current_energy = neighbor_energy
            if current_energy < best_energy:
                best_node = current_node
                best_energy = current_energy
    return best_node


def hybrid_ternary_route(graph, points, phases, t0, alpha):
    """Assign points to their nearest route nodes using hybrid ternary route algorithm."""
    route_nodes = list(graph.keys())
    assignments = {}
    for point in points:
        distances = {}
        for node in route_nodes:
            multivector_a = Multivector({frozenset([point])})
            multivector_b = Multivector({frozenset([node])})
            distance = geometric_product_distance(multivector_a, multivector_b)
            distances[node] = distance
        nearest_node = min(distances, key=distances.get)
        assignments[point] = nearest_node
    leader = hybrid_leader_election(graph, phases, t0, alpha)
    return assignments, leader


if __name__ == "__main__":
    graph = {
        'A': {'B', 'C'},
        'B': {'A', 'D', 'E'},
        'C': {'A', 'F'},
        'D': {'B'},
        'E': {'B', 'F'},
        'F': {'C', 'E'}
    }
    points = [1, 2, 3, 4, 5]
    phases = 5
    t0 = 1.0
    alpha = 0.95

    assignments, leader = hybrid_ternary_route(graph, points, phases, t0, alpha)
    print("Assignments:", assignments)
    print("Leader:", leader)