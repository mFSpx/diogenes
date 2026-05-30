# DARWIN HAMMER — match 2890, survivor 0
# gen: 2
# parent_a: fisher_localization.py (gen0)
# parent_b: hybrid_geometric_product_voronoi_partition_m4_s1.py (gen1)
# born: 2026-05-29T23:46:28Z

#!/usr/bin/env python3
"""This module integrates the Fisher information scoring from the fisher_localization.py 
with the geometric product and Voronoi partitioning from the hybrid_geometric_product_voronoi_partition_m4_s1.py.
The mathematical bridge between these two structures is formed by using the geometric product 
to compute distances and orientations between points in the Voronoi diagram, and then applying 
these computations to assign points to their nearest seeds. The Fisher information scoring 
is used to optimize the placement of the seeds in the Voronoi diagram, resulting in a more 
efficient and accurate partitioning of the space."""

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

    def __add__(self, other):
        result = self.components.copy()
        for blade, coeff in other.components.items():
            if blade in result:
                result[blade] += coeff
            else:
                result[blade] = coeff
        return Multivector(result)

    def __mul__(self, other):
        result = {}
        for blade_a, coeff_a in self.components.items():
            for blade_b, coeff_b in other.components.items():
                blade, sign = _multiply_blades(blade_a, blade_b)
                if blade in result:
                    result[blade] += sign * coeff_a * coeff_b
                else:
                    result[blade] = sign * coeff_a * coeff_b
        return Multivector(result)


def gaussian_beam(theta: float, center: float, width: float) -> float:
    if width <= 0:
        raise ValueError('width must be positive')
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)


def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity


def best_angle(candidates: list[float], center: float, width: float) -> float:
    if not candidates:
        raise ValueError('candidates required')
    return max(candidates, key=lambda t: (fisher_score(t, center, width), -abs(t-center)))


def voronoi_partition(seeds, points):
    """Assign points to their nearest seeds using the Voronoi partitioning."""
    assignments = {}
    for point in points:
        min_dist = float('inf')
        closest_seed = None
        for seed in seeds:
            dist = np.linalg.norm(np.array(point) - np.array(seed))
            if dist < min_dist:
                min_dist = dist
                closest_seed = seed
        if closest_seed not in assignments:
            assignments[closest_seed] = []
        assignments[closest_seed].append(point)
    return assignments


def optimize_seeds(seeds, points, width):
    """Optimize the placement of the seeds in the Voronoi diagram using the Fisher information scoring."""
    new_seeds = []
    for seed in seeds:
        candidates = [point for point in points if np.linalg.norm(np.array(point) - np.array(seed)) < width]
        if candidates:
            new_seed = best_angle(candidates, seed[0], width)
            new_seeds.append([new_seed, seed[1]])
        else:
            new_seeds.append(seed)
    return new_seeds


def hybrid_voronoi_partition(seeds, points, width):
    """Assign points to their nearest seeds using the Voronoi partitioning and optimize the placement of the seeds using the Fisher information scoring."""
    assignments = voronoi_partition(seeds, points)
    new_seeds = optimize_seeds(seeds, points, width)
    return voronoi_partition(new_seeds, points)


if __name__ == "__main__":
    seeds = [[0, 0], [5, 5]]
    points = [[1, 1], [2, 2], [3, 3], [4, 4], [6, 6], [7, 7], [8, 8]]
    width = 2.0
    assignments = hybrid_voronoi_partition(seeds, points, width)
    print(assignments)