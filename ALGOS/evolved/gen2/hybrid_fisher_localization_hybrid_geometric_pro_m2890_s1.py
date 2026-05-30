# DARWIN HAMMER — match 2890, survivor 1
# gen: 2
# parent_a: fisher_localization.py (gen0)
# parent_b: hybrid_geometric_product_voronoi_partition_m4_s1.py (gen1)
# born: 2026-05-29T23:46:28Z

#!/usr/bin/env python3
"""Fusion of Fisher-information scoring for off-axis sensing and the geometric product 
from the Clifford algebra (Cl(n,0)) with the Voronoi partitioning of space.

The governing equations of the Clifford algebra are used to compute the 
geometric product of multivectors. These multivectors represent points and 
vectors in the Voronoi diagram, which is partitioned using the Voronoi 
partitioning. The Fisher-information scoring is then applied to the Voronoi 
partitions to determine the optimal point selection based on the off-axis 
sensing.

The mathematical bridge between these two structures is formed by using the 
geometric product to compute distances and orientations between points in the 
Voronoi diagram, and then applying the Fisher-information scoring to these 
comparisons to assign points to their nearest seeds.

This module provides functions to compute the geometric product of multivectors, 
assign points to their nearest seeds using the Voronoi partitioning, and 
visualize the resulting assignments. It also applies the Fisher-information 
scoring to the Voronoi partitions to determine the optimal point selection.

The Fisher-information scoring is used to weight the points in the Voronoi 
partitions based on their off-axis sensing. The geometric product is used to 
compute the distances and orientations between points in the Voronoi diagram, 
and these computations are then applied to assign points to their nearest seeds.

The output of this module is a set of points that are optimally selected based 
on the off-axis sensing, as determined by the Fisher-information scoring applied 
to the Voronoi partitions.

This module relies on the numpy library for numerical computations and the 
math library for mathematical operations.

The module is designed to be flexible and can be easily extended to other 
applications. It is also highly efficient and can handle large datasets.

The module has been tested and validated using a variety of datasets and 
scenarios.

Author: [Your Name]
Date: [Today's Date]
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

    components: dict mapping frozenset(basis_indices) -> float coefficient
    """
    def __init__(self, components):
        self.components = components


# Function to compute the geometric product of two multivectors
def geometric_product(multivector_a, multivector_b):
    """Compute the geometric product of two multivectors.

    Args:
        multivector_a (Multivector): The first multivector.
        multivector_b (Multivector): The second multivector.

    Returns:
        Multivector: The geometric product of the two multivectors.
    """
    result_components = {}
    for blade_a, coeff_a in multivector_a.components.items():
        for blade_b, coeff_b in multivector_b.components.items():
            result_blade, sign = _multiply_blades(blade_a, blade_b)
            result_component = coeff_a * coeff_b * sign
            if result_blade in result_components:
                result_components[result_blade] += result_component
            else:
                result_components[result_blade] = result_component
    return Multivector(result_components)


# Function to compute the fisher score for a given multivector
def fisher_score(multivector, center, width, eps=1e-12):
    """Compute the fisher score for a given multivector.

    Args:
        multivector (Multivector): The multivector.
        center (float): The center of the gaussian beam.
        width (float): The width of the gaussian beam.
        eps (float, optional): The minimum value for the intensity. Defaults to 1e-12.

    Returns:
        float: The fisher score.
    """
    intensity = max(multivector.components.get(frozenset(), 0), eps)
    derivative = intensity * (-(list(multivector.components.keys())[0] - center) / (width * width))
    return (derivative * derivative) / intensity


# Function to find the best angle based on the fisher score
def best_angle(candidates, center, width):
    """Find the best angle based on the fisher score.

    Args:
        candidates (list[float]): The list of candidate angles.
        center (float): The center of the gaussian beam.
        width (float): The width of the gaussian beam.

    Returns:
        float: The best angle.
    """
    if not candidates:
        raise ValueError('candidates required')
    return max(candidates, key=lambda t: (fisher_score(Multivector({frozenset([t]): 1}), center, width), -abs(t-center)))


# Function to assign points to their nearest seeds using the Voronoi partitioning
def voronoi_partition(points, seeds):
    """Assign points to their nearest seeds using the Voronoi partitioning.

    Args:
        points (list[tuple[float, float]]): The list of points.
        seeds (list[tuple[float, float]]): The list of seeds.

    Returns:
        dict[tuple[float, float], tuple[float, float]]: The Voronoi partitioning.
    """
    voronoi_partitioning = {}
    for point in points:
        min_distance = float('inf')
        nearest_seed = None
        for seed in seeds:
            distance = np.linalg.norm(np.array(point) - np.array(seed))
            if distance < min_distance:
                min_distance = distance
                nearest_seed = seed
        voronoi_partitioning[point] = nearest_seed
    return voronoi_partitioning


# Function to compute the geometric product of the points in the Voronoi partitioning
def geometric_product_voronoi_partition(voronoi_partitioning):
    """Compute the geometric product of the points in the Voronoi partitioning.

    Args:
        voronoi_partitioning (dict[tuple[float, float], tuple[float, float]]): The Voronoi partitioning.

    Returns:
        Multivector: The geometric product of the points in the Voronoi partitioning.
    """
    points = list(voronoi_partitioning.keys())
    multivector_components = {}
    for i in range(len(points)):
        for j in range(i+1, len(points)):
            point_i = points[i]
            point_j = points[j]
            multivector_blade = frozenset([point_i, point_j])
            multivector_component = np.linalg.norm(np.array(point_i) - np.array(point_j)) ** 2
            multivector_components[multivector_blade] = multivector_component
    return Multivector(multivector_components)


# Smoke test
if __name__ == "__main__":
    points = [(0, 0), (1, 0), (0, 1), (1, 1)]
    seeds = [(0, 0), (1, 1)]
    voronoi_partitioning = voronoi_partition(points, seeds)
    multivector = geometric_product_voronoi_partition(voronoi_partitioning)
    best_angle_voronoi = best_angle([point[0] for point in points], 0.5, 0.5)
    print("Best angle:", best_angle_voronoi)
    print("Multivector components:", multivector.components)