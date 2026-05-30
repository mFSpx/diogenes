# DARWIN HAMMER — match 1088, survivor 0
# gen: 4
# parent_a: hybrid_hybrid_hybrid_worksh_hybrid_hybrid_model__m51_s0.py (gen3)
# parent_b: hybrid_hybrid_hybrid_geomet_hybrid_hybrid_pherom_m30_s1.py (gen3)
# born: 2026-05-29T23:32:41Z

"""
Hybrid module combining the sinusoidal rotation and matrix operations from 
'hybrid_hybrid_hybrid_worksh_hybrid_hybrid_model__m51_s0.py' and the geometric product and Voronoi partitioning 
from 'hybrid_hybrid_hybrid_geomet_hybrid_hybrid_pherom_m30_s1.py'. The mathematical bridge lies in applying the 
sinusoidal rotation to the probability distribution of Voronoi partitions obtained from the geometric product, 
and then using these probabilities to weight the matrix operations.

The mathematical interface is established by representing the Voronoi partitions as a probability distribution, 
and then applying the sinusoidal rotation to this distribution. The resulting probabilities are then used to 
weight the matrix operations, allowing for a more informed selection of actions.
"""

import numpy as np
import math
import random
import sys
import pathlib

Point = tuple[float, float]

def distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])

def nearest(point: Point, seeds: list[Point]) -> int:
    if not seeds:
        raise ValueError('seeds required')
    return min(range(len(seeds)), key=lambda i: (distance(point, seeds[i]), i))

def assign(points: list[Point], seeds: list[Point]) -> dict[int, list[Point]]:
    regions = {i: [] for i in range(len(seeds))}
    for p in points:
        regions[nearest(p, seeds)].append(p)
    return regions

def weekday_weight_vector(groups: list[str], dow: int) -> np.ndarray:
    """
    Normalised weight vector for *groups* based on weekday ``dow``.
    Sinusoidal rotation yields a row‑stochastic vector.
    """
    n = len(groups)
    if n == 0:
        raise ValueError("groups must contain at least one element")
    base_angles = np.arange(n) * (2.0 * math.pi) / n
    phase = (2.0 * math.pi) * (dow / 7.0)
    weight_vec = 1.0 + 0.5 * np.sin(base_angles + phase)
    return weight_vec / np.sum(weight_vec)

def geometric_product(seeds: list[Point], points: list[Point]) -> dict[int, list[Point]]:
    regions = assign(points, seeds)
    return regions

def hybrid_operation(seeds: list[Point], points: list[Point], dow: int) -> dict[int, list[Point]]:
    regions = geometric_product(seeds, points)
    groups = [f"seed_{i}" for i in range(len(seeds))]
    weights = weekday_weight_vector(groups, dow)
    weighted_regions = {i: [point for point in region if random.random() < weights[i]] for i, region in regions.items()}
    return weighted_regions

def matrix_operation(seeds: list[Point], points: list[Point], dow: int) -> np.ndarray:
    regions = geometric_product(seeds, points)
    groups = [f"seed_{i}" for i in range(len(seeds))]
    weights = weekday_weight_vector(groups, dow)
    matrix = np.zeros((len(seeds), len(points)))
    for i, region in regions.items():
        for point in region:
            matrix[i, points.index(point)] = weights[i]
    return matrix

if __name__ == "__main__":
    seeds = [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)]
    points = [(0.1, 0.1), (1.1, 1.1), (2.1, 2.1), (0.2, 0.2), (1.2, 1.2), (2.2, 2.2)]
    dow = 3
    weighted_regions = hybrid_operation(seeds, points, dow)
    matrix = matrix_operation(seeds, points, dow)
    print("Weighted Regions:")
    print(weighted_regions)
    print("Matrix:")
    print(matrix)