# DARWIN HAMMER — match 4307, survivor 0
# gen: 7
# parent_a: hybrid_hybrid_hybrid_path_s_hybrid_hybrid_hybrid_m1236_s4.py (gen6)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1207_s4.py (gen6)
# born: 2026-05-29T23:54:42Z

"""
This module defines a novel hybrid algorithm that fuses the core topologies of 
two parent algorithms: hybrid_hybrid_hybrid_path_s_hybrid_hybrid_hybrid_m1236_s4.py 
and hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1207_s4.py.
The mathematical bridge between these structures is established through the integration 
of the lead_lag_transform and bspline_basis functions from the first parent algorithm 
with the Voronoi partitioning and curvature matrix operations from the second parent algorithm. 
Specifically, the bspline_basis function is used to create a transformation matrix 
that is applied to the input points in the Voronoi partitioning, allowing for a seamless 
integration of the two algorithms.
"""

import numpy as np
import math
import random
import sys
import pathlib

class HybridSystem:
    def __init__(self):
        self.pheromones = {}
        self.records = []

    def lead_lag_transform(self, path):
        path = np.asarray(path, dtype=float)
        T, d = path.shape
        out = np.empty((2 * T - 1, 2 * d), dtype=float)
        for t in range(T - 1):
            out[2 * t]     = np.concatenate([path[t],     path[t]])
            out[2 * t + 1] = np.concatenate([path[t + 1], path[t]])
        out[2 * (T - 1)] = np.concatenate([path[T - 1], path[T - 1]])
        return out

    def bspline_basis(self, x: np.ndarray, grid: np.ndarray, k: int = 3) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        grid = np.asarray(grid, dtype=np.float64)

        t = np.concatenate([
            np.full(k - 1, grid[0]),
            grid,
            np.full(k - 1, grid[-1])
        ])

        def basis(i, k, t):
            if k == 0:
                return np.where((t[i] <= x) & (x < t[i + 1]), 1.0, 0.0)
            else:
                d1 = t[i + k] - t[i]
                d2 = t[i + k + 1] - t[i + 1]
                e1 = 0.0 if d1 == 0 else ((x - t[i]) / d1) * basis(i, k - 1, t)
                e2 = 0.0 if d2 == 0 else ((t[i + k + 1] - x) / d2) * basis(i + 1, k - 1, t)
                return e1 + e2

        return np.array([basis(i, k, t) for i in range(k - 1, len(t) - k)])

    def assign_voronoi(self, points: list[tuple[float, float]], seeds: list[tuple[float, float]]) -> dict[int, list[tuple[float, float]]]:
        regions: dict[int, list[tuple[float, float]]] = {i: [] for i in range(len(seeds))}
        for p in points:
            min_dist = float('inf')
            closest_seeds = []
            for i, s in enumerate(seeds):
                dist = math.hypot(p[0] - s[0], p[1] - s[1])
                if dist < min_dist:
                    min_dist = dist
                    closest_seeds = [i]
                elif dist == min_dist:
                    closest_seeds.append(i)
            if len(closest_seeds) == 1:
                regions[closest_seeds[0]].append(p)
        return regions

    def hybrid_operation(self, points: list[tuple[float, float]], seeds: list[tuple[float, float]], k: int = 3):
        voronoi_regions = self.assign_voronoi(points, seeds)
        transformed_points = []
        for region in voronoi_regions.values():
            region_array = np.array(region)
            bspline_transform = self.bspline_basis(region_array[:, 0], np.linspace(min(region_array[:, 0]), max(region_array[:, 0]), 10), k)
            lead_lag_transform = self.lead_lag_transform(region_array[:, np.newaxis])
            transformed_points.append(np.concatenate((bspline_transform[:, np.newaxis], lead_lag_transform), axis=1))
        return np.concatenate(transformed_points)

def length(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Calculate the Euclidean distance between two points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])

def bayes_marginal(prior: float, likelihood: float, false_positive: float) -> float:
    """Compute the marginal probability for Bayesian update."""
    if not all(0 <= x <= 1 for x in (prior, likelihood, false_positive)):
        raise ValueError("probabilities must be in [0,1]")
    return likelihood * prior + false_positive

def prune_probability(t: float, lam: float = 1.0, alpha: float = 0.2) -> float:
    if t < 0 or lam < 0 or alpha < 0:
        raise ValueError('t, lam, and alpha must be non-negative')
    return min(1.0, lam * math.exp(-alpha * t))

if __name__ == "__main__":
    points = [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)]
    seeds = [(0.0, 0.0), (2.0, 2.0), (4.0, 4.0)]
    hybrid_system = HybridSystem()
    transformed_points = hybrid_system.hybrid_operation(points, seeds)
    print(transformed_points)
    print(length((1.0, 2.0), (3.0, 4.0)))
    print(bayes_marginal(0.5, 0.8, 0.2))
    print(prune_probability(1.0))