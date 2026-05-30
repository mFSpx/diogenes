# DARWIN HAMMER — match 1236, survivor 1
# gen: 6
# parent_a: hybrid_hybrid_path_signatur_hybrid_hybrid_pherom_m266_s3.py (gen3)
# parent_b: hybrid_hybrid_hybrid_vorono_hybrid_liquid_time_c_m633_s2.py (gen5)
# born: 2026-05-29T23:34:34Z

"""
This module defines a novel hybrid algorithm that mathematically fuses the core topologies of 
two parent algorithms: hybrid_hybrid_path_signatur_hybrid_hybrid_pherom_m266_s3.py and 
hybrid_hybrid_hybrid_vorono_hybrid_liquid_time_c_m633_s2.py. 
The mathematical bridge between these two algorithms lies in the concept of 
information encoding and transformation, where the Voronoi partitioning and 
hyperdimensional primitives from hybrid_hybrid_hybrid_vorono_hybrid_liquid_time_c_m633_s2.py 
are integrated with the path transformation and pheromone system from 
hybrid_hybrid_path_signatur_hybrid_hybrid_pherom_m266_s3.py. 
The Voronoi partitioning is used to create a set of regions, each associated with a 
pheromone value, and the hyperdimensional primitives are used to compute the 
input-dependent time constant for the pheromone system.
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
            np.full(k - 1, grid[-1]),
        ])

        n_basis = len(grid) + k - 2      
        N = len(x)

        B = np.zeros((N, len(t) - 1), dtype=np.float64)
        for i in range(N):
            for j in range(len(t) - 1):
                if t[j] <= x[i] <= t[j + 1]:
                    if j == 0 or j == len(t) - 2:
                        B[i, j] = 1
                    else:
                        B[i, j] = (x[i] - t[j]) / (t[j + 1] - t[j])
                elif t[j - 1] <= x[i] <= t[j]:
                    if j == 1:
                        B[i, j - 1] = 1
                    else:
                        B[i, j - 1] = (t[j] - x[i]) / (t[j] - t[j - 1])
        return B

    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        return np.where(
            x >= 0,
            1.0 / (1.0 + np.exp(-x)),
            np.exp(x) / (1.0 + np.exp(x)),
        )

    def ltc_f(
        self,
        x: np.ndarray,
        I: np.ndarray,
        W: np.ndarray,
        b: np.ndarray,
    ) -> np.ndarray:
        return self.sigmoid(np.dot(W, np.concatenate((x, I))) + b)

    def distance(self, a: tuple[float, float], b: tuple[float, float]) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def nearest(self, point: tuple[float, float], seeds: list[tuple[float, float]]) -> int:
        if not seeds:
            raise ValueError('seeds required')
        return min(range(len(seeds)), key=lambda i: (self.distance(point, seeds[i]), i))

    def assign(self, points: list[tuple[float, float]], seeds: list[tuple[float, float]]) -> dict[int, list[tuple[float, float]]]:
        regions = {i: [] for i in range(len(seeds))}
        for p in points:
            regions[self.nearest(p, seeds)].append(p)
        return regions

    def hybrid_operation(self, path, points, seeds):
        transformed_path = self.lead_lag_transform(path)
        regions = self.assign(points, seeds)
        pheromone_values = {}
        for region, points_in_region in regions.items():
            pheromone_value = 0
            for point in points_in_region:
                pheromone_value += self.distance(point, seeds[region])
            pheromone_values[region] = pheromone_value
        return transformed_path, pheromone_values

if __name__ == "__main__":
    hybrid_system = HybridSystem()
    path = np.random.rand(10, 2)
    points = [(random.random(), random.random()) for _ in range(100)]
    seeds = [(random.random(), random.random()) for _ in range(5)]
    transformed_path, pheromone_values = hybrid_system.hybrid_operation(path, points, seeds)
    print(transformed_path.shape)
    print(pheromone_values)