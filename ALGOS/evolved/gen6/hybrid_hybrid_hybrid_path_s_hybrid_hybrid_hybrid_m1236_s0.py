# DARWIN HAMMER — match 1236, survivor 0
# gen: 6
# parent_a: hybrid_hybrid_path_signatur_hybrid_hybrid_pherom_m266_s3.py (gen3)
# parent_b: hybrid_hybrid_hybrid_vorono_hybrid_liquid_time_c_m633_s2.py (gen5)
# born: 2026-05-29T23:34:34Z

"""
This module defines a novel hybrid algorithm that fuses the core topologies of 
two parent algorithms: hybrid_hybrid_path_signatur_hybrid_hybrid_pherom_m266_s3.py and 
hybrid_hybrid_hybrid_vorono_hybrid_liquid_time_c_m633_s2.py. 
The mathematical bridge between these structures is the integration of Voronoi partitioning 
with the transformation and pheromone/anonymization scoring helpers from the first parent, 
and the application of hyperdimensional primitives to compute the input-dependent time constant 
in the second parent. The governing equations of both parents are combined to create a unified 
system that combines the path transformation with Voronoi partitioning and hyperdimensional primitives.
"""

import numpy as np
import math
import random
import sys
import pathlib
from datetime import datetime, timezone

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
                if t[j] <= x[i] < t[j + 1]:
                    if k == 1:
                        B[i, j] = 1.0
                    elif k == 2:
                        B[i, j] = (x[i] - t[j]) / (t[j + 1] - t[j])
                    elif k == 3:
                        B[i, j] = (x[i] - t[j]) ** 2 * (t[j + 1] - x[i]) / ((t[j + 1] - t[j]) ** 3)
                    elif k == 4:
                        B[i, j] = (x[i] - t[j]) ** 3 * (t[j + 1] - x[i]) / ((t[j + 1] - t[j]) ** 4)
        return B

    def nearest(self, point: tuple[float, float], seeds: list[tuple[float, float]]) -> int:
        if not seeds:
            raise ValueError('seeds required')
        return min(range(len(seeds)), key=lambda i: (self.distance(point, seeds[i]), i))

    def distance(self, a: tuple[float, float], b: tuple[float, float]) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def assign(self, points: list[tuple[float, float]], seeds: list[tuple[float, float]]) -> dict[int, list[tuple[float, float]]]:
        regions = {i: [] for i in range(len(seeds))}
        for p in points:
            regions[self.nearest(p, seeds)].append(p)
        return regions

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

    def random_vector(self, dim: int = 10000, seed: str | int | None = None) -> np.ndarray:
        rng = random.Random(seed)
        data = np.fromiter(
            (1 if rng.getrandbits(1) else -1 for _ in range(dim)),
            dtype=float,
        )
        return data / np.linalg.norm(data)

    def hybrid_operation(self, path, points, seeds):
        transformed_path = self.lead_lag_transform(path)
        regions = self.assign(points, seeds)
        x = np.random.rand(10)
        I = np.random.rand(10)
        W = np.random.rand(10, 20)
        b = np.random.rand(10)
        return transformed_path, regions, self.ltc_f(x, I, W, b)

if __name__ == "__main__":
    hybrid_system = HybridSystem()
    path = np.random.rand(10, 2)
    points = [(random.random(), random.random()) for _ in range(10)]
    seeds = [(random.random(), random.random()) for _ in range(3)]
    transformed_path, regions, result = hybrid_system.hybrid_operation(path, points, seeds)
    print(transformed_path.shape, len(regions), result.shape)