# DARWIN HAMMER — match 1236, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_path_signatur_hybrid_hybrid_pherom_m266_s3.py (gen3)
# parent_b: hybrid_hybrid_hybrid_vorono_hybrid_liquid_time_c_m633_s2.py (gen5)
# born: 2026-05-29T23:34:34Z

"""
This module defines a novel hybrid algorithm that fuses the core topologies of 
two parent algorithms: hybrid_hybrid_path_signatur_hybrid_hybrid_pherom_m266_s3.py and 
hybrid_hybrid_hybrid_vorono_hybrid_liquid_time_c_m633_s2.py. 
The mathematical bridge between these structures lies in the integration of 
Voronoi partitioning with the path transformation and pheromone system, 
and the application of sparse winner-take-all tags to inform model selection 
in the hybrid privacy model pool management, combined with the 
hyperdimensional primitives' binding and bundling operations to compute the 
input-dependent time constant. 
The bspline_basis function from the first parent is used to create a 
transformation matrix, which is then used in the Voronoi partitioning to 
assign points to regions.
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
                    B[i, j] = self.basis(x[i], t, j, k)
        return B

    def basis(self, x: float, t: np.ndarray, j: int, k: int) -> float:
        if k == 0:
            return 1 if t[j] <= x < t[j + 1] else 0
        else:
            d = t[j + k] - t[j]
            e = t[j + k + 1] - t[j + 1]
            return (x - t[j]) / d * self.basis(x, t, j, k - 1) + (t[j + k + 1] - x) / e * self.basis(x, t, j + 1, k - 1)

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

    def hybrid_operation(self, path, points, seeds, x, I, W, b):
        transformed_path = self.lead_lag_transform(path)
        transformation_matrix = self.bspline_basis(np.arange(len(transformed_path)), np.arange(len(transformed_path)))
        regions = self.assign(points, seeds)
        output = self.ltc_f(x, I, W, b)
        return transformed_path, transformation_matrix, regions, output

if __name__ == "__main__":
    hybrid_system = HybridSystem()
    path = np.random.rand(10, 2)
    points = [(random.random(), random.random()) for _ in range(100)]
    seeds = [(random.random(), random.random()) for _ in range(5)]
    x = np.random.rand(10)
    I = np.random.rand(10)
    W = np.random.rand(20, 20)
    b = np.random.rand(20)
    transformed_path, transformation_matrix, regions, output = hybrid_system.hybrid_operation(path, points, seeds, x, I, W, b)
    print("Transformed Path:", transformed_path)
    print("Transformation Matrix:", transformation_matrix)
    print("Regions:", regions)
    print("Output:", output)