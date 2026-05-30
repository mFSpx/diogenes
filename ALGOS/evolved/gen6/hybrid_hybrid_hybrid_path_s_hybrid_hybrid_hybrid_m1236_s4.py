# DARWIN HAMMER — match 1236, survivor 4
# gen: 6
# parent_a: hybrid_hybrid_path_signatur_hybrid_hybrid_pherom_m266_s3.py (gen3)
# parent_b: hybrid_hybrid_hybrid_vorono_hybrid_liquid_time_c_m633_s2.py (gen5)
# born: 2026-05-29T23:34:34Z

"""
This module defines a novel hybrid algorithm that fuses the core topologies of 
two parent algorithms: hybrid_hybrid_path_signatur_hybrid_hybrid_pherom_m266_s3.py and 
hybrid_hybrid_hybrid_vorono_hybrid_liquid_time_c_m633_s2.py. 
The mathematical bridge between these structures is established through the integration 
of the lead_lag_transform and bspline_basis functions from the first parent algorithm 
with the Voronoi partitioning and hyperdimensional primitives from the second parent algorithm. 
Specifically, the bspline_basis function is used to create a transformation matrix 
that is applied to the input points in the Voronoi partitioning, allowing for a seamless 
integration of the two algorithms.

The hybrid algorithm leverages the concept of transformation to integrate the 
governing equations of both parent algorithms, creating a unified system that 
combines the path transformation with pheromone/anonymization scoring helpers and 
Voronoi partitioning with hyperdimensional primitives.

The mathematical interface between the two algorithms is established through 
the use of basis functions and transformations. The bspline_basis function 
from hybrid_hybrid_path_signatur_hybrid_hybrid_pherom_m266_s3.py is used to create a 
transformation matrix, which is then applied to the input points in the Voronoi 
partitioning from hybrid_hybrid_hybrid_vorono_hybrid_liquid_time_c_m633_s2.py.
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
                B[i, j] = self.basis(t, x[i], j, k)

        return B

    def basis(self, t, x, j, k):
        if k == 1:
            return 1.0 if t[j] <= x < t[j+1] else 0.0
        else:
            a = (x - t[j]) / (t[j+k-1] - t[j])
            b = (t[j+k] - x) / (t[j+k] - t[j+1])
            return a * self.basis(t, x, j, k-1) + b * self.basis(t, x, j+1, k-1)

def distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])

def nearest(point: tuple[float, float], seeds: list[tuple[float, float]]) -> int:
    if not seeds:
        raise ValueError('seeds required')
    return min(range(len(seeds)), key=lambda i: (distance(point, seeds[i]), i))

def assign(points: list[tuple[float, float]], seeds: list[tuple[float, float]]) -> dict[int, list[tuple[float, float]]]:
    regions = {i: [] for i in range(len(seeds))}
    for p in points:
        regions[nearest(p, seeds)].append(p)
    return regions

def sigmoid(x: np.ndarray) -> np.ndarray:
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x)),
    )

def ltc_f(
    x: np.ndarray,
    I: np.ndarray,
    W: np.ndarray,
    b: np.ndarray,
) -> np.ndarray:
    return sigmoid(np.dot(W, np.concatenate((x, I))) + b)

def random_vector(dim: int = 10000, seed: str | int | None = None) -> np.ndarray:
    rng = random.Random(seed)
    data = np.fromiter(
        (1 if rng.getrandbits(1) else -1 for _ in range(dim)),
        dtype=float,
    )
    return data / np.linalg.norm(data)

def hybrid_operation(points: list[tuple[float, float]], seeds: list[tuple[float, float]], path: np.ndarray) -> dict[int, list[tuple[float, float]]]:
    system = HybridSystem()
    transformed_path = system.lead_lag_transform(path)
    grid = np.linspace(0, 1, 10)
    B = system.bspline_basis(transformed_path[:, 0], grid)
    regions = assign(points, seeds)

    # Apply transformation matrix to points in each region
    for i, region in regions.items():
        region_array = np.array(region)
        transformed_region = np.dot(B, region_array)
        regions[i] = transformed_region.tolist()

    return regions

if __name__ == "__main__":
    points = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]
    seeds = [(0, 0), (2, 2), (4, 4)]
    path = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]])
    regions = hybrid_operation(points, seeds, path)
    print(regions)