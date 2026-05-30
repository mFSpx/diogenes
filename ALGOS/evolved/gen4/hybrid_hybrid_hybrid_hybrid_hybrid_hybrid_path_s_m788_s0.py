# DARWIN HAMMER — match 788, survivor 0
# gen: 4
# parent_a: hybrid_hybrid_hybrid_geomet_hybrid_hybrid_pherom_m30_s2.py (gen3)
# parent_b: hybrid_hybrid_path_signatur_hybrid_hybrid_pherom_m266_s2.py (gen3)
# born: 2026-05-29T23:31:02Z

"""
This module defines a novel hybrid algorithm that fuses the core topologies of 
two parent algorithms: hybrid_hybrid_hybrid_geomet_hybrid_hybrid_pherom_m30_s2.py and 
hybrid_hybrid_path_signatur_hybrid_hybrid_pherom_m266_s2.py. 
The mathematical bridge between these two algorithms lies in the concept of 
integrating the Voronoi partitioning with the lead-lag transform and B-spline basis functions. 
This hybrid algorithm leverages the concept of Shannon entropy to integrate the governing equations 
of both parent algorithms, creating a unified system that combines the geometric product with pheromone 
signal calculation and entropy estimation.
"""

import math
import numpy as np
import random
import sys
import pathlib
from collections import Counter

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

def lead_lag_transform(path):
    path = np.asarray(path, dtype=float)
    T, d = path.shape
    out = np.empty((2 * T - 1, 2 * d), dtype=float)
    for t in range(T - 1):
        out[2 * t]     = np.concatenate([path[t],     path[t]])
        out[2 * t + 1] = np.concatenate([path[t + 1], path[t]])
    out[2 * (T - 1)] = np.concatenate([path[T - 1], path[T - 1]])
    return out

def bspline_basis(x: np.ndarray, grid: np.ndarray, k: int = 3) -> np.ndarray:
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
    for i in range(len(t) - 1):
        B[:, i] = np.where((x >= t[i]) & (x < t[i + 1]), 1.0, 0.0)
    B[:, -1] = np.where(x == t[-1], 1.0, B[:, -1])

    for order in range(2, k + 1):
        B_new = np.zeros((N, len(t) - order), dtype=np.float64)
        for i in range(len(t) - order):
            denom_l = t[i + order - 1] - t[i]
            denom_r = t[i + order] - t[i + 1]
            term_l = (
                (x - t[i]) / denom_l * B[:, i]
                if denom_l > 0 else np.zeros(N)
            )
            term_r = (
                (t[i + order] - x) / denom_r * B[:, i + 1]
                if denom_r > 0 else np.zeros(N)
            )
            B_new[:, i] = (term_l + term_r) / (order - 1)
        B = B_new
    return B

def hybrid_voronoi_pheromone(points, seeds, path):
    regions = assign(points, seeds)
    voronoi_entropy = 0
    for i in regions:
        region = regions[i]
        region_path = np.array([point for point in region])
        lead_lag_path = lead_lag_transform(region_path)
        bspline_path = bspline_basis(lead_lag_path[:, 0], lead_lag_path[:, 1])
        voronoi_entropy += len(region) * np.sum(bspline_path)
    return voronoi_entropy

def hybrid_pheromone_voronoi(path, seeds, points):
    lead_lag_path = lead_lag_transform(path)
    bspline_path = bspline_basis(lead_lag_path[:, 0], lead_lag_path[:, 1])
    pheromone_signal = np.sum(bspline_path)
    regions = assign(points, seeds)
    voronoi_pheromone = 0
    for i in regions:
        region = regions[i]
        voronoi_pheromone += len(region) * pheromone_signal
    return voronoi_pheromone

def hybrid_shannon_entropy(points, seeds, path):
    regions = assign(points, seeds)
    shannon_entropy = 0
    for i in regions:
        region = regions[i]
        region_path = np.array([point for point in region])
        lead_lag_path = lead_lag_transform(region_path)
        bspline_path = bspline_basis(lead_lag_path[:, 0], lead_lag_path[:, 1])
        region_entropy = -np.sum(bspline_path * np.log2(bspline_path))
        shannon_entropy += region_entropy
    return shannon_entropy

if __name__ == "__main__":
    points = [(random.random(), random.random()) for _ in range(100)]
    seeds = [(random.random(), random.random()) for _ in range(10)]
    path = np.random.rand(10, 2)
    voronoi_pheromone = hybrid_voronoi_pheromone(points, seeds, path)
    pheromone_voronoi = hybrid_pheromone_voronoi(path, seeds, points)
    shannon_entropy = hybrid_shannon_entropy(points, seeds, path)
    print(f"Voronoi Pheromone: {voronoi_pheromone}")
    print(f"Pheromone Voronoi: {pheromone_voronoi}")
    print(f"Shannon Entropy: {shannon_entropy}")