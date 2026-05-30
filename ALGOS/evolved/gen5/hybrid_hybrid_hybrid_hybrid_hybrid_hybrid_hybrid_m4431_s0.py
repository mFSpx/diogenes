# DARWIN HAMMER — match 4431, survivor 0
# gen: 5
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_path_s_m788_s0.py (gen4)
# parent_b: hybrid_hybrid_hybrid_gliner_hybrid_hybrid_ternar_m1199_s0.py (gen4)
# born: 2026-05-29T23:55:47Z

import math
import numpy as np
import random
import sys
import pathlib

# This module defines a novel hybrid algorithm that fuses the core topologies 
# of two parent algorithms: hybrid_hybrid_hybrid_geomet_hybrid_hybrid_pherom_m30_s2.py 
# and hybrid_hybrid_hybrid_gliner_hybrid_hybrid_ternar_m1199_s0.py. 
# The mathematical bridge between these two algorithms lies in the concept of 
# integrating the Voronoi partitioning with the lead-lag transform and B-spline basis 
# functions, and allocating work units based on the similarity to a prototype vector 
# using the SSIM score as a metric.

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
        B_new = n

def compute_ssim(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute the Structural Similarity Index (SSIM) between two vectors.
    """
    mu_x = np.mean(x)
    mu_y = np.mean(y)
    sigma_x = np.std(x)
    sigma_y = np.std(y)
    sigma_xy = np.mean((x - mu_x) * (y - mu_y))
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    ssim = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x ** 2 + sigma_y ** 2 + c2))
    return ssim

def allocate_workshare_ssim(x: np.ndarray, y: np.ndarray, *, total_units: float, deterministic_target_pct: float = 90.0, groups: tuple[str, ...] = ("codex", "groq", "cohere", "local_models")) -> dict[str, float]:
    """
    Allocate work units among different groups based on the similarity to a prototype vector.
    """
    ssim = compute_ssim(x, y)
    deterministic_units = total_units * deterministic_target_pct / 100.0
    llm_units = total_units - deterministic_units
    per_group = llm_units / len(groups)
    lanes = [
        {
            "group": group,
            "llm_units": round(per_group * ssim, 2),
            "llm_share_pct": round(100.0 / len(groups), 2),
            "proof_required": True,
        }
        for group in groups
    ]
    return {
        "total_units": round(total_units, 2),
        "deterministic_units": round(deterministic_units, 2),
        "llm_units": round(llm_units, 2),
    }

def hybrid_assign(points: list[Point], seeds: list[Point], allocate_units: bool = False) -> dict[int, list[Point]]:
    """
    Assign points to Voronoi regions using the nearest point, and optionally 
    allocate work units among different groups based on the similarity to a prototype 
    vector.
    """
    regions = assign(points, seeds)
    if allocate_units:
        x = np.array([p[0] for p in points])
        y = np.array([p[1] for p in points])
        prototype = np.mean([p[0] for p in seeds], axis=0) + np.mean([p[1] for p in seeds], axis=0)
        regions = allocate_workshare_ssim(x, prototype, total_units=len(points))
    return regions

def hybrid_lead_lag_transform(path):
    """
    Apply the lead-lag transform to the input path, and then allocate work units 
    among different groups based on the similarity to a prototype vector.
    """
    path = np.asarray(path, dtype=float)
    T, d = path.shape
    out = lead_lag_transform(path)
    x = np.array([p[0] for p in out])
    y = np.array([p[1] for p in out])
    prototype = np.mean([p[0] for p in out], axis=0) + np.mean([p[1] for p in out], axis=0)
    allocate_workshare_ssim(x, prototype, total_units=len(out))
    return out

def hybrid_bspline_basis(x: np.ndarray, grid: np.ndarray, k: int = 3) -> np.ndarray:
    """
    Compute the B-spline basis functions for the input data, and then allocate work 
    units among different groups based on the similarity to a prototype vector.
    """
    basis = bspline_basis(x, grid, k)
    x = np.array([p[0] for p in basis])
    y = np.array([p[1] for p in basis])
    prototype = np.mean([p[0] for p in basis], axis=0) + np.mean([p[1] for p in basis], axis=0)
    allocate_workshare_ssim(x, prototype, total_units=len(basis))
    return basis

if __name__ == "__main__":
    points = [(1, 2), (3, 4), (5, 6), (7, 8)]
    seeds = [(0, 0), (10, 10)]
    print(hybrid_assign(points, seeds))
    print(hybrid_lead_lag_transform(points))
    print(hybrid_bspline_basis(points, np.linspace(0, 10, 100), k=3))