# DARWIN HAMMER — match 2342, survivor 0
# gen: 5
# parent_a: hybrid_hybrid_hybrid_worksh_hybrid_hybrid_path_s_m851_s2.py (gen4)
# parent_b: hybrid_hybrid_hybrid_geomet_hybrid_hybrid_pherom_m30_s1.py (gen3)
# born: 2026-05-29T23:41:51Z

"""
This module implements a novel hybrid algorithm that fuses the governing equations of the 
hybrid_hybrid_hybrid_worksh_hybrid_hybrid_path_s_m851_s2 and hybrid_hybrid_hybrid_geomet_hybrid_hybrid_pherom_m30_s1 algorithms.

The mathematical bridge between these two structures lies in the application of Voronoi partitioning 
from the geometric product to the path signature representation, and the use of Shannon entropy calculation 
to weight the pheromone signals in the surface usage tracking. This enables a more detailed understanding 
of the surface usage patterns informed by the geometric structure and the path signature representation.

The mathematical interface is established by representing the Voronoi partitions as a probability distribution, 
and then applying the Shannon entropy calculation to this distribution. The resulting entropy values are then 
used to weight the pheromone signals, allowing for a more informed selection of actions. The B-spline basis 
is used to approximate the path signature, and the weekday weight vector is used to modulate the effective 
learning rate in the hybrid workshare allocation.
"""

import numpy as np
import math
import random
import sys
import pathlib
from collections import Counter

GROUPS = ("codex", "groq", "cohere", "local_models")
MAX64 = (1 << 64) - 1

def _pct(value: float) -> float:
    return round(float(value), 6)

def doomsday(year: int, month: int, day: int) -> int:
    return (date(year, month, day).weekday() + 1) % 7

def weekday_weight_vector(groups: np.ndarray, dow: int) -> np.ndarray:
    n = len(groups)
    if n == 0:
        raise ValueError("groups must contain at least one element")
    base_angles = np.arange(n) * (2.0 * math.pi) / n
    phase = (2.0 * math.pi) * (dow % 7) / 7.0
    amplitude = 0.2
    raw = 1.0 + amplitude * np.sin(base_angles + phase)
    weight_vec = raw / raw.sum()
    return weight_vec.astype(np.float64)

def lead_lag_transform(path):
    path = np.asarray(path, dtype=float)
    T, d = path.shape
    out = np.empty((2 * T - 1, 2 * d), dtype=float)
    for t in range(T - 1):
        out[2 * t]     = np.concatenate([path[t],     path[t]])
        out[2 * t + 1] = np.concatenate([path[t + 1], path[t]])
    out[2 * (T - 1)] = np.concatenate([path[T - 1], path[T - 1]])
    return out

def bspline_basis(x, grid, k=3):
    x = np.asarray(x, dtype=np.float64)
    grid = np.asarray(grid, dtype=np.float64)

    t = np.concatenate([
        np.full(k - 1, grid[0]),
        grid,
        np.full(k - 1, grid[-1]),
    ])

    def basis(i, k):
        if k == 0:
            return np.where((t[i] <= x) & (x < t[i + 1]), 1.0, 0.0)
        if t[i + k] == t[i]:
            return np.zeros_like(x)
        return (x - t[i]) / (t[i + k] - t[i]) * basis(i, k - 1) + (t[i + k + 1] - x) / (t[i + k + 1] - t[i + 1]) * basis(i + 1, k - 1)

    return np.array([basis(i, k) for i in range(len(t) - k)])

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

def shannon_entropy(probabilities):
    return -sum(p * math.log(p, 2) for p in probabilities if p > 0)

def hybrid_operation(path, seeds, dow):
    # Apply Voronoi partitioning to the path
    regions = assign(path, seeds)

    # Calculate the probability distribution of the Voronoi partitions
    probabilities = [len(regions[i]) / len(path) for i in range(len(seeds))]

    # Calculate the Shannon entropy of the probability distribution
    entropy = shannon_entropy(probabilities)

    # Apply the lead-lag transform to the path
    transformed_path = lead_lag_transform(np.array(path))

    # Calculate the B-spline basis for the transformed path
    basis = bspline_basis(np.linspace(0, 1, len(transformed_path)), np.linspace(0, 1, len(transformed_path)))

    # Modulate the effective learning rate using the weekday weight vector
    weight_vec = weekday_weight_vector(np.array([i for i in range(len(seeds))]), dow)
    modulated_basis = basis * weight_vec[:, None]

    return entropy, modulated_basis

def main():
    # Generate a random path
    path = [(random.uniform(0, 1), random.uniform(0, 1)) for _ in range(10)]

    # Generate random seeds
    seeds = [(random.uniform(0, 1), random.uniform(0, 1)) for _ in range(4)]

    # Calculate the hybrid operation
    entropy, modulated_basis = hybrid_operation(path, seeds, 3)

    print(f"Shannon entropy: {entropy}")
    print(f"Modulated basis shape: {modulated_basis.shape}")

if __name__ == "__main__":
    main()