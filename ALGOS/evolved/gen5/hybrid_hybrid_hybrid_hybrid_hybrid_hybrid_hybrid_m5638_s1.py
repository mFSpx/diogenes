# DARWIN HAMMER — match 5638, survivor 1
# gen: 5
# parent_a: hybrid_hybrid_hybrid_worksh_hybrid_hybrid_path_s_m851_s2.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1088_s1.py (gen4)
# born: 2026-05-30T00:03:38Z

"""
This module integrates the mathematical structures of 
hybrid_hybrid_hybrid_worksh_hybrid_hybrid_path_s_m851_s2.py and 
hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1088_s1.py. 
The mathematical bridge lies in applying the B-spline basis functions from the first parent 
to approximate the path signature, and then using the Shannon entropy calculation from 
the second parent to weight the effective learning rate in the hybrid workshare allocation. 
The row-stochastic weights obtained from the sinusoidal rotation are represented as a 
probability distribution, and then the Shannon entropy calculation is applied to this 
distribution. The resulting entropy values are then used to weight the pheromone signals, 
allowing for a more informed selection of actions.
"""

import numpy as np
import math
import random
import sys
import pathlib
from datetime import date

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

def bspline_basis(x, grid, k=3):
    x = np.asarray(x, dtype=np.float64)
    grid = np.asarray(grid, dtype=np.float64)

    t = np.concatenate([
        np.full(k - 1, grid[0]),
        grid,
        np.full(k - 1, grid[-1]),
    ])

    def basis(i, k):
        def b(s):
            if k == 0:
                return 1.0 if s >= t[i] and s < t[i + 1] else 0.0
            if t[i + k] == t[i]:
                return 0.0
            d1 = (s - t[i]) / (t[i + k] - t[i])
            if k > 1 and t[i + k - 1] != t[i + k]:
                d2 = (t[i + k + 1] - s) / (t[i + k + 1] - t[i + 1])
            else:
                d2 = 0.0
            return d1 * basis(i, k - 1) + d2 * basis(i + 1, k - 1)
        return b

    return [basis(i, k) for i in range(len(t) - k)]

def shannon_entropy(probabilities):
    return -sum([p * math.log(p, 2) for p in probabilities if p > 0])

def lead_lag_transform(path):
    path = np.asarray(path, dtype=float)
    T, d = path.shape
    out = np.empty((2 * T - 1, 2 * d), dtype=float)
    for t in range(T - 1):
        out[2 * t]     = np.concatenate([path[t],     path[t]])
        out[2 * t + 1] = np.concatenate([path[t + 1], path[t]])
    out[2 * (T - 1)] = np.concatenate([path[T - 1], path[T - 1]])
    return out

def hybrid_operation(path, groups, grid):
    weight_vec = weekday_weight_vector(groups, doomsday(2026, 5, 29))
    basis = bspline_basis(np.linspace(0, 1, len(path)), grid)
    transformed_path = lead_lag_transform(path)
    probabilities = weight_vec / weight_vec.sum()
    entropy = shannon_entropy(probabilities)
    return transformed_path, entropy

def distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def nearest(point, seeds):
    if not seeds:
        raise ValueError('seeds required')
    return min(range(len(seeds)), key=lambda i: (distance(point, seeds[i]), i))

def assign(points, seeds):
    regions = {i: [] for i in range(len(seeds))}
    for p in points:
        regions[nearest(p, seeds)].append(p)
    return regions

if __name__ == "__main__":
    path = np.random.rand(10, 2)
    groups = np.array([1, 2, 3, 4])
    grid = np.linspace(0, 1, 10)
    transformed_path, entropy = hybrid_operation(path, groups, grid)
    print(transformed_path)
    print(entropy)