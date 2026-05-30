# DARWIN HAMMER — match 1537, survivor 0
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_decisi_m284_s3.py (gen4)
# parent_b: hybrid_hybrid_path_signatur_hybrid_hybrid_hybrid_m572_s2.py (gen5)
# born: 2026-05-29T23:37:23Z

"""
This module fuses the core topologies of hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_decisi_m284_s3.py (parent A)
and hybrid_hybrid_path_signatur_hybrid_hybrid_hybrid_m572_s2.py (parent B) by integrating the sphericity index 
from parent A with the lead-lag transform and B-spline basis from parent B. The mathematical bridge is found 
in the use of geometric transformations and basis functions to analyze physical and logical entities.

Parent A: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_decisi_m284_s3.py
Parent B: hybrid_hybrid_path_signatur_hybrid_hybrid_hybrid_m572_s2.py
"""

import numpy as np
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float

def sphericity_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("All dimensions must be positive.")
    geo_mean = (length * width * height) ** (1.0 / 3.0)
    return geo_mean / max(length, width, height)

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
            B_new[:, i] = term_l + term_r
        B = B_new

    return B

def hybrid_analysis(morphology: Morphology, path: np.ndarray, grid: np.ndarray) -> Tuple[float, np.ndarray]:
    sphericity = sphericity_index(morphology.length, morphology.width, morphology.height)
    lead_lag_path = lead_lag_transform(path)
    bspline = bspline_basis(path[:, 0], grid)
    return sphericity, bspline

def smoke_test():
    morphology = Morphology(1.0, 2.0, 3.0, 4.0)
    path = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    grid = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    sphericity, bspline = hybrid_analysis(morphology, path, grid)
    print(f"Sphericity: {sphericity}")
    print(bspline)

if __name__ == "__main__":
    smoke_test()