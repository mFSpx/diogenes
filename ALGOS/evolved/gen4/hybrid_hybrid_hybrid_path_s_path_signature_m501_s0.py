# DARWIN HAMMER — match 501, survivor 0
# gen: 4
# parent_a: hybrid_hybrid_path_signatur_hybrid_hybrid_pherom_m266_s3.py (gen3)
# parent_b: path_signature.py (gen0)
# born: 2026-05-29T23:29:16Z

"""
This module defines a novel hybrid algorithm that fuses the core topologies of 
two parent algorithms: hybrid_hybrid_path_signatur_hybrid_hybrid_pherom_m266_s3.py and path_signature.py. 
The mathematical bridge between these two algorithms lies in the concept of 
information encoding and transformation. In hybrid_hybrid_path_signatur_hybrid_hybrid_pherom_m266_s3.py, 
the lead_lag_transform and bspline_basis functions perform a transformation on 
the input path, while in path_signature.py, the path signature functions (signature_level1, signature_level2) 
use a similar transformation to calculate the signature values. 
This hybrid algorithm leverages the concept of transformation to integrate the 
governing equations of both parent algorithms, creating a unified system that 
combines the path transformation with signature calculation helpers.
"""

import numpy as np
import math
import random
import sys
from datetime import datetime, timezone
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
                    if k == 1:
                        B[i, j] = 1.0
                    elif k == 2:
                        B[i, j] = (x[i] - t[j]) / (t[j + 1] - t[j])
                        B[i, j + 1] = (t[j + 2] - x[i]) / (t[j + 2] - t[j + 1])
                    elif k == 3:
                        h1 = (t[j + 1] - t[j]) * (t[j + 1] - x[i])**2 / ((t[j + 2] - t[j]) * (t[j + 1] - t[j]))
                        h2 = (x[i] - t[j]) * (t[j + 1] - x[i])**2 / ((t[j + 1] - t[j]) * (t[j + 1] - t[j - 1]))
                        h3 = (x[i] - t[j]) * (x[i] - t[j + 1])**2 / ((t[j + 1] - t[j]) * (t[j + 2] - t[j + 1]))
                        B[i, j] = h1
                        B[i, j + 1] = h2 + h3
        return B

    def signature_level1(self, path):
        path = np.asarray(path, dtype=float)
        return path[-1] - path[0]

    def signature_level2(self, path):
        path = np.asarray(path, dtype=float)
        increments = np.diff(path, axis=0)          # (T-1, d)
        running    = path[:-1] - path[0]            # (T-1, d)  X_{t-1} - X_0
        return running.T @ increments               # (d, d)

    def hybrid_signature(self, path, level):
        transformed_path = self.lead_lag_transform(path)
        if level == 1:
            return self.signature_level1(transformed_path)
        elif level == 2:
            return self.signature_level2(transformed_path)

if __name__ == "__main__":
    hybrid_system = HybridSystem()
    path = np.random.rand(10, 3)
    print(hybrid_system.hybrid_signature(path, 1))
    print(hybrid_system.hybrid_signature(path, 2))