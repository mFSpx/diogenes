# DARWIN HAMMER — match 613, survivor 1
# gen: 5
# parent_a: hybrid_hybrid_path_signatur_hybrid_hybrid_pherom_m266_s0.py (gen3)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_infota_m7_s1.py (gen4)
# born: 2026-05-29T23:29:59Z

import numpy as np
import math
import random
import sys
from pathlib import Path

"""
This module defines a novel hybrid algorithm that mathematically fuses the core 
topologies of two parent algorithms: hybrid_path_signature_kan_m30_s3.py and 
hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_infota_m7_s1.py. The exact mathematical bridge 
between their structures lies in the concept of entropy, which is used in both 
algorithms to measure uncertainty or information. In hybrid_path_signature_kan_m30_s3.py, 
entropy is implicit in the calculation of the pheromone signal decay, while in 
hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_infota_m7_s1.py, entropy is explicit in the 
interpretation of the MinHash signature as a discrete force series. This hybrid algorithm 
leverages the concept of entropy to integrate the governing equations of both parent 
algorithms, creating a unified system that combines the path signature system with pheromone 
signal decay and MinHash-based similarity metric calculation.
"""

def lead_lag_transform(path):
    path = np.asarray(path, dtype=float)
    T, d = path.shape
    out = np.empty((2 * T - 1, 2 * d), dtype=float)
    for t in range(T - 1):
        out[2 * t]     = np.concatenate([path[t],     path[t]])
        out[2 * t + 1] = np.concatenate([path[t + 1], path[t]])
    out[2 * (T - 1)] = np.concatenate([path[T - 1], path[T - 1]])
    return out

def signature_level1(path):
    path = np.asarray(path, dtype=float)
    return path[-1] - path[0]

def signature_level2(path):
    path = np.asarray(path, dtype=float)
    increments = np.diff(path, axis=0)          # (T-1, d)
    running    = path[:-1] - path[0]            # (T-1, d)  X_{t-1} - X_0
    return running.T @ increments               # (d, d)

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
        B_new = np.zeros((N, len(t) - 1), dtype=np.float64)
        for i in range(N):
            for j in range(len(t) - 1):
                B_new[i, j] = (t[j] - x[i]) * B[i, j] / (t[j + 1] - t[j])
        B = B_new

    return B

def gaussian(r: float, epsilon: float = 1.0) -> float:
    return math.exp(-((epsilon * r) ** 2))

def euclidean(a: np.ndarray, b: np.ndarray) -> float:
    return np.linalg.norm(a - b)

def minhash_signature(vector: np.ndarray) -> np.ndarray:
    hash_values = np.zeros((len(vector),), dtype=np.int32)
    for i in range(len(vector)):
        hash_values[i] = int(hashlib.md5(vector[i].tobytes()).hexdigest(), 16)
    return hash_values

def entropic_minhash(vector: np.ndarray, num_bins: int = 256) -> np.ndarray:
    hash_values = minhash_signature(vector)
    bins = np.linspace(0, 1, num_bins + 1)
    bin_indices = np.digitize(hash_values, bins)
    return bin_indices

def hybrid_path_signature_minhash(vector: np.ndarray, path: np.ndarray) -> np.ndarray:
    lead_lag = lead_lag_transform(path)
    signature = signature_level2(lead_lag)
    entropy = np.linalg.inv(signature)
    minhash = entropic_minhash(vector)
    return np.dot(minhash, entropy)

def hybrid_main():
    # Smoke test
    vector = np.random.rand(10, 2)
    path = np.random.rand(10, 2)

    lead_lag = lead_lag_transform(path)
    signature = signature_level2(lead_lag)
    entropy = np.linalg.inv(signature)
    minhash = entropic_minhash(vector)
    hybrid_signature = np.dot(minhash, entropy)

    print(hybrid_signature)

if __name__ == "__main__":
    hybrid_main()