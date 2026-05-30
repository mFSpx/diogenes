# DARWIN HAMMER — match 266, survivor 0
# gen: 3
# parent_a: hybrid_path_signature_kan_m30_s3.py (gen1)
# parent_b: hybrid_hybrid_pheromone_inf_privacy_m54_s0.py (gen2)
# born: 2026-05-29T23:27:55Z

"""
This module defines a novel hybrid algorithm that mathematically fuses the core 
topologies of two parent algorithms: hybrid_path_signature_kan_m30_s3.py and 
hybrid_hybrid_pheromone_inf_privacy_m54_s0.py. The exact mathematical bridge 
between their structures lies in the concept of entropy, which is used in both 
algorithms to measure uncertainty or information. In hybrid_path_signature_kan_m30_s3.py, 
entropy is implicit in the calculation of the pheromone signal decay, while in 
hybrid_hybrid_pheromone_inf_privacy_m54_s0.py, entropy is explicit in the 
calculation of the reconstruction risk score. This hybrid algorithm leverages the 
concept of entropy to integrate the governing equations of both parent algorithms, 
creating a unified system that combines the path signature system with pheromone 
signal decay and reconstruction risk scoring helpers.
"""

import numpy as np
import math
import random
import sys
from pathlib import Path

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

def kan_layer(
    x: np.ndarray,
    spline_weights: np.ndarray,
    grid: np.ndarray,
    k: int = 3,
) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    spline_weights = np.asarray(spline_weights, dtype=np.float64)
    grid = np.asarray(grid, dtype=np.float64)

    batch, n_in = x.shape
    n_out, n_in_w, n_basis = spline_weights.shape
    assert n_in == n_in_w, f"n_in mismatch: x has {n_in}, weights expect {n_in_w}"
    expected_n_basis = len(grid) + k - 2

def calculate_pheromone_signal(path, signal_kind, signal_value, half_life_seconds):
    path = np.asarray(path, dtype=float)
    T, d = path.shape
    current_time = 1  # dummy time for demonstration purposes
    if T not in path_dict:
        path_dict[T] = {'signal_kind': signal_kind, 'signal_value': signal_value, 'half_life_seconds': half_life_seconds, 'created_time': current_time}
    else:
        previous_signal_value = path_dict[T]['signal_value']
        previous_half_life_seconds = path_dict[T]['half_life_seconds']
        previous_created_time = path_dict[T]['created_time']
        elapsed_time = (current_time - previous_created_time)
        decayed_signal_value = previous_signal_value * math.pow(0.5, elapsed_time / previous_half_life_seconds)
        path_dict[T] = {'signal_kind': signal_kind, 'signal_value': signal_value, 'half_life_seconds': half_life_seconds, 'created_time': current_time}
    return signal_value

def calculate_entropy(probabilities, eps=1e-12):
    total = sum(probabilities)
    if total <= 0:
        raise ValueError('positive probability mass required')
    return -sum((p/total) * math.log(max(p/total, eps)) for p in probabilities if p > 0)

def hybrid_path_signature(path):
    path = np.asarray(path, dtype=float)
    signature = signature_level1(path)
    lead_lag = lead_lag_transform(path)
    basis = bspline_basis(lead_lag[:, 0], np.linspace(0, 1, 10))
    return np.dot(basis.T, lead_lag)

def hybrid_pheromone_signature(path, signal_kind, signal_value, half_life_seconds):
    path = np.asarray(path, dtype=float)
    T, d = path.shape
    path_dict = {}
    pheromone_signal = calculate_pheromone_signal(path, signal_kind, signal_value, half_life_seconds)
    entropy = calculate_entropy([pheromone_signal])
    return np.array([entropy] + list(path.flatten()))

def hybrid_path_pheromone(path, signal_kind, signal_value, half_life_seconds):
    path = np.asarray(path, dtype=float)
    T, d = path.shape
    path_signature = hybrid_path_signature(path)
    pheromone_signature = hybrid_pheromone_signature(path, signal_kind, signal_value, half_life_seconds)
    return np.concatenate((path_signature, pheromone_signature))

if __name__ == "__main__":
    path = np.random.rand(10, 10)
    signal_kind = "dummy_signal"
    signal_value = 1.0
    half_life_seconds = 1.0
    result = hybrid_path_pheromone(path, signal_kind, signal_value, half_life_seconds)
    print(result.shape)