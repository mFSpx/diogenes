# DARWIN HAMMER — match 266, survivor 1
# gen: 3
# parent_a: hybrid_path_signature_kan_m30_s3.py (gen1)
# parent_b: hybrid_hybrid_pheromone_inf_privacy_m54_s0.py (gen2)
# born: 2026-05-29T23:27:55Z

"""
This module defines a novel hybrid algorithm that fuses the core topologies of 
two parent algorithms: hybrid_path_signature_kan_m30_s3.py and hybrid_hybrid_pheromone_inf_privacy_m54_s0.py.
The mathematical bridge between these two algorithms lies in the concept of 
transformations on path data, which is used in hybrid_path_signature_kan_m30_s3.py to 
calculate signatures of paths, and in hybrid_hybrid_pheromone_inf_privacy_m54_s0.py to 
calculate pheromone signals. This hybrid algorithm leverages the concept of 
transformations to integrate the governing equations of both parent algorithms, 
creating a unified system that combines path signature calculations with pheromone 
signal calculations.
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

    def calculate_pheromone_signal(self, surface_key, signal_kind, signal_value, half_life_seconds):
        current_time = pathlib.Path('/proc/self/cmdline').stat().st_ctime
        if surface_key not in self.pheromones:
            self.pheromones[surface_key] = {'signal_kind': signal_kind, 'signal_value': signal_value, 'half_life_seconds': half_life_seconds, 'created_time': current_time}
        else:
            previous_signal_value = self.pheromones[surface_key]['signal_value']
            previous_half_life_seconds = self.pheromones[surface_key]['half_life_seconds']
            previous_created_time = self.pheromones[surface_key]['created_time']
            elapsed_time = current_time - previous_created_time
            decayed_signal_value = previous_signal_value * math.pow(0.5, elapsed_time / previous_half_life_seconds)
            self.pheromones[surface_key] = {'signal_kind': signal_kind, 'signal_value': signal_value, 'half_life_seconds': half_life_seconds, 'created_time': current_time}
        return signal_value

    def calculate_entropy(self, probabilities, eps=1e-12):
        total = sum(probabilities)
        if total <= 0:
            raise ValueError('positive probability mass required')
        return -sum((p/total) * math.log(max(p/total, eps)) for p in probabilities if p > 0)

    def signature_level1(self, path):
        path = np.asarray(path, dtype=float)
        return path[-1] - path[0]

    def signature_level2(self, path):
        path = np.asarray(path, dtype=float)
        increments = np.diff(path, axis=0)          # (T-1, d)
        running    = path[:-1] - path[0]            # (T-1, d)  X_{t-1} - X_0
        return running.T @ increments               # (d, d)

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

def test_hybrid_system():
    system = HybridSystem()
    path = np.random.rand(10, 3)
    transformed_path = system.lead_lag_transform(path)
    print("Transformed path shape:", transformed_path.shape)
    signal_value = system.calculate_pheromone_signal("test", "test", 1.0, 3600)
    print("Pheromone signal value:", signal_value)
    probabilities = [0.2, 0.3, 0.5]
    entropy = system.calculate_entropy(probabilities)
    print("Entropy:", entropy)
    signature_level1 = system.signature_level1(path)
    print("Signature level 1:", signature_level1)
    signature_level2 = system.signature_level2(path)
    print("Signature level 2:", signature_level2)
    x = np.array([1, 2, 3])
    grid = np.array([0, 1, 2, 3])
    bspline_basis = system.bspline_basis(x, grid)
    print("B-spline basis shape:", bspline_basis.shape)

if __name__ == "__main__":
    test_hybrid_system()