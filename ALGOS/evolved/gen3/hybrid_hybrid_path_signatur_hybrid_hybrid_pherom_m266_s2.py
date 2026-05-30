# DARWIN HAMMER — match 266, survivor 2
# gen: 3
# parent_a: hybrid_path_signature_kan_m30_s3.py (gen1)
# parent_b: hybrid_hybrid_pheromone_inf_privacy_m54_s0.py (gen2)
# born: 2026-05-29T23:27:55Z

"""
This module defines a novel hybrid algorithm that fuses the core topologies of 
two parent algorithms: hybrid_path_signature_kan_m30_s3.py and hybrid_hybrid_pheromone_inf_privacy_m54_s0.py.
The mathematical bridge between these two algorithms lies in the concept of 
information-theoretic measures, specifically, the use of basis functions in 
hybrid_path_signature_kan_m30_s3.py and the calculation of entropy in 
hybrid_hybrid_pheromone_inf_privacy_m54_s0.py. This hybrid algorithm 
leverages the concept of entropy to integrate the governing equations of both 
parent algorithms, creating a unified system that combines the lead-lag 
transform and B-spline basis functions with pheromone signal calculation and 
entropy estimation.

Parent A: hybrid_path_signature_kan_m30_s3.py
Parent B: hybrid_hybrid_pheromone_inf_privacy_m54_s0.py
"""

import numpy as np
import math
import random
import sys
import pathlib

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

class HybridSystem:
    def __init__(self):
        self.pheromones = {}
        self.records = []

    def calculate_pheromone_signal(self, surface_key, signal_kind, signal_value, half_life_seconds):
        current_time = datetime.now()
        if surface_key not in self.pheromones:
            self.pheromones[surface_key] = {'signal_kind': signal_kind, 'signal_value': signal_value, 'half_life_seconds': half_life_seconds, 'created_time': current_time}
        else:
            previous_signal_value = self.pheromones[surface_key]['signal_value']
            previous_half_life_seconds = self.pheromones[surface_key]['half_life_seconds']
            previous_created_time = self.pheromones[surface_key]['created_time']
            elapsed_time = (current_time - previous_created_time).total_seconds()
            decayed_signal_value = previous_signal_value * math.pow(0.5, elapsed_time / previous_half_life_seconds)
            self.pheromones[surface_key] = {'signal_kind': signal_kind, 'signal_value': signal_value, 'half_life_seconds': half_life_seconds, 'created_time': current_time}
        return signal_value

    def calculate_entropy(self, probabilities, eps=1e-12):
        total = sum(probabilities)
        if total <= 0:
            raise ValueError('positive probability mass required')
        return -sum((p/total) * math.log(max(p/total, eps)) for p in probabilities if p > 0)

    def hybrid_operation(self, path, grid, k=3):
        lead_lag_path = lead_lag_transform(path)
        bspline_weights = bspline_basis(lead_lag_path[:, 0], grid, k)
        probabilities = np.mean(bspline_weights, axis=0)
        entropy = self.calculate_entropy(probabilities)
        return lead_lag_path, bspline_weights, entropy

def main():
    np.random.seed(0)
    random.seed(0)
    path = np.random.rand(10, 2)
    grid = np.linspace(0, 1, 10)
    hybrid_system = HybridSystem()
    lead_lag_path, bspline_weights, entropy = hybrid_system.hybrid_operation(path, grid)
    print("Lead-Lag Path Shape:", lead_lag_path.shape)
    print("B-Spline Weights Shape:", bspline_weights.shape)
    print("Entropy:", entropy)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)