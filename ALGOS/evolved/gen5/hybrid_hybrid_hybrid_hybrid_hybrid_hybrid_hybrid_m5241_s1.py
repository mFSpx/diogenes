# DARWIN HAMMER — match 5241, survivor 1
# gen: 5
# parent_a: hybrid_hybrid_hybrid_path_s_path_signature_m501_s0.py (gen4)
# parent_b: hybrid_hybrid_hybrid_bandit_hybrid_path_signatur_m56_s2.py (gen3)
# born: 2026-05-30T00:00:55Z

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

        B = np.zeros((N, n_basis), dtype=np.float64)
        for i in range(N):
            for j in range(n_basis):
                if t[j] <= x[i] <= t[j + 1]:
                    if k == 1:
                        B[i, j] = 1.0
                    elif k == 2:
                        B[i, j] = (x[i] - t[j]) / (t[j + 1] - t[j])
                        if j < n_basis - 1:
                            B[i, j + 1] = (t[j + 2] - x[i]) / (t[j + 2] - t[j + 1])
                    elif k == 3:
                        if t[j] <= x[i] <= t[j + 1]:
                            B[i, j] = (x[i] - t[j])**2 * (t[j + 1] - x[i]) / ((t[j + 1] - t[j])**3)
                        elif t[j + 1] <= x[i] <= t[j + 2]:
                            B[i, j] = (x[i] - t[j])**2 * (t[j + 2] - x[i]) / ((t[j + 1] - t[j])**2 * (t[j + 2] - t[j + 1]))
                        if j < n_basis - 1:
                            if t[j + 1] <= x[i] <= t[j + 2]:
                                B[i, j + 1] = (x[i] - t[j + 1])**2 * (t[j + 2] - x[i]) / ((t[j + 2] - t[j + 1])**3)
                            elif t[j + 2] <= x[i] <= t[j + 3]:
                                B[i, j + 1] = (x[i] - t[j + 1])**2 * (t[j + 3] - x[i]) / ((t[j + 2] - t[j + 1])**2 * (t[j + 3] - t[j + 2]))
        return B

    def lead_lag_bspline_signature(self, path, grid, k=3):
        lead_lag_path = self.lead_lag_transform(path)
        x = np.arange(len(lead_lag_path))
        return self.bspline_basis(x, grid, k)

    def store_update_from_signature(self, signature, alpha, beta, dt, base, level=0.0):
        inflow = np.sum(signature, axis=0)
        outflow = np.sum(signature, axis=1)
        delta = alpha * np.sum(inflow) - beta * np.sum(outflow)
        level = max(0, level + delta * dt)
        return level

    def adjust_bandit_propensities(self, propensities, dance):
        return propensities * np.tanh(dance)

    def smoke_test(self):
        T = 10
        d = 2
        path = np.random.rand(T, d)
        grid = np.linspace(0, 2 * T - 1, 10)
        k = 3

        signature = self.lead_lag_bspline_signature(path, grid, k)

        level = 0.0
        alpha = 1.0
        beta = 1.0
        dt = 1.0
        base = 1.0

        level = self.store_update_from_signature(signature, alpha, beta, dt, base, level)

        propensities = np.random.rand(10)
        dance = np.tanh(alpha * (beta * np.sum(signature, axis=0) - np.sum(signature, axis=1)))
        adjusted_propensities = self.adjust_bandit_propensities(propensities, dance)

        print("Smoke test completed successfully.")

if __name__ == "__main__":
    hybrid_system = HybridSystem()
    hybrid_system.smoke_test()