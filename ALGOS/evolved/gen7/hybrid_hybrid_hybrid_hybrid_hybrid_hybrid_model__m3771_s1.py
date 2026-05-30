# DARWIN HAMMER — match 3771, survivor 1
# gen: 7
# parent_a: hybrid_hybrid_hybrid_path_s_hybrid_hybrid_hybrid_m1236_s0.py (gen6)
# parent_b: hybrid_hybrid_model_pool_hy_hybrid_hybrid_minimu_m1971_s3.py (gen4)
# born: 2026-05-29T23:51:39Z

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

        B = np.zeros((N, len(t) - 1), dtype=np.float64)
        for i in range(N):
            for j in range(len(t) - 1):
                if t[j] <= x[i] < t[j + 1]:
                    if k == 1:
                        B[i, j] = 1.0
                    elif k == 2:
                        B[i, j] = (x[i] - t[j]) / (t[j + 1] - t[j])
                    elif k == 3:
                        B[i, j] = (x[i] - t[j]) ** 2 * (t[j + 1] - x[i]) / ((t[j + 1] - t[j]) ** 3)
        return B

    def bayes_marginal(self, prior: float, likelihood: float, false_positive: float) -> float:
        if not all(0 <= x <= 1 for x in (prior, likelihood, false_positive)):
            raise ValueError("probabilities must be between 0 and 1")
        return (prior * likelihood) / (prior * likelihood + (1 - prior) * false_positive)

    def compute_curvature(self, features):
        return np.std(features)

    def compute_entropy(self, features):
        hist, _ = np.histogram(features, bins=10, range=(0, 1))
        hist = hist / len(features)
        entropy = -np.sum(hist * np.log2(hist + 1e-12))
        return entropy

    def load_model_with_curvature_and_entropy(self, model, curvature, entropy):
        prior = curvature / (curvature + 1)
        likelihood = entropy / (entropy + 1)
        false_positive = 0.5
        return self.bayes_marginal(prior, likelihood, false_positive)

    def hybrid_summary(self, features, model):
        curvature = self.compute_curvature(features)
        entropy = self.compute_entropy(features)
        load = self.load_model_with_curvature_and_entropy(model, curvature, entropy)
        return load

if __name__ == "__main__":
    hybrid = HybridSystem()
    features = np.random.rand(10)
    model = "example_model"
    print(hybrid.hybrid_summary(features, model))