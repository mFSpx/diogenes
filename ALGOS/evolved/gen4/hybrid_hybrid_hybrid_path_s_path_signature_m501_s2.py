# DARWIN HAMMER — match 501, survivor 2
# gen: 4
# parent_a: hybrid_hybrid_path_signatur_hybrid_hybrid_pherom_m266_s3.py (gen3)
# parent_b: path_signature.py (gen0)
# born: 2026-05-29T23:29:16Z

import numpy as np

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

    def hybrid_signature(self, path, level, basis_degree=3):
        transformed_path = self.lead_lag_transform(path)
        basis = self.bspline_basis(np.arange(len(transformed_path)), np.arange(len(transformed_path)), basis_degree)
        if level == 1:
            return self.signature_level1(basis @ transformed_path)
        elif level == 2:
            return self.signature_level2(basis @ transformed_path)

if __name__ == "__main__":
    hybrid_system = HybridSystem()
    path = np.random.rand(10, 3)
    print(hybrid_system.hybrid_signature(path, 1))
    print(hybrid_system.hybrid_signature(path, 2))