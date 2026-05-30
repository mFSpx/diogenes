# DARWIN HAMMER — match 4395, survivor 2
# gen: 7
# parent_a: hybrid_hybrid_hybrid_path_s_hybrid_hybrid_hybrid_m1236_s1.py (gen6)
# parent_b: hybrid_hybrid_fisher_locali_hybrid_hybrid_label__m4_s1.py (gen4)
# born: 2026-05-29T23:55:29Z

import numpy as np
import math

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
                    if j == 0 or j == len(t) - 2:
                        B[i, j] = 1
                    else:
                        B[i, j] = (x[i] - t[j]) / (t[j + 1] - t[j])
                elif t[j - 1] <= x[i] <= t[j]:
                    if j == 1:
                        B[i, j - 1] = (t[j] - x[i]) / (t[j] - t[j - 1])
        return B

    def gaussian_beam(self, theta: float, center: float, width: float) -> float:
        if width <= 0.0:
            raise ValueError("width must be positive")
        z = (theta - center) / width
        return math.exp(-0.5 * z * z)

    def fisher_score(self, theta: float, center: float, width: float, eps: float = 1e-12) -> float:
        intensity = max(self.gaussian_beam(theta, center, width), eps)
        derivative = intensity * (-(theta - center) / (width * width))
        return (derivative * derivative) / intensity

    def voronoi_partition(self, points, num_regions):
        voronoi = np.zeros((len(points), num_regions))
        for i, point in enumerate(points):
            distances = np.linalg.norm(point - points, axis=1)
            voronoi[i] = np.argsort(distances)[:num_regions]
        return voronoi

    def hyperdimensional_primitives(self, points, num_dimensions):
        primitives = np.zeros((len(points), num_dimensions))
        for i, point in enumerate(points):
            primitives[i] = np.random.rand(num_dimensions)
        return primitives

    def compute_pheromone(self, points, voronoi_partition, hyperdimensional_primitives):
        pheromones = np.zeros(len(points))
        for i, point in enumerate(points):
            pheromones[i] = np.sum(voronoi_partition[i] * hyperdimensional_primitives[i])
        return pheromones

def hybrid_operation(path, theta, center, width):
    system = HybridSystem()
    transformed_path = system.lead_lag_transform(path)
    basis = system.bspline_basis(np.linspace(0, 1, len(path)), np.linspace(0, 1, len(path)))
    fisher = system.fisher_score(theta, center, width)
    voronoi = system.voronoi_partition(path, 5)
    primitives = system.hyperdimensional_primitives(path, 10)
    pheromones = system.compute_pheromone(path, voronoi, primitives)
    return transformed_path, basis, fisher, voronoi, primitives, pheromones

def hybrid_fusion(path1, path2, theta, center, width):
    system = HybridSystem()
    transformed_path1 = system.lead_lag_transform(path1)
    transformed_path2 = system.lead_lag_transform(path2)
    basis1 = system.bspline_basis(np.linspace(0, 1, len(path1)), np.linspace(0, 1, len(path1)))
    basis2 = system.bspline_basis(np.linspace(0, 1, len(path2)), np.linspace(0, 1, len(path2)))
    fisher1 = system.fisher_score(theta, center, width)
    fisher2 = system.fisher_score(theta, center, width)
    voronoi1 = system.voronoi_partition(path1, 5)
    voronoi2 = system.voronoi_partition(path2, 5)
    primitives1 = system.hyperdimensional_primitives(path1, 10)
    primitives2 = system.hyperdimensional_primitives(path2, 10)
    pheromones1 = system.compute_pheromone(path1, voronoi1, primitives1)
    pheromones2 = system.compute_pheromone(path2, voronoi2, primitives2)
    return transformed_path1, transformed_path2, basis1, basis2, fisher1, fisher2, voronoi1, voronoi2, primitives1, primitives2, pheromones1, pheromones2

def hybrid_recovery(path, theta, center, width):
    system = HybridSystem()
    transformed_path = system.lead_lag_transform(path)
    basis = system.bspline_basis(np.linspace(0, 1, len(path)), np.linspace(0, 1, len(path)))
    fisher = system.fisher_score(theta, center, width)
    voronoi = system.voronoi_partition(path, 5)
    primitives = system.hyperdimensional_primitives(path, 10)
    pheromones = system.compute_pheromone(path, voronoi, primitives)
    recovery_priority = fisher * basis.sum() * np.sum(pheromones)
    return transformed_path, basis, fisher, voronoi, primitives, pheromones, recovery_priority

if __name__ == "__main__":
    path = np.random.rand(10, 2)
    theta = 0.5
    center = 0.5
    width = 0.1
    transformed_path, basis, fisher, voronoi, primitives, pheromones = hybrid_operation(path, theta, center, width)
    transformed_path1, transformed_path2, basis1, basis2, fisher1, fisher2, voronoi1, voronoi2, primitives1, primitives2, pheromones1, pheromones2 = hybrid_fusion(path, path, theta, center, width)
    transformed_path, basis, fisher, voronoi, primitives, pheromones, recovery_priority = hybrid_recovery(path, theta, center, width)
    print("Hybrid operation successful")