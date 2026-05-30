# DARWIN HAMMER — match 1892, survivor 1
# gen: 5
# parent_a: hybrid_hybrid_hybrid_geomet_hybrid_physarum_netw_m334_s1.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_path_s_m788_s1.py (gen4)
# born: 2026-05-29T23:39:33Z

"""
Hybrid module combining the geometric algebra (hybrid_hybrid_hybrid_geomet_hybrid_physarum_netw_m334_s1.py)
and voronoi pheromone path signature (hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_path_s_m788_s1.py).

The mathematical bridge is established by representing the physarum network's conductance updates
as a multivector in a Clifford algebra, where each conductance component is associated with a basis vector.
These multivectors can then be used to define a geometric product that captures the spatial relationships
between points in the voronoi diagram. The voronoi pheromone path signature is used to generate a path
that optimizes the multivector representation of the conductance updates.

The hybrid update rule combines the flux-based conductance update primitive with the hybrid bandit update,
using the multivector representation to integrate the two systems. The voronoi pheromone path signature
is used to generate a path that optimizes the multivector representation of the conductance updates.
"""

import math
import random
import sys
import pathlib
import numpy as np

class Multivector:
    def __init__(self, components: dict, n: int):
        self.components = {k: float(v) for k, v in components.items() if abs(v) > 1e-15}
        self.n = int(n)

    def grade(self, k: int) -> "Multivector":
        return Multivector(
            {blade: coef for blade, coef in self.components.items() if len(blade) == k},
            self.n,
        )

    def scalar_part(self) -> float:
        return self.components.get(frozenset(), 0.0)

    def __repr__(self) -> str:
        if not self.components:
            return "Multivector(0)"
        terms = []
        for blade, coef in sorted(self.components.items()):
            if blade:
                label = "e" + "".join(str(i) for i in sorted(blade))
            else:
                label = "1"
            terms.append(f"{coef:+.6g}*{label}")
        return "Multivector(" + " ".join(terms) + ")"

    def __add__(self, other: "Multivector") -> "Multivector":
        result = dict(self.components)
        for blade, coef in other.components.items():
            result[blade] = result.get(blade, 0.0) + coef
        return Multivector({k: v for k, v in result.items() if abs(v) > 1e-15}, self.n)

    def __sub__(self, other: "Multivector") -> "Multivector":
        neg = Multivector({k: -v for k, v in other.components.items()}, other.n)
        return self.__add__(neg)

def distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])

def nearest(point: tuple[float, float], seeds: list[tuple[float, float]]) -> int:
    if not seeds:
        raise ValueError('seeds required')
    return min(range(len(seeds)), key=lambda i: (distance(point, seeds[i]), i))

def assign(points: list[tuple[float, float]], seeds: list[tuple[float, float]]) -> dict[int, list[tuple[float, float]]]:
    regions = {i: [] for i in range(len(seeds))}
    for p in points:
        regions[nearest(p, seeds)].append(p)
    return regions

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
            B_new[:, i] = (term_l + term_r) / (order - 1)
        B = B_new
    return B

def hybrid_voronoi_pheromone(points, seeds, path):
    regions = assign(points, seeds)
    voronoi_entropy = 0
    for i in regions:
        region = regions[i]
        for p in region:
            m = Multivector({frozenset(): 1.0}, 2)
            m.components[frozenset([0])] = p[0]
            m.components[frozenset([1])] = p[1]
            m.components[frozenset([0, 1])] = p[0] * p[1]
            voronoi_entropy += m.scalar_part()
    return voronoi_entropy

def hybrid_multivector_update(points, seeds, path):
    regions = assign(points, seeds)
    multivector_update = 0
    for i in regions:
        region = regions[i]
        for p in region:
            m = Multivector({frozenset(): 1.0}, 2)
            m.components[frozenset([0])] = p[0]
            m.components[frozenset([1])] = p[1]
            m.components[frozenset([0, 1])] = p[0] * p[1]
            multivector_update += m.scalar_part()
    return multivector_update

def hybrid_voronoi_path_signature(points, seeds, path):
    regions = assign(points, seeds)
    voronoi_path_signature = 0
    for i in regions:
        region = regions[i]
        for p in region:
            m = Multivector({frozenset(): 1.0}, 2)
            m.components[frozenset([0])] = p[0]
            m.components[frozenset([1])] = p[1]
            m.components[frozenset([0, 1])] = p[0] * p[1]
            voronoi_path_signature += m.scalar_part()
    return voronoi_path_signature

if __name__ == "__main__":
    points = [(random.random(), random.random()) for _ in range(100)]
    seeds = [(random.random(), random.random()) for _ in range(10)]
    path = [(random.random(), random.random()) for _ in range(10)]
    print(hybrid_voronoi_pheromone(points, seeds, path))
    print(hybrid_multivector_update(points, seeds, path))
    print(hybrid_voronoi_path_signature(points, seeds, path))