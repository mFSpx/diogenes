# DARWIN HAMMER — match 1719, survivor 0
# gen: 5
# parent_a: hybrid_voronoi_partition_hybrid_hybrid_hybrid_m325_s1.py (gen4)
# parent_b: hybrid_hybrid_hybrid_worksh_hybrid_hybrid_path_s_m851_s4.py (gen4)
# born: 2026-05-29T23:38:25Z

# darwin_hammer.py
"""
This module integrates the concepts of Voronoi partitioning from the hybrid_voronoi_partition_hybrid_hybrid_hybrid_m325_s1 algorithm
and the lead-lag transforms from the hybrid_hybrid_hybrid_worksh_hybrid_hybrid_path_s_m851_s4 algorithm.
The mathematical bridge between these two structures lies in the representation of data as vectors
and the use of linear transformations to define the Voronoi regions and lead-lag transforms.
Here, we fuse these concepts by using the lead-lag transforms to organize the data within the Voronoi regions.
"""

import numpy as np
import math
import random
import sys
import pathlib

class Sheaf:
    def __init__(self, node_dims: dict, edges: list, seeds: list):
        self.node_dims: dict = dict(node_dims)
        self.edges: list = list(edges)
        self._restrictions: dict = {}
        self._sections: dict = {}
        self.seeds: list = seeds

    def set_restriction(
        self,
        edge: tuple,
        src_map: np.ndarray,
        dst_map: np.ndarray,
    ) -> None:
        u, v = edge
        if src_map.shape[1] != self.node_dims[u]:
            raise ValueError("src_map column dimension must match dim(node u)")
        if dst_map.shape[1] != self.node_dims[v]:
            raise ValueError("dst_map column dimension must match dim(node v)")
        if src_map.shape[0] != dst_map.shape[0]:
            raise ValueError("src_map and dst_map must have the same row dimension")
        self._restrictions[(u, v)] = (np.asarray(src_map, dtype=float), np.asarray(dst_map, dtype=float))

    def set_section(self, node: any, value: np.ndarray) -> None:
        self._sections[node] = np.asarray(value, dtype=float)

    def assign(self, points: list):
        regions = {}
        for point in points:
            node = self.nearest(point, self.seeds)
            if node not in regions:
                regions[node] = []
            regions[node].append(point)
        return regions

    def nearest(self, point: tuple, seeds: list):
        if not seeds:
            raise ValueError('seeds required')
        return min(range(len(seeds)), key=lambda i: self.distance(point, seeds[i]))

    def distance(self, a: tuple, b: tuple) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

def _softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()

def lead_lag_transform(path: np.ndarray) -> np.ndarray:
    path = np.asarray(path, dtype=float)
    T, d = path.shape
    out = np.empty((2 * T - 1, 2 * d), dtype=float)
    for t in range(T - 1):
        out[2 * t]     = np.concatenate([path[t],     path[t]])
        out[2 * t + 1] = np.concatenate([path[t + 1], path[t]])
    out[2 * (T - 1)] = np.concatenate([path[T - 1], path[T - 1]])
    return out

def weekday_weight_vector(groups: Sequence[str], dow: int) -> np.ndarray:
    n = len(groups)
    if n == 0:
        raise ValueError("groups must contain at least one element")
    base_angles = np.arange(n) * (2.0 * math.pi) / n
    phase = (2.0 * math.pi) * (dow % 7) / 7.0
    amplitude = 0.2
    raw = 1.0 + amplitude * np.sin(base_angles + phase)
    weight_vec = raw / raw.sum()
    return weight_vec.astype(np.float64)

def bspline_basis(x: np.ndarray, grid: np.ndarray, k: int = 3, weights: np.ndarray = None) -> np.ndarray:
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

    if weights is not None:
        B = B * weights[:, np.newaxis]

    for order in range(2, k + 1):
        B_new = np.zeros((N, len(t) - order), dtype=np.float64)
        for i in range(len(t) - order):
            denom_l = t[i + order - 1] - t[i]
            denom_r = t[i + order] - t[i + 1]
            term_l = (
                (x - t[i]) / denom_l * B[:, i]
            )
            term_r = (
                (t[i + order] - x) / denom_r * B[:, i + 1]
            )
            B_new[:, i] = term_l + term_r
        B = B_new
    return B

def fused_lead_lag_transform(self, node: any, value: np.ndarray, groups: Sequence[str], date: dt.date) -> np.ndarray:
    dow = (dt.date(date.year, date.month, date.day).weekday()
           % 7)
    weight_vec = weekday_weight_vector(groups, dow)
    lead_lag = lead_lag_transform(value)
    return lead_lag * weight_vec[:, np.newaxis]

def assign_with_fused_lead_lag(self, points: list, groups: Sequence[str], date: dt.date) -> np.ndarray:
    regions = {}
    for point in points:
        node = self.nearest(point, self.seeds)
        if node not in regions:
            regions[node] = []
        value = self._sections[node]
        lead_lag_value = self.fused_lead_lag_transform(node, value, groups, date)
        regions[node].append(lead_lag_value)
    return np.array(regions)

if __name__ == "__main__":
    np.random.seed(0)
    random.seed(0)
    groups = ['group1', 'group2', 'group3']
    date = dt.date(2022, 1, 1)

    node_dims = {0: 2, 1: 3, 2: 2}
    edges = [(0, 1), (1, 2)]
    seeds = [(0, 0), (1, 1), (2, 0)]
    sheaf = Sheaf(node_dims, edges, seeds)

    sheaf.set_section(0, np.random.rand(2))
    sheaf.set_section(1, np.random.rand(3))
    sheaf.set_section(2, np.random.rand(2))

    points = [(0.5, 0.5), (1.0, 1.0), (2.0, 0.0)]
    regions = sheaf.assign(points)
    fused_regions = sheaf.assign_with_fused_lead_lag(points, groups, date)
    print(fused_regions)