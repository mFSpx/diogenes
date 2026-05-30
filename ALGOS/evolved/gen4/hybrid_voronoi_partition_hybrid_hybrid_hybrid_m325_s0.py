# DARWIN HAMMER — match 325, survivor 0
# gen: 4
# parent_a: voronoi_partition.py (gen0)
# parent_b: hybrid_hybrid_hybrid_sketch_dense_associative_me_m32_s0.py (gen3)
# born: 2026-05-29T23:28:15Z

# voronoi_dense_associative_sheaf.py

"""
This module integrates the concepts of Voronoi partitioning from the `voronoi_partition.py` algorithm
and Dense Associative Memory from the `hybrid_hybrid_hybrid_sketch_dense_associative_me_m32_s0.py` algorithm.
The mathematical bridge between these two structures lies in the use of nearest neighbor distances for data organization,
and linear transformations for pattern retrieval. By fusing these concepts, we create a hybrid system where Voronoi
partitions are used to organize data points and Dense Associative Memory is used to perform pattern retrieval.
"""

import numpy as np
import math
import random
import sys
import pathlib

def distance(a: np.ndarray, b: np.ndarray) -> float:
    return np.linalg.norm(a - b)

def nearest(point: np.ndarray, seeds: np.ndarray) -> int:
    if not seeds.size:
        raise ValueError('seeds required')
    return np.argmin(np.apply_along_axis(lambda x: distance(point, x), 1, seeds))

def assign(points: np.ndarray, seeds: np.ndarray) -> np.ndarray:
    regions = np.zeros((seeds.shape[0], points.shape[0]), dtype=int)
    for i, p in enumerate(points):
        regions[nearest(p, seeds), i] = 1
    return regions

class Sheaf:
    def __init__(self, node_dims: Dict[Any, int], edges: List[Tuple[Any, Any]]):
        self.node_dims: Dict[Any, int] = dict(node_dims)
        self.edges: List[Tuple[Any, Any]] = list(edges)
        self._restrictions: Dict[Tuple[Any, Any], Tuple[np.ndarray, np.ndarray]] = {}
        self._sections: Dict[Any, np.ndarray] = {}

    def set_restriction(
        self,
        edge: Tuple[Any, Any],
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

    def set_section(self, node: Any, value: np.ndarray) -> None:
        self._sections[node] = np.asarray(value, dtype=float)

def _softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()

def _lse(z):
    m = z.max()
    return m + np.log(np.exp(z - m).sum())

def energy(xi, M, beta=1.0):
    xi = np.asarray(xi, dtype=float)
    M = np.asarray(M, dtype=float)
    scores = beta * (M @ xi)
    lse_term = _lse(scores) / beta
    quadratic_term = 0.5 * np.sum(xi ** 2)
    return -np.log(np.exp(beta * (M @ xi)).sum()) + quadratic_term

def hybrid_retrieve(sheaf: Sheaf, query: np.ndarray, beta=1.0):
    sections = np.array([sheaf._sections[nod] for nod in sheaf._sections])
    scores = beta * (sections @ query.T)
    lse_term = _lse(scores) / beta
    quadratic_term = 0.5 * np.sum(query ** 2)
    return -np.log(np.exp(beta * scores).sum()) + quadratic_term

def hybrid_train(points: np.ndarray, seeds: np.ndarray):
    sheaf = Sheaf({i: 2 for i in range(seeds.shape[0])}, [(i, j) for i in range(seeds.shape[0]) for j in range(i+1, seeds.shape[0])])
    regions = assign(points, seeds)
    for i, region in enumerate(regions):
        section = np.mean(points[region == 1], axis=0)
        sheaf.set_section(i, section)
    return sheaf

def hybrid_test(sheaf: Sheaf, points: np.ndarray):
    regions = assign(points, np.array([0]*sheaf.node_dims[0] + [1]*sheaf.node_dims[1]))
    for i, region in enumerate(regions):
        section = sheaf._sections[i]
        similarity = np.dot(section, points[i])
        print(f"Point {i} assigned to region {np.where(region == 1)[0][0]} with similarity {similarity}")

if __name__ == "__main__":
    points = np.random.rand(10, 2)
    seeds = np.random.rand(2, 2)
    sheaf = hybrid_train(points, seeds)
    hybrid_test(sheaf, points)