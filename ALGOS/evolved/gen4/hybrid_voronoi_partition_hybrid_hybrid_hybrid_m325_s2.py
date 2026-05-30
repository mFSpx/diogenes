# DARWIN HAMMER — match 325, survivor 2
# gen: 4
# parent_a: voronoi_partition.py (gen0)
# parent_b: hybrid_hybrid_hybrid_sketch_dense_associative_me_m32_s0.py (gen3)
# born: 2026-05-29T23:28:15Z

"""
This module integrates the concepts of Voronoi partitioning from the voronoi_partition algorithm
and the Dense Associative Memory from the hybrid_hybrid_hybrid_sketch_dense_associative_me_m32_s0 algorithm.
The mathematical bridge between these two structures lies in the representation of data as vectors
and the use of linear transformations to define the Voronoi regions.
Here, we fuse these concepts by using the Voronoi partitioning to organize the data and the Dense Associative Memory
to perform pattern retrieval within each Voronoi region.
"""

import numpy as np
import math
import random
import sys
import pathlib
from typing import List, Tuple, Dict, Any

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
    scores = np.exp(beta * (sections @ query)) / np.sum(np.exp(beta * (sections @ query)))
    return scores

def distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])

def nearest(point: Tuple[float, float], seeds: List[Tuple[float, float]]) -> int:
    if not seeds:
        raise ValueError('seeds required')
    return min(range(len(seeds)), key=lambda i: (distance(point, seeds[i]), i))

def assign(points: List[Tuple[float, float]], seeds: List[Tuple[float, float]]) -> Dict[int, List[Tuple[float, float]]]:
    regions = {i: [] for i in range(len(seeds))}
    for p in points:
        regions[nearest(p, seeds)].append(p)
    return regions

def voronoi_retrieve(sheaf: Sheaf, points: List[Tuple[float, float]], seeds: List[Tuple[float, float]], query: np.ndarray, beta=1.0):
    regions = assign(points, seeds)
    scores = []
    for i in range(len(seeds)):
        region_points = regions[i]
        region_centroid = np.array([sum(x) / len(region_points) for x in zip(*region_points)])
        region_query = region_centroid - query
        region_score = energy(region_query, sheaf._sections[i], beta)
        scores.append(region_score)
    return scores

def hybrid_voronoi_retrieve(sheaf: Sheaf, points: List[Tuple[float, float]], seeds: List[Tuple[float, float]], query: np.ndarray, beta=1.0):
    regions = assign(points, seeds)
    scores = []
    for i in range(len(seeds)):
        region_points = regions[i]
        region_centroid = np.array([sum(x) / len(region_points) for x in zip(*region_points)])
        region_query = region_centroid - query
        region_score = hybrid_retrieve(sheaf, region_query, beta)
        scores.append(region_score)
    return scores

if __name__ == "__main__":
    sheaf = Sheaf({0: 2, 1: 2}, [(0, 1)])
    sheaf.set_restriction((0, 1), np.array([[1, 0], [0, 1]]), np.array([[1, 0], [0, 1]]))
    sheaf.set_section(0, np.array([1, 0]))
    sheaf.set_section(1, np.array([0, 1]))
    points = [(0, 0), (1, 1), (2, 2)]
    seeds = [(0, 0), (1, 1)]
    query = np.array([1, 1])
    print(voronoi_retrieve(sheaf, points, seeds, query))
    print(hybrid_voronoi_retrieve(sheaf, points, seeds, query))