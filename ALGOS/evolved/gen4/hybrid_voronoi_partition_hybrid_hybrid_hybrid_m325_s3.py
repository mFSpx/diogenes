# DARWIN HAMMER — match 325, survivor 3
# gen: 4
# parent_a: voronoi_partition.py (gen0)
# parent_b: hybrid_hybrid_hybrid_sketch_dense_associative_me_m32_s0.py (gen3)
# born: 2026-05-29T23:28:15Z

"""
This module integrates the concepts of Voronoi partitioning from `voronoi_partition.py` 
and cellular sheaf theory with Dense Associative Memory from `hybrid_hybrid_hybrid_sketch_dense_associative_me_m32_s0.py`.
The mathematical bridge between these two structures lies in the representation of data as points in a metric space 
and the use of linear transformations to perform pattern retrieval.

The Voronoi partitioning provides a way to organize the data into regions based on proximity to seed points, 
while the cellular sheaf theory with Dense Associative Memory provides a way to perform pattern retrieval 
using linear transformations.

Here, we fuse these concepts by using the Voronoi partitioning to organize the data and 
the cellular sheaf theory with Dense Associative Memory to perform pattern retrieval.
"""

import numpy as np
import math
import random
from typing import List, Tuple, Dict

Point = tuple[float, float]

def distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])

def nearest(point: Point, seeds: list[Point]) -> int:
    if not seeds:
        raise ValueError('seeds required')
    return min(range(len(seeds)), key=lambda i: (distance(point, seeds[i]), i))

def assign(points: list[Point], seeds: list[Point]) -> dict[int, list[Point]]:
    regions = {i: [] for i in range(len(seeds))}
    for p in points:
        regions[nearest(p, seeds)].append(p)
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
    sections = np.array([sheaf._sections[n] for n in sheaf._sections])
    M = np.random.rand(sections.shape[1], sections.shape[1])
    return energy(query, M, beta)

def voronoi_retrieve(points: list[Point], seeds: list[Point], query: Point):
    regions = assign(points, seeds)
    nearest_seed = nearest(query, seeds)
    return regions[nearest_seed]

def hybrid_voronoi_retrieve(points: list[Point], seeds: list[Point], query: Point):
    sheaf = Sheaf({i: 2 for i in range(len(seeds))}, [(i, i) for i in range(len(seeds))])
    for i, seed in enumerate(seeds):
        sheaf.set_section(i, np.array([seed[0], seed[1]]))
    query_vec = np.array([query[0], query[1]])
    return hybrid_retrieve(sheaf, query_vec), voronoi_retrieve(points, seeds, query)

if __name__ == "__main__":
    points = [(random.random(), random.random()) for _ in range(100)]
    seeds = [(random.random(), random.random()) for _ in range(5)]
    query = (random.random(), random.random())
    hybrid_result, voronoi_result = hybrid_voronoi_retrieve(points, seeds, query)
    print(hybrid_result)
    print(voronoi_result)