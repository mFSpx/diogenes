# DARWIN HAMMER — match 325, survivor 4
# gen: 4
# parent_a: voronoi_partition.py (gen0)
# parent_b: hybrid_hybrid_hybrid_sketch_dense_associative_me_m32_s0.py (gen3)
# born: 2026-05-29T23:28:15Z

"""
This module fuses the Voronoi partitioning from `voronoi_partition.py` and 
the sheaf-based pattern retrieval from `hybrid_hybrid_hybrid_sketch_dense_associative_me_m32_s0.py`.
The mathematical bridge between these two structures lies in the representation 
of data as points in a metric space and the use of geometric transformations.

The Voronoi diagram provides a way to partition the space into regions based 
on proximity to a set of seed points. The sheaf structure, on the other hand, 
organizes data into a network of interconnected nodes, with each node 
representing a specific context or feature.

By using the Voronoi diagram to define the nodes of the sheaf, and the 
distance-based similarity metric to define the sheaf's restriction maps, 
we can create a hybrid system that combines the strengths of both approaches.
"""

import numpy as np
import math
import random
from typing import List, Tuple, Dict

class Point(tuple):
    pass

def distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])

def nearest(point: Point, seeds: List[Point]) -> int:
    if not seeds:
        raise ValueError('seeds required')
    return min(range(len(seeds)), key=lambda i: (distance(point, seeds[i]), i))

def assign(points: List[Point], seeds: List[Point]) -> Dict[int, List[Point]]:
    regions = {i: [] for i in range(len(seeds))}
    for p in points:
        regions[nearest(p, seeds)].append(p)
    return regions

class Sheaf:
    def __init__(self, node_dims: Dict[int, int], edges: List[Tuple[int, int]]):
        self.node_dims: Dict[int, int] = dict(node_dims)
        self.edges: List[Tuple[int, int]] = list(edges)
        self._restrictions: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]] = {}
        self._sections: Dict[int, np.ndarray] = {}

    def set_restriction(
        self,
        edge: Tuple[int, int],
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

    def set_section(self, node: int, value: np.ndarray) -> None:
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
    M = np.random.rand(sections.shape[0], sections.shape[1])
    return energy(query, M, beta)

def voronoi_sheaf(points: List[Point], seeds: List[Point]) -> Sheaf:
    regions = assign(points, seeds)
    node_dims = {i: 2 for i in range(len(seeds))}
    edges = [(u, v) for u in range(len(seeds)) for v in range(len(seeds)) if u != v]
    sheaf = Sheaf(node_dims, edges)
    for i, region in regions.items():
        section = np.array(region).mean(axis=0)
        sheaf.set_section(i, section)
    return sheaf

def hybrid_query(sheaf: Sheaf, query_point: Point) -> np.ndarray:
    query = np.array(query_point)
    return hybrid_retrieve(sheaf, query)

if __name__ == "__main__":
    points = [Point(random.random(), random.random()) for _ in range(100)]
    seeds = [Point(random.random(), random.random()) for _ in range(5)]
    sheaf = voronoi_sheaf(points, seeds)
    query_point = Point(0.5, 0.5)
    result = hybrid_query(sheaf, query_point)
    print(result)