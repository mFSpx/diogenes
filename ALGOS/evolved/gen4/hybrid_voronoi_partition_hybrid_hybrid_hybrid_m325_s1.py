# DARWIN HAMMER — match 325, survivor 1
# gen: 4
# parent_a: voronoi_partition.py (gen0)
# parent_b: hybrid_hybrid_hybrid_sketch_dense_associative_me_m32_s0.py (gen3)
# born: 2026-05-29T23:28:15Z

"""
This module integrates the concepts of Voronoi partitioning from the voronoi_partition algorithm
and cellular sheaf theory from the hybrid_hybrid_hybrid_sketch_dense_associative_me_m32_s0 algorithm.
The mathematical bridge between these two structures lies in the representation of data as vectors
and the use of linear transformations to define the Voronoi regions as sheaf sections.
Here, we fuse these concepts by using the sheaf structure to organize the data and the Voronoi
partitioning to define the regions of the sheaf.
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
    return energy(query, sections, beta)

def voronoi_sheaf(points: list, seeds: list):
    sheaf = Sheaf(node_dims={i: len(points) for i in range(len(seeds))}, edges=[], seeds=seeds)
    regions = sheaf.assign(points)
    for i, region in regions.items():
        section = np.array(region)
        sheaf.set_section(i, section)
    return sheaf

if __name__ == "__main__":
    points = [(random.random(), random.random()) for _ in range(100)]
    seeds = [(random.random(), random.random()) for _ in range(5)]
    sheaf = voronoi_sheaf(points, seeds)
    query = np.array([random.random() for _ in range(100)])
    energy_value = hybrid_retrieve(sheaf, query)
    print("Energy value:", energy_value)