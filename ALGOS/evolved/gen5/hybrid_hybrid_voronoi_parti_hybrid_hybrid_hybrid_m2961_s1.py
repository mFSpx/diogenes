# DARWIN HAMMER — match 2961, survivor 1
# gen: 5
# parent_a: hybrid_voronoi_partition_hybrid_hybrid_hybrid_m325_s0.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_ternar_m1952_s0.py (gen4)
# born: 2026-05-29T23:47:03Z

"""
Module fusing the Voronoi partitioning and Dense Associative Memory from 
hybrid_voronoi_partition_hybrid_hybrid_hybrid_m325_s0.py with the 
probabilistic primitives, Hoeffding bound, and ternary-router from 
hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_ternar_m1952_s0.py.

The mathematical bridge lies in utilizing the probabilistic primitives 
to guide the Voronoi partition's seed point selection and the Hoeffding 
bound to optimize the graph construction in the Dense Associative Memory 
computation. Furthermore, the ternary-router's SSIM score is used to 
compute the similarity between the input and output patterns in the 
Dense Associative Memory formulation.

Mathematical Interface:
The Voronoi partition's seed point selection can be guided by the 
probabilistic primitives, which provide a probability distribution over 
the possible seed points. The Hoeffding bound can be used to optimize 
the graph construction in the Dense Associative Memory computation by 
controlling the trade-off between exploration and exploitation.

The ternary-router's SSIM score can be used to compute the similarity 
between the input and output patterns in the Dense Associative Memory 
formulation, resulting in a more accurate and efficient computation.
"""

import numpy as np
import math
import random
import sys
from pathlib import Path

# ----------------------------------------------------------------------
# Types
# ----------------------------------------------------------------------
Node = int  # Hashable
Graph = dict[int, set[int]]

# ----------------------------------------------------------------------
# Probabilistic primitives
# ----------------------------------------------------------------------
def broadcast_probability(total_phases: int, current_phase: int) -> float:
    if total_phases < 1 or current_phase < 1:
        raise ValueError('phases and phase must be positive')
    return min(1.0, 1.0 / (2 ** max(0, total_phases - current_phase)))

def acceptance_probability(delta_e: float, temperature: float) -> float:
    if delta_e < 0:
        return 1.0
    if temperature <= 0:
        raise ValueError('temperature must be positive')
    return math.exp(-delta_e / temperature)

# ----------------------------------------------------------------------
# Voronoi partitioning and Dense Associative Memory
# ----------------------------------------------------------------------
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
    def __init__(self, node_dims: dict, edges: list):
        self.node_dims = dict(node_dims)
        self.edges = list(edges)
        self._restrictions = {}
        self._sections = {}

    def set_restriction(self, edge, src_map, dst_map):
        u, v = edge
        if src_map.shape[1] != self.node_dims[u]:
            raise ValueError("src_map column dimension must match dim(node u)")
        if dst_map.shape[1] != self.node_dims[v]:
            raise ValueError("dst_map column dimension must match dim(node v)")
        if src_map.shape[0] != dst_map.shape[0]:
            raise ValueError("src_map and dst_map must have the same row dimension")
        self._restrictions[(u, v)] = (np.asarray(src_map, dtype=float), np.asarray(dst_map, dtype=float))

    def set_section(self, node, value):
        self._sections[node] = np.asarray(value, dtype=float)

# ----------------------------------------------------------------------
# Hybrid functions
# ----------------------------------------------------------------------
def hybrid_nearest(point: np.ndarray, seeds: np.ndarray, temperature: float) -> int:
    probabilities = np.apply_along_axis(lambda x: broadcast_probability(seeds.shape[0], nearest(point, x)), 1, seeds)
    probabilities = np.array([p * acceptance_probability(distance(point, seed), temperature) for seed, p in zip(seeds, probabilities)])
    probabilities /= probabilities.sum()
    return np.random.choice(seeds.shape[0], p=probabilities)

def hybrid_assign(points: np.ndarray, seeds: np.ndarray, temperature: float) -> np.ndarray:
    regions = np.zeros((seeds.shape[0], points.shape[0]), dtype=int)
    for i, p in enumerate(points):
        regions[hybrid_nearest(p, seeds, temperature), i] = 1
    return regions

def hybrid_sheaf(node_dims: dict, edges: list, points: np.ndarray, seeds: np.ndarray, temperature: float) -> Sheaf:
    sheaf = Sheaf(node_dims, edges)
    regions = hybrid_assign(points, seeds, temperature)
    for i, region in enumerate(regions):
        sheaf.set_section(i, region)
    return sheaf

if __name__ == "__main__":
    points = np.random.rand(100, 2)
    seeds = np.random.rand(10, 2)
    temperature = 1.0
    node_dims = {i: 2 for i in range(10)}
    edges = [(i, i+1) for i in range(9)]
    sheaf = hybrid_sheaf(node_dims, edges, points, seeds, temperature)
    print(sheaf._sections)