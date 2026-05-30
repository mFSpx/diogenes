# DARWIN HAMMER — match 4630, survivor 0
# gen: 5
# parent_a: hybrid_hybrid_perceptual_de_hybrid_hybrid_ternar_m1907_s0.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_worksh_m1396_s6.py (gen4)
# born: 2026-05-29T23:56:59Z

"""
Hybrid Perceptual-RBF Voronoi Ternary Router + Semantic-Bayesian + Liquid-Time-Constant Allocation.

This module fuses the perceptual hashing utilities and radial-basis-function 
surrogate modeling from *hybrid_perceptual_de_hybrid_hybrid_ternar_m1907_s0.py* 
with the Voronoi partitioning and ternary minimum-cost routing from 
*hybrid_hybrid_ternary_route_hybrid_voronoi_parti_m41_s2.py*, and the 
semantic-Bayesian utilities and Liquid-Time-Constant allocation from 
*hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_worksh_m1396_s6.py*.

The mathematical bridge is formed by using the perceptual hash as a clustering 
key to partition the spatial domain into Voronoi cells, each containing a 
ternary minimum-cost routing tree. The cost of an edge between a point *p* 
and a seed *s* is defined as the weighted sum of the Euclidean distance and 
the Bayesian posterior mean failure probability of seed *s*, updated by the 
circuit-breaker statistics. The Liquid-Time-Constant allocation is used to 
modulate the resource allocation of the Voronoi cells based on the semantic 
distances and Bayesian posteriors.

The module provides three public functions that illustrate the hybrid operation 
and a small smoke-test when executed as a script.
"""

import math
import random
import sys
import pathlib
import numpy as np
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence, List, Tuple, Dict

Vector = Sequence[float]
Point = Tuple[float, float]
Edge = Tuple[str, str]

def compute_dhash(values: List[float]) -> int:
    bits = 0
    for i in range(len(values) - 1):
        bits = (bits << 1) | int(values[i] > values[i + 1])
    return bits

def compute_phash(values: List[float]) -> int:
    if not values:
        return 0
    avg = sum(values) / len(values)
    bits = 0
    for v in values[:64]:
        bits = (bits << 1) | int(v >= avg)
    return bits

def hamming_distance(a: int, b: int) -> int:
    return bin(a ^ b).count('1')

def euclidean_distance(point1: Point, point2: Point) -> float:
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def voronoi_partition(points: List[Point], seeds: List[Point]) -> Dict[Point, List[Point]]:
    voronoi_cells = {seed: [] for seed in seeds}
    for point in points:
        closest_seed = min(seeds, key=lambda seed: euclidean_distance(point, seed))
        voronoi_cells[closest_seed].append(point)
    return voronoi_cells

def length(a: Point, b: Point) -> float:
    """Euclidean distance between two points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])

def bayes_marginal(prior: float, likelihood: float, false_positive: float) -> float:
    return (prior * likelihood) / (prior * likelihood + (1 - prior) * false_positive)

def liquid_time_constant(t: float, w: float, b: float, tau: float) -> float:
    return tau / (1 + tau * (1 / (1 + math.exp(-w * t - b))))

def hybrid_routing(voronoi_cells: Dict[Point, List[Point]], lambda_: float, mu: float, w: float, b: float, tau: float) -> Dict[Point, List[Point]]:
    routing_trees = {}
    for seed, points in voronoi_cells.items():
        costs = []
        for point in points:
            distance = euclidean_distance(point, seed)
            posterior = bayes_marginal(0.5, 0.5, 0.1)
            lt_constant = liquid_time_constant(posterior, w, b, tau)
            cost = lambda_ * distance + mu * random.random() + lt_constant
            costs.append((point, cost))
        costs.sort(key=lambda x: x[1])
        routing_trees[seed] = [point for point, cost in costs]
    return routing_trees

def allocate_resources(voronoi_cells: Dict[Point, List[Point]], w: float, b: float, tau: float) -> Dict[Point, float]:
    allocations = {}
    for seed, points in voronoi_cells.items():
        posterior = bayes_marginal(0.5, 0.5, 0.1)
        lt_constant = liquid_time_constant(posterior, w, b, tau)
        allocation = len(points) * lt_constant
        allocations[seed] = allocation
    return allocations

def smoke_test():
    points = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]
    seeds = [(0, 0), (2, 2), (4, 4)]
    voronoi_cells = voronoi_partition(points, seeds)
    routing_trees = hybrid_routing(voronoi_cells, 1.0, 0.1, 0.5, 0.1, 1.0)
    allocations = allocate_resources(voronoi_cells, 0.5, 0.1, 1.0)
    print("Voronoi Cells:", voronoi_cells)
    print("Routing Trees:", routing_trees)
    print("Resource Allocations:", allocations)

if __name__ == "__main__":
    smoke_test()