# DARWIN HAMMER — match 5509, survivor 1
# gen: 6
# parent_a: hybrid_hybrid_fisher_locali_hybrid_minimum_cost__m29_s1.py (gen2)
# parent_b: hybrid_hybrid_hybrid_vorono_hybrid_liquid_time_c_m633_s2.py (gen5)
# born: 2026-05-30T00:02:29Z

"""
This module integrates the Fisher information scoring and minimum-cost tree scoring from 
hybrid_hybrid_fisher_locali_hybrid_minimum_cost__m29_s1.py, and the Voronoi partitioning 
and hyperdimensional primitives from hybrid_hybrid_hybrid_vorono_hybrid_liquid_time_c_m633_s2.py.
The mathematical bridge between these structures is the use of Gaussian distributions to model 
uncertainty in the Voronoi regions and the application of Fisher information scoring to 
inform model selection in the hyperdimensional primitives.
"""

import math
import numpy as np
import random
import sys
import pathlib
from datetime import datetime

Point = tuple[float, float]
Edge = tuple[str, str]

def gaussian_beam(theta: float, center: float, width: float) -> float:
    if width <= 0:
        raise ValueError('width must be positive')
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)

def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity

def length(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])

def tree_cost(nodes: dict[str, Point], edges: list[Edge], root: str, path_weight: float = 0.2) -> float:
    adj: dict[str, list[str]] = {n: [] for n in nodes}
    material = 0.0
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
        material += length(nodes[a], nodes[b])
    dist = {root: 0.0}
    stack = [root]
    while stack:
        a = stack.pop()
        for b in adj[a]:
            if b not in dist:
                dist[b] = dist[a] + length(nodes[a], nodes[b])
                stack.append(b)
    return material + path_weight * sum(dist.values())

def bayes_marginal(prior: float, likelihood: float, false_positive: float) -> float:
    if not all(0 <= x <= 1 for x in (prior, likelihood, false_positive)):
        raise ValueError("probabilities must be in [0,1]")

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

def sigmoid(x: np.ndarray) -> np.ndarray:
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x)),
    )

def ltc_f(
    x: np.ndarray,
    I: np.ndarray,
    W: np.ndarray,
    b: np.ndarray,
) -> np.ndarray:
    return sigmoid(np.dot(W, np.concatenate((x, I))) + b)

def random_vector(dim: int = 10000, seed: str | int | None = None) -> np.ndarray:
    rng = random.Random(seed)
    data = np.fromiter(
        (1 if rng.getrandbits(1) else -1 for _ in range(dim)),
        dtype=float,
    )
    return data / np.linalg.norm(data)

def hybrid_fisher_voronoi(points: list[Point], seeds: list[Point]) -> dict[int, float]:
    regions = assign(points, seeds)
    fisher_scores = {}
    for i, region in regions.items():
        center = seeds[i]
        width = np.std([distance(p, center) for p in region])
        theta = np.mean([distance(p, center) for p in region])
        fisher_scores[i] = fisher_score(theta, center, width)
    return fisher_scores

def hybrid_voronoi_tree(nodes: dict[str, Point], edges: list[Edge], root: str, seeds: list[Point]) -> float:
    points = list(nodes.values())
    regions = assign(points, seeds)
    tree_costs = {}
    for i, region in regions.items():
        region_nodes = {n: p for n, p in nodes.items() if p in region}
        region_edges = [e for e in edges if e[0] in region_nodes and e[1] in region_nodes]
        tree_costs[i] = tree_cost(region_nodes, region_edges, root)
    return np.mean(list(tree_costs.values()))

def hybrid_ltcf_fusion(x: np.ndarray, I: np.ndarray, W: np.ndarray, b: np.ndarray, seeds: list[Point]) -> np.ndarray:
    ltc_output = ltc_f(x, I, W, b)
    fisher_scores = {}
    for i, seed in enumerate(seeds):
        center = seed
        width = np.std([distance((p[0], p[1]), center) for p in zip(x, I)])
        theta = np.mean([distance((p[0], p[1]), center) for p in zip(x, I)])
        fisher_scores[i] = fisher_score(theta, center, width)
    return ltc_output * np.array(list(fisher_scores.values()))

if __name__ == "__main__":
    points = [(1, 2), (3, 4), (5, 6)]
    seeds = [(0, 0), (10, 10)]
    print(hybrid_fisher_voronoi(points, seeds))
    nodes = {'A': (0, 0), 'B': (10, 10)}
    edges = [('A', 'B')]
    print(hybrid_voronoi_tree(nodes, edges, 'A', seeds))
    x = np.array([1, 2])
    I = np.array([3, 4])
    W = np.random.rand(2, 4)
    b = np.random.rand(2)
    print(hybrid_ltcf_fusion(x, I, W, b, seeds))