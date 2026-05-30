# DARWIN HAMMER — match 4682, survivor 1
# gen: 4
# parent_a: hybrid_hybrid_nlms_omni_cha_hybrid_gliner_zero_s_m26_s5.py (gen2)
# parent_b: hybrid_hybrid_ternary_route_hybrid_voronoi_parti_m41_s3.py (gen3)
# born: 2026-05-29T23:57:20Z

"""
This module fuses two parent algorithms:
* **Parent A** – Hybrid NLMS-Graph-Tree Fusion (hybrid_hybrid_nlms_omni_cha_hybrid_gliner_zero_s_m26_s5.py)
* **Parent B** – Hybrid Ternary Route Hybrid Voronoi Partition (hybrid_hybrid_ternary_route_hybrid_voronoi_parti_m41_s3.py)

The mathematical bridge between the two parents is the use of point-to-point distances and Voronoi diagrams in Parent B, 
which can be used to define the similarity matrix in Parent A. The similarity matrix S is used to update the weight vector 
in the NLMS algorithm, which in turn is used to estimate the cost of edges in the minimum-cost tree problem.

By integrating the Voronoi partitioning from Parent B into the NLMS algorithm of Parent A, we create a hybrid system that 
learns to adaptively weight graph edges based on the geometric relationships between points, while still solving the 
classic minimum-cost tree problem.
"""

import json
import math
import random
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

Point = Tuple[float, float]

def euclidean(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])

def nearest(point: Point, seeds: List[Point]) -> int:
    if not seeds:
        raise ValueError("seed list cannot be empty")
    return min(range(len(seeds)), key=lambda i: (euclidean(point, seeds[i]), i))

def assign_voronoi(points: List[Point], seeds: List[Point]) -> dict:
    regions: dict = {i: [] for i in range(len(seeds))}
    for p in points:
        regions[nearest(p, seeds)].append(p)
    return regions

def predict(weights: np.ndarray, x: np.ndarray) -> float:
    return float(np.dot(weights, x))

def nlms_update(
    weights: np.ndarray,
    x: np.ndarray,
    target: float,
    mu: float = 0.5,
    eps: float = 1e-9,
) -> Tuple[np.ndarray, float]:
    y = predict(weights, x)
    error = target - y
    weights = weights + mu * (error / (eps + np.dot(x, x))) * x
    return weights, error

def compute_edge_cost(
    point: Point,
    seed: Point,
    estimator: float,
) -> float:
    return euclidean(point, seed) * estimator

def hybrid_nlms_voronoi(points: List[Point], seeds: List[Point], mu: float = 0.5, eps: float = 1e-9) -> Tuple[np.ndarray, dict]:
    weights = np.zeros(len(seeds))
    regions = assign_voronoi(points, seeds)
    for i, region in regions.items():
        x = np.array([euclidean(point, seeds[i]) for point in region])
        target = np.mean(x)
        weights[i], _ = nlms_update(np.array([weights[i]]), np.array([1]), target, mu, eps)
    return weights, regions

def hybrid_voronoi_tree(points: List[Point], seeds: List[Point], mu: float = 0.5, eps: float = 1e-9) -> dict:
    weights, regions = hybrid_nlms_voronoi(points, seeds, mu, eps)
    tree = {}
    for i, region in regions.items():
        tree[i] = {}
        for point in region:
            tree[i][point] = compute_edge_cost(point, seeds[i], weights[i])
    return tree

if __name__ == "__main__":
    points = [(1, 2), (3, 4), (5, 6)]
    seeds = [(0, 0), (2, 2), (4, 4)]
    tree = hybrid_voronoi_tree(points, seeds)
    print(tree)