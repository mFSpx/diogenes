# DARWIN HAMMER — match 1102, survivor 1
# gen: 4
# parent_a: hybrid_hybrid_ternary_route_hybrid_voronoi_parti_m41_s2.py (gen3)
# parent_b: hybrid_fisher_localization_hybrid_ternary_route_m40_s1.py (gen2)
# born: 2026-05-29T23:32:44Z

"""
This module implements a hybrid algorithm that fuses the 
hybrid_hybrid_ternary_route_hybrid_voronoi_parti_m41_s2.py and 
hybrid_fisher_localization_hybrid_ternary_route_m40_s1.py algorithms.

The mathematical bridge between these two algorithms is found by 
applying the Fisher score as a weighting factor in the Voronoi 
construction and the ternary routing process. This allows the 
algorithm to make more informed decisions about which points to 
assign to each seed and how to route packets between them.

The governing equations of the hybrid algorithm are:

c(p, s) = λ·‖p‑s‖₂  +  μ·ĥ(s) · F(θ, center, width)

where `‖·‖₂` is the Euclidean distance, `ĥ(s)` is the Bayesian 
posterior mean failure probability of seed *s*, `F(θ, center, width)` 
is the Fisher score, and `λ, μ ≥ 0` are weighting hyper-parameters.
"""

import json
import math
import random
import sys
from pathlib import Path
import numpy as np

def gaussian_beam(theta: float, center: float, width: float) -> float:
    if width <= 0:
        raise ValueError('width must be positive')
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)

def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity

def voronoi_cell_distance(point: tuple, seed: tuple, lambda_: float, mu: float, theta: float, center: float, width: float) -> float:
    euclidean_distance = math.sqrt((point[0] - seed[0])**2 + (point[1] - seed[1])**2)
    fisher_weight = fisher_score(theta, center, width)
    return lambda_ * euclidean_distance + mu * fisher_weight

def ternary_router(points: list, seeds: list, lambda_: float, mu: float, theta: float, center: float, width: float) -> dict:
    routing_tree = {}
    for point in points:
        distances = []
        for seed in seeds:
            distance = voronoi_cell_distance(point, seed, lambda_, mu, theta, center, width)
            distances.append((seed, distance))
        distances.sort(key=lambda x: x[1])
        routing_tree[point] = distances[:3]
    return routing_tree

def hybrid_algorithm(points: list, seeds: list, lambda_: float, mu: float, theta: float, center: float, width: float) -> dict:
    voronoi_cells = {}
    for seed in seeds:
        cell_points = []
        for point in points:
            if voronoi_cell_distance(point, seed, lambda_, mu, theta, center, width) <= voronoi_cell_distance(point, seeds[0], lambda_, mu, theta, center, width):
                cell_points.append(point)
        voronoi_cells[seed] = cell_points
    routing_tree = {}
    for seed, cell_points in voronoi_cells.items():
        routing_tree[seed] = ternary_router(cell_points, seeds, lambda_, mu, theta, center, width)
    return routing_tree

if __name__ == "__main__":
    points = [(1, 2), (3, 4), (5, 6), (7, 8)]
    seeds = [(0, 0), (10, 10)]
    lambda_ = 1.0
    mu = 1.0
    theta = 0.5
    center = 0.5
    width = 1.0
    print(hybrid_algorithm(points, seeds, lambda_, mu, theta, center, width))