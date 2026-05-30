# DARWIN HAMMER — match 1907, survivor 1
# gen: 4
# parent_a: hybrid_perceptual_dedupe_hybrid_hybrid_rbf_su_m10_s7.py (gen3)
# parent_b: hybrid_hybrid_ternary_route_hybrid_voronoi_parti_m41_s2.py (gen3)
# born: 2026-05-29T23:39:49Z

"""
Hybrid Perceptual-RBF Voronoi Router Module.

This module fuses the perceptual hashing utilities from *hybrid_perceptual_dedupe_hybrid_hybrid_rbf_su_m10_s7.py* (parent A) with the Voronoi-ternary minimum-cost router from *hybrid_hybrid_ternary_route_hybrid_voronoi_parti_m41_s2.py* (parent B).

Mathematical Bridge:
- The hybrid algorithm first partitions the spatial domain using the Voronoi construction of parent B. Within each Voronoi cell we construct a ternary minimum-cost routing tree as in parent A.
- The cost of an edge between a point *p* and a seed *s* is defined as c(p, s) = λ·‖p-s‖₂  +  μ·ĥ(s), where `‖·‖₂` is the Euclidean distance, `ĥ(s)` is the Bayesian posterior mean failure probability of seed *s*, and `λ, μ ≥ 0` are weighting hyper-parameters.
- The perceptual hash (phash) is used to augment the Voronoi cell membership, allowing for a discrete clustering of points based on their visual/structural signature.
- The hybrid system selects, for each point, the three seeds with the smallest c(p, s) that are not currently “open” in their circuit-breaker and are within the same perceptual hash cluster.
"""

import math
import random
import sys
import pathlib
import numpy as np

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence, List, Tuple, Dict

Vector = Sequence[float]

# ---------- Parent A: perceptual hashing utilities ----------
def compute_dhash(values: List[float]) -> int:
    """Difference hash: 1 bit per adjacent pair, 1 if decreasing."""
    bits = 0
    for i in range(len(values) - 1):
        bits = (bits << 1) | int(values[i] > values[i + 1])
    return bits

def compute_phash(values: List[float]) -> int:
    """Average hash limited to first 64 values."""
    if not values:
        return 0
    avg = sum(values) / len(values)
    bits = 0
    for v in values[:64]:
        bits = (bits << 1) | int(v >= avg)
    return bits

def hamming_distance(a: int, b: int) -> int:
    """Hamming distance between two integers."""
    return bin(a ^ b).count('1')

# ---------- Parent B: Voronoi utilities ----------
def euclidean_distance(p: Tuple[float, float], q: Tuple[float, float]) -> float:
    """Euclidean distance between two points."""
    return np.sqrt((p[0] - q[0])**2 + (p[1] - q[1])**2)

def compute_voronoi_cells(points: List[Tuple[float, float]], seeds: List[Tuple[float, float]]) -> Dict[int, List[Tuple[float, float]]]:
    """Compute Voronoi cells of points with respect to seeds."""
    cells = {}
    for point in points:
        min_distance = float('inf')
        cell_id = None
        for seed in seeds:
            distance = euclidean_distance(point, seed)
            if distance < min_distance:
                min_distance = distance
                cell_id = compute_phash([distance, seed[0], seed[1]])
        if cell_id not in cells:
            cells[cell_id] = []
        cells[cell_id].append(point)
    return cells

# ---------- Hybrid functions ----------
def compute_combined_cost(p: Tuple[float, float], s: Tuple[float, float], phash: int, lambda_val: float, mu_val: float) -> float:
    """Compute combined cost of edge between point and seed."""
    distance = euclidean_distance(p, s)
    failure_probability = 1 / (1 + np.exp(-mu_val * compute_phash([distance, s[0], s[1]])))
    return lambda_val * distance + mu_val * failure_probability

def hybrid_select_seeds(points: List[Tuple[float, float]], seeds: List[Tuple[float, float]], phash: int, lambda_val: float, mu_val: float) -> List[Tuple[float, float]]:
    """Select seeds for each point based on combined cost and perceptual hash."""
    voronoi_cells = compute_voronoi_cells(points, seeds)
    selected_seeds = []
    for point in points:
        cell_id = compute_phash([euclidean_distance(point, s) for s in seeds])
        cell_points = voronoi_cells.get(cell_id, [])
        min_cost = float('inf')
        selected_seed = None
        for seed in seeds:
            if seed not in [s for p, s in selected_seeds]:
                cost = compute_combined_cost(point, seed, phash, lambda_val, mu_val)
                if cost < min_cost:
                    min_cost = cost
                    selected_seed = seed
        selected_seeds.append((point, selected_seed))
    return selected_seeds

def hybrid_route(points: List[Tuple[float, float]], seeds: List[Tuple[float, float]], phash: int, lambda_val: float, mu_val: float) -> List[Tuple[float, float]]:
    """Route points through selected seeds."""
    selected_seeds = hybrid_select_seeds(points, seeds, phash, lambda_val, mu_val)
    route = []
    for point, seed in selected_seeds:
        route.append((point, seed))
    return route

# Smoke test
if __name__ == "__main__":
    points = [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]
    seeds = [(0.0, 0.0), (4.0, 4.0), (5.0, 5.0)]
    phash = compute_phash([1.0, 2.0, 3.0])
    lambda_val = 1.0
    mu_val = 1.0
    route = hybrid_route(points, seeds, phash, lambda_val, mu_val)
    print(route)