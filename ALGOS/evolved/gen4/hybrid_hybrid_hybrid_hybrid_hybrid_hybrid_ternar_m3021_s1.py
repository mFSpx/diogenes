# DARWIN HAMMER — match 3021, survivor 1
# gen: 4
# parent_a: hybrid_hybrid_hybrid_geomet_hybrid_rbf_surrogate_m409_s6.py (gen3)
# parent_b: hybrid_hybrid_ternary_lens__hybrid_hybrid_model__m716_s2.py (gen3)
# born: 2026-05-29T23:47:26Z

"""
Hybrid Algorithm: Fusing Geometric Voronoi-RBF and Audit-Pruning Topologies

This hybrid algorithm combines the strengths of two parent algorithms:
1. hybrid_hybrid_hybrid_geomet_hybrid_rbf_surrogate_m409_s6.py (Voronoi-RBF)
2. hybrid_hybrid_ternary_lens__hybrid_hybrid_model__m716_s2.py (Audit-Pruning)

Mathematical bridge:
The Voronoi-RBF algorithm generates a similarity matrix between seeds using Gaussian RBF and perceptual hashing.
The Audit-Pruning algorithm modulates a prune probability matrix using a weight vector derived from a count vector.

The hybrid algorithm integrates these by:
1. Computing a weighted similarity matrix between seeds using the Voronoi-RBF method.
2. Deriving a count vector from the weighted similarity matrix.
3. Modulating a prune probability matrix using the count vector as weights.

This fusion enables a stochastic, similarity-aware Voronoi partition with adaptive pruning.
"""

import math
import random
import sys
import numpy as np
from typing import List, Tuple, Dict

# ----------------------------------------------------------------------
# Basic geometric utilities (from Parent A)
# ----------------------------------------------------------------------
Point = Tuple[float, float]

def distance(a: Point, b: Point) -> float:
    """Euclidean distance between two 2‑D points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])

def nearest(point: Point, seeds: List[Point]) -> int:
    """Index of the nearest seed (ties broken by lower index)."""
    if not seeds:
        raise ValueError('seeds required')
    return min(range(len(seeds)), key=lambda i: (distance(point, seeds[i]), i))

def assign(points: List[Point], seeds: List[Point]) -> Dict[int, List[Point]]:
    """Deterministic Voronoi assignment of points to the nearest seed."""
    regions: Dict[int, List[Point]] = {}
    for point in points:
        seed_idx = nearest(point, seeds)
        if seed_idx not in regions:
            regions[seed_idx] = []
        regions[seed_idx].append(point)
    return regions

# ----------------------------------------------------------------------
# Voronoi-RBF utilities
# ----------------------------------------------------------------------
def gaussian_rbf(x: float, sigma: float) -> float:
    """Gaussian radial basis function."""
    return np.exp(-x**2 / (2 * sigma**2))

def weighted_similarity_matrix(seeds: List[Point], sigma: float) -> np.ndarray:
    """Weighted similarity matrix between seeds using Gaussian RBF."""
    num_seeds = len(seeds)
    similarity_matrix = np.zeros((num_seeds, num_seeds))
    for i in range(num_seeds):
        for j in range(i+1, num_seeds):
            dist = distance(seeds[i], seeds[j])
            similarity = gaussian_rbf(dist, sigma)
            similarity_matrix[i, j] = similarity
            similarity_matrix[j, i] = similarity
    return similarity_matrix

# ----------------------------------------------------------------------
# Audit-Pruning utilities
# ----------------------------------------------------------------------
def count_vector(similarity_matrix: np.ndarray) -> np.ndarray:
    """Count vector from a similarity matrix."""
    return np.sum(similarity_matrix, axis=1)

def modulate_prune_probability(count_vector: np.ndarray, prune_probability: float) -> np.ndarray:
    """Modulate prune probability using a count vector as weights."""
    weights = count_vector / np.sum(count_vector)
    return prune_probability * weights

# ----------------------------------------------------------------------
# Hybrid functions
# ----------------------------------------------------------------------
def hybrid_voronoi_rbf_audit_pruning(points: List[Point], seeds: List[Point], sigma: float, prune_probability: float) -> Dict[int, List[Point]]:
    """Hybrid Voronoi-RBF audit-pruning algorithm."""
    regions = assign(points, seeds)
    similarity_matrix = weighted_similarity_matrix(seeds, sigma)
    count_vec = count_vector(similarity_matrix)
    modulated_prune_probability = modulate_prune_probability(count_vec, prune_probability)
    
    # Stochastic re-assignment of points to seeds based on modulated prune probability
    for seed_idx, region in regions.items():
        if random.random() < modulated_prune_probability[seed_idx]:
            # Re-assign points to a new seed
            new_seed_idx = np.random.choice(list(regions.keys()))
            regions[new_seed_idx] = regions.get(new_seed_idx, []) + region
            del regions[seed_idx]
    
    return regions

def hybrid_smoke_test():
    points = [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0), (4.0, 4.0)]
    seeds = [(0.0, 0.0), (5.0, 5.0)]
    sigma = 1.0
    prune_probability = 0.5
    
    regions = hybrid_voronoi_rbf_audit_pruning(points, seeds, sigma, prune_probability)
    print(regions)

if __name__ == "__main__":
    hybrid_smoke_test()