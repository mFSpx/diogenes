# DARWIN HAMMER — match 1769, survivor 0
# gen: 6
# parent_a: hybrid_hybrid_voronoi_parti_hybrid_hybrid_hybrid_m841_s2.py (gen5)
# parent_b: hybrid_hybrid_shannon_entro_sparse_wta_m36_s1.py (gen2)
# born: 2026-05-29T23:38:38Z

"""
This module fuses the concepts of Voronoi partitioning from `hybrid_hybrid_voronoi_parti_hybrid_hybrid_hybrid_m841_s2.py` 
and Sparse Winner-Take-All (WTA) with Shannon Entropy from `hybrid_hybrid_shannon_entro_sparse_wta_m36_s1.py`.
The mathematical bridge between these structures lies in the use of nearest neighbor distances 
for data organization and sparse binary tags for pattern retrieval.
By fusing these concepts, we create a hybrid system where Voronoi partitions are used to organize data points,
Sparse WTA is used to produce high-dimensional similarity vectors,
and Shannon entropy is used to quantify the information content of WTA outputs.

The governing equations of both parents are integrated through the following interface:
1. Voronoi partitioning produces a set of regions, each associated with a seed point.
2. Sparse WTA operates on the similarity vectors between data points and seed points.
3. Shannon entropy is applied to the sparse WTA outputs to quantify information content.

The hybrid operation:
1. Apply Voronoi partitioning to a set of input points, producing a set of regions.
2. Compute similarity vectors between data points and seed points.
3. Apply Sparse WTA to these similarity vectors, producing sparse binary tags.
4. Apply Shannon entropy to these sparse WTA outputs.

"""

import numpy as np
import math
import random
import sys
from pathlib import Path

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

def sparse_wta(similarity_vectors: np.ndarray, k: int = 5) -> np.ndarray:
    """Sparse Winner-Take-All (WTA) operation."""
    num_vectors, num_features = similarity_vectors.shape
    wta_outputs = np.zeros((num_vectors, num_features), dtype=int)
    for i in range(num_vectors):
        top_k_indices = np.argsort(-similarity_vectors[i])[:k]
        wta_outputs[i, top_k_indices] = 1
    return wta_outputs

def shannon_entropy(probability_distribution: np.ndarray) -> float:
    """Shannon entropy computation."""
    epsilon = 1e-10
    probability_distribution = np.clip(probability_distribution, epsilon, 1 - epsilon)
    return -np.sum(probability_distribution * np.log2(probability_distribution))

def hybrid_operation(points: np.ndarray, seeds: np.ndarray) -> float:
    """Hybrid operation: Voronoi partitioning, Sparse WTA, and Shannon entropy."""
    regions = assign(points, seeds)
    similarity_vectors = np.apply_along_axis(lambda x: np.array([distance(x, seed) for seed in seeds]), 1, points)
    wta_outputs = sparse_wta(similarity_vectors)
    probability_distribution = np.mean(wta_outputs, axis=0)
    return shannon_entropy(probability_distribution)

if __name__ == "__main__":
    np.random.seed(0)
    points = np.random.rand(100, 2)
    seeds = np.random.rand(5, 2)
    entropy = hybrid_operation(points, seeds)
    print(f"Shannon entropy: {entropy:.4f}")