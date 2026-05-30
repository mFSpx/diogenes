# DARWIN HAMMER — match 5276, survivor 0
# gen: 7
# parent_a: hybrid_hybrid_hybrid_physar_hybrid_hybrid_ternar_m1140_s0.py (gen5)
# parent_b: hybrid_hybrid_hybrid_vorono_hybrid_hybrid_shanno_m1769_s0.py (gen6)
# born: 2026-05-30T00:01:01Z

"""
Module for integrating Physarum Network Flux-Based Conductance Updates with Voronoi Partitioning,
Sparse Winner-Take-All (WTA), and Shannon Entropy. The exact mathematical bridge lies in applying
Voronoi partitioning to organize data points, Sparse WTA to produce high-dimensional similarity vectors,
and Shannon entropy to quantify the information content of WTA outputs. Then, these similarity vectors
are used to update conductance in the Physarum network and optimize ternary routes.

Parents: hybrid_hybrid_physarum_netw_hybrid_hybrid_hybrid_m373_s0.py, hybrid_hybrid_ternary_route_hybrid_ternary_route_m98_s1.py,
hybrid_hybrid_voronoi_parti_hybrid_hybrid_hybrid_m841_s2.py, hybrid_hybrid_shannon_entro_sparse_wta_m36_s1.py
"""
import numpy as np
import math
import random
import sys
from pathlib import Path

@dataclass(frozen=True)
class BanditAction:
    """Result of a bandit decision."""
    action_id: str
    propensity: float            # interpreted as inflow rate
    expected_reward: float
    confidence_bound: float      # interpreted as outflow rate
    algorithm: str

@dataclass(frozen=True)
class BanditUpdate:
    """Observed reward for a given action."""
    context_id: str
    action_id: str
    reward: float
    propensity: float

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

def shannon_entropy(sparse_wta_outputs: np.ndarray) -> float:
    """Shannon entropy of sparse WTA outputs."""
    entropy = 0.0
    for row in sparse_wta_outputs:
        prob = np.mean(row)
        if prob > 0:
            entropy -= prob * np.log2(prob)
    return entropy

def flux(conductance: float, edge_length: float, pressure_a: float, pressure_b: float, eps: float = 1e-12) -> float:
    if edge_length <= 0:
        raise ValueError('edge_length must be positive')
    return conductance / max(edge_length, eps) * (pressure_a - pressure_b)

def update_conductance(conductance: float, q: float, dt: float = 1.0, gain: float = 1.0, decay: float = 0.05) -> float:
    if dt < 0 or decay < 0:
        raise ValueError('dt and decay must be non-negative')
    return max(0.0, conductance + dt * (gain * abs(q) - decay * conductance))

def hybrid_operation(points: np.ndarray, seeds: np.ndarray, similarity_vectors: np.ndarray, k: int = 5) -> float:
    """Hybrid operation of Voronoi partitioning, Sparse WTA, and Shannon entropy."""
    regions = assign(points, seeds)
    sparse_wta_outputs = sparse_wta(similarity_vectors, k)
    entropy = shannon_entropy(sparse_wta_outputs)
    similarity_vector = np.mean(similarity_vectors, axis=1)
    q = np.sum(entropy * similarity_vector)
    conductance = update_conductance(1.0, q)
    return flux(conductance, np.mean(np.linalg.norm(points, axis=1)), 1.0, 1.0)

if __name__ == "__main__":
    # Smoke test
    points = np.random.rand(10, 2)
    seeds = np.random.rand(5, 2)
    similarity_vectors = np.random.rand(10, 20)
    print(hybrid_operation(points, seeds, similarity_vectors))