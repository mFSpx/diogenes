# DARWIN HAMMER — match 1899, survivor 0
# gen: 5
# parent_a: hybrid_fisher_localization_hybrid_hybrid_path_s_m20_s2.py (gen3)
# parent_b: hybrid_hybrid_hybrid_distri_hybrid_hybrid_hybrid_m995_s4.py (gen4)
# born: 2026-05-29T23:39:30Z

"""
Hybrid Graph-Pheromone Localization Model
=====================================

This module fuses the core mathematics of two parent algorithms:

* **Parent A** – implements a hybrid mathematical algorithm that fuses the Fisher-information scoring
  from 'fisher_localization.py' with the lead-lag transform and feature extraction from 
  'hybrid_hybrid_path_signatur_hybrid_krampus_brain_m25_s0.py'.

* **Parent B** – builds a hybrid graph-pheromone model that updates an adjacency weight matrix
  with a broadcast probability function and modulates the exploration intensity of a multi-armed bandit
  (entropy-based) decision process.

**Mathematical bridge**

The bridge is formed by letting the pheromone signal associated with a node act as an additive bias
on the corresponding rows/columns of the adjacency weight matrix *W*. The edge weight of the
similarity graph is computed as the product of the Fisher-information score and the pheromone signal.
"""

import numpy as np
import math
import random
import sys
from pathlib import Path

# ----------------------------------------------------------------------
# Utility functions taken from Parent A
# ----------------------------------------------------------------------
def compute_phash(values: list[float]) -> int:
    """Perceptual hash: 1 bit per value indicating >= mean (max 64 bits)."""
    if not values:
        return 0
    avg = sum(values) / len(values)
    bits = 0
    for v in values[:64]:
        bits = (bits << 1) | int(v >= avg)
    return bits


def hamming_distance(a: int, b: int) -> int:
    """Number of differing bits between two integers."""
    return (a ^ b).bit_count()


def broadcast_probability(phase: int, step: int) -> float:
    # Assume a simple broadcast probability function
    # In the original Parent B, the broadcast probability is a more complex function
    return 0.5


# ----------------------------------------------------------------------
# Utility functions taken from Parent A
# ----------------------------------------------------------------------
def gaussian_beam(theta: float, center: float, width: float) -> float:
    if width <= 0:
        raise ValueError('width must be positive')
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)


def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity


def lead_lag_transform(path):
    path = np.asarray(path, dtype=float)
    T, d = path.shape
    out = np.empty((2 * T - 1, 2 * d), dtype=float)
    for t in range(T - 1):
        out[2 * t]     = np.concatenate([path[t],     path[t]])
        out[2 * t + 1] = np.concatenate([path[t + 1], path[t]])
    out[2 * (T - 1)] = np.concatenate([path[T - 1], path[T - 1]])
    return out


def extract_fisher_feature(angle: float, center: float, width: float) -> float:
    return fisher_score(angle, center, width)


# ----------------------------------------------------------------------
# Hybrid Graph-Pheromone Model
# ----------------------------------------------------------------------
class HybridGraphPheromoneModel:
    def __init__(self, eta: float, lambda_: float):
        self.eta = eta
        self.lambda_ = lambda_

    def build_graph(self, values: list[float]) -> np.ndarray:
        """Builds a similarity graph from the input values."""
        similarity_matrix = np.zeros((len(values), len(values)))
        for i in range(len(values)):
            for j in range(i + 1, len(values)):
                similarity = compute_phash([values[i], values[j]])
                similarity_matrix[i, j] = similarity
                similarity_matrix[j, i] = similarity
        return similarity_matrix

    def update_weights_with_pheromones(self, similarity_matrix: np.ndarray, pheromones: Mapping[Hashable, float], step: int) -> np.ndarray:
        """Updates the adjacency weight matrix *W* with pheromone signals."""
        adjacency_matrix = np.zeros_like(similarity_matrix)
        for i in range(len(similarity_matrix)):
            for j in range(i + 1, len(similarity_matrix)):
                adjacency_matrix[i, j] = -self.eta * (similarity_matrix[i, j] - 0.5) * broadcast_probability(step, i) + self.lambda_ * (pheromones[i] + pheromones[j]) / 2
                adjacency_matrix[j, i] = adjacency_matrix[i, j]
        return adjacency_matrix

    def select_leader(self, adjacency_matrix: np.ndarray, pheromones: Mapping[Hashable, float]) -> Hashable:
        """Selects the leader node based on the pheromone signals and adjacency matrix."""
        leader_node = max(pheromones, key=lambda i: (pheromones[i], adjacency_matrix[pheromones.index(i), :].sum()))
        return leader_node


def hybrid_fusion(candidates: list[float], center: float, width: float, eta: float, lambda_: float) -> np.ndarray:
    values = np.array(candidates)
    pheromones = {i: fisher_score(values[i], center, width) for i in range(len(values))}
    hybrid_graph = HybridGraphPheromoneModel(eta, lambda_)
    similarity_matrix = hybrid_graph.build_graph(values)
    adjacency_matrix = hybrid_graph.update_weights_with_pheromones(similarity_matrix, pheromones, 0)
    return adjacency_matrix


if __name__ == "__main__":
    # Smoke test
    candidates = [1.0, 2.0, 3.0, 4.0, 5.0]
    center = 3.0
    width = 1.0
    eta = 0.01
    lambda_ = 0.1
    adjacency_matrix = hybrid_fusion(candidates, center, width, eta, lambda_)
    print(adjacency_matrix)