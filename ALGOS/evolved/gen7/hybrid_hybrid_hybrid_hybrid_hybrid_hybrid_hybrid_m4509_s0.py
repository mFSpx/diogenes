# DARWIN HAMMER — match 4509, survivor 0
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_omni_chaotic_sprint_m621_s3.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_krampu_m1451_s2.py (gen6)
# born: 2026-05-29T23:56:21Z

"""
This module fuses the governing equations of hybrid_hybrid_hybrid_hybrid_omni_chaotic_sprint_m621_s3.py (Parent A)
and hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_krampu_m1451_s2.py (Parent B). The mathematical bridge between the two
parents lies in the use of weighted allocations and curvature-based node representations. Specifically, we use the
caputo_kernel and fractional_memory_sum functions from Parent A to compute weighted allocations for the nodes in
the graph constructed by the hybrid_build_adj function from Parent B. These allocations are then used to compute
a hybrid health metric that combines the effects of both curvature and allocation.

Parent A: hybrid_hybrid_hybrid_hybrid_omni_chaotic_sprint_m621_s3.py
Parent B: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_krampu_m1451_s2.py
"""

import numpy as np
import math
import random
from pathlib import Path

def caputo_kernel(alpha: float, delta: int) -> float:
    if delta < 0:
        raise ValueError("Delta must be non-negative")
    if not (0.0 < alpha < 1.0):
        raise ValueError("Alpha must be in (0,1)")
    gamma_term = math.gamma(2.0 - alpha)
    return ((delta + 1) ** (1.0 - alpha) - delta ** (1.0 - alpha)) / gamma_term

def fractional_memory_sum(alpha: float, allocations: list[float]) -> float:
    total = 0.0
    t = len(allocations) - 1
    for k, a in enumerate(allocations):
        delta = t - k
        total += caputo_kernel(alpha, delta) * a
    return total

def hybrid_build_adj(matrix: np.ndarray) -> list[tuple[int, int]]:
    num_nodes = len(matrix)
    adj_list = []
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            dist = np.linalg.norm(matrix[i] - matrix[j])
            if dist < 1.0:
                adj_list.append((i, j))
                adj_list.append((j, i))
    return adj_list

def hybrid_node_curvature(adj_list: list[tuple[int, int]], matrix: np.ndarray) -> np.ndarray:
    num_nodes = len(matrix)
    curvature = np.zeros((num_nodes,))
    for i in range(num_nodes):
        neighbors = [j for j in range(num_nodes) if (i, j) in adj_list]
        kappa_sum = 0.0
        for j in neighbors:
            dist = np.linalg.norm(matrix[i] - matrix[j])
            kappa = 1 - np.linalg.norm(matrix[i] - matrix[j]) / (dist + 1e-6)
            kappa_sum += kappa
        curvature[i] = kappa_sum / len(neighbors) if neighbors else 0.0
    return curvature

def hybrid_health(matrix: np.ndarray, curvature: np.ndarray, adj_list: list[tuple[int, int]], 
                   alpha: float, allocations: list[float]) -> np.ndarray:
    num_nodes = len(matrix)
    health = np.zeros((num_nodes,))
    total_allocation = fractional_memory_sum(alpha, allocations)
    for i in range(num_nodes):
        neighbors = [j for j in range(num_nodes) if (i, j) in adj_list]
        failure_rate = len(neighbors) / (num_nodes - 1)
        weighted_allocation = allocations[i] / total_allocation
        health[i] = (1 - curvature[i] * failure_rate) * (1 - failure_rate) * weighted_allocation
    return health

def compute_hybrid_representation(matrix: np.ndarray, alpha: float, allocations: list[float]) -> np.ndarray:
    adj_list = hybrid_build_adj(matrix)
    curvature = hybrid_node_curvature(adj_list, matrix)
    return hybrid_health(matrix, curvature, adj_list, alpha, allocations)

if __name__ == "__main__":
    np.random.seed(0)
    matrix = np.random.rand(10, 3)
    allocations = [random.random() for _ in range(10)]
    alpha = 0.5
    hybrid_repr = compute_hybrid_representation(matrix, alpha, allocations)
    print(hybrid_repr)