# DARWIN HAMMER — match 1451, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m612_s2.py (gen5)
# parent_b: hybrid_hybrid_krampus_brain_hybrid_hybrid_ternar_m55_s0.py (gen4)
# born: 2026-05-29T23:36:35Z

import numpy as np
from typing import List, Tuple

def hybrid_build_adj(matrix: np.ndarray) -> List[Tuple[int, int]]:
    num_nodes = len(matrix)
    adj_list = []
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            dist = np.linalg.norm(matrix[i] - matrix[j])
            if dist < 1.0:
                adj_list.append((i, j))
                adj_list.append((j, i))
    return adj_list

def hybrid_node_curvature(adj_list: List[Tuple[int, int]], matrix: np.ndarray) -> np.ndarray:
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

def krampus_ollivier_sheaf(matrix: np.ndarray, curvature: np.ndarray, adj_list: List[Tuple[int, int]]) -> np.ndarray:
    num_nodes = len(matrix)
    brain_xyz = np.zeros((num_nodes, 3))
    for i in range(num_nodes):
        neighbors = [j for j in range(num_nodes) if (i, j) in adj_list]
        weights = np.array([curvature[j] for j in neighbors])
        weights /= np.sum(weights)
        brain_xyz[i] = np.sum([matrix[j] * weights[k] for k, j in enumerate(neighbors)], axis=0)
    return brain_xyz

def hybrid_health(matrix: np.ndarray, curvature: np.ndarray, adj_list: List[Tuple[int, int]]) -> np.ndarray:
    num_nodes = len(matrix)
    health = np.zeros((num_nodes,))
    for i in range(num_nodes):
        neighbors = [j for j in range(num_nodes) if (i, j) in adj_list]
        failure_rate = len(neighbors) / (num_nodes - 1)
        health[i] = (1 - curvature[i] * failure_rate) * (1 - failure_rate)
    return health

if __name__ == "__main__":
    np.random.seed(0)
    matrix = np.random.rand(10, 3)
    adj_list = hybrid_build_adj(matrix)
    curvature = hybrid_node_curvature(adj_list, matrix)
    brain_xyz = krampus_ollivier_sheaf(matrix, curvature, adj_list)
    health = hybrid_health(matrix, curvature, adj_list)
    print(brain_xyz)
    print(health)