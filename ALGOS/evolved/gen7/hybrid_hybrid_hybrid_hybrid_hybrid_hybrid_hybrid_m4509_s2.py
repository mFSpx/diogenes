# DARWIN HAMMER — match 4509, survivor 2
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_omni_chaotic_sprint_m621_s3.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_krampu_m1451_s2.py (gen6)
# born: 2026-05-29T23:56:21Z

"""
HYBRID ALGORITHM: hybrid_hybrid_hybrid_hybrid_hybrid_krampu_caputo_fracti_m729_s0.py

This algorithm fuses the topologies of PARENT ALGORITHM A (caputo kernel, fractional memory sum) and PARENT ALGORITHM B (krampus ollivier sheaf, hybrid health).
The bridge between their structures lies in the fact that both use a distance metric (Euclidean norm in PARENT ALGORITHM B and caputo kernel in PARENT ALGORITHM A) to compute weights or curvatures between nodes.
In this fusion, we use the caputo kernel as the distance metric in the krampus ollivier sheaf and hybrid health functions to combine the two paradigms.

Authors: 
    PARENT ALGORITHM A: hybrid_hybrid_hybrid_worksh_hybrid_caputo_fracti_m46_s0.py (gen3)
    PARENT ALGORITHM B: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_krampu_m1451_s2.py (gen6)

Date: 2026-05-29T23:30:15Z
"""

import math
import random
import sys
from pathlib import Path
import numpy as np

def caputo_kernel(alpha: float, delta: int) -> float:
    if delta < 0:
        raise ValueError("Delta must be non-negative")
    if not (0.0 < alpha < 1.0):
        raise ValueError("Alpha must be in (0,1)")
    gamma_term = math.gamma(2.0 - alpha)
    return ((delta + 1) ** (1.0 - alpha) - delta ** (1.0 - alpha)) / gamma_term

def hybrid_node_curvature(adj_list: List[Tuple[int, int]], matrix: np.ndarray, alpha: float, ltc_params: dict) -> np.ndarray:
    num_nodes = len(matrix)
    curvature = np.zeros((num_nodes,))
    for i in range(num_nodes):
        neighbors = [j for j in range(num_nodes) if (i, j) in adj_list]
        kappa_sum = 0.0
        for j in neighbors:
            dist = caputo_kernel(alpha, abs(i - j))
            if dist < 1.0:
                kappa = 1 - dist / (dist + 1e-6)
                kappa_sum += kappa
        curvature[i] = kappa_sum / len(neighbors) if neighbors else 0.0
    return curvature

def krampus_ollivier_sheaf(matrix: np.ndarray, curvature: np.ndarray, adj_list: List[Tuple[int, int]], ltc_params: dict) -> np.ndarray:
    num_nodes = len(matrix)
    brain_xyz = np.zeros((num_nodes, 3))
    for i in range(num_nodes):
        neighbors = [j for j in range(num_nodes) if (i, j) in adj_list]
        weights = np.array([curvature[j] for j in neighbors])
        weights /= np.sum(weights)
        tau = effective_time_constant(i, ltc_params)
        brain_xyz[i] = np.sum([matrix[j] * weights[k] * tau for k, j in enumerate(neighbors)], axis=0)
    return brain_xyz

def hybrid_health(matrix: np.ndarray, curvature: np.ndarray, adj_list: List[Tuple[int, int]], ltc_params: dict) -> np.ndarray:
    num_nodes = len(matrix)
    health = np.zeros((num_nodes,))
    for i in range(num_nodes):
        neighbors = [j for j in range(num_nodes) if (i, j) in adj_list]
        failure_rate = len(neighbors) / (num_nodes - 1)
        health[i] = (1 - curvature[i] * failure_rate) * (1 - failure_rate) * effective_time_constant(i, ltc_params)
    return health

def init_ltc_parameters(base_tau: float = 1.0, amplitude: float = 0.3) -> dict:
    return {
        "base_tau": float(base_tau),
        "amplitude": float(amplitude),
        "gamma": 2 * math.pi / 7.0,  
    }

def effective_time_constant(day: int, params: dict) -> float:
    base = params["base_tau"]
    amp = params["amplitude"]
    gamma = params["gamma"]
    return base * (1.0 + amp * math.sin(gamma * day))

def hybrid_allocate_by_dates(
    days: list[int],
    groups: list[str],
    ltc_params: dict,
    total_daily_budget: float = 100.0,
) -> dict:
    allocations: dict[int, dict[str, float]] = {}
    random.seed(0)  
    for d in days:
        tau = effective_time_constant(d, ltc_params)
        gates = {g: random.random() for g in groups}
        total_gate = sum(gates.values())
        day_alloc = {}
        for g in groups:
            share = (gates[g] / total_gate) * tau
            day_alloc[g] = share * total_daily_budget / sum(
                effective_time_constant(d, ltc_params) for _ in groups
            )
        allocations[d] = day_alloc
    return allocations

def build_random_graph(num_nodes: int, edge_prob: float = 0.4, seed: int = 42) -> dict[int, list[tuple[int, float]]]:
    random.seed(seed)
    adj: dict[int, list[tuple[int, float]]] = {i: [] for i in range(num_nodes)}
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if random.random() < edge_prob:
                weight = 1.0
                adj[i].append((j, weight))
                adj[j].append((i, weight))
    return adj

if __name__ == "__main__":
    np.random.seed(0)
    matrix = np.random.rand(10, 3)
    adj_list = hybrid_build_adj(matrix)
    ltc_params = init_ltc_parameters()
    alpha = 0.8
    curvature = hybrid_node_curvature(adj_list, matrix, alpha, ltc_params)
    brain_xyz = krampus_ollivier_sheaf(matrix, curvature, adj_list, ltc_params)
    health = hybrid_health(matrix, curvature, adj_list, ltc_params)
    print(brain_xyz)