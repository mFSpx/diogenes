# DARWIN HAMMER — match 4509, survivor 3
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_omni_chaotic_sprint_m621_s3.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_krampu_m1451_s2.py (gen6)
# born: 2026-05-29T23:56:21Z

"""
hybrid_unified algorithm: This module fuses the mathematical structures of 
hybrid_hybrid_hybrid_hybrid_omni_chaotic_sprint_m621_s3.py and 
hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_krampu_m1451_s2.py.

The mathematical bridge between these two algorithms is found in their 
ability to process and generate time-dependent allocations and 
node-weighted matrices. This unified system uses the caputo kernel 
fractional memory sum to allocate resources in a hybrid graph, 
and the krampus ollivier sheaf to compute the brain xyz of the nodes.
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

def fractional_memory_sum(alpha: float, allocations: list[float]) -> float:
    total = 0.0
    t = len(allocations) - 1
    for k, a in enumerate(allocations):
        delta = t - k
        total += caputo_kernel(alpha, delta) * a
    return total

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

def krampus_ollivier_sheaf(matrix: np.ndarray, curvature: np.ndarray, adj_list: list[tuple[int, int]]) -> np.ndarray:
    num_nodes = len(matrix)
    brain_xyz = np.zeros((num_nodes, 3))
    for i in range(num_nodes):
        neighbors = [j for j in range(num_nodes) if (i, j) in adj_list]
        weights = np.array([curvature[j] for j in neighbors])
        weights /= np.sum(weights)
        brain_xyz[i] = np.sum([matrix[j] * weights[k] for k, j in enumerate(neighbors)], axis=0)
    return brain_xyz

def hybrid_health(matrix: np.ndarray, curvature: np.ndarray, adj_list: list[tuple[int, int]]) -> np.ndarray:
    num_nodes = len(matrix)
    health = np.zeros((num_nodes,))
    for i in range(num_nodes):
        neighbors = [j for j in range(num_nodes) if (i, j) in adj_list]
        failure_rate = len(neighbors) / (num_nodes - 1)
        health[i] = (1 - curvature[i] * failure_rate) * (1 - failure_rate)
    return health

def hybrid_fusion_allocate(matrix: np.ndarray, days: list[int], groups: list[str], alpha: float = 0.5, base_tau: float = 1.0, amplitude: float = 0.3) -> dict:
    ltc_params = init_ltc_parameters(base_tau, amplitude)
    allocations = hybrid_allocate_by_dates(days, groups, ltc_params)
    adj_list = hybrid_build_adj(matrix)
    curvature = hybrid_node_curvature(adj_list, matrix)
    brain_xyz = krampus_ollivier_sheaf(matrix, curvature, adj_list)
    health = hybrid_health(matrix, curvature, adj_list)
    fusion_allocations = {}
    for day, alloc in allocations.items():
        alpha_sum = fractional_memory_sum(alpha, [alloc[group] for group in groups])
        fusion_allocations[day] = {group: alloc[group] * alpha_sum / sum(alloc.values()) for group in groups}
    return fusion_allocations

def hybrid_fusion_nodes(matrix: np.ndarray, days: list[int], groups: list[str], alpha: float = 0.5, base_tau: float = 1.0, amplitude: float = 0.3) -> dict:
    ltc_params = init_ltc_parameters(base_tau, amplitude)
    allocations = hybrid_allocate_by_dates(days, groups, ltc_params)
    adj_list = hybrid_build_adj(matrix)
    curvature = hybrid_node_curvature(adj_list, matrix)
    brain_xyz = krampus_ollivier_sheaf(matrix, curvature, adj_list)
    health = hybrid_health(matrix, curvature, adj_list)
    node_allocations = {}
    for i in range(len(matrix)):
        neighbors = [j for j in range(len(matrix)) if (i, j) in adj_list]
        alpha_sum = fractional_memory_sum(alpha, [health[j] for j in neighbors])
        node_allocations[i] = {group: health[i] * alpha_sum / sum(health) for group in groups}
    return node_allocations

def hybrid_fusion_curvature(matrix: np.ndarray, days: list[int], groups: list[str], alpha: float = 0.5, base_tau: float = 1.0, amplitude: float = 0.3) -> dict:
    ltc_params = init_ltc_parameters(base_tau, amplitude)
    allocations = hybrid_allocate_by_dates(days, groups, ltc_params)
    adj_list = hybrid_build_adj(matrix)
    curvature = hybrid_node_curvature(adj_list, matrix)
    brain_xyz = krampus_ollivier_sheaf(matrix, curvature, adj_list)
    health = hybrid_health(matrix, curvature, adj_list)
    curvature_allocations = {}
    for day, alloc in allocations.items():
        alpha_sum = fractional_memory_sum(alpha, [curvature[i] for i in range(len(matrix))])
        curvature_allocations[day] = {group: alloc[group] * alpha_sum / sum(alloc.values()) for group in groups}
    return curvature_allocations

if __name__ == "__main__":
    np.random.seed(0)
    matrix = np.random.rand(10, 3)
    days = [1, 2, 3, 4, 5]
    groups = ['A', 'B', 'C']
    fusion_alloc = hybrid_fusion_allocate(matrix, days, groups)
    fusion_nodes = hybrid_fusion_nodes(matrix, days, groups)
    fusion_curvature = hybrid_fusion_curvature(matrix, days, groups)
    print(fusion_alloc)
    print(fusion_nodes)
    print(fusion_curvature)