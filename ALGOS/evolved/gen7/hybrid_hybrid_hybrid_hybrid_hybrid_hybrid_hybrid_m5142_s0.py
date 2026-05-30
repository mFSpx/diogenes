# DARWIN HAMMER — match 5142, survivor 0
# gen: 7
# parent_a: hybrid_hybrid_hybrid_distri_hybrid_minimum_cost__m1186_s4.py (gen4)
# parent_b: hybrid_hybrid_hybrid_physar_hybrid_pheromone_hyb_m2648_s1.py (gen6)
# born: 2026-05-29T23:59:59Z

"""
This module provides a novel HYBRID algorithm, named hybrid_hammer, 
which mathematically fuses the core topologies of two parent algorithms: 
hybrid_hybrid_hybrid_distri_hybrid_minimum_cost__m1186_s4.py and 
hybrid_hybrid_hybrid_physar_hybrid_pheromone_hyb_m2648_s1.py. 

The mathematical bridge between their structures lies in the integration of 
the tree cost calculation from the hybrid_hybrid_hybrid_distri_hybrid_minimum_cost__m1186_s4.py, 
the flux-based conductance update from the hybrid_hybrid_hybrid_physar_hybrid_pheromone_hyb_m2648_s1.py, 
and the structural similarity index measurement (SSIM) from the Hybrid SSIM Decision Hygiene. 
The interface is established through the concept of propensity, 
which influences the conductance update in the Physarum network, 
and the SSIM-based weighting factor, which modulates the pheromone signal.
"""

import numpy as np
import random
import math
import sys
import pathlib

@dataclass(frozen=True)
class Point:
    x: float
    y: float

def length(a: Point, b: Point) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)

def tree_cost(nodes: dict,
              edges: list,
              root: str,
              path_weight: float = 0.2) -> float:
    adj: dict = {n: [] for n in nodes}
    material = 0.0
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
        material += length(nodes[a], nodes[b])

    dist: dict = {root: 0.0}
    stack = [root]
    while stack:
        cur = stack.pop()
        for nxt in adj[cur]:
            if nxt not in dist:
                dist[nxt] = dist[cur] + length(nodes[cur], nodes[nxt])
                stack.append(nxt)

    path_cost = sum(dist.values())
    return material + path_weight * path_cost

def flux(conductance: float, edge_length: float, pressure_a: float, pressure_b: float, eps: float = 1e-12) -> float:
    if edge_length <= 0:
        raise ValueError('edge_length must be positive')
    return conductance / max(edge_length, eps) * (pressure_a - pressure_b)

def update_conductance(conductance: float, q: float, dt: float = 1.0, gain: float = 1.0, decay: float = 0.05) -> float:
    if dt < 0 or decay < 0:
        raise ValueError('dt and decay must be non-negative')
    return max(0.0, conductance + dt * (gain * abs(q) - decay * conductance))

def hybrid_hammer_update(conductance: float, propensity: float, reward: float, dt: float = 1.0, gain: float = 1.0, decay: float = 0.05) -> float:
    q = propensity * reward
    return update_conductance(conductance, q, dt, gain, decay)

def calculate_ssim(feature_vector_a: np.ndarray, feature_vector_b: np.ndarray) -> float:
    mu_a = np.mean(feature_vector_a)
    mu_b = np.mean(feature_vector_b)
    sigma_a = np.std(feature_vector_a)
    sigma_b = np.std(feature_vector_b)
    sigma_ab = np.mean((feature_vector_a - mu_a) * (feature_vector_b - mu_b))
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    ssim = ((2 * mu_a * mu_b + c1) * (2 * sigma_ab + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (sigma_a ** 2 + sigma_b ** 2 + c2))
    return ssim

def hybrid_hammer_tree_cost(nodes: dict, edges: list, root: str, path_weight: float = 0.2, conductance: float = 1.0, dt: float = 1.0, gain: float = 1.0, decay: float = 0.05) -> float:
    tree_cost_value = tree_cost(nodes, edges, root, path_weight)
    propensity = 1.0 - tree_cost_value / (tree_cost_value + conductance)
    reward = calculate_ssim(np.array([length(nodes[a], nodes[b]) for a, b in edges]), np.array([conductance] * len(edges)))
    conductance = hybrid_hammer_update(conductance, propensity, reward, dt, gain, decay)
    return tree_cost_value, conductance

if __name__ == "__main__":
    nodes = {'A': Point(0.0, 0.0), 'B': Point(1.0, 0.0), 'C': Point(1.0, 1.0)}
    edges = [('A', 'B'), ('B', 'C'), ('C', 'A')]
    root = 'A'
    tree_cost_value, conductance = hybrid_hammer_tree_cost(nodes, edges, root)
    print("Tree cost:", tree_cost_value)
    print("Conductance:", conductance)