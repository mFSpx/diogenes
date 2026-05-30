# DARWIN HAMMER — match 880, survivor 0
# gen: 4
# parent_a: hybrid_caputo_fractional_minimum_cost_tree_m35_s7.py (gen1)
# parent_b: hybrid_hybrid_regret_engine_hybrid_ternary_lens__m3_s0.py (gen3)
# born: 2026-05-29T23:31:25Z

"""
Hybrid Fractional-Memory Regret-Weighted Ternary Decision Hygiene Analyzer.

This module fuses the Hybrid Fractional-Memory Tree Scoring (parent A) with the Hybrid Regret-Weighted Ternary Decision Hygiene Analyzer (parent B).
The mathematical bridge between these two structures lies in the application of the Caputo fractional derivative to the regret-weighted strategy's decision-making process.
By incorporating a power-law memory kernel into the regret-weighted strategy, we obtain a fractional-memory variant that weights past contributions with a slowly decaying algebraic factor.

The governing equation of the hybrid system is:

` regret_weighted_score = Σ_k w_k * (Δ_material_k + Δ_path_k) * (action_id_k - E[action_id_k])`

where `w_k = ϕ(T-1-k;α)/∑_j ϕ(T-1-j;α)`, `ϕ(t-τ;α) = (t-τ)^{α-1}/Γ(α)`, and `E[action_id_k]` is the expected value of the action.

The hybrid operation consists of:

1. Compute the incremental material and path contributions for each edge as it is added.
2. Form the Caputo weights `w_k` using the sequence of incremental cost contributions.
3. Compute the regret-weighted score using the hybrid governing equation.

"""

import numpy as np
import math
import random
import sys
from pathlib import Path
from collections import deque
from typing import Dict, List, Tuple
from dataclasses import dataclass

_LANCZOS_G = 7
_LANCZOS_C = np.array([
    0.99999999999980993,
    676.5203681218851,
    -1259.1392167224028,
    771.32342877765313,
    -176.61502916214059,
    12.507343278686905,
    -0.13857109526572012,
    9.9843695780195716e-6,
    1.5056327351493116e-7,
])

def gamma_lanczos(z: float) -> float:
    if z < 0.5:
        return math.pi / (math.sin(math.pi * z) * gamma_lanczos(1 - z))
    zz = z - 1
    x = _LANCZOS_C / (zz + np.arange(_LANCZOS_G) + 1)
    return math.sqrt(2 * math.pi) * (zz + _LANCZOS_G + 0.5) ** (zz + 0.5) * np.exp(-(zz + _LANCZOS_G + 0.5)) * np.prod(x)

def caputo_weight(alpha: float, T: int, k: int) -> float:
    return ((T - 1 - k) ** (alpha - 1)) / gamma_lanczos(alpha)

def compute_hybrid_score(alpha: float, edges: List[Tuple[int, int]], 
                         material_costs: List[float], path_weights: List[float], 
                         actions: List[str], expected_values: List[float]) -> float:
    T = len(edges)
    w_k = [caputo_weight(alpha, T, k) / sum(caputo_weight(alpha, T, j) for j in range(T)) for k in range(T)]
    regret_weighted_score = 0
    for k, (edge, material_cost, path_weight, action, expected_value) in enumerate(zip(edges, material_costs, path_weights, actions, expected_values)):
        delta_material = material_cost
        delta_path = path_weight
        regret_weighted_score += w_k[k] * (delta_material + delta_path) * (action == '1' and 1 or -1 - expected_value)
    return regret_weighted_score

def compute_regret_weighted_score(actions: List[str], expected_values: List[float]) -> float:
    regret_weighted_score = 0
    for action, expected_value in zip(actions, expected_values):
        regret_weighted_score += (action == '1' and 1 or -1) - expected_value
    return regret_weighted_score

def compute_fractional_memory_score(alpha: float, edges: List[Tuple[int, int]], 
                                   material_costs: List[float], path_weights: List[float]) -> float:
    T = len(edges)
    w_k = [caputo_weight(alpha, T, k) / sum(caputo_weight(alpha, T, j) for j in range(T)) for k in range(T)]
    fractional_memory_score = 0
    for k, (edge, material_cost, path_weight) in enumerate(zip(edges, material_costs, path_weights)):
        delta_material = material_cost
        delta_path = path_weight
        fractional_memory_score += w_k[k] * (delta_material + delta_path)
    return fractional_memory_score

if __name__ == "__main__":
    alpha = 0.5
    edges = [(1, 2), (2, 3), (3, 4)]
    material_costs = [1.0, 2.0, 3.0]
    path_weights = [0.5, 0.6, 0.7]
    actions = ['1', '0', '1']
    expected_values = [0.4, 0.5, 0.6]

    hybrid_score = compute_hybrid_score(alpha, edges, material_costs, path_weights, actions, expected_values)
    regret_weighted_score = compute_regret_weighted_score(actions, expected_values)
    fractional_memory_score = compute_fractional_memory_score(alpha, edges, material_costs, path_weights)

    print(hybrid_score)
    print(regret_weighted_score)
    print(fractional_memory_score)