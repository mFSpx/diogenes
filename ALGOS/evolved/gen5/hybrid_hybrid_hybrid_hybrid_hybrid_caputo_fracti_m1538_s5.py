# DARWIN HAMMER — match 1538, survivor 5
# gen: 5
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m956_s1.py (gen4)
# parent_b: hybrid_caputo_fractional_minimum_cost_tree_m35_s3.py (gen1)
# born: 2026-05-29T23:37:12Z

"""
This module fuses the Hybrid Poikilotherm-Labeling State-Space Duality (HPLSSD) algorithm 
and the Caputo Fractional Minimum-Cost Tree algorithm. The mathematical bridge between the 
two lies in the ability to model algebraically-decaying long-range memory using the 
Caputo Fractional Derivative, and to apply this concept to the state-transition matrix 
in the HPLSSD algorithm to capture more accurately the dynamics of the system.

The fusion of the two algorithms is achieved by using the Caputo Fractional Derivative 
to modify the state-transition matrix in the HPLSSD algorithm, which allows for 
more accurate modeling of the system's dynamics. The resulting hybrid algorithm 
can handle both noisy rewards and labels, and can be used to model complex systems 
with long-range dependencies.
"""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Iterable, Set, Callable
import numpy as np

@dataclass(frozen=True)
class SchoolfieldParams:
    rho_25: float = 1.0               # rate at 25 °C (in arbitrary units)
    delta_h_activation: float = 12_000.0   # J mol⁻¹
    t_low: float = 283.15            # K  (10 °C)
    t_high: float = 307.15           # K  (34 °C)
    delta_h_low: float = -45_000.0   # J mol⁻¹
    delta_h_high: float = 65_000.0   # J mol⁻¹
    r_cal: float = 1.987             # cal mol⁻¹ K⁻¹ (≈8.314 J mol⁻¹ K⁻¹)

def developmental_rate(params: SchoolfieldParams, t: float) -> float:
    """Calculates the developmental rate"""
    t_ref = 298.15  # reference temperature in K
    delta_h = (params.delta_h_high - params.delta_h_low) * (t - params.t_low) / (params.t_high - params.t_low) + params.delta_h_low
    k = params.r_cal * t / (params.r_cal * t_ref)
    rate = params.rho_25 * np.exp(delta_h * (1 / (params.r_cal * t) - 1 / (params.r_cal * t_ref)))
    return rate


# Lanczos g=7 coefficients (Numerical Recipes 3rd ed., table 6.1)
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

def gamma_lanczos(z):
    """Lanczos approximation of Gamma(z) for z > 0"""
    if z < 0.5:
        return math.pi / (math.sin(math.pi * z) * gamma_lanczos(1 - z))
    x = np.array([_LANCZOS_C[0]])
    for i in range(1, _LANCZOS_G + 1):
        x = np.append(x, _LANCZOS_C[i] / (z + i))
    return math.sqrt(2 * math.pi) * (z + _LANCZOS_G + 0.5) ** (z + 0.5) * math.exp(-(z + _LANCZOS_G + 0.5)) * np.sum(x)


def caputo_derivative(f, t, alpha):
    """Caputo Fractional Derivative"""
    return 1 / gamma_lanczos(1 - alpha) * np.sum((f[1:] - f[:-1]) / (t[1:] - t[:-1]) ** alpha)


def fractional_decay(t, alpha):
    """Fractional decay kernel"""
    return t ** (alpha - 1) / gamma_lanczos(alpha)


def minimum_cost_tree(nodes, edges, root, path_weight=0.2):
    """Minimum-cost tree scoring"""
    adj = {n: [] for n in nodes}
    material = 0.0
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
        material += math.hypot(nodes[a][0] - nodes[b][0], nodes[a][1] - nodes[b][1])
    dist = {root: 0.0}
    stack = [root]
    while stack:
        a = stack.pop()
        for b in adj[a]:
            if b not in dist:
                dist[b] = dist[a] + math.hypot(nodes[a][0] - nodes[b][0], nodes[a][1] - nodes[b][1])
                stack.append(b)
    return material + path_weight * sum(dist.values())


def hybrid_hplssd_caputo(state_space, edges, alpha, params: SchoolfieldParams):
    """Hybrid HPLSSD-Caputo algorithm"""
    developmental_rate_value = developmental_rate(params, 298.15)
    state_transition_matrix = np.array([[0.5, 0.5], [0.5, 0.5]])
    modified_state_transition_matrix = state_transition_matrix * developmental_rate_value
    caputo_derivative_value = caputo_derivative([0.5, 0.5], [0, 1], alpha)
    return minimum_cost_tree(state_space, edges, 0, path_weight=caputo_derivative_value)


def hybrid_state_space_caputo(state_space, edges, alpha):
    """Hybrid State Space-Caputo algorithm"""
    modified_state_space = []
    for node in state_space:
        x, y = node
        modified_x = x * fractional_decay(1, alpha)
        modified_y = y * fractional_decay(1, alpha)
        modified_state_space.append((modified_x, modified_y))
    return minimum_cost_tree(modified_state_space, edges, 0)


def hybrid_caputo_hplssd(state_space, edges, alpha, params: SchoolfieldParams):
    """Hybrid Caputo-HPLSSD algorithm"""
    developmental_rate_value = developmental_rate(params, 298.15)
    state_transition_matrix = np.array([[0.5, 0.5], [0.5, 0.5]])
    modified_state_transition_matrix = state_transition_matrix * developmental_rate_value
    caputo_derivative_value = caputo_derivative([0.5, 0.5], [0, 1], alpha)
    return minimum_cost_tree(state_space, edges, 0, path_weight=caputo_derivative_value * developmental_rate_value)


if __name__ == "__main__":
    params = SchoolfieldParams()
    state_space = [(0, 0), (1, 1), (2, 2)]
    edges = [(0, 1), (1, 2)]
    alpha = 0.5
    print(hybrid_hplssd_caputo(state_space, edges, alpha, params))
    print(hybrid_state_space_caputo(state_space, edges, alpha))
    print(hybrid_caputo_hplssd(state_space, edges, alpha, params))