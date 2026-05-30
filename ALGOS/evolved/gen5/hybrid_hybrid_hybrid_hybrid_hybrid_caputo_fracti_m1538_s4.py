# DARWIN HAMMER — match 1538, survivor 4
# gen: 5
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m956_s1.py (gen4)
# parent_b: hybrid_caputo_fractional_minimum_cost_tree_m35_s3.py (gen1)
# born: 2026-05-29T23:37:12Z

"""
Hybrid Fractional Poikilotherm-Labeling State-Space Duality (HFPSSD)

This module fuses two parent algorithms:

* **Parent A** – the Hybrid Poikilotherm-Labeling State-Space Duality (HPLSSD) 
  that bridges the Schoolfield-Rollinson poikilotherm developmental rate with the 
  State-Space Duality (SSD) sequential and semiseparable parallel forms, 
  and weak supervision labeling primitives.
* **Parent B** – the Hybrid Caputo Fractional Derivative and Minimum-Cost Tree scoring algorithm.

The mathematical bridge between the two parents lies in the ability to model 
algebraically-decaying long-range memory in the state-transition matrix `A` 
using the Caputo Fractional Derivative, while using the developmental rate 
to modulate the state-transition matrix `A` and the fractional decay kernel 
to model the decay of path costs over time.

By combining these two approaches, we can create a hybrid algorithm that can 
handle both noisy rewards and labels, and model the decay of path costs over time.
"""

import numpy as np
import math
import random
import sys
import pathlib
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Iterable, Set, Callable

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

@dataclass(frozen=True)
class SchoolfieldParams:
    rho_25: float = 1.0               # rate at 25 °C (in arbitrary units)
    delta_h_activation: float = 12_000.0   # J mol⁻¹
    t_low: float = 283.15            # K  (10 °C)
    t_high: float = 307.15           # K  (34 °C)
    delta_h_low: float = -45_000.0   # J mol⁻¹
    delta_h_high: float = 65_000.0   # J mol⁻¹
    r_cal: float = 1.987             # cal mol⁻¹ K⁻¹ (≈8.314 J mol⁻¹ K⁻¹)

def gamma_lanczos(z):
    if z < 0.5:
        return math.pi / (math.sin(math.pi * z) * gamma_lanczos(1 - z))
    x = np.array([_LANCZOS_C[0]])
    for i in range(1, _LANCZOS_G + 1):
        x = np.append(x, _LANCZOS_C[i] / (z + i))
    return math.sqrt(2 * math.pi) * (z + _LANCZOS_G + 0.5) ** (z + 0.5) * math.exp(-(z + _LANCZOS_G + 0.5)) * np.sum(x)

def caputo_derivative(f, t, alpha):
    return 1 / gamma_lanczos(1 - alpha) * np.sum((f[1:] - f[:-1]) / (t[1:] - t[:-1]) ** alpha)

def developmental_rate(T, params: SchoolfieldParams):
    R = params.r_cal * 1e3  # convert to J mol^-1 K^-1
    T_low, T_high = params.t_low, params.t_high
    rho_25 = params.rho_25
    Delta_H_low, Delta_H_high = params.delta_h_low, params.delta_h_high
    Delta_H_activation = params.delta_h_activation

    num = rho_25 * np.exp((Delta_H_activation / R) * (1 / 298.15 - 1 / T))
    den = 1 + np.exp((Delta_H_low / R) * (1 / T_low - 1 / T)) + np.exp((Delta_H_high / R) * (1 / T - 1 / T_high))
    return num / den

def fractional_decay(t, alpha):
    return t ** (alpha - 1) / gamma_lanczos(alpha)

def hybrid_state_transition(T, alpha, params: SchoolfieldParams):
    rate = developmental_rate(T, params)
    decay = fractional_decay(np.arange(1, 10), alpha)
    A = np.eye(10) * rate * decay[:, None]
    return A

def hybrid_minimum_cost_tree(nodes, edges, root, path_weight=0.2, alpha=0.5, T=298.15, params: SchoolfieldParams=None):
    A = hybrid_state_transition(T, alpha, params)
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
                dist[b] = dist[a] + math.hypot(nodes[a][0] - nodes[b][0], nodes[a][1] - nodes[b][1]) * A[a, b]
                stack.append(b)
    return material + path_weight * sum(dist.values())

def hybrid_caputo_derivative(nodes, edges, root, alpha=0.5):
    f = np.array([math.hypot(nodes[a][0] - nodes[b][0], nodes[a][1] - nodes[b][1]) for a, b in edges])
    t = np.arange(len(f))
    return caputo_derivative(f, t, alpha)

if __name__ == "__main__":
    nodes = [(0, 0), (1, 0), (1, 1), (0, 1)]
    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    root = 0
    alpha = 0.5
    T = 298.15
    params = SchoolfieldParams()
    print(hybrid_minimum_cost_tree(nodes, edges, root, alpha=alpha, T=T, params=params))
    print(hybrid_caputo_derivative(nodes, edges, root, alpha=alpha))