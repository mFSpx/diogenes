# DARWIN HAMMER — match 3123, survivor 0
# gen: 4
# parent_a: hybrid_caputo_fractional_minimum_cost_tree_m35_s3.py (gen1)
# parent_b: hybrid_hybrid_pheromone_hyb_chelydrid_ambush_m183_s1.py (gen3)
# born: 2026-05-29T23:47:55Z

"""
This module fuses the mathematical structures of 'hybrid_caputo_fractional_minimum_cost_tree_m35_s3.py' 
and 'hybrid_hybrid_pheromone_hyb_chelydrid_ambush_m183_s1.py' to create a novel hybrid algorithm. 
The mathematical bridge between the two algorithms is formed by applying the burst action admission model 
from 'hybrid_hybrid_pheromone_hyb_chelydrid_ambush_m183_s1.py' to the fractional decay kernel 
from 'hybrid_caputo_fractional_minimum_cost_tree_m35_s3.py', and then using the resulting scores 
to inform the minimum-cost tree scoring process. The governing equations of the burst action admission model 
are integrated with the Caputo Fractional Derivative to create a hybrid system that not only records 
fractional decay signals but also evaluates the trade-offs between length and path costs based on burst action scores.
"""

import numpy as np
import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Iterable

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
    if z < 0.5:
        return math.pi / (math.sin(math.pi * z) * gamma_lanczos(1 - z))
    x = np.array([_LANCZOS_C[0]])
    for i in range(1, _LANCZOS_G + 1):
        x = np.append(x, _LANCZOS_C[i] / (z + i))
    return math.sqrt(2 * math.pi) * (z + _LANCZOS_G + 0.5) ** (z + 0.5) * math.exp(-(z + _LANCZOS_G + 0.5)) * np.sum(x)

def caputo_derivative(f, t, alpha):
    return 1 / gamma_lanczos(1 - alpha) * np.sum((f[1:] - f[:-1]) / (t[1:] - t[:-1]) ** alpha)

def fractional_decay(t, alpha):
    return t ** (alpha - 1) / gamma_lanczos(alpha)

def burst_admission_score(work_value: float, cost_drag: float, urgency_force: float, steps: int = 12) -> float:
    state = integrate_strike(pulse_force(max(0.0, urgency_force), steps), dt=0.01, m_head=1.0, drag_cd=max(0.0, cost_drag), fluid_density=1.0, area=1.0)
    return work_value * state.distance

def integrate_strike(force_series: Iterable[float], dt: float, m_head: float, drag_cd: float = 0.3, fluid_density: float = 1000.0, area: float = 0.001, v0: float = 0.0) -> 'StrikeState':
    if dt <= 0 or m_head <= 0 or drag_cd < 0 or fluid_density < 0 or area < 0:
        raise ValueError("invalid strike parameters")
    v = v0
    x = 0.0
    peak = v0
    for force in force_series:
        drag = drag_cd * fluid_density * area * v * abs(v) / (2.0 * m_head)
        acc = force / m_head - drag
        v = max(0.0, v + acc * dt)
        x += v * dt
        peak = max(peak, v)
    return StrikeState(v, x, peak)

@dataclass(frozen=True)
class StrikeState:
    velocity: float
    distance: float
    peak_velocity: float

def pulse_force(peak_force: float, steps: int):
    return [peak_force * (1 - i / steps) for i in range(steps)]

def hybrid_minimum_cost_tree(nodes, edges, root, alpha, path_weight=0.2):
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
    burst_scores = []
    for i in range(len(nodes)):
        burst_score = burst_admission_score(1.0, 0.1, 1.0)
        burst_scores.append(burst_score)
    hybrid_scores = []
    for i in range(len(nodes)):
        hybrid_score = fractional_decay(dist[nodes[i]], alpha) * burst_scores[i]
        hybrid_scores.append(hybrid_score)
    return material + path_weight * sum(hybrid_scores)

def test_hybrid_minimum_cost_tree():
    nodes = [(0, 0), (1, 0), (1, 1), (0, 1)]
    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    root = 0
    alpha = 0.5
    print(hybrid_minimum_cost_tree(nodes, edges, root, alpha))

if __name__ == "__main__":
    test_hybrid_minimum_cost_tree()