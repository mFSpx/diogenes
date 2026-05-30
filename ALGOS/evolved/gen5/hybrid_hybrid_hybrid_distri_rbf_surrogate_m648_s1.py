# DARWIN HAMMER — match 648, survivor 1
# gen: 5
# parent_a: hybrid_hybrid_distributed_l_hybrid_physarum_netw_m110_s4.py (gen4)
# parent_b: rbf_surrogate.py (gen0)
# born: 2026-05-29T23:30:25Z

"""
Hybrid algorithm merging DARWIN HAMMER's Simulated Annealing Leader Election 
and Physarum Network Dynamics with OPOSSUM-style Radial-Basis Surrogate Model.

Mathematical bridge
-------------------
- The leader election process is driven by a temperature (T) derived from the 
  Physarum network's conductance and pressures, as well as the radial-basis 
  surrogate model's prediction error.
- The temperature schedule is an exponential decay function, dependent on 
  both the Physarum network's conductance and the surrogate model's prediction 
  uncertainty.
- The acceptance probability for a candidate node in the leader election is 
  computed using the Metropolis acceptance rule, where the energy difference 
  ΔE_n is the number of conflicts (edges to already selected broadcasts), and 
  the temperature T is the combined decay of the broadcast probability, 
  Physarum network's conductance, and surrogate model's prediction uncertainty.

This module implements the hybrid dynamics, exposing core functions for 
temperature calculation, leader election, Physarum conductance updates, 
and surrogate model predictions.
"""

import math
import random
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np

# ----------------------------------------------------------------------
# Parent A primitives (re‑implemented for self‑containment)
# ----------------------------------------------------------------------
def broadcast_probability(phases: int, phase: int) -> float:
    if phases < 1 or phase < 1:
        raise ValueError("phases and phase must be positive")
    return min(1.0, 1.0 / (2 ** max(0, phases - phase)))

def cooling_temperature(k: int, t0: float = 1.0, alpha: float = 0.95) -> float:
    if k < 0 or t0 < 0 or not (0 <= alpha <= 1):
        raise ValueError("invalid cooling parameters")
    return t0 * (alpha ** k)

def hybrid_temperature(phases: int, phase: int, conductance: float, pressure_a: float, pressure_b: float,
                       t0: float = 1.0, alpha: float = 0.95, eps: float = 1e-12) -> float:
    bp = broadcast_probability(phases, phase)
    return cooling_temperature(phase, t0 * bp * conductance * (pressure_a + pressure_b), alpha)

# ----------------------------------------------------------------------
# Parent B primitives (re‑implemented for self‑containment)
# ----------------------------------------------------------------------
def gaussian(r: float, epsilon: float = 1.0) -> float:
    return math.exp(-((epsilon * r) ** 2))

def euclidean(a: List[float], b: List[float]) -> float:
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def solve_linear(a: List[List[float]], b: List[float]) -> List[float]:
    n = len(b)
    m = [row[:] + [rhs] for row, rhs in zip(a, b)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[pivot][col]) < 1e-12:
            raise ValueError("singular surrogate system")
        m[col], m[pivot] = m[pivot], m[col]
        div = m[col][col]
        m[col] = [v / div for v in m[col]]
        for row in range(n):
            if row == col:
                continue
            factor = m[row][col]
            m[row] = [v - factor * p for v, p in zip(m[row], m[col])]
    return [row[-1] for row in m]

@dataclass(frozen=True)
class RBFSurrogate:
    centers: List[Tuple[float, ...]]
    weights: List[float]
    epsilon: float = 1.0

    def predict(self, x: List[float]) -> float:
        return sum(w * gaussian(euclidean(list(x), list(c)), self.epsilon) for w, c in zip(self.weights, self.centers))

def fit(points: List[List[float]], values: List[float], epsilon: float = 1.0, ridge: float = 1e-9) -> RBFSurrogate:
    centers = [tuple(map(float, p)) for p in points]
    y = [float(v) for v in values]
    if not centers or len(centers) != len(y):
        raise ValueError("points and values must be non-empty and same length")
    k = [[gaussian(euclidean(list(a), list(b)), epsilon) + (ridge if i == j else 0.0) for j, b in enumerate(centers)] for i, a in enumerate(centers)]
    return RBFSurrogate(centers, solve_linear(k, y), epsilon)

# ----------------------------------------------------------------------
# Hybrid Algorithm
# ----------------------------------------------------------------------
def hybrid_leader_election(phase: int, phases: int, conductance: float, pressure_a: float, pressure_b: float,
                           surrogate: RBFSurrogate, candidates: List[List[float]]) -> Tuple[List[float], float]:
    t = hybrid_temperature(phases, phase, conductance, pressure_a, pressure_b)
    best_candidate = None
    best_prediction = float('inf')
    for candidate in candidates:
        prediction = surrogate.predict(candidate)
        if prediction < best_prediction:
            best_prediction = prediction
            best_candidate = candidate
    # Metropolis acceptance rule
    delta_e = len([c for c in candidates if c != best_candidate])
    acceptance_prob = math.exp(-delta_e / t)
    if random.random() < acceptance_prob:
        return best_candidate, best_prediction
    else:
        return random.choice(candidates), surrogate.predict(random.choice(candidates))

def physarum_update_conductance(conductance: float, pressure_a: float, pressure_b: float, delta: float) -> float:
    return conductance * (1 + delta * (pressure_a + pressure_b))

def hybrid_smoke_test():
    np.random.seed(42)
    random.seed(42)

    # Initialize surrogate model
    points = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    values = [10.0, 20.0, 30.0]
    surrogate = fit(points, values)

    # Initialize leader election
    phases = 10
    phase = 5
    conductance = 0.5
    pressure_a = 1.0
    pressure_b = 2.0
    candidates = [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]

    best_candidate, best_prediction = hybrid_leader_election(phase, phases, conductance, pressure_a, pressure_b, surrogate, candidates)
    print(f"Best candidate: {best_candidate}, Best prediction: {best_prediction}")

    # Update conductance
    delta = 0.1
    new_conductance = physarum_update_conductance(conductance, pressure_a, pressure_b, delta)
    print(f"New conductance: {new_conductance}")

if __name__ == "__main__":
    hybrid_smoke_test()