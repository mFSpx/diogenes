# DARWIN HAMMER — match 2013, survivor 1
# gen: 5
# parent_a: hybrid_hybrid_hybrid_hybrid_omni_chaotic_sprint_m621_s3.py (gen4)
# parent_b: hybrid_hybrid_hybrid_bandit_hybrid_hybrid_rbf_su_m971_s0.py (gen4)
# born: 2026-05-29T23:40:26Z

"""
Hybrid Algorithm: fusion of hybrid_hybrid_hybrid_hybrid_omni_chaotic_sprint_m621_s3.py and hybrid_hybrid_hybrid_bandit_hybrid_hybrid_rbf_su_m971_s0.py.
The mathematical bridge between these two structures is the use of the fractional memory sum from the Caputo kernel as a weighting function in the RBF Surrogate model, 
allowing the surrogate to incorporate insights from the chaotic sprint allocations.

"""

import numpy as np
import math
import random
import sys
import pathlib
from dataclasses import dataclass

@dataclass(frozen=True)
class SchoolfieldParams:
    rho_25: float = 1.0
    delta_h_activation: float = 12_000.0
    t_low: float = 283.15
    t_high: float = 307.15
    delta_h_low: float = -45_000.0
    delta_h_high: float = 65_000.0
    r_cal: float = 1.987

@dataclass(frozen=True)
class RBFSurrogate:
    centers: list[tuple[float, ...]]
    weights: list[float]
    epsilon: float = 1.0

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

def developmental_rate(temp_k: float, params: SchoolfieldParams = SchoolfieldParams()) -> float:
    if temp_k <= 0 or params.rho_25 < 0:
        raise ValueError("temperature must be Kelvin-positive and rho_25 non-negative")
    numerator = params.rho_25 * (temp_k / 298.15) * math.exp((params.delta_h_activation / params.r_cal) * ((1.0 / 298.15) - (1.0 / temp_k)))
    low = math.exp((params.delta_h_low / params.r_cal) * ((1.0 / params.t_low) - (1.0 / temp_k)))
    high = math.exp((params.delta_h_high / params.r_cal) * ((1.0 / params.t_high) - (1.0 / temp_k)))
    return numerator / (low * high)

def gaussian(r: float, epsilon: float = 1.0) -> float:
    return math.exp(-((epsilon * r) ** 2))

def hybrid_rbf_surrogate(allocations: list[float], alpha: float, centers: list[tuple[float, ...]], weights: list[float]) -> float:
    fractional_sum = fractional_memory_sum(alpha, allocations)
    surrogate_output = 0.0
    for i, (center, weight) in enumerate(zip(centers, weights)):
        distance = np.linalg.norm(np.array(center) - np.array([fractional_sum]))
        surrogate_output += weight * gaussian(distance)
    return surrogate_output

def effective_time_constant(day: int, base_tau: float = 1.0, amplitude: float = 0.3) -> float:
    gamma = 2 * math.pi / 7.0
    return base_tau * (1.0 + amplitude * math.sin(gamma * day))

def hybrid_allocate_and_surrogate(days: list[int], groups: list[str], total_daily_budget: float = 100.0, 
                                 alpha: float = 0.5, centers: list[tuple[float, ...]] = [(0.0,)], 
                                 weights: list[float] = [1.0]) -> dict:
    allocations: dict[int, dict[str, float]] = {}
    random.seed(0)  
    for d in days:
        tau = effective_time_constant(d)
        gates = {g: random.random() for g in groups}
        total_gate = sum(gates.values())
        day_alloc = {}
        for g in groups:
            share = (gates[g] / total_gate) * tau
            day_alloc[g] = share * total_daily_budget / sum(effective_time_constant(d) for _ in groups)
        allocations[d] = day_alloc
        surrogate_output = hybrid_rbf_surrogate(list(day_alloc.values()), alpha, centers, weights)
        print(f"Day {d} Surrogate Output: {surrogate_output}")
    return allocations

if __name__ == "__main__":
    days = [1, 2, 3, 4, 5]
    groups = ['A', 'B', 'C']
    allocations = hybrid_allocate_and_surrogate(days, groups)
    print(allocations)