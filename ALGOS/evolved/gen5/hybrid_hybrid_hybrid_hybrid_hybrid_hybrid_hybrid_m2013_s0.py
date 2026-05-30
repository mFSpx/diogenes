# DARWIN HAMMER — match 2013, survivor 0
# gen: 5
# parent_a: hybrid_hybrid_hybrid_hybrid_omni_chaotic_sprint_m621_s3.py (gen4)
# parent_b: hybrid_hybrid_hybrid_bandit_hybrid_hybrid_rbf_su_m971_s0.py (gen4)
# born: 2026-05-29T23:40:26Z

"""
Hybrid Algorithm: fusion of hybrid_hybrid_hybrid_omni_chaotic_sprint_m621_s3 and hybrid_hybrid_hybrid_bandit_hybrid_hybrid_rbf_su_m971_s0.
The mathematical bridge between these two structures is the use of the temperature-dependent activity curve from the Schoolfield algorithm
as a weighting function in the RBF Surrogate model, incorporated into the fractional memory sum and hybrid allocation by dates.
This allows the surrogate to incorporate insights from the temperature-dependent activity, while the fractional memory sum and hybrid allocation
by dates provide a framework for incorporating the Schoolfield algorithm into the allocation process.
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

def fractional_memory_sum(alpha: float, allocations: list[float], temperatures: list[float]) -> float:
    total = 0.0
    t = len(allocations) - 1
    schoolfield_params = type('SchoolfieldParams', (), {
        'rho_25': 1.0,
        'delta_h_activation': 12_000.0,
        't_low': 283.15,
        't_high': 307.15,
        'delta_h_low': -45_000.0,
        'delta_h_high': 65_000.0,
        'r_cal': 1.987
    })
    for k, a in enumerate(allocations):
        delta = t - k
        celsius = temperatures[k]
        temp_k = celsius + 273.15
        developmental_rate_val = developmental_rate(temp_k, schoolfield_params)
        total += caputo_kernel(alpha, delta) * a * developmental_rate_val
    return total

def developmental_rate(temp_k: float, params: object = None) -> float:
    if temp_k <= 0 or (params is None or params.rho_25 < 0):
        raise ValueError("temperature must be Kelvin-positive and rho_25 non-negative")
    if params is None:
        params = type('SchoolfieldParams', (), {
            'rho_25': 1.0,
            'delta_h_activation': 12_000.0,
            't_low': 283.15,
            't_high': 307.15,
            'delta_h_low': -45_000.0,
            'delta_h_high': 65_000.0,
            'r_cal': 1.987
        })
    numerator = params.rho_25 * (temp_k / 298.15) * math.exp((params.delta_h_activation / params.r_cal) * ((1.0 / 298.15) - (1.0 / temp_k)))
    low = math.exp((params.delta_h_low / params.r_cal) * ((1.0 / params.t_low) - (1.0 / temp_k)))
    high = math.exp((params.delta_h_high / params.r_cal) * ((1.0 / params.t_high) - (1.0 / temp_k)))
    return numerator / (low * high)

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
    temperatures: list[float],
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

def hybrid_edge_cost(
    base_weight: float,
    day: int,
    temperature: float,
    params: dict,
    alpha: float,
    allocations: list[float],
    temperatures: list[float]
) -> float:
    schoolfield_params = type('SchoolfieldParams', (), {
        'rho_25': 1.0,
        'delta_h_activation': 12_000.0,
        't_low': 283.15,
        't_high': 307.15,
        'delta_h_low': -45_000.0,
        'delta_h_high': 65_000.0,
        'r_cal': 1.987
    })
    temp_k = temperature + 273.15
    developmental_rate_val = developmental_rate(temp_k, schoolfield_params)
    effective_time = effective_time_constant(day, params)
    fractional_sum = fractional_memory_sum(alpha, allocations, temperatures)
    return base_weight * developmental_rate_val * effective_time * fractional_sum

if __name__ == "__main__":
    days = [1, 2, 3]
    groups = ['A', 'B', 'C']
    ltc_params = init_ltc_parameters()
    temperatures = [20, 25, 30]
    allocations = hybrid_allocate_by_dates(days, groups, ltc_params, temperatures)
    alpha = 0.5
    base_weight = 1.0
    for d in days:
        cost = hybrid_edge_cost(base_weight, d, temperatures[d-1], ltc_params, alpha, [1.0]*len(days), temperatures)
        print(f"Day {d}, Cost: {cost}")