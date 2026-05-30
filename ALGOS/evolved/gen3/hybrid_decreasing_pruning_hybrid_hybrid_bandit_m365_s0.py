# DARWIN HAMMER — match 365, survivor 0
# gen: 3
# parent_a: decreasing_pruning.py (gen0)
# parent_b: hybrid_hybrid_bandit_router_state_space_duality_m143_s1.py (gen2)
# born: 2026-05-29T23:28:26Z

"""
This module fuses the decreasing-rate pruning schedule from decreasing_pruning.py and 
the temperature-dependent state transition and output projection from 
hybrid_hybrid_bandit_router_state_space_duality_m143_s1.py.

The mathematical bridge between the two algorithms lies in the exponential functions 
used in both. The decreasing_pruning.py algorithm uses an exponential function to 
calculate the pruning probability, while the 
hybrid_hybrid_bandit_router_state_space_duality_m143_s1.py algorithm uses 
exponential functions to calculate the developmental rate.

The hybrid algorithm combines these two by using the developmental rate as a 
temperature-dependent factor in the pruning probability calculation.
"""

import math
import numpy as np
from dataclasses import dataclass
from typing import List
from collections.abc import Hashable

@dataclass(frozen=True)
class SchoolfieldParams:
    rho_25: float = 1.0
    delta_h_activation: float = 12_000.0
    t_low: float = 283.15
    t_high: float = 307.15
    delta_h_low: float = -45_000.0
    delta_h_high: float = 65_000.0
    r_cal: float = 1.987  # cal mol^-1 K^-1

def c_to_k(celsius: float) -> float:
    return celsius + 273.15

def developmental_rate(temp_k: float, params: SchoolfieldParams = SchoolfieldParams()) -> float:
    if temp_k <= 0 or params.rho_25 < 0:
        raise ValueError("temperature must be Kelvin-positive and rho_25 non-negative")
    numerator = params.rho_25 * (temp_k / 298.15) * math.exp((params.delta_h_activation / params.r_cal) * ((1.0 / 298.15) - (1.0 / temp_k)))
    low = math.exp((params.delta_h_low / params.r_cal) * ((1.0 / params.t_low) - (1.0 / temp_k)))
    high = math.exp((params.delta_h_high / params.r_cal) * ((1.0 / params.t_high) - (1.0 / temp_k)))
    denominator = 1.0 + low + high
    if denominator == 0:
        raise ValueError("denominator cannot be zero")
    return numerator / denominator

def prune_probability(t: float, lam: float = 1.0, alpha: float = 0.2, temp_k: float = 298.15) -> float:
    rate = developmental_rate(temp_k)
    return min(1.0, lam * math.exp(-alpha * t * rate))

def prune_edges(edges: list[Hashable], t: float, lam: float = 1.0, alpha: float = 0.2, temp_k: float = 298.15, seed: int | str | None = None) -> list[Hashable]:
    import random
    rng = random.Random(seed)
    p = prune_probability(t, lam, alpha, temp_k)
    return [e for e in edges if rng.random() >= p]

def temperature_dependent_state_transition(A: np.ndarray, temp_k: float, params: SchoolfieldParams = SchoolfieldParams()) -> np.ndarray:
    rate = developmental_rate(temp_k, params)
    return A * rate

def hybrid_pruning_step(edges: list[Hashable], A: np.ndarray, temp_k: float, t: float, lam: float = 1.0, alpha: float = 0.2) -> (list[Hashable], np.ndarray):
    pruned_edges = prune_edges(edges, t, lam, alpha, temp_k)
    transitioned_A = temperature_dependent_state_transition(A, temp_k)
    return pruned_edges, transitioned_A

if __name__ == "__main__":
    edges = [1, 2, 3, 4, 5]
    A = np.array([[1, 2], [3, 4]])
    temp_k = c_to_k(25)
    t = 1.0
    pruned_edges, transitioned_A = hybrid_pruning_step(edges, A, temp_k, t)
    print(pruned_edges)
    print(transitioned_A)