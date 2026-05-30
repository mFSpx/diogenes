# DARWIN HAMMER — match 1182, survivor 2
# gen: 5
# parent_a: hybrid_hybrid_physarum_netw_hybrid_sparse_wta_hy_m336_s0.py (gen4)
# parent_b: hybrid_hybrid_hybrid_bandit_hybrid_voronoi_parti_m169_s0.py (gen3)
# born: 2026-05-29T23:33:15Z

"""
Module for the hybrid algorithm that combines the Flux-based conductance update 
primitive from hybrid_hybrid_physarum_netw_hybrid_sparse_wta_hy_m336_s0.py and 
the bandit router with Voronoi partition from hybrid_hybrid_hybrid_bandit_hybrid_voronoi_parti_m169_s0.py.
The mathematical bridge between these two structures lies in the use of 
propensity and conductance to inform the bandit router's action selection process, 
which is then geometrically contextualized using Voronoi partitions.
"""

import numpy as np
import random
import math
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple, Dict, Any

# ----------------------------------------------------------------------
# Physarum core
# ----------------------------------------------------------------------
def flux(conductance, edge_length, pressure_a, pressure_b, eps=1e-12):
    if edge_length <= 0:
        raise ValueError('edge_length must be positive')
    return conductance / max(edge_length, eps) * (pressure_a - pressure_b)

def update_conductance(conductance, q, dt=1.0, gain=1.0, decay=0.05):
    if dt < 0 or decay < 0:
        raise ValueError('dt and decay must be non-negative')
    return max(0.0, conductance + dt * (gain * abs(q) - decay * conductance))

# ----------------------------------------------------------------------
# Bandit core
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str

@dataclass(frozen=True)
class BanditUpdate:
    context_id: str
    action_id: str
    reward: float
    propensity: float

def hybrid_bandit_update(conductance, propensity, reward, dt=1.0, gain=1.0, decay=0.05):
    q = propensity * reward
    return update_conductance(conductance, q, dt, gain, decay)

# ----------------------------------------------------------------------
# Voronoi helpers
# ----------------------------------------------------------------------
Point = Tuple[float, float]

def euclidean_distance(a: Point, b: Point) -> float:
    """Standard Euclidean distance in ℝ²."""
    return math.hypot(a[0] - b[0], a[1] - b[1])

# ----------------------------------------------------------------------
# Hybrid core
# ----------------------------------------------------------------------
def hybrid_update(bandit_action: BanditAction, 
                  conductance: float, 
                  edge_length: float, 
                  pressure_a: float, 
                  pressure_b: float) -> Tuple[float, float]:
    flux_value = flux(conductance, edge_length, pressure_a, pressure_b)
    updated_conductance = hybrid_bandit_update(conductance, bandit_action.propensity, 1.0)
    contextualized_distance = euclidean_distance((bandit_action.propensity, bandit_action.expected_reward), 
                                                 (flux_value, updated_conductance))
    return updated_conductance, contextualized_distance

def expand(values: List[float], m: int) -> List[float]:
    """Deterministically project `values` into an m-dimensional sparse vector."""
    if m <= 0:
        raise ValueError('m must be positive')
    out = [0.0] * m
    for i, v in enumerate(values):
        out[i] = v
    return out

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    bandit_action = BanditAction("action_1", 0.5, 1.0, 0.1, "algorithm_1")
    conductance = 1.0
    edge_length = 2.0
    pressure_a = 3.0
    pressure_b = 4.0

    updated_conductance, contextualized_distance = hybrid_update(bandit_action, 
                                                                  conductance, 
                                                                  edge_length, 
                                                                  pressure_a, 
                                                                  pressure_b)
    print(f"Updated conductance: {updated_conductance}, Contextualized distance: {contextualized_distance}")
    values = [1.0, 2.0, 3.0]
    m = 10
    expanded_values = expand(values, m)
    print(expanded_values)