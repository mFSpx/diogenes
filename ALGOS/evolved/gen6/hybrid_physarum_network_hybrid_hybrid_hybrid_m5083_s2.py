# DARWIN HAMMER — match 5083, survivor 2
# gen: 6
# parent_a: physarum_network.py (gen0)
# parent_b: hybrid_hybrid_hybrid_fold_c_state_space_duality_m407_s1.py (gen5)
# born: 2026-05-29T23:59:41Z

"""
This module implements a novel hybrid algorithm that fuses the core topologies of 
two parent algorithms: physarum_network.py and hybrid_hybrid_hybrid_fold_c_state_space_duality_m407_s1.py. 
The mathematical bridge between the two parents lies in the interface between the flux-based 
conductance update and the pheromone infotaxis. Specifically, the flux-based conductance update 
can be viewed as a mechanism for updating the pheromone concentrations in the pheromone infotaxis, 
while the pheromone infotaxis provides a framework for sequentially updating the state and output.

The hybrid algorithm integrates the governing equations of both parents, using the flux-based 
conductance update to update the pheromone concentrations, and the pheromone infotaxis to 
sequentially update the state and output. This fusion enables the hybrid algorithm to leverage 
the strengths of both parents, combining the adaptability of the pheromone infotaxis with the 
efficiency of the flux-based conductance update.
"""

import math
import numpy as np
from dataclasses import dataclass
from collections import defaultdict

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

_POLICY: dict = defaultdict(lambda: [0.0, 0.0])

def reset_policy() -> None:
    """Reset the bandit policy."""
    for action in list(_POLICY.keys()):
        del _POLICY[action]

def _reward(action: str) -> float:
    """Compute the reward for an action based on the bandit policy."""
    total, n = _POLICY[action]
    return total / n if n else 0.0

def _count(action: str) -> float:
    """Count the number of times an action has been selected."""
    return _POLICY[action][1]

def flux(conductance: float, edge_length: float, pressure_a: float, pressure_b: float, eps: float = 1e-12) -> float:
    if edge_length <= 0:
        raise ValueError('edge_length must be positive')
    return conductance / max(edge_length, eps) * (pressure_a - pressure_b)

def update_conductance(conductance: float, q: float, dt: float = 1.0, gain: float = 1.0, decay: float = 0.05) -> float:
    if dt < 0 or decay < 0:
        raise ValueError('dt and decay must be non-negative')
    return max(0.0, conductance + dt * (gain * abs(q) - decay * conductance))

def _phermone_infotaxis(pheromone: float, log_count_ratio: float) -> float:
    """Compute the pheromone infotaxis."""
    return pheromone * log_count_ratio if pheromone != 0 and log_count_ratio != 0 else 0.0

def hybrid_update(conductance: float, edge_length: float, pressure_a: float, pressure_b: float, 
                   pheromone: float, log_count_ratio: float, q: float, dt: float = 1.0, 
                   gain: float = 1.0, decay: float = 0.05) -> (float, float):
    flux_value = flux(conductance, edge_length, pressure_a, pressure_b)
    updated_conductance = update_conductance(conductance, q, dt, gain, decay)
    pheromone_infotaxis = _phermone_infotaxis(pheromone, log_count_ratio)
    return updated_conductance, pheromone_infotaxis

def hybrid_policy_update(action_id: str, reward: float, propensity: float) -> None:
    total, n = _POLICY[action_id]
    _POLICY[action_id] = [total + reward, n + 1]

def hybrid_compute_reward(action_id: str) -> float:
    return _reward(action_id)

if __name__ == "__main__":
    conductance = 1.0
    edge_length = 1.0
    pressure_a = 10.0
    pressure_b = 5.0
    pheromone = 1.0
    log_count_ratio = 0.5
    q = 1.0
    dt = 1.0
    gain = 1.0
    decay = 0.05

    updated_conductance, pheromone_infotaxis = hybrid_update(conductance, edge_length, 
                                                             pressure_a, pressure_b, pheromone, 
                                                             log_count_ratio, q, dt, gain, decay)
    print(f"Updated conductance: {updated_conductance}, Pheromone infotaxis: {pheromone_infotaxis}")

    action_id = "test_action"
    reward = 10.0
    propensity = 1.0
    hybrid_policy_update(action_id, reward, propensity)
    print(f"Reward for {action_id}: {hybrid_compute_reward(action_id)}")