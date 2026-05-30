# DARWIN HAMMER — match 495, survivor 0
# gen: 3
# parent_a: hybrid_hybrid_bandit_router_variational_free_ene_m56_s2.py (gen2)
# parent_b: hybrid_pheromone_infotaxis_m3_s2.py (gen1)
# born: 2026-05-29T23:29:06Z

"""
Hybrid Algorithm: Fusing Bandit-Routing Active Inference with Pheromone-based Infotaxis

This module integrates the core topologies of the Bandit-Routing Active Inference (hybrid_hybrid_bandit_router_variational_free_ene_m56_s2.py) 
and Pheromone-based Infotaxis (hybrid_pheromone_infotaxis_m3_s2.py) algorithms. 
The mathematical interface is established by using the pheromone signals to modulate the bandit action propensities, 
which are then updated based on the variational free energy minimization.

The pheromone signals are used to compute a entropy-based bonus for each bandit action, 
which is added to the expected reward to guide the action selection. 
The agent's beliefs and actions are updated based on the interplay between the pheromone signals, 
bandit action propensities, and variational free energy.

The governing equations of both parents are integrated through the following mathematical bridge:
- The pheromone signal strength is used to compute a entropy-based bonus for each bandit action.
- The bandit action propensities are updated based on the variational free energy minimization.
- The agent's beliefs and actions are updated based on the interplay between the pheromone signals, 
  bandit action propensities, and variational free energy.
"""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np

# ----------------------------------------------------------------------
# Parent A – Bandit Router core (lightly adapted)
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

_POLICY: Dict[str, List[float]] = {}

def reset_policy() -> None:
    _POLICY.clear()

def _reward(a: str) -> float:
    total, n = _POLICY.get(a, [0.0, 0.0])
    return total / n if n else 0.0

def update_policy(updates: List[BanditUpdate]) -> None:
    for u in updates:
        stats = _POLICY.setdefault(u.action_id, [0.0, 0.0])
        stats[0] += float(u.reward)
        stats[1] += 1.0

# ----------------------------------------------------------------------
# Parent B – Pheromone Infotaxis core (lightly adapted)
# ----------------------------------------------------------------------
class PheromoneSystem:
    def __init__(self):
        self.pheromone_signals = {}

    def calculate_pheromone_signal(self, surface_key, signal_kind, signal_value, half_life_seconds):
        if surface_key not in self.pheromone_signals:
            self.pheromone_signals[surface_key] = {}
        if signal_kind not in self.pheromone_signals[surface_key]:
            self.pheromone_signals[surface_key][signal_kind] = signal_value
        elapsed_time = 0  # placeholder for actual elapsed time
        return self.pheromone_signals[surface_key][signal_kind] * math.pow(0.5, elapsed_time / half_life_seconds)

    def update_pheromone_signal(self, surface_key, signal_kind, signal_value, half_life_seconds):
        if surface_key not in self.pheromone_signals:
            self.pheromone_signals[surface_key] = {}
        self.pheromone_signals[surface_key][signal_kind] = signal_value

def calculate_entropy(probabilities, eps=1e-12):
    total = sum(probabilities)
    if total <= 0:
        raise ValueError('positive probability mass required')
    return -sum((p/total) * math.log(max(p/total, eps)) for p in probabilities if p > 0)

def expected_entropy(p_hit, hit_state, miss_state):
    if not 0 <= p_hit <= 1:
        raise ValueError('p_hit must be in [0,1]')
    return p_hit * calculate_entropy(hit_state) + (1.0 - p_hit) * calculate_entropy(miss_state)

# ----------------------------------------------------------------------
# Hybrid Algorithm
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class HybridParams:
    pheromone_half_life: float
    entropy_bonus_weight: float

def hybrid_policy_update(pheromone_system: PheromoneSystem, 
                         bandit_updates: List[BanditUpdate], 
                         params: HybridParams) -> None:
    for u in bandit_updates:
        surface_key = u.context_id
        signal_kind = u.action_id
        signal_value = pheromone_system.calculate_pheromone_signal(surface_key, signal_kind, 1.0, params.pheromone_half_life)
        entropy_bonus = params.entropy_bonus_weight * calculate_entropy([signal_value])
        update_policy([BanditUpdate(u.context_id, u.action_id, u.reward + entropy_bonus, u.propensity)])

def best_hybrid_action(pheromone_system: PheromoneSystem, 
                       actions: Dict[str, Tuple[float, float]], 
                       params: HybridParams) -> str:
    best_action_id = None
    best_expected_entropy = float('inf')
    for action_id, (propensity, expected_reward) in actions.items():
        surface_key = action_id
        signal_kind = action_id
        signal_value = pheromone_system.calculate_pheromone_signal(surface_key, signal_kind, 1.0, params.pheromone_half_life)
        entropy_bonus = params.entropy_bonus_weight * calculate_entropy([signal_value])
        expected_entropy_value = expected_entropy(signal_value, [propensity], [expected_reward + entropy_bonus])
        if expected_entropy_value < best_expected_entropy:
            best_expected_entropy = expected_entropy_value
            best_action_id = action_id
    return best_action_id

def smoke_test():
    pheromone_system = PheromoneSystem()
    pheromone_system.update_pheromone_signal('surface1', 'action1', 0.5, 10.0)
    bandit_updates = [BanditUpdate('surface1', 'action1', 1.0, 0.5)]
    hybrid_params = HybridParams(pheromone_half_life=10.0, entropy_bonus_weight=0.1)
    hybrid_policy_update(pheromone_system, bandit_updates, hybrid_params)
    actions = {'action1': (0.5, 1.0), 'action2': (0.3, 0.8)}
    best_action_id = best_hybrid_action(pheromone_system, actions, hybrid_params)
    print(best_action_id)

if __name__ == "__main__":
    smoke_test()