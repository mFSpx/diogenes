# DARWIN HAMMER — match 547, survivor 0
# gen: 4
# parent_a: hybrid_bandit_router_poikilotherm_schoolf_m20_s4.py (gen1)
# parent_b: hybrid_hybrid_hybrid_decisi_fisher_localization_m153_s2.py (gen3)
# born: 2026-05-29T23:29:33Z

"""
Hybrid algorithm combining the bandit router core from hybrid_bandit_router_poikilotherm_schoolf_m20_s4.py
with the Fisher information angle selection from hybrid_hybrid_hybrid_decisi_fisher_localization_m153_s2.py.
The mathematical bridge lies in mapping the action space from the bandit router to the angular domain
of the Fisher information, allowing for a novel exploration-exploitation trade-off.
"""

import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

# ----------------------------------------------------------------------
# Parent A – bandit router core (lightly adapted)
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
# Parent B – Schoolfield temperature model (lightly adapted)
# ----------------------------------------------------------------------
R_CAL = 1.987  
K25 = 298.15  

@dataclass(frozen=True)
class SchoolfieldParams:
    rho_25: float = 1.0
    delta_h_activation: float = 12_000.0
    t_low: float = 283.15
    t_high: float = 307.15
    delta_h_low: float = -45_000.0
    delta_h_high: float = 65_000.0
    r_cal: float = R_CAL

def c_to_k(celsius: float) -> float:
    return celsius + 273.15

def developmental_rate(temp_k: float, params: SchoolfieldParams = SchoolfieldParams()) -> float:
    if temp_k <= 0 or params.rho_25 < 0:
        raise ValueError("temperature must be Kelvin-positive and rho_25 non-negative")
    numerator = params.rho_25 * (temp_k / K25) * math.exp(
        (params.delta_h_activation / params.r_cal) * ((1.0 / K25) - (1.0 / temp_k))
    )
    low = math.exp(
        (params.delta_h_low / params.r_cal) * ((1.0 / params.t_low) - (1.0 / temp_k))
    )
    high = math.exp(
        (params.delta_h_high / params.r_cal) * ((1.0 / params.t_high) - (1.0 / temp_k))
    )
    return numerator / (1.0 + low + high)

# ----------------------------------------------------------------------
# Hybrid algorithm
# ----------------------------------------------------------------------
def action_to_angle(action_id: str) -> float:
    """Map action space to angular domain."""
    return float(action_id) % (2 * math.pi)

def fisher_information(angle: float, params: SchoolfieldParams = SchoolfieldParams()) -> float:
    """Compute Fisher information for a given angle."""
    temp_k = c_to_k(20)  # assuming a constant temperature of 20°C
    rate = developmental_rate(temp_k, params)
    return rate ** 2 / (1 + math.cos(angle)**2)

def hybrid_update(updates: List[BanditUpdate]) -> None:
    """Update policy using both bandit router and Fisher information."""
    for u in updates:
        stats = _POLICY.setdefault(u.action_id, [0.0, 0.0])
        stats[0] += float(u.reward)
        stats[1] += 1.0
        angle = action_to_angle(u.action_id)
        fisher_info = fisher_information(angle)
        stats[0] += fisher_info * u.propensity

def get_hybrid_action() -> str:
    """Select an action using the hybrid algorithm."""
    best_action = None
    best_value = -float('inf')
    for action_id in _POLICY:
        stats = _POLICY[action_id]
        value = stats[0] / stats[1]
        angle = action_to_angle(action_id)
        fisher_info = fisher_information(angle)
        value += fisher_info
        if value > best_value:
            best_value = value
            best_action = action_id
    return best_action

if __name__ == "__main__":
    reset_policy()
    updates = [
        BanditUpdate("context1", "action1", 1.0, 0.5),
        BanditUpdate("context1", "action2", 0.5, 0.3),
        BanditUpdate("context1", "action3", 0.2, 0.2),
    ]
    update_policy(updates)
    hybrid_update(updates)
    print(get_hybrid_action())