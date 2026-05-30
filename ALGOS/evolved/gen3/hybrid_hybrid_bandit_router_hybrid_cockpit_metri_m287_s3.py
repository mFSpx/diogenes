# DARWIN HAMMER — match 287, survivor 3
# gen: 3
# parent_a: hybrid_bandit_router_poikilotherm_schoolf_m20_s2.py (gen1)
# parent_b: hybrid_cockpit_metrics_hybrid_pheromone_inf_m64_s0.py (gen2)
# born: 2026-05-29T23:28:06Z

"""
Hybrid algorithm fusing 'hybrid_bandit_router_poikilotherm_schoolf_m20_s2.py' and 'hybrid_cockpit_metrics_hybrid_pheromone_inf_m64_s0.py'.

The mathematical bridge between these two structures lies in the concept of fusing the 
temperature-aware exploration term from the bandit router with the honesty-weighted 
pheromone signal system from the hybrid pheromone infotaxis. Specifically, we 
integrate the temperature-dependent activity gate A(T) into the calculation of the 
honesty-weighted pheromone signal strength, allowing the system to adapt its 
exploration-exploitation balance based on the operating temperature.

The core fusion occurs in the `calculate_hybrid_pheromone_signal` function, where 
the temperature-aware scale S_T is used to modulate the pheromone signal strength.
"""

import numpy as np
import math
import random
import sys
import pathlib
from dataclasses import dataclass
from datetime import datetime

@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float
    expected_reward: float

def temperature_activity(celsius: float) -> float:
    """
    Compute the normalized activity gate from Celsius.
    """
    T_opt = 25.0  # Optimal temperature
    T_low = 10.0  # Lower bound temperature
    T_high = 40.0  # Upper bound temperature
    A_low = 0.01  # Activity at low temperature
    A_high = 0.01  # Activity at high temperature
    A_opt = 1.0  # Activity at optimal temperature

    if celsius < T_low:
        return A_low + (A_opt - A_low) * (celsius - T_low) / (T_opt - T_low)
    elif celsius > T_high:
        return A_high + (A_opt - A_high) * (T_high - celsius) / (T_high - T_opt)
    else:
        return A_opt

def calculate_context_norm(context: np.ndarray) -> float:
    """
    Compute the Euclidean norm of the context vector.
    """
    return np.linalg.norm(context)

def hybrid_bandit_scale(context: np.ndarray, celsius: float) -> float:
    """
    Compute the temperature-aware scale S_T.
    """
    A_T = temperature_activity(celsius)
    context_norm = calculate_context_norm(context)
    return A_T * context_norm

def anti_slop_ratio(claims_with_evidence: int, total_claims_emitted: int) -> float:
    """
    Calculate the anti-slop ratio based on claims with evidence and total claims emitted.
    """
    return 1.0 if total_claims_emitted <= 0 else max(0.0, min(1.0, claims_with_evidence / total_claims_emitted))

def calculate_hybrid_pheromone_signal(surface_key, signal_kind, signal_value, half_life_seconds, 
                                       claims_with_evidence, total_claims_emitted, context: np.ndarray, celsius: float):
    """
    Calculate the honesty-weighted pheromone signal strength modulated by the temperature-aware scale.
    """
    honesty_weight = anti_slop_ratio(claims_with_evidence, total_claims_emitted)
    S_T = hybrid_bandit_scale(context, celsius)
    return signal_value * math.pow(0.5, (datetime.now(pathlib.PurePath().root) - datetime.now(pathlib.PurePath().root)).total_seconds() / half_life_seconds) * honesty_weight * S_T

def hybrid_select_action(actions: list[BanditAction], context: np.ndarray, celsius: float) -> BanditAction:
    """
    Select an action based on the temperature-aware bandit algorithm.
    """
    S_T = hybrid_bandit_scale(context, celsius)
    best_action = max(actions, key=lambda a: a.expected_reward + S_T / math.sqrt(1 + a.propensity))
    return best_action

if __name__ == "__main__":
    context = np.array([1.0, 2.0, 3.0])
    celsius = 25.0
    actions = [BanditAction("action1", 10, 5.0), BanditAction("action2", 5, 3.0)]
    surface_key = "test_surface"
    signal_kind = "test_signal"
    signal_value = 1.0
    half_life_seconds = 3600
    claims_with_evidence = 10
    total_claims_emitted = 20

    hybrid_pheromone_signal = calculate_hybrid_pheromone_signal(surface_key, signal_kind, signal_value, half_life_seconds, 
                                                               claims_with_evidence, total_claims_emitted, context, celsius)
    selected_action = hybrid_select_action(actions, context, celsius)

    print(f"Hybrid pheromone signal: {hybrid_pheromone_signal}")
    print(f"Selected action: {selected_action.action_id}")