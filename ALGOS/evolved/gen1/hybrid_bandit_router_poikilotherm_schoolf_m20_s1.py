# DARWIN HAMMER — match 20, survivor 1
# gen: 1
# parent_a: bandit_router.py (gen0)
# parent_b: poikilotherm_schoolfield.py (gen0)
# born: 2026-05-29T23:23:03Z

"""This module fuses the LinUCB/Thompson/epsilon-greedy-lite action router from bandit_router.py 
and the Schoolfield-Rollinson poikilotherm rate primitive from poikilotherm_schoolfield.py. 
The mathematical bridge between these two structures is the incorporation of the temperature-dependent 
developmental rate from the poikilotherm model into the bandit algorithm's reward function and action selection process.
This allows the bandit algorithm to adapt its exploration-exploitation trade-off based on the current temperature or state of the system."""

import math
import random
import numpy as np
from dataclasses import dataclass
from typing import List, Dict

R_CAL = 1.987  # cal mol^-1 K^-1
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

def update_policy(updates: List[BanditUpdate]) -> None:
    for u in updates:
        s = _POLICY.setdefault(u.action_id, [0.0, 0.0])
        s[0] += float(u.reward)
        s[1] += 1.0

def c_to_k(celsius: float) -> float:
    return celsius + 273.15

def developmental_rate(temp_k: float, params: SchoolfieldParams = SchoolfieldParams()) -> float:
    if temp_k <= 0 or params.rho_25 < 0:
        raise ValueError("temperature must be Kelvin-positive and rho_25 non-negative")
    numerator = params.rho_25 * (temp_k / K25) * math.exp((params.delta_h_activation / params.r_cal) * ((1.0 / K25) - (1.0 / temp_k)))
    low = math.exp((params.delta_h_low / params.r_cal) * ((1.0 / params.t_low) - (1.0 / temp_k)))
    high = math.exp((params.delta_h_high / params.r_cal) * ((1.0 / params.t_high) - (1.0 / temp_k)))
    return numerator / (1.0 + low + high)

def temperature_dependent_reward(action_id: str, temp_c: float) -> float:
    params = SchoolfieldParams(t_low=c_to_k(5.0), t_high=c_to_k(40.0))
    rate = developmental_rate(c_to_k(temp_c), params)
    max_rate = max(developmental_rate(c_to_k(5.0 + (40.0 - 5.0) * i / 140), params) for i in range(141))
    return rate / max_rate if max_rate > 0 else 0.0

def _reward(a: str) -> float:
    total, n = _POLICY.get(a, [0.0, 0.0])
    return total / n if n else 0.0

def select_action(context: Dict[str, float], actions: List[str], algorithm: str = 'linucb', epsilon: float = 0.1, seed: int | str | None = 7, temp_c: float = 20.0) -> BanditAction:
    if not actions:
        raise ValueError('actions required')
    rng = random.Random(seed)
    if algorithm == 'epsilon_greedy' and rng.random() < epsilon:
        chosen = rng.choice(actions)
    elif algorithm == 'thompson':
        chosen = max(actions, key=lambda a: rng.betavariate(1 + max(0, _reward(a)), 1 + max(0, 1 - _reward(a))))
    else:
        scale = math.sqrt(sum(float(v) * float(v) for v in context.values())) if context else 1.0
        chosen = max(actions, key=lambda a: temperature_dependent_reward(a, temp_c) * _reward(a) + 0.1 * scale / math.sqrt(1 + _POLICY.get(a, [0, 0])[1]))
    return BanditAction(chosen, 1.0 / len(actions), _reward(chosen), 1.0 / math.sqrt(1 + _POLICY.get(chosen, [0, 0])[1]), algorithm)

def update_bandit_policy(updates: List[BanditUpdate]) -> None:
    update_policy(updates)

def get_bandit_action(context: Dict[str, float], actions: List[str], algorithm: str = 'linucb', epsilon: float = 0.1, seed: int | str | None = 7, temp_c: float = 20.0) -> BanditAction:
    return select_action(context, actions, algorithm, epsilon, seed, temp_c)

if __name__ == "__main__":
    reset_policy()
    actions = ['action1', 'action2', 'action3']
    context = {'feature1': 1.0, 'feature2': 2.0}
    temp_c = 25.0
    action = get_bandit_action(context, actions, 'linucb', 0.1, 7, temp_c)
    print(action)
    update = BanditUpdate('context1', action.action_id, 1.0, 0.5)
    update_bandit_policy([update])
    action = get_bandit_action(context, actions, 'linucb', 0.1, 7, temp_c)
    print(action)