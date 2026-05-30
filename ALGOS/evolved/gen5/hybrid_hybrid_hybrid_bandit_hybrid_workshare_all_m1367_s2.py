# DARWIN HAMMER — match 1367, survivor 2
# gen: 5
# parent_a: hybrid_hybrid_bandit_router_hybrid_hybrid_hybrid_m896_s1.py (gen4)
# parent_b: hybrid_workshare_allocator_doomsday_calendar_m14_s2.py (gen1)
# born: 2026-05-29T23:35:39Z

import math
import random
import numpy as np
from dataclasses import dataclass
from typing import List, Dict
import datetime as dt

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
    return numerator * (low + high)

def doomsday(year: int, month: int, day: int) -> int:
    return (dt.date(year, month, day).weekday() + 1) % 7

def weekday_weight_vector(groups: List[str], dow: int) -> np.ndarray:
    N = len(groups)
    theta = [2 * math.pi * i / N for i in range(N)]
    phi = 2 * math.pi * dow / 7
    alpha = 0.2
    weights = [1 + alpha * math.sin(t + phi) for t in theta]
    return np.array(weights) / sum(weights)

def hybrid_bandit_router(updates: List[BanditUpdate], temp_c: float, year: int, month: int, day: int) -> None:
    temp_k = c_to_k(temp_c)
    schoolfield_params = SchoolfieldParams()
    rate = developmental_rate(temp_k, schoolfield_params)
    
    dow = doomsday(year, month, day)
    groups = list(_POLICY.keys())
    weights = weekday_weight_vector(groups, dow)
    
    for i, u in enumerate(updates):
        s = _POLICY.setdefault(u.action_id, [0.0, 0.0])
        propensity = s[1] * rate * weights[i]
        print(f"Action ID: {u.action_id}, Propensity: {propensity}")

def improved_hybrid_bandit_router(updates: List[BanditUpdate], temp_c: float, year: int, month: int, day: int) -> None:
    temp_k = c_to_k(temp_c)
    schoolfield_params = SchoolfieldParams()
    rate = developmental_rate(temp_k, schoolfield_params)
    
    dow = doomsday(year, month, day)
    groups = list(_POLICY.keys())
    weights = weekday_weight_vector(groups, dow)
    
    for u in updates:
        s = _POLICY.setdefault(u.action_id, [0.0, 0.0])
        propensity = s[1] * rate * weights[list(_POLICY.keys()).index(u.action_id)]
        s[0] += float(u.reward)
        s[1] += 1.0
        print(f"Action ID: {u.action_id}, Propensity: {propensity}")

if __name__ == "__main__":
    reset_policy()
    updates = [BanditUpdate("context1", "action1", 1.0, 0.5), BanditUpdate("context2", "action2", 2.0, 0.7)]
    update_policy(updates)
    improved_hybrid_bandit_router(updates, 25.0, 2024, 9, 16)