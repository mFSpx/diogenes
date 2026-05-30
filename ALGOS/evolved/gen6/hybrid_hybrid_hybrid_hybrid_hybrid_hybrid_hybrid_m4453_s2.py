# DARWIN HAMMER — match 4453, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_bandit_m1221_s4.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_worksh_m1380_s1.py (gen4)
# born: 2026-05-29T23:55:58Z

import numpy as np
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

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
class WorkshareParams:
    total_units: float
    deterministic_target_pct: float
    groups: tuple[str, ...]

_POLICY: dict[str, list[float]] = {}
def reset_policy() -> None: 
    _POLICY.clear()

def update_policy(updates: list[BanditUpdate]) -> None:
    for u in updates:
        s=_POLICY.setdefault(u.action_id,[0.0,0.0]); s[0]+=float(u.reward); s[1]+=1.0

def _reward(a: str) -> float:
    total,n=_POLICY.get(a,[0.0,0.0]); return total/n if n else 0.0

def c_to_k(celsius: float) -> float:
    return celsius + 273.15

def developmental_rate(temp_k: float, params: SchoolfieldParams) -> float:
    if temp_k <= 0 or params.rho_25 < 0:
        raise ValueError("Invalid temperature or rho_25")
    return params.rho_25 * math.exp((params.delta_h_activation / params.r_cal) * ((1 / params.t_low) - (1 / temp_k)))

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))

def ltc_f(
    x: np.ndarray,
    I: np.ndarray,
    W: np.ndarray,
    b: np.ndarray,
) -> np.ndarray:
    concat = np.concatenate([x, I], axis=0)
    return sigmoid(W @ concat + b)

def allocate_workshare(params: WorkshareParams, developmental_rate: float) -> dict[str, float]:
    if params.total_units <= 0:
        raise ValueError("total_units must be positive")
    if not 0 <= params.deterministic_target_pct <= 100:
        raise ValueError("deterministic_target_pct must be between 0 and 100")
    if not params.groups:
        raise ValueError("groups required")
    deterministic_units = params.total_units * params.deterministic_target_pct / 100.0
    modulated_units = deterministic_units * developmental_rate
    return {group: modulated_units / len(params.groups) for group in params.groups}

def temperature_dependent_reward(temp_k: float, schoolfield_params: SchoolfieldParams) -> float:
    rate = developmental_rate(temp_k, schoolfield_params)
    return rate * (1 - rate)

def hybrid_operation(schoolfield_params: SchoolfieldParams, workshare_params: WorkshareParams, temp_c: float) -> dict[str, float]:
    temp_k = c_to_k(temp_c)
    rate = developmental_rate(temp_k, schoolfield_params)
    reward = temperature_dependent_reward(temp_k, schoolfield_params)
    return allocate_workshare(workshare_params, rate)

def optimize_bandit_router(actions: List[BanditAction], updates: List[BanditUpdate]) -> None:
    update_policy(updates)
    action_rewards = {action.action_id: _reward(action.action_id) for action in actions}
    best_action_id = max(action_rewards, key=action_rewards.get)
    print(f"Best action: {best_action_id}")

if __name__ == "__main__":
    schoolfield_params = SchoolfieldParams()
    workshare_params = WorkshareParams(total_units=100.0, deterministic_target_pct=90.0, groups=("codex", "groq", "cohere", "local_models"))
    temp_c = 25.0
    result = hybrid_operation(schoolfield_params, workshare_params, temp_c)
    print(result)
    
    actions = [
        BanditAction(action_id="action1", propensity=0.5, expected_reward=10.0, confidence_bound=0.1, algorithm="UCB"),
        BanditAction(action_id="action2", propensity=0.3, expected_reward=20.0, confidence_bound=0.2, algorithm="epsilon-greedy"),
    ]
    updates = [
        BanditUpdate(context_id="context1", action_id="action1", reward=10.0, propensity=0.5),
        BanditUpdate(context_id="context2", action_id="action2", reward=20.0, propensity=0.3),
    ]
    optimize_bandit_router(actions, updates)