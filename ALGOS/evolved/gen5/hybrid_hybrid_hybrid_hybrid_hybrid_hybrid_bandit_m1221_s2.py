# DARWIN HAMMER — match 1221, survivor 2
# gen: 5
# parent_a: hybrid_hybrid_hybrid_bandit_hybrid_hybrid_doomsd_m988_s0.py (gen4)
# parent_b: hybrid_hybrid_bandit_router_hybrid_hybrid_bandit_m534_s1.py (gen4)
# born: 2026-05-29T23:34:30Z

import numpy as np
import math
import random
import sys
import pathlib

"""
This module integrates the hybrid_hybrid_hybrid_bandit_hybrid_hybrid_doomsd_m988_s0 and 
hybrid_hybrid_bandit_router_hybrid_hybrid_bandit_m534_s1 algorithms. The mathematical 
bridge between the two structures is the use of the bandit_router's action selection 
mechanism to update the graph structure in a manner that resembles the Gini coefficient 
computation from the first parent, and the temperature-dependent reward function from 
the second parent. This allows for the extraction of relevant features from the environment, 
which can then be used in the NLMS prediction, while optimizing the exploration of the 
solution space using the temperature-dependent constraints.
"""

@dataclass(frozen=True)
class BanditAction:
    action_id: str; propensity: float; expected_reward: float; confidence_bound: float; algorithm: str

@dataclass(frozen=True)
class BanditUpdate:
    context_id: str; action_id: str; reward: float; propensity: float

@dataclass(frozen=True)
class SchoolfieldParams:
    rho_25: float = 1.0
    delta_h_activation: float = 12_000.0
    t_low: float = 283.15
    t_high: float = 307.15
    delta_h_low: float = -45_000.0
    delta_h_high: float = 65_000.0
    r_cal: float = 1.987

_POLICY: dict[str, list[float]] = {}
def reset_policy() -> None: 
    _POLICY.clear()

def update_policy(updates: list[BanditUpdate]) -> None:
    for u in updates:
        s=_POLICY.setdefault(u.action_id,[0.0,0.0]); s[0]+=float(u.reward); s[1]+=1.0

def _reward(a: str) -> float:
    total,n=_POLICY.get(a,[0.0,0.0]); return total/n if n else 0.0

def select_action(context: dict[str,float], actions: list[str], algorithm: str='linucb', epsilon: float=0.1, seed: int|str|None=7) -> BanditAction:
    if not actions: raise ValueError('actions required')
    rng=random.Random(seed)
    if algorithm=='epsilon_greedy' and rng.random()<epsilon: chosen=rng.choice(actions)
    elif algorithm=='thompson': chosen=max(actions, key=lambda a: rng.betavariate(1+max(0,_reward(a)),1+max(0,1-_reward(a))))
    else:
        scale=math.sqrt(sum(float(v)*float(v) for v in context.values())) if context else 1.0
        chosen=max(actions, key=lambda a: _reward(a)+0.1*scale/math.sqrt(1+_POLICY.get(a,[0,0])[1]))
    return BanditAction(chosen,1.0/len(actions),_reward(chosen),1.0/math.sqrt(1+_POLICY.get(chosen,[0,0])[1]),algorithm)

def c_to_k(celsius: float) -> float:
    return celsius + 273.15

def developmental_rate(temp_k: float, params: SchoolfieldParams) -> float:
    if temp_k <= 0 or params.rho_25 < 0:
        raise ValueError("temperature must be Kelvin-positive and rho_25 non-negative")
    numerator = params.rho_25 * (temp_k / 298.15) * math.exp(
        (params.delta_h_activation / params.r_cal) * (1 / 298.15 - 1 / temp_k)
    )
    denominator = 1 + math.exp(
        (params.delta_h_high / params.r_cal) * (1 / 298.15 - 1 / temp_k)
    ) + math.exp(
        (params.delta_h_low / params.r_cal) * (1 / 298.15 - 1 / temp_k)
    )
    return numerator / denominator

def temperature_dependent_reward(action: str, temp_k: float, params: SchoolfieldParams) -> float:
    return developmental_rate(temp_k, params) * _reward(action)

def hybrid_action_selection(context: dict[str,float], actions: list[str], algorithm: str='linucb', epsilon: float=0.1, seed: int|str|None=7, temp_k: float=300.0, params: SchoolfieldParams=SchoolfieldParams()) -> BanditAction:
    if not actions: raise ValueError('actions required')
    rng=random.Random(seed)
    if algorithm=='epsilon_greedy' and rng.random()<epsilon: chosen=rng.choice(actions)
    elif algorithm=='thompson': chosen=max(actions, key=lambda a: rng.betavariate(1+max(0,temperature_dependent_reward(a, temp_k, params)),1+max(0,1-temperature_dependent_reward(a, temp_k, params))))
    else:
        scale=math.sqrt(sum(float(v)*float(v) for v in context.values())) if context else 1.0
        chosen=max(actions, key=lambda a: temperature_dependent_reward(a, temp_k, params)+0.1*scale/math.sqrt(1+_POLICY.get(a,[0,0])[1]))
    return BanditAction(chosen,1.0/len(actions),temperature_dependent_reward(chosen, temp_k, params),1.0/math.sqrt(1+_POLICY.get(chosen,[0,0])[1]),algorithm)

if __name__ == "__main__":
    reset_policy()
    actions = ["action1", "action2", "action3"]
    context = {"feature1": 1.0, "feature2": 2.0}
    params = SchoolfieldParams()
    temp_k = 300.0
    action = hybrid_action_selection(context, actions, temp_k=temp_k, params=params)
    print(action)
    update_policy([BanditUpdate("context1", action.action_id, 1.0, action.propensity)])
    print(_reward(action.action_id))