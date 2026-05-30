# DARWIN HAMMER — match 20, survivor 0
# gen: 1
# parent_a: bandit_router.py (gen0)
# parent_b: poikilotherm_schoolfield.py (gen0)
# born: 2026-05-29T23:23:03Z

"""Hybrid algorithm combining the bandit routing of bandit_router.py with the nonlinear temperature-dependent activity curve of poikilotherm_schoolfield.py.
The mathematical bridge between the two algorithms is the concept of temperature, which in bandit_router.py can be seen as the 'context' and in poikilotherm_schoolfield.py is the temperature-dependent variable.
This hybrid algorithm integrates the two by applying the Schoolfield-Rollinson poikilotherm rate primitive to the context of the bandit router, effectively creating a temperature-dependent routing mechanism."""

import numpy as np
import math, random, sys, pathlib
from dataclasses import dataclass

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
def reset_policy() -> None: _POLICY.clear()
def update_policy(updates: list[BanditUpdate]) -> None:
    for u in updates:
        s=_POLICY.setdefault(u.action_id,[0.0,0.0]); s[0]+=float(u.reward); s[1]+=1.0
def _reward(a: str) -> float:
    total,n=_POLICY.get(a,[0.0,0.0]); return total/n if n else 0.0

def c_to_k(celsius: float) -> float:
    return celsius + 273.15

def developmental_rate(temp_k: float, params: SchoolfieldParams = SchoolfieldParams()) -> float:
    if temp_k <= 0 or params.rho_25 < 0:
        raise ValueError("temperature must be Kelvin-positive and rho_25 non-negative")
    numerator = params.rho_25 * (temp_k / 298.15) * math.exp((params.delta_h_activation / params.r_cal) * ((1.0 / 298.15) - (1.0 / temp_k)))
    low = math.exp((params.delta_h_low / params.r_cal) * ((1.0 / params.t_low) - (1.0 / temp_k)))
    high = math.exp((params.delta_h_high / params.r_cal) * ((1.0 / params.t_high) - (1.0 / temp_k)))
    return numerator / (1.0 + low + high)

def normalized_activity(temp_c: float, low_c: float = 5.0, high_c: float = 40.0, samples: int = 141) -> float:
    params = SchoolfieldParams(t_low=c_to_k(low_c), t_high=c_to_k(high_c))
    rate = developmental_rate(c_to_k(temp_c), params)
    max_rate = max(developmental_rate(c_to_k(low_c + (high_c - low_c) * i / max(1, samples - 1)), params) for i in range(samples))
    return 0.0 if max_rate <= 0 else max(0.0, min(1.0, rate / max_rate))

def temperature_dependent_routing(context: dict[str,float], actions: list[str], algorithm: str='linucb', epsilon: float=0.1, seed: int|str|None=7) -> BanditAction:
    temp_c = np.mean(list(context.values()))
    activity = normalized_activity(temp_c)
    if not actions: raise ValueError('actions required')
    rng=random.Random(seed)
    if algorithm=='epsilon_greedy' and rng.random()<epsilon: chosen=rng.choice(actions)
    elif algorithm=='thompson': chosen=max(actions, key=lambda a: rng.betavariate(1+max(0,_reward(a)),1+max(0,1-_reward(a))))
    else:
        scale=math.sqrt(sum(float(v)*float(v) for v in context.values())) if context else 1.0
        chosen=max(actions, key=lambda a: _reward(a)+0.1*scale/math.sqrt(1+_POLICY.get(a,[0,0])[1]))
    return BanditAction(chosen, activity, _reward(chosen), 1.0/math.sqrt(1+_POLICY.get(chosen,[0,0])[1]), algorithm)

def update_context(updates: list[BanditUpdate]) -> dict[str, float]:
    context = {}
    for u in updates:
        context[u.context_id] = _reward(u.action_id)
    return context

def get_action(context: dict[str,float], actions: list[str]) -> str:
    return temperature_dependent_routing(context, actions).action_id

if __name__ == "__main__":
    reset_policy()
    context = {'temp1': 20.0, 'temp2': 30.0}
    actions = ['action1', 'action2']
    action = get_action(context, actions)
    updates = [BanditUpdate('context1', action, 1.0, 0.5), BanditUpdate('context2', action, 0.5, 0.5)]
    update_policy(updates)
    action = get_action(update_context(updates), actions)
    print(action)