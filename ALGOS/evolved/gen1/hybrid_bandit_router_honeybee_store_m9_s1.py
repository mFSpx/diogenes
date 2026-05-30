# DARWIN HAMMER — match 9, survivor 1
# gen: 1
# parent_a: bandit_router.py (gen0)
# parent_b: honeybee_store.py (gen0)
# born: 2026-05-29T23:16:48Z

#!/usr/bin/env python3
"""Hybrid algorithm combining LinUCB/Thompson/epsilon-greedy-lite action router 
(bandit_router.py) and common-store feedback primitive for decentralized resource rate control 
(honeybee_store.py) through a mathematical bridge that utilizes the propensity scores from 
the bandit router as inflow rates and the confidence bounds as outflow rates in the common 
store feedback primitive."""

from __future__ import annotations
import math, random
from dataclasses import dataclass
import numpy as np
import sys
import pathlib

@dataclass(frozen=True)
class BanditAction:
    action_id: str; propensity: float; expected_reward: float; confidence_bound: float; algorithm: str

@dataclass(frozen=True)
class BanditUpdate:
    context_id: str; action_id: str; reward: float; propensity: float

_POLICY: dict[str, list[float]] = {}
_STORE: dict[str, float] = {}

def reset_policy() -> None: 
    _POLICY.clear()
    _STORE.clear()

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

def update_store(store: float, inflow: list[float], outflow: list[float], alpha: float = 1.0, beta: float = 1.0, dt: float = 1.0) -> tuple[float, float]:
    delta = alpha * sum(inflow) - beta * sum(outflow)
    return max(0.0, store + dt * delta), delta

def dance_duration(delta_store: float, base: float = 1.0, gain: float = 1.0, limit: float = 10.0) -> float:
    return max(0.0, min(limit, base + gain * delta_store))

def hybrid_update(actions: list[BanditAction], store: float, alpha: float = 1.0, beta: float = 1.0, dt: float = 1.0) -> tuple[float, float]:
    inflow = [a.propensity for a in actions]
    outflow = [a.confidence_bound for a in actions]
    return update_store(store, inflow, outflow, alpha, beta, dt)

def hybrid_action_store(context: dict[str,float], actions: list[str], algorithm: str='linucb', epsilon: float=0.1, seed: int|str|None=7, alpha: float = 1.0, beta: float = 1.0, dt: float = 1.0) -> tuple[BanditAction, float, float]:
    bandit_action = select_action(context, actions, algorithm, epsilon, seed)
    store = _STORE.get(actions[0], 0.0)
    new_store, delta = hybrid_update([bandit_action], store, alpha, beta, dt)
    _STORE[actions[0]] = new_store
    return bandit_action, new_store, delta

def hybrid_dance_duration(actions: list[BanditAction], delta_store: float, base: float = 1.0, gain: float = 1.0, limit: float = 10.0) -> float:
    return dance_duration(delta_store, base, gain, limit)

if __name__ == "__main__":
    reset_policy()
    actions = ["action1", "action2", "action3"]
    context = {"context1": 0.5, "context2": 0.3}
    bandit_action, new_store, delta = hybrid_action_store(context, actions)
    dance_duration_value = hybrid_dance_duration([bandit_action], delta)
    print("Bandit Action:", bandit_action)
    print("New Store:", new_store)
    print("Delta:", delta)
    print("Dance Duration:", dance_duration_value)
    sys.exit(0)