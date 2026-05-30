# DARWIN HAMMER — match 1354, survivor 1
# gen: 6
# parent_a: bayes_update.py (gen0)
# parent_b: hybrid_hybrid_hybrid_bandit_hybrid_hybrid_endpoi_m403_s0.py (gen5)
# born: 2026-05-29T23:35:29Z

"""
This module defines the bayes_hybrid_bandit algorithm, a mathematical fusion of the 
bayes_update and hybrid_hybrid_bandit_hybrid_hybrid_endpoi_m403_s0 algorithms.

The governing equations of these two algorithms can be bridged through the use of the 
Bayesian update rule to inform the bandit router's propensity scores, and the 
confidence bounds as outputs from the bandit router to update the prior 
probabilities in the Bayesian update rule.

The mathematical bridge is formed by the following steps:
1. The bandit router generates a set of propensity scores for each action.
2. These propensity scores are used as inputs to the Bayesian update rule.
3. The Bayesian update rule generates a set of updated prior probabilities, which 
   are used to update the confidence bounds of the bandit router.

This bridge allows for the integration of the exploration-exploitation trade-off 
from the bandit router with the Bayesian update rule's ability to learn from the 
propensity scores.
"""

import numpy as np
import math
import random
import sys
import pathlib
from dataclasses import dataclass, field
from typing import Dict, List

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
_STORE: Dict[str, float] = {}

def reset_policy() -> None:
    _POLICY.clear()
    _STORE.clear()

def update_policy(updates: List[BanditUpdate]) -> None:
    for u in updates:
        s = _POLICY.setdefault(u.action_id, [0.0, 0.0])
        s[0] += float(u.reward)
        s[1] += 1.0

def _reward(a: str) -> float:
    total, n = _POLICY.get(a, [0.0, 0.0])
    return total / n if n else 0.0

def bayes_marginal(prior: float, likelihood: float, false_positive: float) -> float:
    if not all(0 <= x <= 1 for x in (prior, likelihood, false_positive)):
        raise ValueError("probabilities must be in [0,1]")
    return likelihood * prior + false_positive * (1.0 - prior)

def bayes_update(prior: float, likelihood: float, marginal: float) -> float:
    if marginal <= 0:
        raise ValueError("P(E) must be > 0")
    return prior * likelihood / marginal

def select_action(context: Dict[str, float], actions: List[str], algorithm: str = 'linucb', epsilon: float = 0.1, seed: int | str | None = 7) -> BanditAction:
    if not actions:
        raise ValueError('actions required')
    rng = random.Random(seed)
    if algorithm == 'epsilon_greedy' and rng.random() < epsilon:
        chosen = rng.choice(actions)
    elif algorithm == 'thompson':
        chosen = max(actions, key=lambda a: rng.betavariate(1 + max(0, _POLICY.get(a, [0.0, 0.0])[0]), 1 + max(0, _POLICY.get(a, [0.0, 0.0])[1])))
    else:
        raise ValueError("Invalid algorithm")
    
    propensity = _POLICY.get(chosen, [0.0, 0.0])[0] / (_POLICY.get(chosen, [0.0, 0.0])[1] + 1e-6)
    expected_reward = _reward(chosen)
    confidence_bound = 1.0
    return BanditAction(action_id=chosen, propensity=propensity, expected_reward=expected_reward, confidence_bound=confidence_bound, algorithm=algorithm)

def update_bandit_action(action: BanditAction, likelihood: float, false_positive: float) -> float:
    marginal = bayes_marginal(action.propensity, likelihood, false_positive)
    updated_propensity = bayes_update(action.propensity, likelihood, marginal)
    return updated_propensity

def hybrid_bandit_update(updates: List[BanditUpdate], likelihood: float, false_positive: float) -> None:
    for u in updates:
        s = _POLICY.setdefault(u.action_id, [0.0, 0.0])
        s[0] += float(u.reward)
        s[1] += 1.0
        updated_propensity = update_bandit_action(BanditAction(action_id=u.action_id, propensity=s[0] / s[1], expected_reward=s[0] / s[1], confidence_bound=1.0, algorithm='linucb'), likelihood, false_positive)
        s[0] = updated_propensity * s[1]

if __name__ == "__main__":
    reset_policy()
    updates = [BanditUpdate(context_id='context1', action_id='action1', reward=1.0, propensity=0.5), 
                BanditUpdate(context_id='context2', action_id='action2', reward=0.0, propensity=0.3)]
    hybrid_bandit_update(updates, likelihood=0.8, false_positive=0.2)
    print(_POLICY)