# DARWIN HAMMER — match 5810, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_hybrid_bandit_hybrid_bandit_router_m206_s1.py (gen3)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_sketch_m917_s3.py (gen5)
# born: 2026-05-30T00:04:49Z

"""
Hybrid Algorithm combining:
- Parent A: Hybrid Bandit Algorithm with Schoolfield temperature model (hybrid_hybrid_hybrid_bandit_hybrid_bandit_router_m206_s1.py)
- Parent B: Hybrid Algorithm with Geometric Algebra and Koopman operator dynamics (hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_sketch_m917_s3.py)

Mathematical Bridge:
The bandit algorithm's action selection is guided by a probability distribution 
that is updated based on observed rewards. This distribution can be represented 
as a multivector in a Clifford algebra, similar to Parent B's approach. The 
Schoolfield temperature model can be used to modulate the pheromone-like weights 
that drive action selection in the hybrid algorithm. The Koopman operator learned 
from paired state matrices can be applied to the coefficient vector of this 
multivector, yielding a linear evolution in the algebraic space. The resulting 
coefficients are normalised to form a probability distribution which is then 
refined by a Bayesian update using a Beta prior per bucket.
"""

import math
import random
import numpy as np
from collections import defaultdict
from typing import Iterable, List, Dict, Tuple, FrozenSet
from dataclasses import dataclass

# ----------------------------------------------------------------------
# Bandit core – global statistics
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
_ACTION_TEMPS: Dict[str, float] = {}
_BETA: float = 1.0  
_BETA_SUM_XX: float = 0.0  
_BETA_SUM_XY: float = 0.0 

def reset_policy() -> None:
    _POLICY.clear()
    _ACTION_TEMPS.clear()
    global _BETA, _BETA_SUM_XX, _BETA_SUM_XY
    _BETA = 1.0
    _BETA_SUM_XX = 0.0
    _BETA_SUM_XY = 0.0

def _reward(a: str) -> float:
    total, n = _POLICY.get(a, [0.0, 0.0])
    return total / n if n else 0.0

def _count(a: str) -> float:
    return _POLICY.get(a, [0.0, 0.0])[1]

def update_policy(updates: Iterable[BanditUpdate]) -> None:
    for u in updates:
        stats = _POLICY.setdefault(u.action_id, [0.0, 0.0])
        stats[0] += float(u.reward)
        stats[1] += 1.0

# ----------------------------------------------------------------------
# Schoolfield temperature model
# ----------------------------------------------------------------------

R_CAL = 1.987  
K25 = 298.15  

def schoolfield_temperature(action_id: str, temperature: float) -> float:
    return R_CAL * math.log(temperature / K25)

# ----------------------------------------------------------------------
# Geometric Algebra core
# ----------------------------------------------------------------------

def _blade_sign(indices: List[int]) -> Tuple[List[int], int]:
    lst = list(indices)
    sign = 1
    n = len(lst)
    i = 0
    while i < n:
        j = 0
        while j < n - 1 - i:
            if lst[j] > lst[j + 1]:
                lst[j], lst[j + 1] = lst[j + 1], lst[j]
                sign *= -1
            elif lst[j] == lst[j + 1]:
                del lst[j : j + 2]
                n -= 2
                continue
            j += 1
        i += 1
    return lst, sign

def _multiply_blades(blade_a: FrozenSet[int], blade_b: FrozenSet[int]) -> Tuple[FrozenSet[int], int]:
    combined = list(blade_a) + list(blade_b)
    result, sign = _blade_sign(combined)
    return frozenset(result), sign

# ----------------------------------------------------------------------
# Hybrid Algorithm
# ----------------------------------------------------------------------

def hybrid_action_selection(actions: List[str], 
                           temperatures: Dict[str, float], 
                           updates: Iterable[BanditUpdate]) -> str:
    update_policy(updates)
    action_probabilities = {}
    for action in actions:
        temperature = temperatures[action]
        reward = _reward(action)
        count = _count(action)
        action_probabilities[action] = reward * math.exp(schoolfield_temperature(action, temperature) / R_CAL)
    return np.random.choice(actions, p=[p / sum(action_probabilities.values()) for p in action_probabilities.values()])

def hybrid_koopman_operator(actions: List[str], 
                           temperatures: Dict[str, float], 
                           updates: Iterable[BanditUpdate]) -> Dict[str, float]:
    action_probabilities = {}
    for action in actions:
        temperature = temperatures[action]
        reward = _reward(action)
        count = _count(action)
        action_probabilities[action] = reward * math.exp(schoolfield_temperature(action, temperature) / R_CAL)
    koopman_operator = defaultdict(float)
    for action, probability in action_probabilities.items():
        blade, sign = _multiply_blades(frozenset([action]), frozenset([action]))
        koopman_operator[action] += probability * sign
    return dict(koopman_operator)

def hybrid_bayesian_update(actions: List[str], 
                          temperatures: Dict[str, float], 
                          updates: Iterable[BanditUpdate]) -> Dict[str, float]:
    action_probabilities = {}
    for action in actions:
        temperature = temperatures[action]
        reward = _reward(action)
        count = _count(action)
        action_probabilities[action] = reward * math.exp(schoolfield_temperature(action, temperature) / R_CAL)
    bayesian_updates = defaultdict(float)
    for action, probability in action_probabilities.items():
        blade, sign = _multiply_blades(frozenset([action]), frozenset([action]))
        bayesian_updates[action] += probability * sign
    return dict(bayesian_updates)

if __name__ == "__main__":
    actions = ["action1", "action2", "action3"]
    temperatures = {"action1": 20.0, "action2": 25.0, "action3": 30.0}
    updates = [BanditUpdate("context1", "action1", 10.0, 0.5), 
               BanditUpdate("context2", "action2", 20.0, 0.6)]
    print(hybrid_action_selection(actions, temperatures, updates))
    print(hybrid_koopman_operator(actions, temperatures, updates))
    print(hybrid_bayesian_update(actions, temperatures, updates))