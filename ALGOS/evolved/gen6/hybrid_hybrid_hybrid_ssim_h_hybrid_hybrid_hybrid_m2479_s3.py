# DARWIN HAMMER — match 2479, survivor 3
# gen: 6
# parent_a: hybrid_hybrid_ssim_hybrid_h_hybrid_sketches_rlct_m1064_s0.py (gen5)
# parent_b: hybrid_hybrid_hybrid_bandit_hybrid_capybara_opti_m41_s4.py (gen3)
# born: 2026-05-29T23:42:38Z

import math
import random
import sys
from pathlib import Path
from typing import Sequence, List, Dict, Tuple, FrozenSet, Any
import numpy as np

class Multivector:
    def __init__(self, components: Dict[FrozenSet[int], float], n: int):
        self.components = {
            k: float(v) for k, v in components.items() if abs(v) > 1e-15
        }
        self.n = int(n)

    def scalar_part(self) -> float:
        return self.components.get(frozenset(), 0.0)

    def __add__(self, other: "Multivector") -> "Multivector":
        result = dict(self.components)
        for blade, value in other.components.items():
            result[blade] = result.get(blade, 0.0) + value
        return Multivector(result, self.n)

    def __mul__(self, other: "Multivector") -> "Multivector":
        result: Dict[FrozenSet[int], float] = {}
        for b1, v1 in self.components.items():
            for b2, v2 in other.components.items():
                new_blade = frozenset(b1.union(b2))
                result[new_blade] = result.get(new_blade, 0.0) + v1 * v2
        return Multivector(result, self.n)

def stats_to_multivector(seq: Sequence[float]) -> Multivector:
    arr = np.asarray(seq, dtype=float)
    mean = float(np.mean(arr)) if arr.size else 0.0
    var = float(np.var(arr)) if arr.size else 0.0
    cov = 0.0
    comps: Dict[FrozenSet[int], float] = {
        frozenset(): mean,
        frozenset([0]): var,
        frozenset([0, 1]): cov,
    }
    return Multivector(comps, 2)

def geometric_ssim(x: Sequence[float], y: Sequence[float]) -> float:
    mx = stats_to_multivector(x)
    my = stats_to_multivector(y)
    return (mx * my).scalar_part()

ALPHA = 0.6
BETA = 0.4
DT = 1.0
ETA0 = 0.01
HOEFFDING_DELTA = 0.05
CLAMP_LO = -5.0
CLAMP_HI = 5.0

class BanditAction:
    __slots__ = ("action_id", "propensity", "expected_reward",
                 "confidence_bound", "algorithm")
    def __init__(self, action_id: str, propensity: float,
                 expected_reward: float, confidence_bound: float,
                 algorithm: str):
        self.action_id = action_id
        self.propensity = propensity
        self.expected_reward = expected_reward
        self.confidence_bound = confidence_bound
        self.algorithm = algorithm

    def __repr__(self) -> str:
        return (f"BanditAction(action_id={self.action_id!r}, propensity={self.propensity:.3f}, "
                f"expected_reward={self.expected_reward:.3f}, "
                f"confidence_bound={self.confidence_bound:.3f}, algorithm={self.algorithm!r})")

_POLICY: Dict[str, Tuple[float, float, int]] = {}   
_STORE: Dict[str, float] = {}                      

def reset_policy() -> None:
    _POLICY.clear()
    _STORE.clear()

def _reward_estimate(action_id: str) -> Tuple[float, float]:
    total, total_sq, n = _POLICY.get(action_id, (0.0, 0.0, 0))
    if n == 0:
        return 0.0, 0.0
    mean = total / n
    var = max(0.0, total_sq / n - mean * mean)
    return mean, var

def _confidence_bound(action_id: str) -> float:
    _, _, n = _POLICY.get(action_id, (0.0, 0.0, 0))
    if n == 0:
        return float("inf")
    return math.sqrt(math.log(2.0 / HOEFFDING_DELTA) / (2.0 * n))

def context_to_multivector(context: Dict[str, float]) -> Multivector:
    values = list(context.values())
    return stats_to_multivector(values)

def action_to_multivector(action_id: str) -> Multivector:
    mean, var = _reward_estimate(action_id)
    seq = [mean, mean + math.sqrt(var) if var > 0 else mean]
    return stats_to_multivector(seq)

def geometric_bandit_score(context: Dict[str, float], action_id: str) -> float:
    ctx_mv = context_to_multivector(context)
    act_mv = action_to_multivector(action_id)
    return (ctx_mv * act_mv).scalar_part()

def select_action_hybrid(
    context: Dict[str, float],
    actions: List[str],
    algorithm: str = "geometric_ucb",
    epsilon: float = 0.1,
    seed: int | str | None = 7,
) -> BanditAction:
    if seed is not None:
        random.seed(seed)

    if not actions:
        raise ValueError("action list cannot be empty")

    if random.random() < epsilon:
        chosen = random.choice(actions)
    else:
        scores = []
        for a in actions:
            score = geometric_bandit_score(context, a) + _confidence_bound(a) - _STORE.get(a, 0.0)
            scores.append((score, a))
        scores.sort(reverse=True)
        chosen = scores[0][1]

    mean, var = _reward_estimate(chosen)
    propensity = geometric_bandit_score(context, chosen)
    confidence_bound = _confidence_bound(chosen)
    return BanditAction(chosen, propensity, mean, confidence_bound, algorithm)

def update_hybrid(
    context: Dict[str, float],
    action_id: str,
    reward: float,
) -> None:
    if action_id not in _POLICY:
        _POLICY[action_id] = (0.0, 0.0, 0)
    total, total_sq, n = _POLICY[action_id]
    total += reward
    total_sq += reward * reward
    n += 1
    _POLICY[action_id] = (total, total_sq, n)

    score = geometric_bandit_score(context, action_id)
    _STORE[action_id] = _STORE.get(action_id, 0.0) + ALPHA * score - BETA * DT

# Example usage:
context = {"feature1": 1.0, "feature2": 2.0}
actions = ["action1", "action2", "action3"]
action = select_action_hybrid(context, actions)
print(action)
update_hybrid(context, action.action_id, 1.0)