# DARWIN HAMMER — match 2649, survivor 0
# gen: 5
# parent_a: bandit_router.py (gen0)
# parent_b: hybrid_hybrid_hybrid_bandit_hybrid_hybrid_hybrid_m18_s6.py (gen4)
# born: 2026-05-29T23:43:18Z

# hybrid_bandit_rbf.py
# This module combines LinUCB, Thompson Sampling, epsilon-greedy with a 
# Radial Basis Function (RBF) surrogate model. The RBF surrogate uses the 
# empirical rewards from LinUCB/Thompson Sampling/epsilon-greedy to estimate 
# the expected rewards, and then uses these estimates to select the next 
# action.
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple, Callable, Sequence
import numpy as np

# ----------------------------------------------------------------------
# Shared Types
# ----------------------------------------------------------------------
Vector = Sequence[float]

# ----------------------------------------------------------------------
# Bandit core (from Parent A)
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

_POLICY: Dict[str, List[float]] = {}          # action_id → [total_reward, count]
_STORE: Dict[str, float] = {}                 # placeholder VRAM store (unused)
_SURROGATE = None                             # will hold an RBFSurrogate instance

def reset_policy() -> None:
    """Clear all learned statistics and the surrogate model."""
    _POLICY.clear()
    _STORE.clear()
    global _SURROGATE
    _SURROGATE = RBFSurrogate(centers=[], weights=[], epsilon=1.0)

def _empirical_reward(a: str) -> float:
    total, n = _POLICY.get(a, [0.0, 0.0])
    return total / n if n else 0.0

# ----------------------------------------------------------------------
# RBF surrogate (from Parent B)
# ----------------------------------------------------------------------
def gaussian(r: float, epsilon: float = 1.0) -> float:
    """Gaussian kernel exp(‑(ε·r)²)."""
    return math.exp(-((epsilon * r) ** 2))

def euclidean(a: Vector, b: Vector) -> float:
    """Euclidean distance between two equal‑length vectors."""
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

class RBFSurrogate:
    def __init__(self, centers: List[float], weights: List[float], epsilon: float = 1.0):
        self.centers = centers
        self.weights = weights
        self.epsilon = epsilon

    def predict(self, x: Vector) -> float:
        """Predict the expected reward for a given context."""
        result = 0.0
        for i, center in enumerate(self.centers):
            result += self.weights[i] * gaussian(euclidean(x, center), self.epsilon)
        return result

# ----------------------------------------------------------------------
# Hybrid bandit core
# ----------------------------------------------------------------------
def update_policy(updates: List[BanditUpdate]) -> None:
    """Update the learned statistics and the surrogate model."""
    for u in updates:
        s = _POLICY.setdefault(u.action_id, [0.0, 0.0])
        s[0] += float(u.reward)
        s[1] += 1.0
        if u.action_id not in _STORE:
            _STORE[u.action_id] = float(u.reward)

def select_action(context: Dict[str, float], actions: List[str], algorithm: str = 'linucb', epsilon: float = 0.1, seed: int | str | None = 7) -> BanditAction:
    """Select the next action based on the given algorithm."""
    if not actions:
        raise ValueError('actions required')
    rng = random.Random(seed)
    if algorithm == 'epsilon_greedy' and rng.random() < epsilon:
        chosen = rng.choice(actions)
    elif algorithm == 'thompson':
        chosen = max(actions, key=lambda a: rng.betavariate(1 + max(0, _empirical_reward(a)), 1 + max(0, 1 - _empirical_reward(a))))
    else:
        scale = math.sqrt(sum(float(v) * float(v) for v in context.values())) if context else 1.0
        chosen = max(actions, key=lambda a: _empirical_reward(a) + 0.1 * scale / math.sqrt(1 + _POLICY.get(a, [0, 0])[1]))
    
    # Use the RBF surrogate to estimate the expected reward
    rbf = _SURROGATE if _SURROGATE else RBFSurrogate(centers=[], weights=[], epsilon=1.0)
    rbf.centers = [np.array([_empirical_reward(a) for a in actions])]
    rbf.weights = [1.0]
    expected_reward = rbf.predict([float(context.get(a, 0.0)) for a in actions])
    
    return BanditAction(chosen, 1.0 / len(actions), expected_reward, 1.0 / math.sqrt(1 + _POLICY.get(chosen, [0, 0])[1]), algorithm)

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    update_policy([BanditUpdate("context1", "action1", 1.0, 1.0), BanditUpdate("context2", "action2", 2.0, 2.0)])
    print(select_action({"action1": 1.0, "action2": 2.0}, ["action1", "action2"]))