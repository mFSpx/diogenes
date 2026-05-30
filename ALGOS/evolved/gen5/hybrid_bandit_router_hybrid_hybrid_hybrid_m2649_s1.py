# DARWIN HAMMER — match 2649, survivor 1
# gen: 5
# parent_a: bandit_router.py (gen0)
# parent_b: hybrid_hybrid_hybrid_bandit_hybrid_hybrid_hybrid_m18_s6.py (gen4)
# born: 2026-05-29T23:43:18Z

"""
This module defines a novel HYBRID algorithm that mathematically fuses the core topologies of 
bandit_router.py and hybrid_hybrid_hybrid_bandit_hybrid_hybrid_hybrid_m18_s6.py into a single 
unified system. The mathematical bridge between the two parents lies in the integration of 
the bandit update logic with the radial basis function (RBF) surrogate model. The hybrid 
algorithm combines the exploration-exploitation trade-off of the bandit algorithm with the 
function approximation capabilities of the RBF surrogate model.
"""

import math
import random
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np

# ----------------------------------------------------------------------
# Shared Types
# ----------------------------------------------------------------------
Vector = list[float]

# ----------------------------------------------------------------------
# Bandit core
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

# ----------------------------------------------------------------------
# RBF surrogate
# ----------------------------------------------------------------------
def gaussian(r: float, epsilon: float = 1.0) -> float:
    """Gaussian kernel exp(‑(ε·r)²)."""
    return math.exp(-((epsilon * r) ** 2))

def euclidean(a: Vector, b: Vector) -> float:
    """Euclidean distance between two equal‑length vectors."""
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

# ----------------------------------------------------------------------
# Bandit update logic
# ----------------------------------------------------------------------
_POLICY: dict[str, list[float]] = {}          # action_id → [total_reward, count]
_STORE: dict[str, float] = {}                 # placeholder VRAM store (unused)
_SURROGATE = None                             # will hold an RBFSurrogate instance

def reset_policy() -> None:
    """Clear all learned statistics and the surrogate model."""
    global _POLICY, _STORE, _SURROGATE
    _POLICY.clear()
    _STORE.clear()
    _SURROGATE = None

def update_policy(updates: list[BanditUpdate]) -> None:
    for u in updates:
        s = _POLICY.setdefault(u.action_id, [0.0, 0.0])
        s[0] += float(u.reward)
        s[1] += 1.0

def _empirical_reward(a: str) -> float:
    total, n = _POLICY.get(a, [0.0, 0.0])
    return total / n if n else 0.0

# ----------------------------------------------------------------------
# Hybrid algorithm
# ----------------------------------------------------------------------
class RBFSurrogate:
    def __init__(self, centers: list[Vector], weights: list[float], epsilon: float = 1.0):
        self.centers = centers
        self.weights = weights
        self.epsilon = epsilon

    def evaluate(self, x: Vector) -> float:
        return sum(w * gaussian(euclidean(x, c), self.epsilon) for w, c in zip(self.weights, self.centers))

def select_action(context: dict[str, float], actions: list[str], algorithm: str = 'linucb', epsilon: float = 0.1, seed: int | str | None = 7) -> BanditAction:
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
    return BanditAction(chosen, 1.0 / len(actions), _empirical_reward(chosen), 1.0 / math.sqrt(1 + _POLICY.get(chosen, [0, 0])[1]), algorithm)

def update_surrogate(actions: list[Vector], rewards: list[float]) -> None:
    global _SURROGATE
    if _SURROGATE is None:
        _SURROGATE = RBFSurrogate(centers=actions, weights=rewards, epsilon=1.0)
    else:
        _SURROGATE.centers.extend(actions)
        _SURROGATE.weights.extend(rewards)

def hybrid_select_action(context: dict[str, float], actions: list[Vector], algorithm: str = 'linucb', epsilon: float = 0.1, seed: int | str | None = 7) -> BanditAction:
    if not actions:
        raise ValueError('actions required')
    rng = random.Random(seed)
    if algorithm == 'epsilon_greedy' and rng.random() < epsilon:
        chosen = rng.choice(actions)
    elif algorithm == 'thompson':
        chosen = max(actions, key=lambda a: rng.betavariate(1 + max(0, _SURROGATE.evaluate(a) if _SURROGATE else 0), 1 + max(0, 1 - _SURROGATE.evaluate(a) if _SURROGATE else 0)))
    else:
        scale = math.sqrt(sum(float(v) * float(v) for v in context.values())) if context else 1.0
        chosen = max(actions, key=lambda a: _SURROGATE.evaluate(a) if _SURROGATE else 0 + 0.1 * scale / math.sqrt(1 + sum(1 for c in _SURROGATE.centers if c == a) if _SURROGATE else 0))
    return BanditAction(str(chosen), 1.0 / len(actions), _SURROGATE.evaluate(chosen) if _SURROGATE else 0, 1.0 / math.sqrt(1 + sum(1 for c in _SURROGATE.centers if c == chosen) if _SURROGATE else 0), algorithm)

if __name__ == "__main__":
    reset_policy()
    actions = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    rewards = [1.0, 2.0, 3.0]
    update_surrogate(actions, rewards)
    context = {'feature1': 1.0, 'feature2': 2.0}
    action = hybrid_select_action(context, actions)
    print(action)