# DARWIN HAMMER — match 2649, survivor 2
# gen: 5
# parent_a: bandit_router.py (gen0)
# parent_b: hybrid_hybrid_hybrid_bandit_hybrid_hybrid_hybrid_m18_s6.py (gen4)
# born: 2026-05-29T23:43:18Z

"""
Hybrid bandit algorithm fusing LinUCB/Thompson/epsilon-greedy-lite (bandit_router.py) 
and RBF surrogate model (hybrid_hybrid_hybrid_bandit_hybrid_hybrid_hybrid_m18_s6.py).

The mathematical bridge between the two parents lies in the integration of the 
empirical reward estimation from the bandit core with the RBF surrogate model. 
The hybrid algorithm uses the RBF surrogate to estimate the reward for unexplored 
actions and combines it with the empirical reward estimation for explored actions.

The governing equations of the parents are fused as follows:

- The empirical reward estimation from the bandit core is used to update the 
  RBF surrogate model.
- The RBF surrogate model is used to estimate the reward for unexplored actions.
- The hybrid algorithm selects the action with the highest estimated reward, 
  which is a combination of the empirical reward estimation and the RBF surrogate 
  estimation.
"""

import math
import random
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple, Callable, Sequence

# ----------------------------------------------------------------------
# Shared Types
# ----------------------------------------------------------------------
Vector = Sequence[float]

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

_POLICY: Dict[str, List[float]] = {}  # action_id → [total_reward, count]
_SURROGATE = None                     # will hold an RBFSurrogate instance

def reset_policy() -> None:
    """Clear all learned statistics and the surrogate model."""
    _POLICY.clear()
    global _SURROGATE
    _SURROGATE = RBFSurrogate(centers=[], weights=[], epsilon=1.0)

def _empirical_reward(a: str) -> float:
    total, n = _POLICY.get(a, [0.0, 0.0])
    return total / n if n else 0.0

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

class RBFSurrogate:
    def __init__(self, centers: List[Vector], weights: List[float], epsilon: float):
        self.centers = centers
        self.weights = weights
        self.epsilon = epsilon

    def estimate(self, x: Vector) -> float:
        return sum(self.weights[i] * gaussian(euclidean(x, self.centers[i]), self.epsilon) for i in range(len(self.centers)))

def update_surrogate(updates: List[BanditUpdate], context: Vector) -> None:
    global _SURROGATE
    centers = [context]
    weights = [_empirical_reward(update.action_id) for update in updates]
    _SURROGATE = RBFSurrogate(centers, weights, 1.0)

def select_action(context: Vector, actions: List[str], algorithm: str = 'linucb', epsilon: float = 0.1, seed: int | str | None = 7) -> BanditAction:
    if not actions:
        raise ValueError('actions required')
    rng = random.Random(seed)
    if algorithm == 'epsilon_greedy' and rng.random() < epsilon:
        chosen = rng.choice(actions)
    elif algorithm == 'thompson':
        chosen = max(actions, key=lambda a: rng.betavariate(1 + max(0, _empirical_reward(a)), 1 + max(0, 1 - _empirical_reward(a))))
    else:
        scale = math.sqrt(sum(float(v) * float(v) for v in context)) if context else 1.0
        chosen = max(actions, key=lambda a: _empirical_reward(a) + 0.1 * scale / math.sqrt(1 + _POLICY.get(a, [0, 0])[1]))
    # Update surrogate model
    update_surrogate([BanditUpdate("context_id", chosen, _empirical_reward(chosen), 1.0)], context)
    return BanditAction(chosen, 1.0 / len(actions), _empirical_reward(chosen), 1.0 / math.sqrt(1 + _POLICY.get(chosen, [0, 0])[1]), algorithm)

def update_policy(updates: List[BanditUpdate]) -> None:
    for u in updates:
        s = _POLICY.setdefault(u.action_id, [0.0, 0.0])
        s[0] += float(u.reward)
        s[1] += 1.0

if __name__ == "__main__":
    reset_policy()
    context = [1.0, 2.0, 3.0]
    actions = ["action1", "action2", "action3"]
    update_policy([BanditUpdate("context_id", "action1", 10.0, 1.0)])
    action = select_action(context, actions)
    print(action)