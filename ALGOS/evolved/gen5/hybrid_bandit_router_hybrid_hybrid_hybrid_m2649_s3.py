# DARWIN HAMMER — match 2649, survivor 3
# gen: 5
# parent_a: bandit_router.py (gen0)
# parent_b: hybrid_hybrid_hybrid_bandit_hybrid_hybrid_hybrid_m18_s6.py (gen4)
# born: 2026-05-29T23:43:18Z

"""
This module fuses the bandit router from bandit_router.py and the hybrid bandit with RBF surrogate from hybrid_hybrid_hybrid_bandit_hybrid_hybrid_hybrid_m18_s6.py.
The mathematical bridge between the two structures is the use of a radial basis function (RBF) surrogate to model the expected rewards of the bandit actions.
The RBF surrogate is used to augment the empirical reward estimates from the bandit router, allowing for more accurate predictions and better decision-making.
"""

import math
import random
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np

# Shared Types
Vector = list[float]

# Bandit core
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

_POLICY: dict[str, list[float]] = {}  # action_id → [total_reward, count]
_STORE: dict[str, float] = {}  # placeholder VRAM store (unused)
_SURROGATE = None  # will hold an RBFSurrogate instance

def reset_policy() -> None:
    """Clear all learned statistics and the surrogate model."""
    _POLICY.clear()
    _STORE.clear()
    global _SURROGATE
    _SURROGATE = RBFSurrogate(centers=[], weights=[], epsilon=1.0)

def _empirical_reward(a: str) -> float:
    total, n = _POLICY.get(a, [0.0, 0.0])
    return total / n if n else 0.0

# RBF surrogate
def gaussian(r: float, epsilon: float = 1.0) -> float:
    """Gaussian kernel exp(‑(ε·r)²)."""
    return math.exp(-((epsilon * r) ** 2))

def euclidean(a: Vector, b: Vector) -> float:
    """Euclidean distance between two equal‑length vectors."""
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

class RBFSurrogate:
    def __init__(self, centers: list[Vector], weights: list[float], epsilon: float):
        self.centers = centers
        self.weights = weights
        self.epsilon = epsilon

    def evaluate(self, x: Vector) -> float:
        return sum(w * gaussian(euclidean(x, c), self.epsilon) for w, c in zip(self.weights, self.centers))

def solve_linear(a: list[list[float]], b: list[float]) -> list[float]:
    """Solve Ax = b with Gauss‑Jordan elimination (no external libs)."""
    n = len(b)
    m = [row[:] + [rhs] for row, rhs in zip(a, b)]
    for col in range(n):
        # pivot selection
        pivot_row = col
        for row in range(col + 1, n):
            if abs(m[row][col]) > abs(m[pivot_row][col]):
                pivot_row = row
        # swap rows
        m[col], m[pivot_row] = m[pivot_row], m[col]
        # eliminate pivot column
        for row in range(n):
            if row != col:
                factor = m[row][col] / m[col][col]
                for col_idx in range(n + 1):
                    m[row][col_idx] -= factor * m[col][col_idx]
    # back substitution
    x = [0.0] * n
    for col in range(n - 1, -1, -1):
        x[col] = m[col][n] / m[col][col]
    return x

def update_policy(updates: list[BanditUpdate]) -> None:
    """Update the policy with the given updates."""
    for u in updates:
        s = _POLICY.setdefault(u.action_id, [0.0, 0.0])
        s[0] += float(u.reward)
        s[1] += 1.0

def select_action(context: dict[str, float], actions: list[str], algorithm: str = 'linucb', epsilon: float = 0.1, seed: int | str | None = 7) -> BanditAction:
    """Select an action using the given algorithm."""
    if not actions:
        raise ValueError('actions required')
    rng = random.Random(seed)
    if algorithm == 'epsilon_greedy' and rng.random() < epsilon:
        chosen = rng.choice(actions)
    elif algorithm == 'thompson':
        chosen = max(actions, key=lambda a: rng.betavariate(1 + max(0, _reward(a)), 1 + max(0, 1 - _reward(a))))
    else:
        scale = math.sqrt(sum(float(v) * float(v) for v in context.values())) if context else 1.0
        chosen = max(actions, key=lambda a: _reward(a) + 0.1 * scale / math.sqrt(1 + _POLICY.get(a, [0, 0])[1]))
    return BanditAction(chosen, 1.0 / len(actions), _reward(chosen), 1.0 / math.sqrt(1 + _POLICY.get(chosen, [0, 0])[1]), algorithm)

def _reward(a: str) -> float:
    """Get the reward for the given action."""
    total, n = _POLICY.get(a, [0.0, 0.0])
    return total / n if n else 0.0

def hybrid_select_action(context: dict[str, float], actions: list[str], algorithm: str = 'linucb', epsilon: float = 0.1, seed: int | str | None = 7) -> BanditAction:
    """Select an action using the hybrid algorithm."""
    if not actions:
        raise ValueError('actions required')
    rng = random.Random(seed)
    if algorithm == 'epsilon_greedy' and rng.random() < epsilon:
        chosen = rng.choice(actions)
    elif algorithm == 'thompson':
        chosen = max(actions, key=lambda a: rng.betavariate(1 + max(0, _reward(a)), 1 + max(0, 1 - _reward(a))))
    else:
        scale = math.sqrt(sum(float(v) * float(v) for v in context.values())) if context else 1.0
        chosen = max(actions, key=lambda a: _reward(a) + 0.1 * scale / math.sqrt(1 + _POLICY.get(a, [0, 0])[1]) + _SURROGATE.evaluate([float(_reward(a))]))
    return BanditAction(chosen, 1.0 / len(actions), _reward(chosen), 1.0 / math.sqrt(1 + _POLICY.get(chosen, [0, 0])[1]), algorithm)

if __name__ == "__main__":
    reset_policy()
    update_policy([BanditUpdate("context1", "action1", 1.0, 1.0)])
    update_policy([BanditUpdate("context1", "action2", 0.0, 1.0)])
    select_action({"feature1": 1.0}, ["action1", "action2"])
    hybrid_select_action({"feature1": 1.0}, ["action1", "action2"])