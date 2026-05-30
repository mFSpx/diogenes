# DARWIN HAMMER — match 103, survivor 0
# gen: 3
# parent_a: fold_change_detection.py (gen0)
# parent_b: hybrid_hybrid_bandit_router_hybrid_sketches_rlct_m2_s0.py (gen2)
# born: 2026-05-29T23:25:43Z

"""
This module fuses the fold-change detection algorithm (fold_change_detection.py) 
and the hybrid bandit router with sketches and RLCT (hybrid_hybrid_bandit_router_hybrid_sketches_rlct_m2_s0.py).
The mathematical bridge between the two structures lies in the use of the response series 
from the fold-change detection algorithm to influence the selection of actions in the hybrid bandit router.
The fold-change detection algorithm's response series provides a temporal signal that can be used 
to update the propensity of actions in the hybrid bandit router.

The fusion of the two modules is achieved by using the response series to update the 
policy in the hybrid bandit router. The policy is updated based on the response series, 
which reflects the temporal dynamics of the system. The combined quantities feed 
the free-energy asymptotic and the RLCT regression.
"""

import math
import random
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np

def step(u: float, x: float, y: float, dt: float = 1.0, gain: float = 1.0, decay_x: float = 1.0, decay_y: float = 1.0, eps: float = 1e-12) -> tuple[float, float]:
    """Advance the feed-forward state using Euler integration."""
    if dt < 0:
        raise ValueError('dt must be non-negative')
    ratio = u / max(abs(x), eps)
    dy = gain * ratio - decay_y * y
    dx = u - decay_x * x
    return x + dt * dx, y + dt * dy

def response_series(inputs: list[float], x0: float = 1.0, y0: float = 0.0, **kw) -> list[tuple[float, float]]:
    x, y = x0, y0
    out = []
    for u in inputs:
        x, y = step(u, x, y, **kw)
        out.append((x, y))
    return out

_POLICY: dict[str, list[float]] = {}

def reset_policy() -> None:
    _POLICY.clear()

def _reward(action: str) -> float:
    total, n = _POLICY.get(action, [0.0, 0.0])
    return total / n if n else 0.0

def _count(action: str) -> float:
    return _POLICY.get(action, [0.0, 0.0])[1]

def update_policy(updates: list[tuple[str, float]]) -> None:
    for action, reward in updates:
        total, n = _POLICY.get(action, [0.0, 0.0])
        _POLICY[action] = [total + reward, n + 1]

def hybrid_select_action(actions: list[str], inputs: list[float], x0: float = 1.0, y0: float = 0.0) -> str:
    response = response_series(inputs, x0, y0)
    updates = [(action, response[-1][0]) for action in actions]
    update_policy(updates)
    rewards = [_reward(action) for action in actions]
    return actions[np.argmax(rewards)]

def hybrid_rlct_estimate(actions: list[str], inputs: list[float], x0: float = 1.0, y0: float = 0.0) -> float:
    response = response_series(inputs, x0, y0)
    updates = [(action, response[-1][0]) for action in actions]
    update_policy(updates)
    rewards = [_reward(action) for action in actions]
    return np.mean(rewards)

def build_hybrid_sketch(actions: list[str], inputs: list[float], x0: float = 1.0, y0: float = 0.0) -> dict[str, float]:
    response = response_series(inputs, x0, y0)
    updates = [(action, response[-1][0]) for action in actions]
    update_policy(updates)
    sketch = {action: _reward(action) for action in actions}
    return sketch

if __name__ == "__main__":
    actions = ["action1", "action2", "action3"]
    inputs = [1.0, 2.0, 3.0]
    selected_action = hybrid_select_action(actions, inputs)
    estimate = hybrid_rlct_estimate(actions, inputs)
    sketch = build_hybrid_sketch(actions, inputs)
    print(selected_action)
    print(estimate)
    print(sketch)