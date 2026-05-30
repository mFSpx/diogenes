# DARWIN HAMMER — match 103, survivor 3
# gen: 3
# parent_a: fold_change_detection.py (gen0)
# parent_b: hybrid_hybrid_bandit_router_hybrid_sketches_rlct_m2_s0.py (gen2)
# born: 2026-05-29T23:25:43Z

import math
import random
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

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

def reset_policy() -> None:
    """Reset the bandit policy."""
    global _POLICY
    _POLICY.clear()

def _reward(action: str) -> float:
    """Compute the reward for an action based on the bandit policy."""
    total, n = _POLICY.get(action, [0.0, 0.0])
    return total / n if n else 0.0

def _count(action: str) -> float:
    """Count the number of times an action has been selected."""
    return _POLICY.get(action, [0.0, 0.0])[1]

def _hybrid_store_factor(action_id: str, count: float, log_count_ratio: float) -> float:
    """Compute the hybrid store factor using the log-count ratio."""
    if count == 0:
        return 1.0
    return max(1.0, math.exp(log_count_ratio * count))

def step(u: float, x: float, y: float, dt: float = 1.0, gain: float = 1.0, decay_x: float = 1.0, decay_y: float = 1.0, eps: float = 1e-12) -> tuple[float, float]:
    """Advance the feed-forward state using Euler integration."""
    if dt < 0:
        raise ValueError('dt must be non-negative')
    ratio = u / max(abs(x), eps)
    dy = gain * ratio - decay_y * y
    dx = u - decay_x * x
    return x + dt * dx, y + dt * dy

def response_series(inputs: List[float], x0: float = 1.0, y0: float = 0.0, **kw) -> List[Tuple[float, float]]:
    """Generate a series of responses to the input stimuli."""
    x, y = x0, y0
    out = []
    for u in inputs:
        x, y = step(u, x, y, **kw)
        out.append((x, y))
    return out

def update_policy(updates: List[BanditUpdate]) -> None:
    """Update the bandit policy based on the new reward and propensity."""
    for update in updates:
        action_id = update.action_id
        reward = update.reward
        propensity = update.propensity
        if action_id not in _POLICY:
            _POLICY[action_id] = [0.0, 0.0]
        _POLICY[action_id] = [_POLICY[action_id][0] + propensity * reward, _POLICY[action_id][1] + 1.0]

def hybrid_select_action(actions: List[BanditAction], context_id: str) -> str:
    """Select an action based on the hybrid bandit router with the influence of the store factor and the log-count ratio."""
    count = _count(context_id)
    if count == 0:
        log_count_ratio = 0.0
    else:
        log_count_ratio = math.log(count) / math.log(len(_POLICY)) if _POLICY else 0.0
    store_factor = _hybrid_store_factor(context_id, count, log_count_ratio)
    best_action = max(actions, key=lambda action: action.propensity * store_factor)
    return best_action.action_id

def hybrid_rlct_estimate(loss_curve: List[float]) -> float:
    """Derive an RLCT estimate from the sketch-based loss curve and evaluate the asymptotic free energy."""
    if not loss_curve:
        return 0.0
    # Compute the sketch-based loss curve
    sketch_loss_curve = [math.exp(-x) for x in loss_curve]
    # Compute the RLCT estimate
    rlct_estimate = np.mean(sketch_loss_curve)
    return rlct_estimate

def fold_change_detection_series(inputs: List[float], x0: float = 1.0, y0: float = 0.0, **kw) -> List[Tuple[float, float]]:
    """Apply the fold-change detection to a series of inputs."""
    x, y = x0, y0
    out = []
    for u in inputs:
        x, y = step(u, x, y, **kw)
        out.append((x, y))
    return out

if __name__ == "__main__":
    # Smoke test
    reset_policy()
    update_policy([BanditUpdate("context_id1", "action_id1", 1.0, 0.5)])
    print(hybrid_select_action([BanditAction("action_id1", 0.5, 1.0, 0.1, "algorithm1")], "context_id1"))
    print(hybrid_rlct_estimate([1.0, 2.0, 3.0]))
    print(fold_change_detection_series([1.0, 2.0, 3.0]))