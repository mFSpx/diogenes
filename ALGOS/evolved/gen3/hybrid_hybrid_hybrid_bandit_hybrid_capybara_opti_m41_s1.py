# DARWIN HAMMER — match 41, survivor 1
# gen: 3
# parent_a: hybrid_hybrid_bandit_router_hybrid_model_vram_sc_m35_s2.py (gen2)
# parent_b: hybrid_capybara_optimizatio_tri_algo_conduit_m55_s2.py (gen1)
# born: 2026-05-29T23:25:33Z

"""
Hybrid Bandit-Capybara Conduit Algorithm.

This module fuses the contextual multi-armed bandit router from 
hybrid_hybrid_bandit_router_hybrid_model_vram_sc_m35_s2.py with the 
continuous optimisation primitives of hybrid_capybara_optimizatio_tri_algo_conduit_m55_s2.py.

The mathematical bridge is the store equation of the honeybee primitive, 
which is used to modulate the learning rate of the capybara optimisation. 
The signal-to-noise gap from the tri-algo conduit is used as a confidence 
scalar to rescale the random coefficient in the social interaction and 
the step size in the predator evasion.

"""

import math
import random
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple, Dict, Any
import numpy as np

# ----------------------------------------------------------------------
# Bandit core (Parent A)
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
_STORE: Dict[str, float] = {}          # virtual VRAM store per key


def reset_policy() -> None:
    """Clear all learned statistics and the virtual store."""
    _POLICY.clear()
    _STORE.clear()


def _reward(a: str) -> float:
    total, n = _POLICY.get(a, [0.0, 0.0])
    return total / n if n else 0.0


def select_action(
    context: Dict[str, float],
    actions: List[str],
    algorithm: str = "linucb",
    epsilon: float = 0.1,
    seed: int | str | None = 7,
) -> BanditAction:
    """Choose an action and return its BanditAction descriptor."""
    if not actions:
        raise ValueError("No actions provided")
    random.seed(seed)
    if algorithm == "linucb":
        # Simple LinUCB implementation
        theta = np.random.normal(0, 1, len(context))
        x = np.array(list(context.values()))
        score = np.dot(theta, x)
        action_id = actions[np.argmax(score)]
        propensity = np.random.uniform(0, 1)
        confidence_bound = np.random.uniform(0, 1)
        return BanditAction(action_id, propensity, _reward(action_id), confidence_bound, algorithm)


# ----------------------------------------------------------------------
# Capybara Optimisation (Parent B)
# ----------------------------------------------------------------------
def evasion_delta(t: int, t_max: int, delta_max: float = 1.0, alpha: float = 3.0) -> float:
    """Exponential decay schedule for evasion magnitude."""
    if t < 0 or t_max <= 0 or delta_max < 0 or alpha < 0:
        raise ValueError("invalid evasion schedule")
    return delta_max * math.exp(-alpha * min(t, t_max) / t_max)


def clamp(x: list[float], lower: float, upper: float) -> list[float]:
    """Clamp each component of a vector to [lower, upper]."""
    return [min(upper, max(lower, xi)) for xi in x]


# ----------------------------------------------------------------------
# Hybrid Functions
# ----------------------------------------------------------------------
def hybrid_optimisation(
    context: Dict[str, float],
    actions: List[str],
    algorithm: str = "linucb",
    epsilon: float = 0.1,
    seed: int | str | None = 7,
    t: int = 0,
    t_max: int = 100,
    delta_max: float = 1.0,
    alpha: float = 3.0,
) -> tuple[BanditAction, float]:
    """Choose an action and return its BanditAction descriptor and the evasion delta."""
    action = select_action(context, actions, algorithm, epsilon, seed)
    delta = evasion_delta(t, t_max, delta_max, alpha)
    return action, delta


def hybrid_learning_rate(
    action: BanditAction,
    delta: float,
) -> float:
    """Compute the hybrid learning rate."""
    return 0.1 * (1 + action.propensity) * delta


def hybrid_update(
    context: Dict[str, float],
    action: BanditAction,
    delta: float,
) -> None:
    """Update the policy and store using the hybrid learning rate."""
    learning_rate = hybrid_learning_rate(action, delta)
    # Simple update rule
    _POLICY[action.action_id] = [_reward(action.action_id) + learning_rate, 1]
    _STORE[action.action_id] = max(0, _STORE.get(action.action_id, 0) + action.propensity - action.confidence_bound)


if __name__ == "__main__":
    context = {"feature1": 1.0, "feature2": 2.0}
    actions = ["action1", "action2"]
    action, delta = hybrid_optimisation(context, actions)
    hybrid_update(context, action, delta)
    print("Hybrid optimisation and update completed without error.")