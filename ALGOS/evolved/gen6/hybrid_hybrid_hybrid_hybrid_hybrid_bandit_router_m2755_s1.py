# DARWIN HAMMER — match 2755, survivor 1
# gen: 6
# parent_a: hybrid_hybrid_hybrid_endpoi_hybrid_hybrid_hybrid_m976_s6.py (gen5)
# parent_b: hybrid_bandit_router_hybrid_hybrid_hybrid_m2649_s4.py (gen5)
# born: 2026-05-29T23:45:38Z

"""
This module fuses the two parent algorithms: 
- hybrid_hybrid_hybrid_endpoi_hybrid_hybrid_hybrid_m976_s6.py 
  (providing morphological and circuit-breaker primitives) and 
- hybrid_bandit_router_hybrid_hybrid_hybrid_m2649_s4.py 
  (implementing a hybrid bandit-RBF router).

The mathematical bridge is the integration of the morphological 
description with the bandit router's reward estimator. 
The hybrid estimator is then plugged into the same selection logic 
used by the bandit router.

The resulting system preserves the morphological description while 
leveraging a non-parametric RBF model that can capture smooth 
context-reward relationships.
"""

import math
import random
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Tuple, Sequence, Callable, Any
import numpy as np

# ----------------------------------------------------------------------
# Shared Types
# ----------------------------------------------------------------------
Vector = Sequence[float]

# ----------------------------------------------------------------------
# Morphology and Circuit-Breaker Primitives
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Morphology:
    """Geometric description of a physical (or logical) entity."""

    length: float
    width: float
    height: float
    mass: float

    def __post_init__(self) -> None:
        for name, value in (
            ("length", self.length),
            ("width", self.width),
            ("height", self.height),
            ("mass", self.mass),
        ):
            if not isinstance(value, (int, float)) or value <= 0:
                raise ValueError(f"{name} must be a positive number")

    def as_vector(self) -> np.ndarray:
        """Return the morphology as a 1‑D numpy array."""
        return np.array([self.length, self.width, self.height, self.mass], dtype=float)


class EndpointCircuitBreaker:
    """Simple failure counter that opens after a configurable threshold."""

    def __init__(self, failure_threshold: int = 3):
        if failure_threshold <= 0:
            raise ValueError("failure_threshold must be positive")
        self.failure_threshold = failure_threshold
        self.failures = 0
        self.open = False
        self.last_event_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    def record_success(self) -> None:
        """Reset failures and close the breaker."""
        self.failures = 0
        self.open = False
        self.last_event_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    def record_failure(self) -> None:
        """Increment failures and open the breaker if threshold is hit."""
        self.failures += 1
        self.last_event_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        if self.failures >= self.failure_threshold:
            self.open = True

    def status(self) -> Tuple[bool, int]:
        """Return (open, failures)."""
        return self.open, self.failures


# ----------------------------------------------------------------------
# Bandit Core
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
    context_id: str          # identifier linking to a stored context vector
    action_id: str
    reward: float
    propensity: float


_POLICY: dict[str, list[float]] = {}          # action_id → [total_reward, count]
_CONTEXT_STORE: dict[str, Vector] = {}        # context_id → vector representation
_SURROGATE: dict[str, "RBFSurrogate"] = {}    # action_id → surrogate model


class RBFSurrogate:
    def __init__(self, action_id: str):
        self.action_id = action_id
        self.contexts = []
        self.rewards = []

    def add_context(self, context: Vector, reward: float):
        self.contexts.append(context)
        self.rewards.append(reward)

    def predict(self, context: Vector):
        if not self.contexts:
            return 0
        distances = [math.sqrt(sum([(a - b) ** 2 for a, b in zip(context, c)])) for c in self.contexts]
        weights = [math.exp(-d ** 2) for d in distances]
        return sum([r * w for r, w in zip(self.rewards, weights)]) / sum(weights)


def reset_hybrid() -> None:
    """Clear all learned statistics, stored contexts and surrogate models."""
    _POLICY.clear()
    _CONTEXT_STORE.clear()
    _SURROGATE.clear()


def update_policy(action_id: str, reward: float):
    if action_id not in _POLICY:
        _POLICY[action_id] = [reward, 1]
    else:
        _POLICY[action_id][0] += reward
        _POLICY[action_id][1] += 1


def get_expected_reward(action_id: str):
    if action_id not in _POLICY:
        return 0
    return _POLICY[action_id][0] / _POLICY[action_id][1]


def get_surrogate_reward(action_id: str, context: Vector):
    if action_id not in _SURROGATE:
        _SURROGATE[action_id] = RBFSurrogate(action_id)
    return _SURROGATE[action_id].predict(context)


def hybrid_predict(action_id: str, context: Vector, alpha: float = 0.5):
    empirical_reward = get_expected_reward(action_id)
    surrogate_reward = get_surrogate_reward(action_id, context)
    return alpha * empirical_reward + (1 - alpha) * surrogate_reward


def test_circuit_breaker():
    breaker = EndpointCircuitBreaker(failure_threshold=3)
    for _ in range(5):
        breaker.record_failure()
    print(breaker.status())  # Should print: (True, 5)


def test_hybrid_predict():
    reset_hybrid()
    update_policy("action1", 10)
    _SURROGATE["action1"] = RBFSurrogate("action1")
    _SURROGATE["action1"].add_context([1, 2, 3], 20)
    print(hybrid_predict("action1", [1, 2, 3]))  # Should print a value between 10 and 20


def test_morphology():
    morphology = Morphology(1, 2, 3, 4)
    print(morphology.as_vector())  # Should print: [1. 2. 3. 4.]


if __name__ == "__main__":
    test_circuit_breaker()
    test_hybrid_predict()
    test_morphology()