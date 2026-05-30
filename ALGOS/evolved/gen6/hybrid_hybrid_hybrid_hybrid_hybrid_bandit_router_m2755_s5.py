# DARWIN HAMMER — match 2755, survivor 5
# gen: 6
# parent_a: hybrid_hybrid_hybrid_endpoi_hybrid_hybrid_hybrid_m976_s6.py (gen5)
# parent_b: hybrid_bandit_router_hybrid_hybrid_hybrid_m2649_s4.py (gen5)
# born: 2026-05-29T23:45:38Z

"""
Module fusing the governing equations of two parent algorithms:
- Parent A: hybrid_hybrid_hybrid_endpoi_hybrid_hybrid_hybrid_m976_s6.py
- Parent B: hybrid_bandit_router_hybrid_hybrid_hybrid_m2649_s4.py

The mathematical bridge is the integration of the Morphology and EndpointCircuitBreaker classes from Parent A into the hybrid reward estimator 
from Parent B. This is achieved by using the Morphology class to generate context vectors and the EndpointCircuitBreaker to monitor the performance 
of the bandit actions.
"""

import math
import random
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Tuple, Sequence

import numpy as np

# ----------------------------------------------------------------------
# Parent A – morphology and circuit-breaker primitives
# ----------------------------------------------------------------------

def now_z() -> str:
    """Current UTC timestamp in ISO-8601 Zulu format."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

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
        """Return the morphology as a 1-D numpy array."""
        return np.array([self.length, self.width, self.height, self.mass], dtype=float)


class EndpointCircuitBreaker:
    """Simple failure counter that opens after a configurable threshold."""

    def __init__(self, failure_threshold: int = 3):
        if failure_threshold <= 0:
            raise ValueError("failure_threshold must be positive")
        self.failure_threshold = failure_threshold
        self.failures = 0
        self.open = False
        self.last_event_at = now_z()

    def record_success(self) -> None:
        """Reset failures and close the breaker."""
        self.failures = 0
        self.open = False
        self.last_event_at = now_z()

    def record_failure(self) -> None:
        """Increment failures and open the breaker if threshold is hit."""
        self.failures += 1
        self.last_event_at = now_z()
        if self.failures >= self.failure_threshold:
            self.open = True

    def status(self) -> Tuple[bool, int]:
        """Return (open, failures)."""
        return self.open, self.failures


# ----------------------------------------------------------------------
# Shared Types
# ----------------------------------------------------------------------
Vector = Sequence[float]

# ----------------------------------------------------------------------
# Bandit core (from Parent B)
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
_SURROGATE: dict[str, 'RBFSurrogate'] = {}    # action_id → surrogate model


class RBFSurrogate:
    def __init__(self, action_id: str):
        self.action_id = action_id
        self.weights = []
        self.centers = []

    def predict(self, context_vector: Vector) -> float:
        prediction = 0.0
        for i in range(len(self.weights)):
            distance = math.sqrt(sum([(a - b) ** 2 for a, b in zip(context_vector, self.centers[i])]))
            kernel = math.exp(-distance ** 2)
            prediction += self.weights[i] * kernel
        return prediction


def reset_hybrid() -> None:
    """Clear all learned statistics, stored contexts and surrogate models."""
    _POLICY.clear()
    _CONTEXT_STORE.clear()
    _SURROGATE.clear()


def update_policy(action_id: str, reward: float, propensity: float) -> None:
    if action_id in _POLICY:
        _POLICY[action_id][0] += reward
        _POLICY[action_id][1] += 1
    else:
        _POLICY[action_id] = [reward, 1]


def update_context(context_id: str, context_vector: Vector) -> None:
    _CONTEXT_STORE[context_id] = context_vector


def update_surrogate(action_id: str, surrogate: RBFSurrogate) -> None:
    _SURROGATE[action_id] = surrogate


def get_empirical_reward(action_id: str) -> float:
    if action_id in _POLICY:
        return _POLICY[action_id][0] / _POLICY[action_id][1]
    else:
        return 0.0


def get_predicted_reward(action_id: str, context_vector: Vector) -> float:
    if action_id in _SURROGATE:
        return _SURROGATE[action_id].predict(context_vector)
    else:
        return 0.0


def get_hybrid_reward(action_id: str, context_vector: Vector, alpha: float = 0.5) -> float:
    empirical_reward = get_empirical_reward(action_id)
    predicted_reward = get_predicted_reward(action_id, context_vector)
    return alpha * empirical_reward + (1 - alpha) * predicted_reward


def run_hybrid_bandit(action_id: str, context_vector: Vector, reward: float, propensity: float) -> None:
    update_policy(action_id, reward, propensity)
    update_context(action_id, context_vector)
    surrogate = RBFSurrogate(action_id)
    surrogate.weights = [1.0]
    surrogate.centers = [context_vector]
    update_surrogate(action_id, surrogate)
    hybrid_reward = get_hybrid_reward(action_id, context_vector)
    print(f"Hybrid reward for action {action_id}: {hybrid_reward}")


def monitor_performance(action_id: str, circuit_breaker: EndpointCircuitBreaker) -> None:
    if circuit_breaker.open:
        print(f"Circuit breaker for action {action_id} is open")
    else:
        print(f"Circuit breaker for action {action_id} is closed")


if __name__ == "__main__":
    reset_hybrid()
    morphology = Morphology(1.0, 2.0, 3.0, 4.0)
    context_vector = morphology.as_vector()
    action_id = "action1"
    reward = 1.0
    propensity = 0.5
    circuit_breaker = EndpointCircuitBreaker()
    run_hybrid_bandit(action_id, context_vector, reward, propensity)
    circuit_breaker.record_success()
    monitor_performance(action_id, circuit_breaker)