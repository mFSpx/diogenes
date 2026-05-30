# DARWIN HAMMER — match 5445, survivor 5
# gen: 6
# parent_a: hybrid_gini_coefficient_hybrid_hybrid_endpoi_m1364_s0.py (gen3)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m950_s0.py (gen5)
# born: 2026-05-30T00:02:06Z

import math
import random
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Dict, Sequence, Any
import numpy as np
from pathlib import Path

Vector = Sequence[float]

# ----------------------------------------------------------------------
# Core data structures
# ----------------------------------------------------------------------
@dataclass
class BanditAction:
    """Immutable description of a possible action."""
    action_id: str
    propensity: float                     # raw probability estimate (0‑1)
    expected_reward: float                # prior mean reward
    confidence_bound: float               # exploration term
    algorithm: str                        # originating algorithm name


@dataclass
class BanditUpdate:
    context_id: str
    action_id: str
    reward: float
    propensity: float


@dataclass
class Endpoint:
    """Mutable endpoint that tracks health and circuit‑breaker state."""
    value: float
    failure_threshold: int = 3
    circuit_breaker_failures: int = 0
    health: float = 0.0
    workshare: float = 0.0
    _circuit_open: bool = field(default=False, init=False, repr=False)

    def record_success(self) -> None:
        self.circuit_breaker_failures = 0
        self._circuit_open = False

    def record_failure(self) -> None:
        self.circuit_breaker_failures += 1
        if self.circuit_breaker_failures >= self.failure_threshold:
            self._circuit_open = True

    def allow(self) -> bool:
        """Endpoint is usable only when the circuit is closed."""
        return not self._circuit_open


# ----------------------------------------------------------------------
# Global stores (simulating a lightweight learning layer)
# ----------------------------------------------------------------------
_POLICY: Dict[str, List[float]] = {}   # action_id -> [cumulative_reward, count]
_STORE: Dict[str, float] = {}         # placeholder for any future per‑key state


def reset_policy() -> None:
    """Clear all learned statistics and the virtual store."""
    _POLICY.clear()
    _STORE.clear()


# ----------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------
def _reward(action_id: str) -> float:
    total, n = _POLICY.get(action_id, [0.0, 0.0])
    return total / n if n else 0.0


def gaussian(r: float, epsilon: float = 1.0) -> float:
    """Radial basis function (Gaussian) with width parameter epsilon."""
    return math.exp(-((epsilon * r) ** 2))


def euclidean(a: Vector, b: Vector) -> float:
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


# ----------------------------------------------------------------------
# RBF surrogate model (stateful – can be updated incrementally)
# ----------------------------------------------------------------------
@dataclass
class RBFSurrogate:
    centers: List[Tuple[float, ...]]
    weights: List[float]
    epsilon: float = 1.0

    def predict(self, x: Vector) -> float:
        """Return a weighted sum of Gaussian kernels."""
        return sum(
            w * gaussian(euclidean(x, c), self.epsilon)
            for w, c in zip(self.weights, self.centers)
        )

    @classmethod
    def from_actions(cls, actions: List[BanditAction], epsilon: float = 1.0) -> "RBFSurrogate":
        """Factory that builds a surrogate using propensity and health as a 2‑D feature."""
        centers = [(a.propensity, a.expected_reward) for a in actions]
        # Initialise weights to the prior expected reward; this gives a sensible baseline.
        weights = [a.expected_reward for a in actions]
        return cls(centers, weights, epsilon)


# ----------------------------------------------------------------------
# Gini coefficient utilities (robust to zero‑sum inputs)
# ----------------------------------------------------------------------
def gini_coefficient(values: List[float]) -> float:
    """Compute the Gini coefficient; returns 0 when all values are equal or sum to 0."""
    if not values:
        return 0.0
    values = np.asarray(values, dtype=float)
    if np.allclose(values, 0):
        return 0.0
    sorted_vals = np.sort(values)                     # ascending
    n = len(sorted_vals)
    cumulative = np.cumsum(sorted_vals)
    sum_vals = cumulative[-1]
    if sum_vals == 0:
        return 0.0
    # Standard Gini formula
    gini = (n + 1 - 2 * np.sum(cumulative) / sum_vals) / n
    return float(gini)


def compute_endpoint_health(values: List[float], endpoint_value: float) -> float:
    """Health is scaled by (1‑Gini) to reward homogeneity across endpoints."""
    gini = gini_coefficient(values)
    health = (1.0 - gini) * endpoint_value
    # Clamp to a sensible range [0, endpoint_value] to avoid runaway values.
    return max(0.0, min(health, endpoint_value))


# ----------------------------------------------------------------------
# Core algorithm: action selection with deeper integration of health
# ----------------------------------------------------------------------
def select_action(
    actions: List[BanditAction],
    endpoint: Endpoint,
    surrogate: RBFSurrogate | None = None,
) -> BanditAction:
    """
    Choose an action for a given endpoint.

    The selection uses a hybrid of:
      * Upper‑confidence bound (UCB) style scoring,
      * RBF surrogate predictions that incorporate both propensity and health,
      * The endpoint's health as an explicit multiplier to encourage
        healthier endpoints to receive higher‑reward actions.
    """
    if not actions:
        raise ValueError("action list cannot be empty")

    # Build or reuse the surrogate; health is added as a second dimension.
    if surrogate is None:
        surrogate = RBFSurrogate.from_actions(actions, epsilon=1.0)

    # Compute a UCB‑like score for each action.
    scores: List[float] = []
    for action in actions:
        # Surrogate prediction based on (propensity, health) pair.
        pred = surrogate.predict([action.propensity, endpoint.health])

        # Classic UCB term: mean reward + confidence * sqrt(log(t)/n)
        mean_reward = _reward(action.action_id)
        count = _POLICY.get(action.action_id, [0.0, 0.0])[1]
        exploration = (
            action.confidence_bound
            * math.sqrt(math.log(max(1, sum(p[1] for p in _POLICY.values()))) / (count + 1))
        )
        # Blend surrogate prediction with UCB; health acts as a scaling factor.
        score = (pred + mean_reward) * (1.0 + endpoint.health) + exploration
        scores.append(score)

    # Choose the action with the highest composite score.
    best_idx = int(np.argmax(scores))
    return actions[best_idx]


# ----------------------------------------------------------------------
# Policy update routine (incremental learning)
# ----------------------------------------------------------------------
def update_policy(action: BanditAction, reward: float) -> None:
    """Incrementally update the running average reward for the given action."""
    if action.action_id in _POLICY:
        _POLICY[action.action_id][0] += reward
        _POLICY[action.action_id][1] += 1
    else:
        _POLICY[action.action_id] = [reward, 1]


# ----------------------------------------------------------------------
# Example driver (executed only when run as a script)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Seed for reproducibility in examples
    random.seed(42)
    np.random.seed(42)

    # Create example endpoints
    endpoints = [
        Endpoint(value=1.0),
        Endpoint(value=2.0),
        Endpoint(value=0.5),
    ]

    # Example actions (propensity in [0,1], expected_reward as prior)
    actions = [
        BanditAction("action1", propensity=0.3, expected_reward=5.0, confidence_bound=1.0, algorithm="algA"),
        BanditAction("action2", propensity=0.6, expected_reward=8.0, confidence_bound=1.0, algorithm="algB"),
        BanditAction("action3", propensity=0.9, expected_reward=3.0, confidence_bound=1.0, algorithm="algC"),
    ]

    # Compute health for each endpoint based on the distribution of endpoint values
    values = [ep.value for ep in endpoints]
    for ep in endpoints:
        ep.health = compute_endpoint_health(values, ep.value)

    # Global surrogate can be reused across endpoints to save work
    global_surrogate = RBFSurrogate.from_actions(actions, epsilon=0.8)

    # Iterate over endpoints, respecting circuit‑breaker state
    for ep in endpoints:
        if not ep.allow():
            print(f"Endpoint {ep.value:.2f} blocked by circuit breaker.")
            continue

        chosen = select_action(actions, ep, surrogate=global_surrogate)
        print(f"Endpoint {ep.value:.2f} (health={ep.health:.3f}) -> selected {chosen.action_id}")

        # Simulate a stochastic reward (for demo purposes)
        simulated_reward = random.gauss(chosen.expected_reward, 2.0)
        update_policy(chosen, simulated_reward)

        # Randomly decide if the endpoint fails to showcase circuit‑breaker logic
        if random.random() < 0.2:  # 20 % failure chance
            ep.record_failure()
            print(f"  Failure recorded; failures={ep.circuit_breaker_failures}")
        else:
            ep.record_success()
            print("  Success recorded; circuit closed.")