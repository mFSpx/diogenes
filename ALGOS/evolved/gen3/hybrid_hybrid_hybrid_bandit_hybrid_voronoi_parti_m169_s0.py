# DARWIN HAMMER — match 169, survivor 0
# gen: 3
# parent_a: hybrid_hybrid_bandit_router_hybrid_model_vram_sc_m35_s3.py (gen2)
# parent_b: hybrid_voronoi_partition_hybrid_endpoint_circ_m47_s3.py (gen2)
# born: 2026-05-29T23:25:52Z

"""
Module for the hybrid algorithm that combines the bandit router from 
hybrid_bandit_router_honeybee_store_m9_s1.py and the Voronoi partition 
and circuit-breaker functionality from hybrid_voronoi_partition_hybrid_endpoint_circ_m47_s3.py.
The mathematical bridge between these two structures is the use of 
Euclidean distance in the Voronoi partition, which can be applied to the 
bandit router's action selection process. This allows the bandit router 
to consider the geometric relationships between actions and contexts.
"""

import math
import random
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple, Dict, Any
import numpy as np

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


_POLICY: Dict[str, List[float]] = {}
_STORE: Dict[str, float] = {}          # virtual VRAM store per key
DEFAULT_BUDGET_MB = 1024 * 4  # Assuming 4GB as default budget


def reset_policy() -> None:
    """Clear all learned statistics and the virtual store."""
    _POLICY.clear()
    _STORE.clear()


def _reward(a: str) -> float:
    total, n = _POLICY.get(a, [0.0, 0.0])
    return total / n if n else 0.0


# ----------------------------------------------------------------------
# Voronoi helpers
# ----------------------------------------------------------------------
Point = Tuple[float, float]

def euclidean_distance(a: Point, b: Point) -> float:
    """Standard Euclidean distance in ℝ²."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


# ----------------------------------------------------------------------
# Circuit-breaker and Morphology
# ----------------------------------------------------------------------
class EndpointCircuitBreaker:
    """Failure counter that opens after a configurable threshold."""

    def __init__(self, failure_threshold: int = 3):
        if failure_threshold <= 0:
            raise ValueError("failure_threshold must be positive")
        self.failure_threshold = failure_threshold
        self.failures = 0
        self.open = False
        self.last_event_at = ""

    def record_success(self) -> None:
        self.failures = 0
        self.open = False
        self.last_event_at = ""

    def record_failure(self) -> None:
        self.failures += 1
        self.open = self.failures >= self.failure_threshold
        self.last_event_at = ""

    def allow(self) -> bool:
        """True if the circuit is closed (i.e. endpoint is usable)."""
        return not self.open

    def as_dict(self) -> dict[str, Any]:
        return {
            "failure_threshold": self.failure_threshold,
            "failures": self.failures,
            "open": self.open,
            "last_event_at": self.last_event_at,
        }


# ----------------------------------------------------------------------
# Hybrid functions
# ----------------------------------------------------------------------
def select_action_with_voronoi(
    context: Dict[str, float],
    actions: List[str],
    algorithm: str = "linucb",
    epsilon: float = 0.1,
    seed: int | str | None = 7,
    circuit_breaker: EndpointCircuitBreaker = None,
) -> BanditAction:
    """Choose an action and return its BanditAction descriptor, 
    using Voronoi partition to consider geometric relationships."""
    if not actions:
        raise ValueError("actions required")
    rng = random.Random(seed)

    if algorithm == "epsilon_greedy" and rng.random() < epsilon:
        chosen = rng.choice(actions)
    elif algorithm == "thompson":
        # Beta-Bernoulli posterior with pseudo-counts derived from rewards
        chosen = max(
            actions,
            key=lambda a: rng.betavariate(
                1 + max(0, _reward(a)),
                1 + max(0, 1 - _reward(a)),
            ),
        )
    else:  # linucb-style surrogate
        scale = math.sqrt(
            sum(float(v) * float(v) for v in context.values())
        ) if context else 1.0
        chosen = max(
            actions,
            key=lambda a: _reward(a)
            + 0.1 * scale / math.sqrt(1 + _POLICY.get(a, [0, 0])[1]),
        )

    if circuit_breaker and not circuit_breaker.allow():
        # If circuit is open, choose a different action with Voronoi partition
        points = [(random.random(), random.random()) for _ in range(len(actions))]
        chosen_point = points[actions.index(chosen)]
        chosen = min(
            actions,
            key=lambda a: euclidean_distance(points[actions.index(a)], chosen_point),
        )

    propensity = 1.0 / len(actions)
    confidence = 1.0 / math.sqrt(1 + _POLICY.get(chosen, [0, 0])[1])
    return BanditAction(
        action_id=chosen,
        propensity=propensity,
        expected_reward=_reward(chosen),
        confidence_bound=confidence,
        algorithm=algorithm,
    )


def update_policy(action: BanditAction, reward: float) -> None:
    """Update the policy with the given action and reward."""
    _POLICY[action.action_id] = [_POLICY.get(action.action_id, [0, 0])[0] + reward, _POLICY.get(action.action_id, [0, 0])[1] + 1]


def get_circuit_breaker_status(circuit_breaker: EndpointCircuitBreaker) -> dict[str, Any]:
    """Get the status of the circuit breaker."""
    return circuit_breaker.as_dict()


if __name__ == "__main__":
    # Smoke test
    reset_policy()
    circuit_breaker = EndpointCircuitBreaker()
    action = select_action_with_voronoi({}, ["action1", "action2"], circuit_breaker=circuit_breaker)
    update_policy(action, 1.0)
    print(get_circuit_breaker_status(circuit_breaker))