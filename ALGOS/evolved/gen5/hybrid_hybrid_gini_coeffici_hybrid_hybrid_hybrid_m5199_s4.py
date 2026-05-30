# DARWIN HAMMER — match 5199, survivor 4
# gen: 5
# parent_a: hybrid_gini_coefficient_hybrid_hybrid_endpoi_m1364_s3.py (gen3)
# parent_b: hybrid_hybrid_hybrid_regret_hybrid_hybrid_hybrid_m142_s2.py (gen4)
# born: 2026-05-30T00:00:42Z

"""
Module hybrid_gini_pheromone_regret: combines the Gini coefficient and endpoint circuit-breaker logic 
from hybrid_gini_coefficient_hybrid_hybrid_endpoi_m1364_s3 with the regret-weighted action selection 
and pheromone-modulated store dynamics from hybrid_hybrid_hybrid_regret_hybrid_hybrid_hybrid_m142_s2.

The mathematical bridge between the two parents is established by using the Gini coefficient to 
weight the regret accumulated for each action in the regret-weighted action selection, and by 
integrating the endpoint circuit-breaker logic into the store dynamics to modulate the pheromone 
levels based on the circuit-breaker's failure rate.
"""

import math
import random
import sys
import pathlib
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Tuple
import numpy as np

@dataclass(frozen=True)
class HybridAction:
    """Result of an action selection."""
    id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0


@dataclass(frozen=True)
class HybridUpdate:
    """Single observation used to update the policy."""
    context_id: str
    action_id: str
    reward: float
    propensity: float


@dataclass
class StoreState:
    """Honeybee-style store with a bounded control signal (dance)."""
    level: float = 0.0
    alpha: float = 1.0
    beta: float = 1.0
    dt: float = 1.0
    base: float = 1.0
    gain: float = 1.0
    limit: float = 10.0
    _last_delta: float = 0.0  # internal cache for dance

    def update(self, inflow: List[float], outflow: List[float]) -> Tuple[float, float]:
        delta = self.alpha * sum(inflow) - self.beta * sum(outflow)
        self.level = max(0.0, self.level + self.dt * delta)
        self._last_delta = delta
        return self.level, delta

    @property
    def dance(self) -> float:
        """Bounded control signal derived from the last Δ."""
        return max(0.0, min(1.0, self._last_delta / self.limit))


class EndpointCircuitBreaker:
    """Simple failure counter that opens after a configurable threshold."""

    def __init__(self, failure_threshold: int = 3):
        self.failure_threshold = failure_threshold
        self.failures = 0
        self.open = False
        self.last_event_at = ""

    def record_success(self) -> None:
        self.failures = 0
        self.open = False
        self.last_event_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    def record_failure(self) -> None:
        self.failures += 1
        self.open = self.failures >= self.failure_threshold
        self.last_event_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    def allow(self) -> bool:
        """True if the circuit is closed (i.e. endpoint is usable)."""
        return not self.open

    @property
    def failure_rate(self) -> float:
        """Failures normalised by the threshold (clamped to [0,1])."""
        return min(self.failures / self.failure_threshold, 1.0)


def gini_coefficient(values: Iterable[float]) -> float:
    """Return the Gini coefficient of a non-empty iterable of non-negative numbers."""
    xs = sorted(float(x) for x in values)
    if not xs or sum(xs) == 0.0:
        return 0.0
    if xs[0] < 0:
        raise ValueError("values must be non-negative")
    n = len(xs)
    return sum((2 * i - n - 1) * x for i, x in enumerate(xs, 1)) / (n * sum(xs))


def hybrid_action_selection(actions: List[HybridAction], store_state: StoreState, circuit_breaker: EndpointCircuitBreaker) -> HybridAction:
    """Select an action based on the hybrid score, which combines the regret-weighted action selection 
    and the pheromone-modulated store dynamics, weighted by the Gini coefficient."""
    hybrid_scores = []
    for action in actions:
        regret = action.expected_reward - action.confidence_bound
        pheromone = store_state.level * (1 + circuit_breaker.failure_rate)
        hybrid_score = (action.expected_value + store_state.dance * pheromone) * (1 - gini_coefficient([action.propensity]))
        hybrid_scores.append((action, hybrid_score))
    return max(hybrid_scores, key=lambda x: x[1])[0]


def update_store_state(store_state: StoreState, updates: List[HybridUpdate]) -> StoreState:
    """Update the store state based on the aggregated weighted regrets."""
    inflow = []
    outflow = []
    for update in updates:
        regret = update.reward - update.propensity
        inflow.append(regret)
        outflow.append(regret * update.propensity)
    level, _ = store_state.update(inflow, outflow)
    return StoreState(level=level, alpha=store_state.alpha, beta=store_state.beta, dt=store_state.dt, base=store_state.base, gain=store_state.gain, limit=store_state.limit)


def update_circuit_breaker(circuit_breaker: EndpointCircuitBreaker, updates: List[HybridUpdate]) -> EndpointCircuitBreaker:
    """Update the circuit breaker based on the failure rate."""
    for update in updates:
        if update.reward < 0:
            circuit_breaker.record_failure()
        else:
            circuit_breaker.record_success()
    return circuit_breaker


if __name__ == "__main__":
    # Test the hybrid algorithm
    actions = [HybridAction("action1", 0.5, 0.2, 0.1, "algorithm1", 0.3), HybridAction("action2", 0.3, 0.1, 0.2, "algorithm2", 0.4)]
    store_state = StoreState()
    circuit_breaker = EndpointCircuitBreaker()
    selected_action = hybrid_action_selection(actions, store_state, circuit_breaker)
    updates = [HybridUpdate("context1", "action1", 0.2, 0.5), HybridUpdate("context2", "action2", 0.1, 0.3)]
    updated_store_state = update_store_state(store_state, updates)
    updated_circuit_breaker = update_circuit_breaker(circuit_breaker, updates)
    print(selected_action)
    print(updated_store_state.level)
    print(updated_circuit_breaker.failure_rate)