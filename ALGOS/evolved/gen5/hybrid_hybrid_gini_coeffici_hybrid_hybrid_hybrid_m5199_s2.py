# DARWIN HAMMER — match 5199, survivor 2
# gen: 5
# parent_a: hybrid_gini_coefficient_hybrid_hybrid_endpoi_m1364_s3.py (gen3)
# parent_b: hybrid_hybrid_hybrid_regret_hybrid_hybrid_hybrid_m142_s2.py (gen4)
# born: 2026-05-30T00:00:42Z

"""
Hybrid algorithm merging the concepts of Gini coefficient and endpoint circuit-breaking 
with regret-weighted action selection and pheromone-modulated store dynamics.

The mathematical bridge is formed by using the Gini coefficient to modulate the 
regret updates in the regret engine, and using the endpoint circuit-breaker to 
determine the availability of actions in the hybrid action selection. The store 
state is updated based on the inflow and outflow derived from the aggregated 
weighted regrets, and the dance signal is used to weight the regret updates.

Parent A: hybrid_gini_coefficient_hybrid_hybrid_endpoi_m1364_s3.py
Parent B: hybrid_hybrid_hybrid_regret_hybrid_hybrid_hybrid_m142_s2.py
"""

import math
import random
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
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

@dataclass
class EndpointCircuitBreaker:
    """Simple failure counter that opens after a configurable threshold."""
    failure_threshold: int = 3
    failures: int = 0
    open: bool = False
    last_event_at: str = ""

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
        return max(0.0, min(self._last_delta, self.limit))

def gini_coefficient(values: Iterable[float]) -> float:
    """Return the Gini coefficient of a non-empty iterable of non-negative numbers."""
    xs = sorted(float(x) for x in values)
    if not xs or sum(xs) == 0.0:
        return 0.0
    if xs[0] < 0:
        raise ValueError("values must be non-negative")
    n = len(xs)
    return sum((2 * i - n - 1) * x for i, x in enumerate(xs, 1)) / (n * sum(xs))

def regret_weighted_action_selection(actions: List[HybridAction], store_state: StoreState, 
                                     endpoint_circuit_breaker: EndpointCircuitBreaker) -> HybridAction:
    """Select an action based on regret-weighted action selection and pheromone-modulated store dynamics."""
    if not actions:
        raise ValueError("No actions available")
    
    # Use the Gini coefficient to modulate the regret updates
    gini = gini_coefficient([action.expected_reward for action in actions])
    regret_updates = [action.expected_reward * (1 + gini) for action in actions]
    
    # Use the endpoint circuit-breaker to determine the availability of actions
    available_actions = [action for action in actions if endpoint_circuit_breaker.allow()]
    
    # Select the action with the highest regret-weighted score
    if available_actions:
        selected_action = max(available_actions, key=lambda action: action.expected_reward * store_state.dance)
    else:
        selected_action = None
    
    return selected_action

def update_store_state(store_state: StoreState, actions: List[HybridAction], 
                        endpoint_circuit_breaker: EndpointCircuitBreaker) -> StoreState:
    """Update the store state based on the inflow and outflow derived from the aggregated weighted regrets."""
    inflow = [action.expected_reward * (1 + endpoint_circuit_breaker.failure_rate) for action in actions]
    outflow = [action.expected_reward * (1 - endpoint_circuit_breaker.failure_rate) for action in actions]
    
    level, delta = store_state.update(inflow, outflow)
    
    return StoreState(level=level, alpha=store_state.alpha, beta=store_state.beta, dt=store_state.dt, 
                       base=store_state.base, gain=store_state.gain, limit=store_state.limit, 
                       _last_delta=delta)

if __name__ == "__main__":
    endpoint_circuit_breaker = EndpointCircuitBreaker()
    store_state = StoreState()
    actions = [HybridAction(id="action1", propensity=0.5, expected_reward=10.0, confidence_bound=0.1, 
                            algorithm="hybrid", expected_value=5.0, cost=1.0, risk=0.1),
               HybridAction(id="action2", propensity=0.3, expected_reward=20.0, confidence_bound=0.2, 
                            algorithm="hybrid", expected_value=10.0, cost=2.0, risk=0.2)]
    
    selected_action = regret_weighted_action_selection(actions, store_state, endpoint_circuit_breaker)
    updated_store_state = update_store_state(store_state, actions, endpoint_circuit_breaker)
    
    print(selected_action)
    print(updated_store_state.level)