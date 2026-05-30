# DARWIN HAMMER — match 5199, survivor 5
# gen: 5
# parent_a: hybrid_gini_coefficient_hybrid_hybrid_endpoi_m1364_s3.py (gen3)
# parent_b: hybrid_hybrid_hybrid_regret_hybrid_hybrid_hybrid_m142_s2.py (gen4)
# born: 2026-05-30T00:00:42Z

"""Hybrid Gini‑Endpoint‑Regret Engine
===================================

This module fuses two distinct parent algorithms:

* **Parent A** – *gini_coefficient* and an *EndpointCircuitBreaker* that
  monitors per‑endpoint failure rates.
* **Parent B** – a regret‑weighted action selection engine enriched with
  pheromone dynamics and a honey‑bee‑style *StoreState* whose “dance”
  signal modulates scores.

**Mathematical bridge**

The bridge is the *inequality* of endpoint failures measured by the Gini
coefficient.  Let  


f_i = circuit_i.failure_rate          (i = 1 … N)
G   = Gini(f_1,…,f_N)


The scalar `G ∈ [0,1]` quantifies how uneven the failure distribution is.
We inject `G` into the hybrid score used for action selection as a
multiplicative factor on the store’s dance signal `D(t)`:


S_a(t) = (E_a + G·D(t)·σ(a,c)) · (1 + φ_a(t)) – ΔR_a(t)


where  

* `E_a` – baseline expected reward,  
* `σ(a,c)` – a (placeholder) similarity between action *a* and the current
  context,  
* `φ_a(t)` – pheromone level for action *a*,  
* `ΔR_a(t)` – accumulated regret for *a*.

Thus the Gini coefficient couples the endpoint reliability layer (Parent A)
to the regret‑pheromone engine (Parent B), yielding a single unified
decision‑making system.

The module provides three public functions that illustrate the hybrid
behaviour:

1. ``endpoint_failure_gini`` – computes the Gini coefficient of all
   endpoints’ failure rates.
2. ``hybrid_score`` – evaluates the above equation for a given action.
3. ``hybrid_update`` – updates regret, pheromone, store dynamics and the
   circuit‑breaker state after observing a reward.

All components are pure‑Python and depend only on the standard library and
NumPy."""

from __future__ import annotations

import math
import random
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Parent A – Gini coefficient and endpoint circuit‑breaker
# ----------------------------------------------------------------------


def gini_coefficient(values: Iterable[float]) -> float:
    """Return the Gini coefficient of a non‑empty iterable of non‑negative numbers."""
    xs = sorted(float(x) for x in values)
    if not xs or sum(xs) == 0.0:
        return 0.0
    if xs[0] < 0:
        raise ValueError("values must be non‑negative")
    n = len(xs)
    return sum((2 * i - n - 1) * x for i, x in enumerate(xs, 1)) / (n * sum(xs))


def now_z() -> str:
    """Current UTC timestamp in ISO‑8601 Zulu format."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


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
        self.last_event_at = now_z()

    def record_failure(self) -> None:
        self.failures += 1
        self.open = self.failures >= self.failure_threshold
        self.last_event_at = now_z()

    def allow(self) -> bool:
        """True if the circuit is closed (i.e. endpoint is usable)."""
        return not self.open

    @property
    def failure_rate(self) -> float:
        """Failures normalised by the threshold (clamped to [0,1])."""
        return min(self.failures / self.failure_threshold, 1.0)


@dataclass
class Endpoint:
    """Container for an external service endpoint."""
    name: str
    circuit: EndpointCircuitBreaker = field(default_factory=EndpointCircuitBreaker)
    recovery_priority: float = 0.5  # ∈ [0,1]; higher means quicker recovery


# ----------------------------------------------------------------------
# Parent B – Regret‑pheromone engine with honey‑bee store dynamics
# ----------------------------------------------------------------------


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
    """Observation used to update the policy."""
    context_id: str
    action_id: str
    reward: float
    propensity: float


@dataclass
class StoreState:
    """Honeybee‑style store with a bounded control signal (dance)."""
    level: float = 0.0
    alpha: float = 1.0
    beta: float = 1.0
    dt: float = 1.0
    base: float = 1.0
    gain: float = 1.0
    limit: float = 10.0
    _last_delta: float = 0.0  # internal cache for dance

    def update(self, inflow: List[float], outflow: List[float]) -> Tuple[float, float]:
        """Apply inflow/outflow and return new level and delta."""
        delta = self.alpha * sum(inflow) - self.beta * sum(outflow)
        self.level = max(0.0, min(self.limit, self.level + self.dt * delta))
        self._last_delta = delta
        return self.level, delta

    @property
    def dance(self) -> float:
        """Bounded control signal derived from the last Δ."""
        # Normalise the raw delta to [0, limit] and apply gain
        raw = self._last_delta * self.gain
        return max(0.0, min(self.limit, raw))


# ----------------------------------------------------------------------
# Fusion utilities
# ----------------------------------------------------------------------


class RegretTracker:
    """Tracks cumulative regret per action."""
    def __init__(self):
        self._regret: Dict[str, float] = {}

    def add(self, action_id: str, delta: float) -> None:
        self._regret[action_id] = self._regret.get(action_id, 0.0) + delta

    def get(self, action_id: str) -> float:
        return self._regret.get(action_id, 0.0)


class PheromoneTable:
    """Simple pheromone level table with evaporation."""
    def __init__(self, evaporation: float = 0.1):
        self._phi: Dict[str, float] = {}
        self.evaporation = evaporation  # per update step

    def deposit(self, action_id: str, amount: float) -> None:
        self._phi[action_id] = self._phi.get(action_id, 0.0) + amount

    def evaporate(self) -> None:
        for aid in list(self._phi):
            self._phi[aid] *= (1.0 - self.evaporation)
            if self._phi[aid] < 1e-6:
                del self._phi[aid]

    def get(self, action_id: str) -> float:
        return self._phi.get(action_id, 0.0)


def placeholder_similarity(action_id: str, context_id: str) -> float:
    """A deterministic pseudo‑similarity in [0,1] based on hash mixing."""
    rnd = random.Random(hash(action_id + context_id))
    return rnd.random()


# ----------------------------------------------------------------------
# Hybrid public API
# ----------------------------------------------------------------------


def endpoint_failure_gini(endpoints: List[Endpoint]) -> float:
    """
    Compute the Gini coefficient of the failure rates of *endpoints*.
    This scalar quantifies the inequality of unreliability across the set.
    """
    rates = [ep.circuit.failure_rate for ep in endpoints]
    return gini_coefficient(rates)


def hybrid_score(
    action: HybridAction,
    store: StoreState,
    gini_factor: float,
    similarity: float,
    regret: float,
    pheromone: float,
) -> float:
    """
    Compute the hybrid selection score

        S = (E + G·D·σ) · (1 + φ) – ΔR

    Parameters
    ----------
    action : HybridAction
        The candidate action (E = expected_reward, id used for logging).
    store : StoreState
        Provides the current dance signal D.
    gini_factor : float
        Gini coefficient of endpoint failure rates (G ∈ [0,1]).
    similarity : float
        σ(a, c) – similarity between the action and current context.
    regret : float
        Accumulated regret ΔR_a for the action.
    pheromone : float
        Current pheromone level φ_a for the action.

    Returns
    -------
    float
        The hybrid score; larger values are preferred.
    """
    D = store.dance
    E = action.expected_reward
    S = (E + gini_factor * D * similarity) * (1.0 + pheromone) - regret
    return S


def hybrid_update(
    upd: HybridUpdate,
    store: StoreState,
    regret_tracker: RegretTracker,
    pheromone_tbl: PheromoneTable,
    endpoints: List[Endpoint],
    baseline_expected: float = 1.0,
) -> None:
    """
    Perform a full hybrid update after observing a reward.

    Steps
    -----
    1. Update regret: ΔR = max(0, baseline – reward).
    2. Deposit pheromone proportional to reward.
    3. Evaporate pheromone globally.
    4. Update the store: inflow = [reward], outflow = [ΔR].
    5. Update the circuit‑breaker of every endpoint based on a simple
       success/failure heuristic (reward < 0.5 → failure).
    """
    # 1. Regret
    delta_regret = max(0.0, baseline_expected - upd.reward)
    regret_tracker.add(upd.action_id, delta_regret)

    # 2. Pheromone deposit (reward scaled)
    pheromone_tbl.deposit(upd.action_id, upd.reward)

    # 3. Global evaporation
    pheromone_tbl.evaporate()

    # 4. Store dynamics
    inflow = [upd.reward]
    outflow = [delta_regret]
    store.update(inflow, outflow)

    # 5. Endpoint circuit‑breaker handling
    #    For demonstration we treat a low reward as a failure for all endpoints.
    for ep in endpoints:
        if upd.reward < 0.5:
            ep.circuit.record_failure()
        else:
            ep.circuit.record_success()


# ----------------------------------------------------------------------
# Demonstration / smoke test
# ----------------------------------------------------------------------


def _demo() -> None:
    # Create a handful of endpoints with varying initial failure states
    endpoints = [
        Endpoint(name="svc-A", circuit=EndpointCircuitBreaker(3), recovery_priority=0.9),
        Endpoint(name="svc-B", circuit=EndpointCircuitBreaker(3), recovery_priority=0.4),
        Endpoint(name="svc-C", circuit=EndpointCircuitBreaker(3), recovery_priority=0.7),
    ]

    # Pretend we observed some failures already
    endpoints[0].circuit.record_failure()  # 1/3
    endpoints[1].circuit.record_failure()
    endpoints[1].circuit.record_failure()  # 2/3
    # svc-C stays healthy

    # Compute Gini of failure rates
    gini = endpoint_failure_gini(endpoints)
    print(f"Gini of endpoint failure rates: {gini:.4f}")

    # Initialise the hybrid engine components
    store = StoreState()
    regret_tracker = RegretTracker()
    pheromone_tbl = PheromoneTable(evaporation=0.05)

    # Define a few mock actions
    actions = [
        HybridAction(
            id="act-1",
            propensity=0.2,
            expected_reward=0.8,
            confidence_bound=0.1,
            algorithm="demo",
            expected_value=0.75,
        ),
        HybridAction(
            id="act-2",
            propensity=0.5,
            expected_reward=0.6,
            confidence_bound=0.2,
            algorithm="demo",
            expected_value=0.55,
        ),
    ]

    # Simulate a single decision round
    context_id = "ctx-xyz"
    scores = {}
    for act in actions:
        sim = placeholder_similarity(act.id, context_id)
        regret = regret_tracker.get(act.id)
        phi = pheromone_tbl.get(act.id)
        score = hybrid_score(
            action=act,
            store=store,
            gini_factor=gini,
            similarity=sim,
            regret=regret,
            pheromone=phi,
        )
        scores[act.id] = score
        print(f"Score for {act.id}: {score:.4f}")

    # Choose best action
    chosen_id = max(scores, key=scores.get)
    chosen_action = next(a for a in actions if a.id == chosen_id)
    print(f"Chosen action: {chosen_id}")

    # Mock reward generation (random for demo)
    reward = random.random()
    print(f"Observed reward: {reward:.4f}")

    # Perform hybrid update
    upd = HybridUpdate(
        context_id=context_id,
        action_id=chosen_id,
        reward=reward,
        propensity=chosen_action.propensity,
    )
    hybrid_update(
        upd=upd,
        store=store,
        regret_tracker=regret_tracker,
        pheromone_tbl=pheromone_tbl,
        endpoints=endpoints,
        baseline_expected=1.0,
    )

    # Post‑update diagnostics
    print(f"Store level after update: {store.level:.4f}, dance: {store.dance:.4f}")
    for ep in endpoints:
        print(
            f"Endpoint {ep.name} – failures: {ep.circuit.failures}, open: {ep.circuit.open}"
        )
    for act in actions:
        print(
            f"Action {act.id} – regret: {regret_tracker.get(act.id):.4f}, pheromone: {pheromone_tbl.get(act.id):.4f}"
        )


if __name__ == "__main__":
    _demo()