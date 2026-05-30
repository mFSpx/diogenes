# DARWIN HAMMER — match 1364, survivor 3
# gen: 3
# parent_a: gini_coefficient.py (gen0)
# parent_b: hybrid_hybrid_endpoint_circ_hybrid_workshare_all_m12_s2.py (gen2)
# born: 2026-05-29T23:35:41Z

from __future__ import annotations

import math
import random
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Parent A – Gini coefficient
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


# ----------------------------------------------------------------------
# Parent B – Endpoint circuit‑breaker + workshare logic
# ----------------------------------------------------------------------

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
    """Data container for an endpoint."""
    name: str
    circuit: EndpointCircuitBreaker
    recovery_priority: float  # Morphology‑driven righting‑time factor ∈ [0,1]

    def health(self) -> float:
        """Health = (1‑failure_rate) * (1‑recovery_priority)."""
        fr = self.circuit.failure_rate
        rp = max(min(self.recovery_priority, 1.0), 0.0)
        return (1.0 - fr) * (1.0 - rp)


# ----------------------------------------------------------------------
# Hybrid functions
# ----------------------------------------------------------------------

def compute_endpoint_healths(endpoints: List[Endpoint]) -> List[float]:
    """Return a list of health scores for the supplied endpoints."""
    return [ep.health() for ep in endpoints]


def deterministic_share(total_units: float, deterministic_target_pct: float) -> float:
    """
    Compute the deterministic portion of the workshare using the Doomsday factor.

    d = (weekday + 1) % 7   where Monday == 0.
    """
    weekday = datetime.now(timezone.utc).weekday()
    d = (weekday + 1) % 7
    return total_units * deterministic_target_pct / 100.0 * (1.0 + d / 7.0)


def allocate_workshare(
    endpoints: List[Endpoint],
    total_units: float,
    deterministic_target_pct: float = 30.0,
    epsilon: float = 1e-9,
) -> Dict[str, float]:
    """
    Allocate `total_units` across `endpoints`.

    The deterministic part is split evenly; the residual part is weighted by health
    scores and modulated by the Gini inequality of those scores.
    """
    if not endpoints:
        raise ValueError("At least one endpoint required")

    n = len(endpoints)

    # 1️⃣ Deterministic component (even split)
    det_total = deterministic_share(total_units, deterministic_target_pct)
    det_per_endpoint = det_total / n

    # 2️⃣ Residual component
    residual_units = total_units - det_total
    if residual_units < 0:
        # In pathological cases (e.g. very high deterministic_target_pct) clamp.
        residual_units = 0.0

    # Health scores
    healths = compute_endpoint_healths(endpoints)

    # Gini coefficient of the health distribution
    G = gini_coefficient(healths)

    # Fairness‑adjusted weights
    weights = np.array([h * (1.0 - G) + epsilon for h in healths])
    weights = np.maximum(weights, epsilon)  # Ensure weights are at least epsilon
    weight_sum = np.sum(weights)

    allocations: Dict[str, float] = {}
    for ep, w in zip(endpoints, weights):
        residual = residual_units * w / weight_sum if weight_sum > 0 else 0.0
        allocations[ep.name] = det_per_endpoint + residual

    # Normalize allocations to ensure they sum to total_units
    total_allocated = sum(allocations.values())
    if not math.isclose(total_allocated, total_units, rel_tol=1e-6):
        allocations = {k: v / total_allocated * total_units for k, v in allocations.items()}

    return allocations


def summarize_allocation(
    endpoints: List[Endpoint],
    allocation: Dict[str, float],
) -> Tuple[float, float]:
    """
    Return a tuple (total_allocated, gini_of_allocation) for diagnostic purposes.
    """
    allocated_vals = [allocation[ep.name] for ep in endpoints]
    total = sum(allocated_vals)
    gini = gini_coefficient(allocated_vals)
    return total, gini


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create a small fleet of endpoints with varying reliability and recovery priorities
    random.seed(42)

    fleet: List[Endpoint] = []
    for i in range(5):
        cb = EndpointCircuitBreaker(failure_threshold=3)
        # Simulate a random number of failures (0‑4)
        for _ in range(random.randint(0, 4)):
            cb.record_failure()
        # Random recovery priority in [0, 0.9]
        rp = random.random() * 0.9
        fleet.append(Endpoint(name=f"ep{i+1}", circuit=cb, recovery_priority=rp))

    total_work_units = 1000.0
    deterministic_pct = 30.0

    alloc = allocate_workshare(fleet, total_work_units, deterministic_pct)
    total, alloc_gini = summarize_allocation(fleet, alloc)

    print("Endpoint allocations:")
    for ep in fleet:
        print(f"  {ep.name:>4}: {alloc[ep.name]:8.2f} units (health={ep.health():.3f})")
    print(f"\nTotal allocated: {total:.2f} (expected {total_work_units})")
    print(f"Gini of final allocation: {alloc_gini:.4f}")

    # Simple sanity checks
    assert math.isclose(total, total_work_units, rel_tol=1e-6), "Allocation sum mismatch"
    sys.exit(0)