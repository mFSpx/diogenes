# DARWIN HAMMER — match 3608, survivor 1
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_bandit_m647_s0.py (gen5)
# parent_b: hybrid_hybrid_endpoint_circ_state_space_duality_m1_s0.py (gen2)
# born: 2026-05-29T23:50:52Z

"""
Hybrid Algorithm combining:
- hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_bandit_m647_s0 (Parent A)
- hybrid_hybrid_endpoint_circ_state_space_duality_m1_s0 (Parent B)

Mathematical Bridge:
Both parents expose a *Morphology* based geometric descriptor (sphericity_index) and an
*EndpointCircuitBreaker* that gates updates.  Parent B adds a linear state‑space model
(xₖ₊₁ = A·xₖ + B·uₖ, yₖ = C·xₖ + D·uₖ) while Parent A introduces a Shapley‑kernel weight
derived from combinatorial subsets.  The hybrid builds the state‑space matrices
(A, B, C, D) directly from the morphology’s sphericity and the Shapley weight, then
uses the circuit breaker to decide whether the state advances.  The resulting system
simultaneously performs work‑share allocation (deterministic target percentage)
and state estimation, with recovery priority guiding fallback behaviour.
"""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

import numpy as np


# ----------------------------------------------------------------------
# Shared data structures
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float


def sphericity_index(length: float, width: float, height: float) -> float:
    """Geometric sphericity (Parent A & B compatible)."""
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length * width * height) ** (1.0 / 3.0) / max(length, width, height)


def shapley_kernel_weight(subset_size: int, feature_count: int) -> float:
    """Combinatorial Shapley kernel weight (Parent A)."""
    return math.comb(feature_count, subset_size)


def flatness_index(length: float, width: float, height: float) -> float:
    """Flatness index (Parent B)."""
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length + width) / (2.0 * height)


def righting_time_index(
    m: Morphology, b: float = 1.0 / 3.0, k: float = 0.35, neck_lever: float = 1.0
) -> float:
    """Righting time index (Parent B)."""
    if m.mass <= 0 or neck_lever <= 0:
        raise ValueError("mass and neck_lever must be positive")
    fi = flatness_index(m.length, m.width, m.height)
    return (m.mass ** b) * math.exp(k * fi) / neck_lever


def recovery_priority(m: Morphology, max_index: float = 10.0) -> float:
    """Recovery priority scaled to [0,1] (Parent B)."""
    return max(0.0, min(1.0, righting_time_index(m) / max_index))


# ----------------------------------------------------------------------
# Endpoint circuit breaker (merged)
# ----------------------------------------------------------------------
class EndpointCircuitBreaker:
    """Circuit breaker that gates state updates."""

    def __init__(self, failure_threshold: int = 3):
        self.failure_threshold = failure_threshold
        self.failures = 0
        self.open = False
        self.last_event_at = ""

    def record_success(self) -> None:
        self.failures = 0
        self.open = False
        self.last_event_at = datetime.now(timezone.utc).isoformat().replace(
            "+00:00", "Z"
        )

    def record_failure(self) -> None:
        self.failures += 1
        self.open = self.failures >= self.failure_threshold
        self.last_event_at = datetime.now(timezone.utc).isoformat().replace(
            "+00:00", "Z"
        )

    def allow(self) -> bool:
        """Return True if the breaker is closed (updates allowed)."""
        return not self.open

    def as_dict(self) -> dict[str, Any]:
        return {
            "failure_threshold": self.failure_threshold,
            "failures": self.failures,
            "open": self.open,
            "last_event_at": self.last_event_at,
        }


# ----------------------------------------------------------------------
# StoreState – holds the internal state vector
# ----------------------------------------------------------------------
@dataclass
class StoreState:
    """Container for the mutable state vector used by the state‑space model."""
    x: np.ndarray

    def as_dict(self) -> Dict[str, Any]:
        return {"x": self.x.tolist()}


# ----------------------------------------------------------------------
# Core hybrid functions
# ----------------------------------------------------------------------
def build_state_matrices(
    morphology: Morphology,
    feature_count: int,
    state_dim: int = 3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Construct (A, B, C, D) matrices where:

    * A is a diagonal matrix scaled by the morphology sphericity.
    * B columns are weighted by the Shapley kernel for each feature.
    * C projects the state onto a single output (sum of states).
    * D is a small identity‑like leakage term.

    This creates a deterministic link between geometry (Parent B) and combinatorial
    weighting (Parent A).
    """
    sph = sphericity_index(morphology.length, morphology.width, morphology.height)
    # A: stable dynamics, eigenvalues in (0,1) proportional to sphericity
    A = np.diag(np.full(state_dim, 0.5 + 0.4 * sph))

    # B: each input dimension gets a Shapley weight; we assume feature_count inputs
    B = np.zeros((state_dim, feature_count))
    for i in range(state_dim):
        for j in range(feature_count):
            weight = shapley_kernel_weight(j + 1, feature_count)
            B[i, j] = (0.1 + 0.01 * weight) * sph

    # C: sum all state components (simple observable)
    C = np.ones((1, state_dim))

    # D: small direct‑feedthrough to keep system observable when breaker blocks A
    D = np.eye(1, feature_count) * 0.05

    return A, B, C, D


def state_update(
    store: StoreState,
    u: np.ndarray,
    cb: EndpointCircuitBreaker,
    morphology: Morphology,
    feature_count: int,
) -> Tuple[StoreState, float]:
    """
    Perform one discrete‑time state‑space update if the circuit breaker permits it.
    Returns the updated StoreState and the scalar output y.

    If the breaker is open, the state is held constant and only the direct term D·u
    contributes to the output (mirroring a fallback path in Parent A).
    """
    A, B, C, D = build_state_matrices(morphology, feature_count, state_dim=store.x.shape[0])

    if cb.allow():
        # Normal update
        x_next = A @ store.x + B @ u
        cb.record_success()
    else:
        # Breaker open: state frozen, only record a failure for diagnostic purposes
        x_next = store.x.copy()
        cb.record_failure()

    y = (C @ x_next + D @ u).item()
    new_store = StoreState(x_next)
    return new_store, y


def allocate_workshare(
    base_target_pct: float,
    morphology: Morphology,
    feature_count: int,
    subset_size: int = 2,
) -> float:
    """
    Compute a deterministic target percentage for work‑share allocation.

    The base target is modulated by:
    * sphericity_index (higher sphericity → higher allocation)
    * Shapley kernel weight for the given subset size (captures combinatorial importance)

    The result is clipped to [0, 100].
    """
    sph = sphericity_index(morphology.length, morphology.width, morphology.height)
    shapley = shapley_kernel_weight(subset_size, feature_count)

    # Normalise shapley weight to a factor in [0,1]
    max_shapley = shapley_kernel_weight(feature_count, feature_count)
    shapley_factor = shapley / max_shapley if max_shapley else 0.0

    # Blend the factors linearly
    modulated = base_target_pct * (0.5 + 0.5 * sph) * (0.5 + 0.5 * shapley_factor)

    return max(0.0, min(100.0, modulated))


def hybrid_step(
    store: StoreState,
    u: np.ndarray,
    cb: EndpointCircuitBreaker,
    morphology: Morphology,
    feature_count: int,
    base_target_pct: float,
) -> Tuple[StoreState, float, float]:
    """
    One high‑level hybrid iteration:

    1. Update the internal state (state‑space) respecting the circuit breaker.
    2. Produce a work‑share allocation percentage using the same geometric data.
    3. Return the new StoreState, system output y, and allocation pct.
    """
    new_store, y = state_update(store, u, cb, morphology, feature_count)
    allocation = allocate_workshare(base_target_pct, morphology, feature_count)
    return new_store, y, allocation


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Define a sample morphology
    morph = Morphology(length=2.0, width=1.5, height=1.0, mass=3.0)

    # Initialise components
    cb = EndpointCircuitBreaker(failure_threshold=2)
    initial_state = StoreState(x=np.zeros(3))
    feature_cnt = 4
    input_vec = np.random.rand(feature_cnt)  # random control/input vector
    base_target = 40.0  # base work‑share target percent

    # Run a few hybrid steps, toggling breaker state artificially
    for step in range(5):
        if step == 2:  # simulate a failure burst
            cb.record_failure()
            cb.record_failure()  # open the breaker

        store, output, alloc = hybrid_step(
            store=initial_state,
            u=input_vec,
            cb=cb,
            morphology=morph,
            feature_count=feature_cnt,
            base_target_pct=base_target,
        )
        initial_state = store  # feed forward

        print(
            f"Step {step}: y={output:.4f}, allocation={alloc:.2f}%, breaker_open={not cb.allow()}"
        )
    sys.exit(0)