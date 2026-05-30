# DARWIN HAMMER — match 5212, survivor 0
# gen: 6
# parent_a: hybrid_hybrid_hybrid_pherom_hybrid_hybrid_hybrid_m2606_s5.py (gen5)
# parent_b: hybrid_hybrid_hybrid_nlms_h_hybrid_hybrid_hybrid_m2571_s0.py (gen5)
# born: 2026-05-30T00:00:39Z

"""Hybrid Pheromone‑NLMS System
================================

Parent A: ``hybrid_hybrid_hybrid_pherom_hybrid_hybrid_hybrid_m2606_s5.py`` – a
pheromone store where each entry decays exponentially with a configurable
half‑life.  The decay factor is a simple exponential:  

``signal(t) = signal(0)·0.5^{t/half_life}``.

Parent B: ``hybrid_hybrid_hybrid_nlms_h_hybrid_hybrid_hybrid_m2571_s0.py`` – a
normalized‑least‑mean‑squares (NLMS) adaptive filter whose learning‑rate
``μ`` is modulated by a temperature‑like variable and guarded by a circuit
breaker.

**Mathematical bridge**  
The pheromone values constitute a time‑ordered high‑dimensional vector
``p(t)``.  We treat this vector as the input signal to the NLMS filter.
Conversely, the instantaneous NLMS error ``e(t)`` is used to adapt the
half‑life of each pheromone entry, i.e. the decay “temperature”.  The
temperature ``T`` is defined as the exponential moving average of the
absolute error and scales the learning‑rate:

``μ(t) = μ₀·exp(-T(t))``  

and the half‑life update rule

``half_life(t+Δ) = half_life(t)·(1 + κ·|e(t)|)``

where ``κ`` is a small sensitivity constant.  This creates a single unified
system that simultaneously evolves pheromone concentrations and an adaptive
filter, each influencing the other.

The module below implements this hybrid dynamics with three public
functions that illustrate the combined operation.
"""

import uuid
import random
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import dataclass, asdict

import numpy as np

# ----------------------------------------------------------------------
# Pheromone subsystem (Parent A)
# ----------------------------------------------------------------------
class PheromoneEntry:
    __slots__ = ("uuid", "surface_key", "signal_kind", "signal_value",
                 "half_life_seconds", "created_at", "last_decay")

    def __init__(self, surface_key: str, signal_kind: str,
                 signal_value: float, half_life_seconds: int):
        self.uuid = str(uuid.uuid4())
        self.surface_key = surface_key
        self.signal_kind = signal_kind
        self.signal_value = float(signal_value)
        self.half_life_seconds = max(1, int(half_life_seconds))
        now = datetime.now(timezone.utc)
        self.created_at = now
        self.last_decay = now

    def age_seconds(self) -> float:
        return (datetime.now(timezone.utc) - self.last_decay).total_seconds()

    def decay_factor(self) -> float:
        """Multiplicative factor since the last decay."""
        if self.half_life_seconds <= 0:
            return 0.0
        return 0.5 ** (self.age_seconds() / self.half_life_seconds)

    def apply_decay(self) -> None:
        factor = self.decay_factor()
        self.signal_value *= factor
        self.last_decay = datetime.now(timezone.utc)


class PheromoneStore:
    """In‑memory singleton‑like store."""
    _entries: dict = {}

    @classmethod
    def add(cls, entry: PheromoneEntry) -> None:
        cls._entries[entry.uuid] = entry

    @classmethod
    def get_all(cls):
        return list(cls._entries.values())

    @classmethod
    def decay_all(cls) -> None:
        for e in cls._entries.values():
            e.apply_decay()

    @classmethod
    def average_signal(cls) -> float:
        vals = [e.signal_value for e in cls._entries.values()]
        return float(np.mean(vals)) if vals else 0.0

    @classmethod
    def clear(cls) -> None:
        cls._entries.clear()


# ----------------------------------------------------------------------
# Adaptive NLMS subsystem (Parent B)
# ----------------------------------------------------------------------
class EndpointCircuitBreaker:
    """Simple failure counter that opens after a configurable threshold."""
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
        self.last_event_at = datetime.now(timezone.utc).isoformat()

    def record_failure(self) -> None:
        self.failures += 1
        self.last_event_at = datetime.now(timezone.utc).isoformat()
        if self.failures >= self.failure_threshold:
            self.open = True


class NLMSFilter:
    """Normalized Least‑Mean‑Squares adaptive filter with temperature‑scaled μ."""
    def __init__(self, order: int, mu0: float = 0.5, eps: float = 1e-8,
                 temp_coeff: float = 0.1):
        self.order = order
        self.mu0 = float(mu0)            # base learning rate
        self.eps = float(eps)
        self.temp_coeff = float(temp_coeff)  # how strongly temperature damps μ
        self.w = np.zeros(order)        # adaptive weights
        self.buffer = np.zeros(order)   # input history
        self.temp = 0.0                 # exponential moving average of |error|
        self.alpha = 0.9                # smoothing factor for temperature

    def _temperature_scaled_mu(self) -> float:
        """μ(t) = μ₀·exp(-temp_coeff·T)."""
        return self.mu0 * math.exp(-self.temp_coeff * self.temp)

    def predict(self, x: float) -> float:
        """Shift buffer, insert new sample, and return current prediction."""
        self.buffer = np.roll(self.buffer, -1)
        self.buffer[-1] = x
        return float(np.dot(self.w, self.buffer))

    def adapt(self, desired: float) -> float:
        """Perform one NLMS adaptation step and return the error."""
        y = self.predict(self.buffer[-1])
        e = desired - y
        # Update temperature (EMA of absolute error)
        self.temp = self.alpha * self.temp + (1 - self.alpha) * abs(e)

        norm = float(np.dot(self.buffer, self.buffer) + self.eps)
        mu = self._temperature_scaled_mu()
        self.w += (mu / norm) * e * self.buffer
        return e


# ----------------------------------------------------------------------
# Hybrid operations – the mathematical bridge in code
# ----------------------------------------------------------------------
def compute_path_signature(pheromone_values: np.ndarray) -> np.ndarray:
    """
    Very lightweight “signature”: cumulative sum of the pheromone vector.
    In a full implementation this would be a true iterated‑integral signature.
    """
    return np.cumsum(pheromone_values)


def update_pheromone_half_life(entry: PheromoneEntry, error: float,
                               sensitivity: float = 0.05) -> None:
    """
    Adjust the half‑life of a pheromone entry based on NLMS error.
    Larger error ⇒ longer half‑life (slower decay) so the signal persists.
    """
    factor = 1.0 + sensitivity * abs(error)
    entry.half_life_seconds = int(entry.half_life_seconds * factor)
    # Enforce a reasonable upper bound to avoid overflow
    entry.half_life_seconds = min(entry.half_life_seconds, 86400)  # 1 day


def hybrid_step(store: PheromoneStore,
                filter_: NLMSFilter,
                breaker: EndpointCircuitBreaker,
                target_signal: float,
                sensitivity: float = 0.05) -> dict:
    """
    Execute one hybrid iteration:
      1. Decay all pheromones.
      2. Build a path signature from current signals.
      3. Feed the signature (mean value) into the NLMS filter.
      4. Use the filter error to adapt half‑lives and possibly trigger the
         circuit breaker.
    Returns a dictionary with diagnostic information.
    """
    # 1 – decay
    store.decay_all()

    # 2 – signature (use mean of all current signals as a scalar feature)
    signals = np.array([e.signal_value for e in store.get_all()], dtype=float)
    if signals.size == 0:
        signature_feature = 0.0
    else:
        signature = compute_path_signature(signals)
        signature_feature = float(np.mean(signature))

    # 3 – NLMS prediction / adaptation
    error = 0.0
    if not breaker.open:
        filter_.predict(signature_feature)          # fill buffer with feature
        error = filter_.adapt(target_signal)        # adapt using desired output
        # 4 – adapt half‑lives
        for e in store.get_all():
            update_pheromone_half_life(e, error, sensitivity)
        breaker.record_success()
    else:
        # When open we still compute error for diagnostics but skip adaptation
        pred = filter_.predict(signature_feature)
        error = target_signal - pred
        breaker.record_failure()

    diagnostics = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "average_signal": float(np.mean(signals)) if signals.size else 0.0,
        "signature_feature": signature_feature,
        "nlms_error": error,
        "learning_rate": filter_._temperature_scaled_mu(),
        "circuit_breaker_open": breaker.open,
    }
    return diagnostics


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Initialise a small pheromone store
    PheromoneStore.clear()
    for i in range(5):
        entry = PheromoneEntry(
            surface_key=f"node_{i}",
            signal_kind="heat",
            signal_value=random.uniform(0.5, 1.5),
            half_life_seconds=random.randint(30, 120)
        )
        PheromoneStore.add(entry)

    # Initialise NLMS filter (order = 3) and circuit breaker
    nlms = NLMSFilter(order=3, mu0=0.6, temp_coeff=0.2)
    breaker = EndpointCircuitBreaker(failure_threshold=2)

    # Run a few hybrid steps with a synthetic target that slowly drifts
    target = 1.0
    for step in range(10):
        target += 0.05 * math.sin(step)  # gentle oscillation
        info = hybrid_step(PheromoneStore, nlms, breaker, target)
        print(f"Step {step+1:02d}: err={info['nlms_error']:.4f}, "
              f"μ={info['learning_rate']:.4f}, "
              f"avg_sig={info['average_signal']:.4f}, "
              f"breaker={'OPEN' if info['circuit_breaker_open'] else 'CLOSED'}")
    sys.exit(0)