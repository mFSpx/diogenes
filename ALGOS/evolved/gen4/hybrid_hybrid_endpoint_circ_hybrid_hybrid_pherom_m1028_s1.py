# DARWIN HAMMER — match 1028, survivor 1
# gen: 4
# parent_a: hybrid_endpoint_circuit_bre_serpentina_self_righ_m18_s0.py (gen1)
# parent_b: hybrid_hybrid_pheromone_hyb_chelydrid_ambush_m183_s1.py (gen3)
# born: 2026-05-29T23:32:27Z

"""Hybrid Endpoint Circuit Breaker + Pheromone Burst Model
Parents:
- hybrid_endpoint_circuit_bre_serpentina_self_righ_m18_s0.py (EndpointCircuitBreaker, Morphology indices)
- hybrid_hybrid_pheromone_hyb_chelydrid_ambush_m183_s1.py (perceptual hash, burst admission score)

Mathematical bridge:
The morphology-derived *sphericity* (S) and *flatness* (F) indices are used to
parameterise the burst‑admission model (work value, drag cost, urgency force).
The resulting burst score modulates the circuit‑breaker failure threshold and
decides whether a recorded event is treated as a success or a failure.
Conversely, the circuit‑breaker state influences the admissibility of pheromone
signals via a hash‑distance penalty. This creates a closed feedback loop that
fuses both topologies into a single hybrid system.
"""

import math
import random
import sys
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Any, Iterable, List

import numpy as np

# ----------------------------------------------------------------------
# Parent A components (EndpointCircuitBreaker & Morphology)
# ----------------------------------------------------------------------


def now_z() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class EndpointCircuitBreaker:
    def __init__(self, failure_threshold: int = 3):
        self.base_failure_threshold = failure_threshold
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
        return not self.open

    def as_dict(self) -> dict[str, Any]:
        return {
            "failure_threshold": self.failure_threshold,
            "failures": self.failures,
            "open": self.open,
            "last_event_at": self.last_event_at,
        }


@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float


def sphericity_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length * width * height) ** (1.0 / 3.0) / length


def flatness_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length + width) / (2.0 * height)


def righting_time_index(
    m: Morphology, b: float = 1.0 / 3.0, k: float = 0.35, neck_lever: float = 1.0
) -> float:
    if m.mass <= 0 or neck_lever <= 0:
        raise ValueError("mass and neck_lever must be positive")
    fi = flatness_index(m.length, m.width, m.height)
    return (m.mass ** b) * math.exp(k * fi) / neck_lever


# ----------------------------------------------------------------------
# Parent B components (Pheromone hash & burst admission)
# ----------------------------------------------------------------------


def compute_phash(values: List[float]) -> int:
    if not values:
        return 0
    avg = sum(values) / len(values)
    bits = 0
    for v in values[:64]:
        bits = (bits << 1) | int(v >= avg)
    return bits


def hamming_distance(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def pulse_force(peak_force: float, steps: int = 12) -> List[float]:
    """Generate a simple linearly decreasing pulse force series."""
    if steps <= 0:
        raise ValueError("steps must be positive")
    return [peak_force * (1.0 - i / steps) for i in range(steps)]


def integrate_strike(
    force_series: Iterable[float],
    dt: float,
    m_head: float,
    drag_cd: float = 0.3,
    fluid_density: float = 1000.0,
    area: float = 0.001,
    v0: float = 0.0,
) -> "StrikeState":
    if dt <= 0 or m_head <= 0 or drag_cd < 0 or fluid_density < 0 or area < 0:
        raise ValueError("invalid strike parameters")
    v = v0
    x = 0.0
    peak = v0
    for force in force_series:
        drag = drag_cd * fluid_density * area * v * abs(v) / (2.0 * m_head)
        acc = force / m_head - drag
        v = max(0.0, v + acc * dt)
        x += v * dt
        peak = max(peak, v)
    return StrikeState(v, x, peak)


@dataclass(frozen=True)
class StrikeState:
    velocity: float
    distance: float
    peak_velocity: float


def burst_admission_score(
    work_value: float,
    cost_drag: float,
    urgency_force: float,
    steps: int = 12,
) -> float:
    """Score = work * travelled distance of a forced strike."""
    force_series = pulse_force(max(0.0, urgency_force), steps)
    state = integrate_strike(
        force_series,
        dt=0.01,
        m_head=1.0,
        drag_cd=max(0.0, cost_drag),
        fluid_density=1.0,
        area=1.0,
    )
    return work_value * state.distance


# ----------------------------------------------------------------------
# Hybrid layer – mathematical bridge
# ----------------------------------------------------------------------


def hybrid_failure_threshold(m: Morphology, base: int = 3) -> int:
    """
    Adjust the circuit breaker's failure threshold using morphology.
    Higher sphericity (more sphere‑like) relaxes the threshold,
    while higher flatness (more plate‑like) tightens it.
    """
    s = sphericity_index(m.length, m.width, m.height)
    f = flatness_index(m.length, m.width, m.height)
    factor = 1.0 + 0.5 * (s - 1.0) - 0.3 * (f - 1.0)
    adj = max(1, int(round(base * factor)))
    return adj


def hybrid_burst_score(signals: List[float], m: Morphology) -> float:
    """
    Convert pheromone signals into a burst admission score.
    - work_value   : number of 'high' bits in the perceptual hash.
    - cost_drag    : flatness index (larger flatness → more drag).
    - urgency_force: sphericity index (more spherical → higher urgency).
    """
    ph = compute_phash(signals)
    work = ph.bit_count()  # number of 1‑bits
    cost = flatness_index(m.length, m.width, m.height)
    urgency = sphericity_index(m.length, m.width, m.height)
    return burst_admission_score(work, cost, urgency)


def hybrid_hash_distance_penalty(reference_hash: int, signals: List[float]) -> float:
    """
    Compute a penalty based on Hamming distance between a reference hash
    (e.g., stored from a healthy state) and the current signal hash.
    The penalty is scaled to [0,1] and later used to bias circuit‑breaker decisions.
    """
    current = compute_phash(signals)
    max_bits = max(1, min(64, len(signals)))  # avoid division by zero
    dist = hamming_distance(reference_hash, current)
    return dist / max_bits


class HybridCircuitBreaker(EndpointCircuitBreaker):
    """
    Extends the basic circuit breaker by:
    * Dynamically recomputing its failure threshold from morphology.
    * Deciding success/failure based on a combined burst score and hash penalty.
    """

    def __init__(self, morphology: Morphology, reference_signals: List[float]):
        super().__init__(failure_threshold=hybrid_failure_threshold(morphology))
        self.morphology = morphology
        self.reference_hash = compute_phash(reference_signals)

    def evaluate_event(self, signals: List[float]) -> None:
        """
        Record an event as success if the burst score exceeds a morphology‑scaled
        baseline *and* the hash penalty is below a tolerance; otherwise record a failure.
        """
        burst = hybrid_burst_score(signals, self.morphology)
        penalty = hybrid_hash_distance_penalty(self.reference_hash, signals)

        # Baseline derived from righting_time_index (higher index → higher tolerance)
        baseline = righting_time_index(self.morphology) * 0.1

        # Decision rule
        if burst >= baseline and penalty < 0.4:
            self.record_success()
        else:
            self.record_failure()

        # Re‑adjust threshold after each event
        self.failure_threshold = hybrid_failure_threshold(self.morphology)


# ----------------------------------------------------------------------
# Demonstration functions (at least three)
# ----------------------------------------------------------------------


def hybrid_step(cb: HybridCircuitBreaker, signals: List[float]) -> dict[str, Any]:
    """
    Perform a single hybrid step:
    1. Evaluate the incoming pheromone signals.
    2. Return a snapshot of the breaker state together with computed metrics.
    """
    cb.evaluate_event(signals)
    return {
        "allow": cb.allow(),
        "failures": cb.failures,
        "threshold": cb.failure_threshold,
        "burst_score": hybrid_burst_score(signals, cb.morphology),
        "hash_penalty": hybrid_hash_distance_penalty(cb.reference_hash, signals),
    }


def simulate_hybrid_system(
    steps: int = 10,
    signal_dim: int = 20,
) -> List[dict[str, Any]]:
    """
    Run a short simulation of the hybrid system with random signals.
    Returns the history of state dictionaries.
    """
    # Arbitrary but plausible morphology
    morph = Morphology(length=1.2, width=0.8, height=0.5, mass=2.3)

    # Reference (healthy) signal pattern
    reference = [random.uniform(0.4, 0.6) for _ in range(signal_dim)]

    cb = HybridCircuitBreaker(morphology=morph, reference_signals=reference)

    history = []
    for _ in range(steps):
        # Randomly generate a new signal vector
        signals = [random.random() for _ in range(signal_dim)]
        state = hybrid_step(cb, signals)
        history.append(state)
    return history


def summarize_history(history: List[dict[str, Any]]) -> None:
    """
    Print a concise summary of the simulation history.
    """
    for i, entry in enumerate(history, 1):
        print(
            f"Step {i:02d} | Allow: {entry['allow']} | Failures: {entry['failures']}"
            f" | Threshold: {entry['threshold']} | Burst: {entry['burst_score']:.3f}"
            f" | Penalty: {entry['hash_penalty']:.2f}"
        )


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    hist = simulate_hybrid_system(steps=15, signal_dim=30)
    summarize_history(hist)