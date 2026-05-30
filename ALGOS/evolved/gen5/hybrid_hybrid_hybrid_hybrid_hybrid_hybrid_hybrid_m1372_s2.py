# DARWIN HAMMER — match 1372, survivor 2
# gen: 5
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_fisher_m984_s0.py (gen4)
# parent_b: hybrid_hybrid_hybrid_doomsd_hybrid_hybrid_hybrid_m192_s0.py (gen4)
# born: 2026-05-29T23:35:42Z

import numpy as np
import math
import random
import sys
import pathlib
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

@dataclass(frozen=True)
class Morphology:
    """Geometric description of a physical (or logical) entity."""
    length: float
    width: float
    height: float
    mass: float

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

    def as_dict(self) -> dict[str, any]:
        return {
            "failure_threshold": self.failure_threshold,
            "failures": self.failures,
            "open": self.open,
            "last_event_at": self.last_event_at,
        }

def sphericity_index(length: float, width: float, height: float) -> float:
    """Ratio of the geometric mean of dimensions to the longest dimension."""
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length * width * height) ** (1.0 / 3.0) / max(length, width, height)

def shapley_kernel_weight(subset_size: int, feature_count: int) -> float:
    return math.factorial(subset_size) * math.factorial(feature_count - subset_size - 1) / math.factorial(feature_count)

def exact_shapley_value(
    value_fn: callable[[frozenset[int]], float],
    feature_index: int,
    feature_count: int,
) -> float:
    """Exact generic Shapley value by enumerating every coalition.

    Intended for small didactic/state-vector cases. 
    """

    sum = 0
    for i in range(1 << feature_count):
        mask = frozenset([j for j in range(feature_count) if (i & (1 << j))])
        if feature_index in mask:
            sum += value_fn(mask) * shapley_kernel_weight(len(mask), feature_count)
    return sum / (1 << feature_count)

def doomsday_numpy(
    years: np.ndarray,
    months: np.ndarray,
    days: np.ndarray,
) -> np.ndarray:
    """Return weekday numbers (Mon=0 … Sun=6) for vectorised dates."""
    dates = np.stack([years, months, days], axis=-1).astype("datetime64[D]")
    flat = dates.ravel()
    py_weekday = np.fromiter(
        (
            datetime.utcfromtimestamp(int(d.astype("datetime64[s]"))).weekday()
            for d in flat
        ),
        dtype=np.int8,
        count=flat.size,
    )
    py_weekday = py_weekday.reshape(dates.shape[:-1])
    # shift so that Sunday becomes 0, Monday 1, … Saturday 6 
    return (py_weekday + 1) % 7

def weekday_counts(
    dates: list,
) -> np.ndarray:
    """Count occurrences of each weekday for an iterable of dates."""
    years, months, days = [], [], []
    for d in dates:
        if isinstance(d, datetime):
            y, m, day = d.year, d.month, d.day
        else:
            y, m, day = d
        years.append(y)
        months.append(m)
        days.append(day)
    years = np.array(years)
    months = np.array(months)
    days = np.array(days)
    return np.bincount(doomsday_numpy(years, months, days).flatten(), minlength=7)

def health_score(
    reconstruction_risk_score: float,
    failure_rate: float,
    recovery_priority: float,
) -> float:
    """Novel health metric."""
    if reconstruction_risk_score < 0 or reconstruction_risk_score > 1:
        raise ValueError("reconstruction_risk_score must be in [0, 1]")
    if failure_rate < 0 or failure_rate > 1:
        raise ValueError("failure_rate must be in [0, 1]")
    if recovery_priority < 0 or recovery_priority > 1:
        raise ValueError("recovery_priority must be in [0, 1]")
    return (1 - (reconstruction_risk_score * failure_rate)) * (1 - recovery_priority)

def morphology_driven_righting_time(
    morphology: Morphology,
) -> float:
    """Novel recovery priority model."""
    sphericity = sphericity_index(morphology.length, morphology.width, morphology.height)
    volume = morphology.length * morphology.width * morphology.height
    return 1 / (sphericity * volume)

def hybrid_decision_engine(
    health_score: float,
    weekday_counts: np.ndarray,
    morphology: Morphology,
) -> float:
    """Bandit-based decision engine informed by health scores, weekday counts and morphology."""
    recovery_priority = morphology_driven_righting_time(morphology)
    failure_rate = health_score / (1 - health_score)
    # for simplicity, assume a uniform prior over weekdays
    prior = np.ones(7) / 7
    posterior = prior * weekday_counts
    posterior /= posterior.sum()
    return np.sum(posterior * health_score * (1 - recovery_priority))

def failure_rate_from_circuit_breaker(circuit_breaker: EndpointCircuitBreaker) -> float:
    return circuit_breaker.failures / circuit_breaker.failure_threshold

if __name__ == "__main__":
    # smoke test
    morphology = Morphology(length=1.0, width=2.0, height=3.0, mass=4.0)
    circuit_breaker = EndpointCircuitBreaker(failure_threshold=3)
    circuit_breaker.record_success()
    print(circuit_breaker.allow())  # should print: True
    dates = [datetime(2022, 1, 1), datetime(2022, 1, 2), datetime(2022, 1, 3)]
    weekday_count = weekday_counts(dates)
    print(weekday_count)  # should print: [0 1 1 1 0 0 0]
    health = health_score(reconstruction_risk_score=0.5, failure_rate=0.2, recovery_priority=0.1)
    decision = hybrid_decision_engine(health, weekday_count, morphology)
    print(decision)  # should print a float value
    failure_rate = failure_rate_from_circuit_breaker(circuit_breaker)
    print(failure_rate)  # should print: 0.0