# DARWIN HAMMER — match 1372, survivor 1
# gen: 5
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_fisher_m984_s0.py (gen4)
# parent_b: hybrid_hybrid_hybrid_doomsd_hybrid_hybrid_hybrid_m192_s0.py (gen4)
# born: 2026-05-29T23:35:42Z

"""
This module fuses the core ideas of two parents: 
- hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_fisher_m984_s0.py (Shapley value and morphology-driven analysis)
- hybrid_hybrid_hybrid_doomsd_hybrid_hybrid_hybrid_m192_s0.py (Doomsday-Calendar Gini analysis and Bandit-based decision engine)

The mathematical bridge between these two structures lies in the application of Shapley values to inform the Bandit-based decision engine. 
Specifically, we use the Shapley kernel weight to weigh the contribution of each feature in the Doomsday-Calendar Gini analysis.
"""

import numpy as np
import math
import random
import sys
from pathlib import Path
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Callable, Any

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

    def as_dict(self) -> dict[str, Any]:
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
    # shift so that Sunday becomes 0, Monday 1, … Saturday 6 (as in parent)
    return (py_weekday + 1) % 7

def weekday_counts(
    dates: list,
) -> np.ndarray:
    """Count occurrences of each weekday for an iterable of dates."""
    years, months, days = [], [], []
    for d in dates:
        if isinstance(d, datetime.date):
            y, m, day = d.year, d.month, d.day
        else:
            y, m, day = d
        years.append(y)
        months.append(m)
        days.append(day)
    return doomsday_numpy(np.array(years), np.array(months), np.array(days))

def hybrid_shapley_gini(
    value_fn: Callable[[frozenset[int]], float],
    feature_index: int,
    feature_count: int,
    dates: list,
) -> float:
    """Compute the Shapley value and Gini coefficient for a given feature and date set."""
    shapley_value = 0
    for subset_size in range(feature_count + 1):
        shapley_value += shapley_kernel_weight(subset_size, feature_count) * value_fn(frozenset(range(subset_size)))
    weekday_counts_vector = weekday_counts(dates)
    gini_coefficient = 1 - np.sum(np.square(weekday_counts_vector) / len(dates))
    return shapley_value * gini_coefficient

def hybrid_decision_engine(
    morphology: Morphology,
    circuit_breaker: EndpointCircuitBreaker,
    dates: list,
) -> bool:
    """Make a decision based on the hybrid Shapley-Gini analysis and circuit breaker state."""
    sphericity = sphericity_index(morphology.length, morphology.width, morphology.height)
    allow = circuit_breaker.allow()
    if allow:
        shapley_gini = hybrid_shapley_gini(lambda x: 1.0, 0, 10, dates)
        return sphericity > 0.5 and shapley_gini > 0.2
    return False

if __name__ == "__main__":
    morphology = Morphology(10.0, 5.0, 2.0, 1.0)
    circuit_breaker = EndpointCircuitBreaker()
    dates = [datetime(2022, 1, 1), datetime(2022, 1, 2), datetime(2022, 1, 3)]
    decision = hybrid_decision_engine(morphology, circuit_breaker, dates)
    print(decision)