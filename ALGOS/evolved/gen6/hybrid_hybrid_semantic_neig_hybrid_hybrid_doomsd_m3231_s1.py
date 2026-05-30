# DARWIN HAMMER — match 3231, survivor 1
# gen: 6
# parent_a: hybrid_semantic_neighbors_hybrid_endpoint_circ_m17_s0.py (gen2)
# parent_b: hybrid_hybrid_doomsday_cale_hybrid_hybrid_hybrid_m2697_s2.py (gen5)
# born: 2026-05-29T23:48:39Z

import numpy as np
from dataclasses import dataclass
from datetime import datetime, timezone
from math import exp, sqrt
from random import random
from sys import exit
from pathlib import Path

@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float

@dataclass(frozen=True)
class Document:
    id: str
    vector: list[float]

def sphericity_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length * width * height) ** (1.0 / 3.0) / length

def flatness_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length + width) / (2.0 * height)

def righting_time_index(m: Morphology, b: float = 1.0 / 3.0, k: float = 0.35, neck_lever: float = 1.0) -> float:
    if m.mass <= 0 or neck_lever <= 0:
        raise ValueError("mass and neck_lever must be positive")
    fi = flatness_index(m.length, m.width, m.height)
    return (m.mass ** b) * exp(k * fi) / neck_lever

def recovery_priority(m: Morphology, max_index: float = 10.0) -> float:
    return max(0.0, min(1.0, righting_time_index(m) / max_index))

def _cos(a, b):
    den = sqrt(sum(x * x for x in a)) * sqrt(sum(y * y for y in b))
    return 0.0 if den == 0 else sum(x * y for x, y in zip(a, b)) / den

def doomsday_numpy(
    years: np.ndarray,
    months: np.ndarray,
    days: np.ndarray,
) -> np.ndarray:
    dates = np.stack([years, months, days], axis=-1).astype('datetime64[D]')
    np_weekday = dates.astype('datetime64[D]').astype('datetime64[ns]').astype('int64')
    flat = dates.ravel()
    py_weekday = np.fromiter(
        (dt.datetime.utcfromtimestamp(int(d.astype('datetime64[s]'))).weekday() for d in flat),
        dtype=np.int8,
        count=flat.size,
    )
    py_weekday = py_weekday.reshape(dates.shape[:-1])
    return (py_weekday + 1) % 7


def gini_coefficient_numpy(values: np.ndarray) -> float:
    if values.ndim != 1:
        raise ValueError("values must be a 1‑D array")
    xs = np.sort(values.astype(float))
    if xs.size == 0 or xs.sum() == 0.0:
        return 0.0
    if xs[0] < 0:
        raise ValueError("values must be non‑negative")
    n = xs.size
    i = np.arange(1, n + 1)  # 1‑based index
    numerator = np.sum((2 * i - n - 1) * xs)
    denominator = n * xs.sum()
    return numerator / denominator


def weekday_counts(dates: list) -> np.ndarray:
    years, months, days = [], [], []
    for d in dates:
        if isinstance(d, dt.date):
            y, m, day = d.year, d.month, d.day
        else:
            y, m, day = d
        years.append(y)
        months.append(m)
        days.append(day)
    return np.array([years, months, days]).T


def hybrid_gini_morphology_analysis(dates: list, morphology: Morphology) -> float:
    """Calculate the weighted sphericity index using the Gini coefficient of weekday counts."""
    weekday_gini = gini_coefficient_numpy(weekday_counts(dates))
    return sphericity_index(morphology.length, morphology.width, morphology.height) * weekday_gini


class EndpointCircuitBreaker:
    def __init__(self, failure_threshold: int = 3, morphology: Morphology = None):
        self.failure_threshold = failure_threshold
        self.morphology = morphology

    def calculate_recovery_priority(self, dates: list) -> float:
        """Calculate the recovery priority based on the morphology and weekday counts."""
        weighted_sphericity = hybrid_gini_morphology_analysis(dates, self.morphology)
        return recovery_priority(self.morphology, max_index=10.0) * weighted_sphericity

    def determine_circuit_status(self, dates: list) -> bool:
        """Determine the circuit status based on the recovery priority."""
        recovery_priority_value = self.calculate_recovery_priority(dates)
        return recovery_priority_value > 0.5


def test_hybrid_algorithm():
    morphology = Morphology(length=5.0, width=3.0, height=2.0, mass=10.0)
    circuit_breaker = EndpointCircuitBreaker(morphology=morphology)
    dates = [dt.date(2022, 1, 1), dt.date(2022, 1, 2), dt.date(2022, 1, 3)]
    circuit_status = circuit_breaker.determine_circuit_status(dates)
    print(f"Circuit status: {circuit_status}")


if __name__ == "__main__":
    test_hybrid_algorithm()