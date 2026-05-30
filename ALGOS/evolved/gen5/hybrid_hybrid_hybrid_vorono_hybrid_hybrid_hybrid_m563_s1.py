# DARWIN HAMMER — match 563, survivor 1
# gen: 5
# parent_a: hybrid_hybrid_voronoi_parti_hybrid_endpoint_circ_m314_s0.py (gen2)
# parent_b: hybrid_hybrid_hybrid_doomsd_hybrid_hybrid_hybrid_m192_s2.py (gen4)
# born: 2026-05-29T23:29:45Z

import math
import numpy as np
from datetime import date, datetime, timezone
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

def distance(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def nearest(point: tuple[float, float], seeds: list[tuple[float, float]]) -> int:
    if not seeds:
        raise ValueError('seeds required')
    return min(range(len(seeds)), key=lambda i: (distance(point, seeds[i]), i))

def assign(points: list[tuple[float, float]], seeds: list[tuple[float, float]]) -> dict[int, list[tuple[float, float]]]:
    regions = {i: [] for i in range(len(seeds))}
    for p in points:
        regions[nearest(p, seeds)].append(p)
    return regions

class EndpointCircuitBreaker:
    def __init__(self, failure_threshold: int = 3):
        self.failure_threshold = failure_threshold
        self.failures = 0
        self.open = False
        self.last_event_at = None

    def record_success(self) -> None:
        self.failures = 0
        self.open = False
        self.last_event_at = None

    def record_failure(self) -> None:
        self.failures += 1
        self.open = self.failures >= self.failure_threshold
        self.last_event_at = datetime.now(timezone.utc)

    def allow(self) -> bool:
        return not self.open

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
    # shift so that Sunday becomes 6, Monday 0, … Saturday 5 (as in parent)
    return (py_weekday + 6) % 7

def weekday_counts(
    dates: List[date],
) -> np.ndarray:
    """Count occurrences of each weekday for an iterable of dates."""
    counts = np.zeros(7, dtype=int)
    for d in dates:
        counts[d.weekday()] += 1
    return counts

def gini_coefficient(c: np.ndarray) -> float:
    """The Gini coefficient is the area between the Lorenz curve and the line of equality."""
    if c.size == 0:
        return 0.0
    c = c.flatten()
    if np.any(c < 0):
        raise ValueError("counts must be non-negative")
    c = c / c.sum()
    max_val = np.arange(c.size) / c.size
    return np.sum((c - max_val) * np.arange(c.size)) / np.sum(max_val)

def hybrid_operation(points: list[tuple[float, float]], seeds: list[tuple[float, float]], 
                    dates: List[date]) -> Tuple[Dict[int, List[tuple[float, float]]], float]:
    circuit_breaker = EndpointCircuitBreaker()
    regions = assign(points, seeds)
    
    counts = weekday_counts(dates)
    gini = gini_coefficient(counts)
    
    if circuit_breaker.allow():
        circuit_breaker.record_success()
        return regions, gini
    else:
        circuit_breaker.record_failure()
        return {}, gini  # return gini even if circuit breaker is open

def reconstruction_risk_score(regions: Dict[int, List[tuple[float, float]]], 
                             gini: float) -> float:
    score = 0.0
    for region in regions.values():
        score += len(region) * gini
    return score

def improved_hybrid_operation(points: list[tuple[float, float]], seeds: list[tuple[float, float]], 
                             dates: List[date], 
                             weekday_risk_weights: Dict[int, float]) -> Tuple[Dict[int, List[tuple[float, float]]], float]:
    circuit_breaker = EndpointCircuitBreaker()
    regions = assign(points, seeds)
    
    counts = weekday_counts(dates)
    gini = gini_coefficient(counts)
    
    weighted_gini = gini
    for i, count in enumerate(counts):
        weighted_gini += count * weekday_risk_weights.get(i, 0)
    
    if circuit_breaker.allow():
        circuit_breaker.record_success()
        return regions, weighted_gini
    else:
        circuit_breaker.record_failure()
        return {}, weighted_gini

if __name__ == "__main__":
    points = [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)]
    seeds = [(0.0, 0.0), (10.0, 10.0)]
    dates = [date(2022, 1, 1), date(2022, 1, 2), date(2022, 1, 3)]
    weekday_risk_weights = {0: 0.1, 1: 0.2, 2: 0.3, 3: 0.4, 4: 0.5, 5: 0.6, 6: 0.7}
    regions, gini = improved_hybrid_operation(points, seeds, dates, weekday_risk_weights)
    score = reconstruction_risk_score(regions, gini)
    print(regions, gini, score)