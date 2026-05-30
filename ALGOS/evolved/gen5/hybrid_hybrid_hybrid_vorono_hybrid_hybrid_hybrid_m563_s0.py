# DARWIN HAMMER — match 563, survivor 0
# gen: 5
# parent_a: hybrid_hybrid_voronoi_parti_hybrid_endpoint_circ_m314_s0.py (gen2)
# parent_b: hybrid_hybrid_hybrid_doomsd_hybrid_hybrid_hybrid_m192_s2.py (gen4)
# born: 2026-05-29T23:29:45Z

"""
This module fuses the mathematical structures of the 'hybrid_hybrid_voronoi_parti_hybrid_endpoint_circ_m314_s0' 
and 'hybrid_hybrid_hybrid_doomsd_hybrid_hybrid_hybrid_m192_s2' algorithms. The bridge between the two 
structures lies in the application of the circuit-breaker mechanism to gate the assignment of points to 
thermal regions based on their distances to seeds, and the use of weekday distribution statistics to 
inform reconstruction risk scores.

The mathematical interface is formed by using the circuit-breaker state to modulate the reward fed back 
to the bandit algorithm, which is used to weigh the split of rewards in the reconstruction risk score 
calculation.

Authors: based on 'hybrid_hybrid_voronoi_parti_hybrid_endpoint_circ_m314_s0' and 
         'hybrid_hybrid_hybrid_doomsd_hybrid_hybrid_hybrid_m192_s2'
"""

import math
import numpy as np
import random
import sys
from pathlib import Path
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
        self.last_event_at = ""

    def record_success(self) -> None:
        self.failures = 0
        self.open = False
        self.last_event_at = ""

    def record_failure(self) -> None:
        self.failures += 1
        self.open = self.failures >= self.failure_threshold
        self.last_event_at = ""

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
    # shift so that Sunday becomes 0, Monday 1, … Saturday 6 (as in parent)
    return (py_weekday + 1) % 7

def weekday_counts(
    dates: List[date],
) -> np.ndarray:
    """Count occurrences of each weekday for an iterable of dates."""
    counts = np.zeros(7, dtype=int)
    for d in dates:
        counts[(d.weekday() + 1) % 7] += 1
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
        return {}, 0.0

def reconstruction_risk_score(regions: Dict[int, List[tuple[float, float]]], 
                             gini: float) -> float:
    score = 0.0
    for region in regions.values():
        score += len(region) * gini
    return score

if __name__ == "__main__":
    points = [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)]
    seeds = [(0.0, 0.0), (10.0, 10.0)]
    dates = [date(2022, 1, 1), date(2022, 1, 2), date(2022, 1, 3)]
    regions, gini = hybrid_operation(points, seeds, dates)
    score = reconstruction_risk_score(regions, gini)
    print(regions, gini, score)