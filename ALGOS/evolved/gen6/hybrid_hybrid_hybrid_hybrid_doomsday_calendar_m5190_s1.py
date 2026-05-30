# DARWIN HAMMER — match 5190, survivor 1
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_fisher_m2731_s0.py (gen5)
# parent_b: doomsday_calendar.py (gen0)
# born: 2026-05-30T00:00:22Z

"""
This module fuses the hybrid endpoint circuit breaker and morphology from 
`hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_fisher_m2731_s0.py` with the 
doomsday/calendar weekday helper from `doomsday_calendar.py`. The mathematical 
bridge between the two structures is the use of the cyclic day index from the 
doomsday/calendar weekday helper to modulate the failure threshold in the 
endpoint circuit breaker.

The governing equations of both parents are integrated by using the cyclic day 
index to adjust the failure threshold in the endpoint circuit breaker. This 
adjusted failure threshold is then used to compute the normalized failure rate 
in the endpoint circuit breaker.

The interface between the two parents is the use of the doomsday function to 
compute the cyclic day index, which is then used to modulate the failure 
threshold in the endpoint circuit breaker.
"""

import math
import numpy as np
import random
import sys
from pathlib import Path
from datetime import date

GROUPS = ("codex", "groq", "cohere", "local_models")

def _pct(value: float) -> float:
    return round(float(value), 6)

def doomsday(year: int, month: int, day: int) -> int:
    """Return a cyclic day index in [0,6] (0 = Monday, 6 = Sunday)."""
    return (date(year, month, day).weekday() + 1) % 7

class EndpointCircuitBreaker:
    """Simple circuit‑breaker tracking consecutive failures."""
    def __init__(self, failure_threshold: int = 3):
        self.failure_threshold = failure_threshold
        self.failures = 0
        self.open = False

    def record_success(self) -> None:
        self.failures = 0
        self.open = False

    def record_failure(self) -> None:
        self.failures += 1
        self.open = self.failures >= self.failure_threshold

    def allow(self) -> bool:
        """True if the circuit is closed (i.e. endpoint may receive work)."""
        return not self.open

    def failure_rate(self, day_index: int) -> float:
        """Normalized failure rate in [0,1]."""
        adjusted_failure_threshold = self.failure_threshold * (1 + day_index / 7)
        return min(self.failures / adjusted_failure_threshold, 1.0)

def hybrid_doomsday(year: int, month: int, day: int, failure_threshold: int = 3) -> float:
    day_index = doomsday(year, month, day)
    circuit_breaker = EndpointCircuitBreaker(failure_threshold)
    circuit_breaker.record_failure()
    return circuit_breaker.failure_rate(day_index)

def simulate_day(year: int, month: int, day: int, num_failures: int) -> float:
    day_index = doomsday(year, month, day)
    circuit_breaker = EndpointCircuitBreaker()
    for _ in range(num_failures):
        circuit_breaker.record_failure()
    return circuit_breaker.failure_rate(day_index)

def stress_test(num_days: int) -> None:
    for day in range(num_days):
        year = 2022
        month = 1
        day_of_month = 1 + day % 30
        failure_rate = hybrid_doomsday(year, month, day_of_month)
        print(f"Day {day_of_month}/{month}/{year}: failure rate = {failure_rate:.4f}")

if __name__ == "__main__":
    print(hybrid_doomsday(2022, 1, 1))
    print(simulate_day(2022, 1, 1, 5))
    stress_test(30)