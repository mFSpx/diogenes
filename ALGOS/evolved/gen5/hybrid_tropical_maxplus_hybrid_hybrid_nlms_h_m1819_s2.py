# DARWIN HAMMER — match 1819, survivor 2
# gen: 5
# parent_a: tropical_maxplus.py (gen0)
# parent_b: hybrid_hybrid_nlms_hybrid_h_hybrid_endpoint_circ_m715_s1.py (gen4)
# born: 2026-05-29T23:39:00Z

import numpy as np
import math
import random
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float

    def __post_init__(self) -> None:
        for name, value in (
            ("length", self.length),
            ("width", self.width),
            ("height", self.height),
            ("mass", self.mass),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive")

class EndpointCircuitBreaker:
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
        self.last_event_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    def record_failure(self) -> None:
        self.failures += 1
        self.open = self.failures >= self.failure_threshold
        self.last_event_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    def allow(self) -> bool:
        return not self.open

class TropicalMaxPlus:
    def __init__(self, coeffs):
        self.coeffs = np.asarray(coeffs, dtype=float)

    def evaluate(self, x):
        exponents = np.arange(len(self.coeffs), dtype=float)
        terms = self.coeffs.reshape((-1,) + (1,) * x.ndim) + exponents.reshape((-1,) + (1,) * x.ndim) * x
        return np.max(terms, axis=0)

class HybridEndpointCircuit:
    def __init__(self, coeffs, failure_threshold=3):
        self.tropical_max_plus = TropicalMaxPlus(coeffs)
        self.endpoint_circuit_breaker = EndpointCircuitBreaker(failure_threshold)
        self.coeffs_history = [coeffs]

    def predict(self, x):
        outcome = self.tropical_max_plus.evaluate(x)
        if self.endpoint_circuit_breaker.allow():
            return outcome
        else:
            return np.nan

    def update(self, x, outcome):
        self.endpoint_circuit_breaker.record_success() if outcome else self.endpoint_circuit_breaker.record_failure()
        self.update_coefficients(x, outcome)

    def update_coefficients(self, x, outcome):
        if self.endpoint_circuit_breaker.open:
            new_coefficients = self.coeffs_history[-1] * 0.9
        else:
            new_coefficients = self.coeffs_history[-1] * 1.1
        self.coeffs_history.append(new_coefficients)
        self.tropical_max_plus = TropicalMaxPlus(new_coefficients)

def hybrid_operation(coeffs, x, failure_threshold=3):
    hybrid_endpoint_circuit = HybridEndpointCircuit(coeffs, failure_threshold)
    outcome = hybrid_endpoint_circuit.predict(x)
    hybrid_endpoint_circuit.update(x, outcome)
    return outcome

def smoke_test():
    coeffs = np.array([1, 2, 3])
    x = np.array([1, 2, 3])
    outcome = hybrid_operation(coeffs, x)
    print(outcome)

if __name__ == "__main__":
    smoke_test()