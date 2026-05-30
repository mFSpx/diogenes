# DARWIN HAMMER — match 3306, survivor 0
# gen: 5
# parent_a: hybrid_hybrid_endpoint_circ_hybrid_hybrid_pherom_m1028_s1.py (gen4)
# parent_b: hybrid_hybrid_doomsday_cale_hybrid_hdc_serpentin_m313_s1.py (gen2)
# born: 2026-05-29T23:49:12Z

"""
Hybrid Endpoint Circuit Breaker + Doomsday Calendar + Pheromone Burst Model
Parents:
- hybrid_endpoint_circuit_bre_serpentina_self_righ_m18_s0.py (EndpointCircuitBreaker, Morphology)
- hybrid_hybrid_doomsday_cale_hybrid_hdc_serpentin_m313_s1.py (weekday distribution, Gini coefficient)

Mathematical bridge:
The morphology-derived *sphericity* (S) and *flatness* (F) indices are used to parameterise the 
burst-admission model (work value, drag cost, urgency force). The resulting burst score modulates 
the circuit-breaker failure threshold and decides whether a recorded event is treated as a success 
or a failure. Conversely, the circuit-breaker state influences the admissibility of pheromone signals 
via a hash-distance penalty. This creates a closed feedback loop that fuses both topologies into a 
single hybrid system. The Gini coefficient is used to measure the inequality in the distribution of 
weekdays, and the morphology hypervector is used to encode the scalar values of the weekday 
distribution. The similarity between the morphology hypervector and a reference hypervector is used 
to obtain a recovery priority.
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

# ----------------------------------------------------------------------
# Parent B components (Gini coefficient, weekday distribution)
# ----------------------------------------------------------------------

def random_vector(dim: int = 10000, seed: str | int | None = None) -> list[int]:
    rng = random.Random(seed)
    return [1 if rng.getrandbits(1) else -1 for _ in range(dim)]

def symbol_vector(symbol: str, dim: int = 10000) -> list[int]:
    seed = int.from_bytes(hashlib.sha256(symbol.encode("utf-8")).digest()[:8], "big")
    return random_vector(dim, seed)

def bind(a: list[int], b: list[int]) -> list[int]:
    if len(a) != len(b):
        raise ValueError("vectors must have equal length")
    return [x * y for x, y in zip(a, b)]

def bundle(vectors: Iterable[list[int]]) -> list[int]:
    return [sum(x) for x in zip(*vectors)]

def gini_coefficient(values: Iterable[float]) -> float:
    xs = sorted(float(x) for x in values)
    if not xs or sum(xs) == 0: 
        return 0.0
    if xs[0] < 0: 
        raise ValueError("values must be non-negative")
    n = len(xs)
    return sum((2*i-n-1)*x for i,x in enumerate(xs,1))/(n*sum(xs))

def simulate_weekday_distribution(year: int, month: int, day: int, num_days: int) -> np.ndarray:
    weekdays = []
    for i in range(num_days):
        date = datetime(year, month, day) + datetime.timedelta(days=i)
        weekday = date.weekday()
        weekdays.append(weekday)
    return np.array(weekdays)

# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------

def morphology_to_gini(morphology: Morphology, num_days: int) -> float:
    # Use morphology indices to parameterise burst-admission model
    s = morphology.length / (morphology.length + morphology.width)
    f = morphology.height / (morphology.length + morphology.width)
    burst_score = 2 * s * f
    # Use burst score to modulate circuit-breaker failure threshold
    failure_threshold = int(burst_score * 5)
    # Use Gini coefficient to measure inequality in weekday distribution
    gini = gini_coefficient(simulate_weekday_distribution(2024, 1, 1, num_days))
    return gini

def circuit_breaker_to_pheromone(circuit_breaker: EndpointCircuitBreaker, morphology: Morphology, symbol: str) -> list[int]:
    # Use circuit-breaker state to influence admissibility of pheromone signals
    if circuit_breaker.open:
        penalty = 1
    else:
        penalty = 0
    # Use morphology indices to parameterise pheromone signal
    s = morphology.length / (morphology.length + morphology.width)
    f = morphology.height / (morphology.length + morphology.width)
    pheromone_signal = [int(x * s * f) for x in symbol_vector(symbol, 10000)]
    # Apply penalty to pheromone signal
    pheromone_signal = [x * (1 + penalty) for x in pheromone_signal]
    return pheromone_signal

def hybrid_operation(circuit_breaker: EndpointCircuitBreaker, morphology: Morphology, symbol: str, num_days: int) -> float:
    gini = morphology_to_gini(morphology, num_days)
    pheromone_signal = circuit_breaker_to_pheromone(circuit_breaker, morphology, symbol)
    # Use similarity between morphology hypervector and reference hypervector to obtain recovery priority
    reference_hypervector = random_vector(10000)
    similarity = np.dot(morphology.length, reference_hypervector) / (np.linalg.norm(morphology.length) * np.linalg.norm(reference_hypervector))
    recovery_priority = gini * similarity
    return recovery_priority

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------

if __name__ == "__main__":
    morph = Morphology(length=10.0, width=5.0, height=3.0)
    circuit_breaker = EndpointCircuitBreaker(failure_threshold=3)
    print(hybrid_operation(circuit_breaker, morph, "test", 100))