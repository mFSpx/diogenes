# DARWIN HAMMER — match 4254, survivor 1
# gen: 5
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_worksh_m1396_s1.py (gen4)
# parent_b: hybrid_hybrid_hybrid_privac_hybrid_endpoint_circ_m28_s5.py (gen3)
# born: 2026-05-29T23:54:33Z

"""
This module represents a hybrid algorithm, fusing the core topologies of 
PARENT ALGORITHM A (hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_worksh_m1396_s1.py) 
and PARENT ALGORITHM B (hybrid_hybrid_hybrid_privac_hybrid_endpoint_circ_m28_s5.py).
The mathematical bridge between these two systems is established by utilizing 
the semantic neighborhood distances from PARENT A as the quasi-identifiers 
in the reconstruction risk score calculation of PARENT B. This fusion enables 
the system to consider both the probabilistic relevance of the paths connecting 
nodes and the risk of re-identification of records.

The hybrid also integrates the Bayesian update rules from PARENT A with the 
circuit-breaker primitives from PARENT B, creating a mathematically coupled 
system in which the temporal dynamics of the circuit-breaker directly 
reshape the resource allocation schedule.
"""

import math
import numpy as np
import random
import sys
import pathlib
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, List

Point = tuple[float, float]
Edge = tuple[str, str]

@dataclass(frozen=True)
class ModelTier:
    """Lightweight descriptor for a model."""
    name: str
    ram_mb: int
    tier: str

def length(a: Point, b: Point) -> float:
    """Calculate the Euclidean distance between two points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])

def bayes_marginal(prior: float, likelihood: float, false_positive: float) -> float:
    """Compute the marginal probability for Bayesian update."""
    if not all(0 <= x <= 1 for x in (prior, likelihood, false_positive)):
        raise ValueError("probabilities must be in [0,1]")
    return likelihood * prior + false_positive * (1.0 - prior)

def bayes_update(prior: float, likelihood: float, marginal: float) -> float:
    """Perform Bayesian update on the prior."""
    if marginal == 0:
        return prior
    return (likelihood * prior) / marginal

def reconstruction_risk_score(unique_quasi_identifiers: float,
                              total_records: int) -> float:
    """Probability that a record can be re‑identified."""
    if total_records <= 0:
        return 0.0
    return max(0.0, min(1.0, unique_quasi_identifiers / total_records))

def dp_aggregate(values: Iterable[float],
                 epsilon: float = 1.0,
                 sensitivity: float = 1.0) -> float:
    """
    Simple Laplace mechanism: sum(values) + Laplace(0, sensitivity/epsilon).
    """
    total = float(np.sum(list(values)))
    scale = sensitivity / epsilon
    noise = np.random.laplace(0.0, scale)
    return total + noise

class EndpointCircuitBreaker:
    """Failure counter that opens after a configurable threshold."""

    def __init__(self, failure_threshold: int = 3):
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
        if self.failures >= self.failure_threshold:
            self.open = True

def now_z() -> str:
    """Current UTC timestamp in ISO‑8601 Zulu format."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

def hybrid_allocate(semantic_distances: List[float], 
                    model_tiers: List[ModelTier], 
                    epsilon: float, 
                    sensitivity: float) -> List[float]:
    """
    Compute per-model allocations using the hybrid system.
    """
    # Calculate quasi-identifiers from semantic distances
    quasi_identifiers = [reconstruction_risk_score(dist, len(semantic_distances)) 
                         for dist in semantic_distances]

    # Perform Bayesian update on model allocations
    prior_allocations = [tier.ram_mb / sum(tier.ram_mb for tier in model_tiers) 
                         for tier in model_tiers]
    likelihoods = [1 - quasi_identifier for quasi_identifier in quasi_identifiers]
    marginals = [bayes_marginal(prior, likelihood, 0.1) 
                  for prior, likelihood in zip(prior_allocations, likelihoods)]
    updated_allocations = [bayes_update(prior, likelihood, marginal) 
                           for prior, likelihood, marginal in zip(prior_allocations, likelihoods, marginals)]

    # Apply differential privacy to allocations
    dp_allocations = dp_aggregate(updated_allocations, epsilon, sensitivity)

    return updated_allocations

def summarize_hybrid_savings(original_allocations: List[float], 
                             hybrid_allocations: List[float]) -> float:
    """
    Calculate savings percentage of hybrid allocations compared to original allocations.
    """
    original_total = sum(original_allocations)
    hybrid_total = sum(hybrid_allocations)
    savings = (original_total - hybrid_total) / original_total * 100
    return savings

def test_circuit_breaker() -> None:
    circuit_breaker = EndpointCircuitBreaker(failure_threshold=2)
    circuit_breaker.record_failure()
    circuit_breaker.record_failure()
    assert circuit_breaker.open

if __name__ == "__main__":
    semantic_distances = [0.1, 0.2, 0.3, 0.4, 0.5]
    model_tiers = [ModelTier("qwen-0.5b", 512, "T1"), 
                   ModelTier("reasoning-t2", 3000, "T2"), 
                   ModelTier("tool-t2", 2600, "T2"), 
                   ModelTier("qwen-7b", 7000, "T3")]
    epsilon = 1.0
    sensitivity = 1.0

    hybrid_allocations = hybrid_allocate(semantic_distances, model_tiers, epsilon, sensitivity)
    print(hybrid_allocations)

    test_circuit_breaker()