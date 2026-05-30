# DARWIN HAMMER — match 4510, survivor 1
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_endpoi_m416_s5.py (gen5)
# parent_b: hybrid_hybrid_hybrid_privac_hybrid_endpoint_circ_m28_s2.py (gen3)
# born: 2026-05-29T23:56:14Z

"""Hybrid Algorithm combining Parent A (network, SSIM, circuit breaker) and Parent B (risk, DP aggregation, sphericity).

Mathematical Bridge:
Both parents expose a linear transformation followed by a non‑linear activation:
- Parent A: `TropicalNetwork.evaluate` computes ReLU( W·x + b ).
- Parent B: risk‑aware resource allocation needs a scalar score; we obtain it by
  aggregating the network output (dot‑product) with a reconstruction‑risk score
  and a differentially‑private aggregate of the same vector.

The hybrid score `S` is defined as  

    S = α * Σ ReLU(W·x + b)               (network contribution)  
        + β * reconstruction_risk_score(q, N)   (privacy risk)  
        + γ * dp_aggregate(x)                 (DP‑smoothed sum)

where α, β, γ are tunable weights.  The resulting scalar drives a circuit‑breaker
decision: if the breaker is closed the allocation proceeds, otherwise it is
blocked.  Morphology enters through a recovery‑priority term that can modulate
α dynamically, linking physical attributes to the computational pathway.

The module implements this fused pipeline and provides three core functions:
`hybrid_network_output`, `compute_hybrid_score`, and `allocate_resources`.
"""

import math
import random
import sys
import pathlib
from dataclasses import dataclass, asdict
from typing import List, Iterable, Any

import numpy as np

# ----------------------------------------------------------------------
# Shared data structures
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Morphology:
    """Geometric description of a physical (or logical) entity."""
    length: float
    width: float
    height: float
    mass: float


@dataclass(frozen=True)
class ModelTier:
    """Lightweight descriptor for a model."""
    name: str
    ram_mb: int
    tier: str


# ----------------------------------------------------------------------
# Parent A components
# ----------------------------------------------------------------------
class TropicalNetwork:
    """Simple ReLU‑activated linear network."""

    def __init__(self, weights: np.ndarray, biases: np.ndarray):
        self.weights = weights  # shape (n_out, n_in)
        self.biases = biases    # shape (n_out,)

    def evaluate(self, input_vector: np.ndarray) -> np.ndarray:
        """ReLU( W·x + b ) element‑wise."""
        lin = self.weights @ input_vector + self.biases
        return np.maximum(0.0, lin)


def ssim(x: List[float], y: List[float],
         dynamic_range: float = 255.0,
         k1: float = 0.01,
         k2: float = 0.03) -> float:
    """Structural similarity index (simplified)."""
    mu_x = np.mean(x)
    mu_y = np.mean(y)
    sigma_x = np.std(x)
    sigma_y = np.std(y)
    sigma_xy = np.mean((np.array(x) - mu_x) * (np.array(y) - mu_y))
    return ((2 * mu_x * mu_y + k1 * dynamic_range ** 2) *
            (2 * sigma_xy + k2 * dynamic_range ** 2) /
            ((mu_x ** 2 + mu_y ** 2 + k1 * dynamic_range ** 2) *
             (sigma_x ** 2 + sigma_y ** 2 + k2 * dynamic_range ** 2)))


class EndpointCircuitBreaker:
    """Keeps track of failure counts and opens/closes accordingly."""

    def __init__(self, failure_threshold: int = 3):
        self.failure_threshold = failure_threshold
        self.failures = 0
        self.open = False
        self.last_event_at = ""

    def record_success(self) -> None:
        self.failures = 0
        self.open = False
        self.last_event_at = str(pathlib.Path(__file__).resolve())

    def record_failure(self) -> None:
        self.failures += 1
        self.open = self.failures >= self.failure_threshold
        self.last_event_at = str(pathlib.Path(__file__).resolve())

    def allow(self) -> bool:
        return not self.open

    def as_dict(self) -> dict[str, Any]:
        return {
            "failure_threshold": self.failure_threshold,
            "failures": self.failures,
            "open": self.open,
            "last_event_at": self.last_event_at,
        }


def compute_recovery_priority(morph: Morphology) -> float:
    """Priority proportional to mass divided by volume."""
    volume = morph.length * morph.width * morph.height
    return morph.mass / volume if volume > 0 else 0.0


# ----------------------------------------------------------------------
# Parent B components
# ----------------------------------------------------------------------
def reconstruction_risk_score(unique_quasi_identifiers: int,
                              total_records: int) -> float:
    """Probability that a record can be re‑identified."""
    if total_records <= 0:
        return 0.0
    ratio = unique_quasi_identifiers / total_records
    return max(0.0, min(1.0, ratio))


def dp_aggregate(values: Iterable[float],
                 epsilon: float = 1.0,
                 sensitivity: float = 1.0) -> float:
    """Differential‑privacy‑style aggregate (mean here, noise omitted)."""
    vals = list(values)
    if not vals:
        return 0.0
    return sum(vals) / len(vals)


def sphericity_index(length: float, width: float, height: float) -> float:
    """Geometric sphericity: geometric mean / max dimension."""
    if min(length, width, height) <= 0:
        raise ValueError("All dimensions must be positive")
    geo_mean = (length * width * height) ** (1.0 / 3.0)
    return geo_mean / max(length, width, height)


# ----------------------------------------------------------------------
# Hybrid core functions (the mathematical fusion)
# ----------------------------------------------------------------------
def hybrid_network_output(network: TropicalNetwork,
                          input_vector: np.ndarray) -> np.ndarray:
    """Compute the ReLU‑activated linear output."""
    return network.evaluate(input_vector)


def compute_hybrid_score(input_vector: np.ndarray,
                         network: TropicalNetwork,
                         morphology: Morphology,
                         unique_qi: int,
                         total_records: int,
                         α: float = 0.5,
                         β: float = 0.3,
                         γ: float = 0.2) -> float:
    """
    Fuse three pillars:

    1. Network contribution – sum of ReLU activations, scaled by a priority
       factor derived from morphology.
    2. Reconstruction risk – bounded probability of re‑identification.
    3. Differential‑privacy aggregate – a smoothed mean of the raw vector.

    The final scalar `S` drives downstream decisions.
    """
    # 1. Network term
    net_out = hybrid_network_output(network, input_vector)
    net_sum = float(net_out.sum())
    priority = compute_recovery_priority(morphology)  # >0 modifies influence
    network_term = α * net_sum * priority

    # 2. Risk term
    risk = reconstruction_risk_score(unique_qi, total_records)
    risk_term = β * risk

    # 3. DP aggregate term
    dp_term = γ * dp_aggregate(input_vector)

    return network_term + risk_term + dp_term


def allocate_resources(hybrid_score: float,
                       circuit_breaker: EndpointCircuitBreaker,
                       max_allocation: float = 100.0) -> float:
    """
    Convert a hybrid score into a concrete resource allocation.

    If the circuit breaker is open, allocation is forced to zero and a failure
    is recorded; otherwise the allocation is a proportion of `max_allocation`
    clipped to the interval [0, max_allocation].
    """
    if not circuit_breaker.allow():
        circuit_breaker.record_failure()
        return 0.0

    # Simple linear mapping with saturation
    allocation = max(0.0, min(max_allocation, hybrid_score))
    circuit_breaker.record_success()
    return allocation


def evaluate_similarity_and_score(x: List[float],
                                 y: List[float],
                                 network: TropicalNetwork,
                                 input_vector: np.ndarray,
                                 morphology: Morphology,
                                 unique_qi: int,
                                 total_records: int) -> dict:
    """
    Demonstrates the interaction of SSIM (from Parent A) with the hybrid score.
    Returns a dictionary useful for downstream logging or telemetry.
    """
    similarity = ssim(x, y)
    hybrid = compute_hybrid_score(input_vector,
                                  network,
                                  morphology,
                                  unique_qi,
                                  total_records)
    return {
        "ssim": similarity,
        "hybrid_score": hybrid,
        "combined_metric": similarity * hybrid,
    }


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Deterministic seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    # Synthetic network (3 outputs, 5 inputs)
    W = np.random.randn(3, 5)
    b = np.random.randn(3)
    net = TropicalNetwork(weights=W, biases=b)

    # Random input vector
    x_vec = np.random.randn(5)

    # Morphology instance
    morph = Morphology(length=2.0, width=1.5, height=1.0, mass=3.0)

    # Circuit breaker
    cb = EndpointCircuitBreaker(failure_threshold=2)

    # Compute hybrid score
    score = compute_hybrid_score(
        input_vector=x_vec,
        network=net,
        morphology=morph,
        unique_qi=42,
        total_records=1000,
        α=0.6,
        β=0.25,
        γ=0.15,
    )
    print(f"Hybrid score: {score:.4f}")

    # Allocate resources based on the score
    allocation = allocate_resources(score, cb, max_allocation=200.0)
    print(f"Allocated resources: {allocation:.2f}")

    # SSIM demo
    a = list(np.random.randint(0, 256, size=10))
    b = list(np.random.randint(0, 256, size=10))
    metrics = evaluate_similarity_and_score(
        x=a,
        y=b,
        network=net,
        input_vector=x_vec,
        morphology=morph,
        unique_qi=42,
        total_records=1000,
    )
    print(f"SSIM: {metrics['ssim']:.4f}, Combined metric: {metrics['combined_metric']:.4f}")

    # Show circuit breaker state
    print("Circuit breaker state:", cb.as_dict())