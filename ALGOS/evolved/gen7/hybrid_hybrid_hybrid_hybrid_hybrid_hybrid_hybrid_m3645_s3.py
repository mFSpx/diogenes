# DARWIN HAMMER — match 3645, survivor 3
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2344_s1.py (gen6)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_endpoi_m416_s3.py (gen5)
# born: 2026-05-29T23:51:09Z

"""
Hybrid Algorithm: Fusion of Hoeffding‑Fisher Certainty Framework (Parent A) with
Tropical Max‑Plus Endpoint Evaluation (Parent B).

Mathematical Bridge
-------------------
* The Hoeffding bound ε = sqrt(R²·ln(1/δ)/(2n)) provides a confidence radius for
  statistical estimates.  In Parent B the tropical max‑plus algebra uses the
  tropical addition `⊕ = max` and tropical multiplication `⊗ = +`.  We map the
  inverse confidence radius (1/ε) onto the tropical semiring by treating it as
  a “weight” that can be combined with other tropical scores via `max`.

* Fisher information `I(θ)` (Parent A) is a measure of certainty for a continuous
  parameter.  We embed it into the tropical network by converting it to a
  tropical log‑weight `log(I)` and then applying the tropical addition with the
  (inverse) Hoeffding bound.

* The pheromone‑driven pruning probability of Parent A is interpreted as a
  dynamic adjustment of the Hoeffding error probability `δ`.  Capability flags
  of an `EngineEndpoint` act as pheromone intensities; higher‑rank capabilities
  reduce `δ`, tightening the Hoeffding bound.

* The resulting tropical‑combined certainty `C_trop = max(log(I), -log(ε))`
  multiplies (tropical multiplication = addition) the output of the
  TropicalNetwork, yielding a unified score that simultaneously reflects
  statistical certainty, information‑theoretic certainty, and the geometric /
  reliability model of Parent B.

The code below implements this fusion and provides three core functions that
demonstrate the hybrid operation.
"""

import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np


# ----------------------------------------------------------------------
# Parent A building blocks
# ----------------------------------------------------------------------
def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Gaussian intensity I(θ) of a beam centred at *center* with *width*."""
    if width <= 0.0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)


def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    """Fisher information for a single angle θ."""
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity


def hoeffding_bound(range_: float, delta: float, n: int) -> float:
    """Hoeffding bound ε = sqrt( (R² * ln(1/δ)) / (2n) )."""
    if range_ <= 0 or not (0 < delta < 1) or n <= 0:
        raise ValueError("range_ > 0, 0 < delta < 1, n > 0")
    return math.sqrt((range_ * range_ * math.log(1 / delta)) / (2 * n))


def epistemic_flags(text: str) -> float:
    """
    Very light‑weight epistemic certainty estimator.
    Counts certainty‑related tokens and returns a normalized score in [0,1].
    """
    tokens = {"certain", "definite", "sure", "guarantee", "proof", "evidence"}
    words = set(text.lower().split())
    return len(tokens & words) / len(tokens)


# ----------------------------------------------------------------------
# Parent B building blocks
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float


@dataclass(frozen=True)
class EngineEndpoint:
    engine_id: str
    channel: str
    residency: str
    runtime: str
    resource_class: str
    always_on: bool
    endpoint: str
    capabilities: List[str]
    morphology: Morphology
    outbound_state: str = "draft_only"


class TropicalNetwork:
    """
    Simple tropical max‑plus network.
    Tropical addition = max, tropical multiplication = +
    The forward pass computes max(0, w·x + b) element‑wise.
    """

    def __init__(self, weights: np.ndarray, biases: np.ndarray):
        if weights.shape[0] != biases.shape[0]:
            raise ValueError("weights and biases must have same output dimension")
        self.weights = weights
        self.biases = biases

    def evaluate(self, input_vector: np.ndarray) -> np.ndarray:
        output = np.empty_like(input_vector)
        for i in range(len(input_vector)):
            output[i] = max(0.0, float(np.dot(self.weights[i], input_vector) + self.biases[i]))
        return output


class EndpointCircuitBreaker:
    """
    Classic circuit‑breaker: opens after `failure_threshold` consecutive failures.
    """

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
        self.last_event_at = "success"

    def record_failure(self) -> None:
        self.failures += 1
        self.last_event_at = "failure"
        if self.failures >= self.failure_threshold:
            self.open = True

    def status(self) -> str:
        return "OPEN" if self.open else "CLOSED"


# ----------------------------------------------------------------------
# Hybrid core functions (mathematical bridge)
# ----------------------------------------------------------------------
def pheromone_delta(capabilities: List[str]) -> float:
    """
    Derive a Hoeffding error probability δ from capability “pheromones”.
    More capabilities → stronger pheromone → smaller δ (tighter bound).
    """
    base_delta = 0.25
    # Each capability reduces δ by a factor of 0.9, but never below 1e-4
    factor = 0.9 ** len(capabilities)
    return max(1e-4, base_delta * factor)


def combined_tropical_certainty(
    theta: float,
    center: float,
    width: float,
    range_: float,
    delta: float,
    n: int,
) -> float:
    """
    Compute a tropical‑combined certainty score C_trop.

    Steps:
    1. Fisher information I(θ).
    2. Hoeffding bound ε.
    3. Convert to log‑scale: log_I = log(I), log_eps = -log(ε) (larger is better).
    4. Tropical addition (max) yields log‑certainty.
    5. Exponentiate back to a linear weight.
    """
    I = fisher_score(theta, center, width)
    eps = hoeffding_bound(range_, delta, n)

    # Protect against non‑positive values
    log_I = math.log(max(I, 1e-12))
    log_eps = -math.log(max(eps, 1e-12))

    # Tropical addition (max)
    log_certainty = max(log_I, log_eps)

    return math.exp(log_certainty)  # linear weight ≥ 1


def evaluate_endpoint(
    endpoint: EngineEndpoint,
    network: TropicalNetwork,
    input_vector: np.ndarray,
    theta: float,
    center: float,
    width: float,
    range_: float,
    n: int,
) -> np.ndarray:
    """
    Hybrid evaluation:
    * Compute a dynamic δ from endpoint capabilities (pheromone effect).
    * Obtain a tropical certainty weight via `combined_tropical_certainty`.
    * Run the tropical network and scale its output by the weight.
    """
    delta = pheromone_delta(endpoint.capabilities)
    weight = combined_tropical_certainty(theta, center, width, range_, delta, n)
    raw_output = network.evaluate(input_vector)
    return raw_output * weight


def circuit_breaker_logic(cb: EndpointCircuitBreaker, success: bool) -> str:
    """
    Update the circuit breaker based on success/failure and return its status.
    If the breaker is OPEN we treat the endpoint as unavailable and force a reset
    on the next successful call.
    """
    if cb.open:
        # If open, ignore current outcome but allow a forced reset on success
        if success:
            cb.record_success()
    else:
        if success:
            cb.record_success()
        else:
            cb.record_failure()
    return cb.status()


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create a dummy morphology and endpoint
    morph = Morphology(length=1.2, width=0.5, height=0.8, mass=3.4)
    endpoint = EngineEndpoint(
        engine_id="eng-001",
        channel="alpha",
        residency="us-east",
        runtime="python3.11",
        resource_class="standard",
        always_on=True,
        endpoint="http://localhost:8000",
        capabilities=["cpu", "gpu", "ssd"],
        morphology=morph,
    )

    # Simple tropical network (2‑dimensional)
    w = np.array([[0.6, -0.2], [0.1, 0.9]])
    b = np.array([0.05, -0.03])
    net = TropicalNetwork(weights=w, biases=b)

    # Input vector
    x = np.array([0.7, 0.3])

    # Parameters for the statistical side
    theta = 0.45
    center = 0.5
    width = 0.1
    range_ = 1.0
    n = 150

    # Evaluate endpoint
    out = evaluate_endpoint(
        endpoint,
        net,
        x,
        theta,
        center,
        width,
        range_,
        n,
    )
    print("Hybrid output vector:", out)

    # Circuit breaker demo
    cb = EndpointCircuitBreaker(failure_threshold=2)
    # Simulate a failure, then success
    print("CB status (init):", cb.status())
    print("CB status (failure):", circuit_breaker_logic(cb, success=False))
    print("CB status (failure again):", circuit_breaker_logic(cb, success=False))
    print("CB status (success, should reset):", circuit_breaker_logic(cb, success=True))
    print("Final CB status:", cb.status())