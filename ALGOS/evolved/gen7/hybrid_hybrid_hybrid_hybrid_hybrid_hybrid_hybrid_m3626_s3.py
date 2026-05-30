# DARWIN HAMMER — match 3626, survivor 3
# gen: 7
# parent_a: hybrid_hybrid_hybrid_endpoi_shap_attribution_m45_s0.py (gen3)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2286_s3.py (gen6)
# born: 2026-05-29T23:50:56Z

"""Hybrid Fusion of Endpoint Circuit Breaker, Morphology Metrics, and Privacy Risk

Parents:
- hybrid_hybrid_hybrid_endpoi_shap_attribution_m45_s0 (Algorithm A)
- hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2286_s3 (Algorithm B)

Mathematical Bridge:
Both parents expose an `EndpointCircuitBreaker` (failure counter) and a `Morphology` with
geometric indices (`sphericity_index`).  Algorithm B adds a complementary `flatness_index`
and a privacy‑risk estimator (`reconstruction_risk_score`).  Algorithm A intends to weight
SHAP attributions with a health score derived from the circuit breaker.

The fusion therefore constructs a single scalar *system health*:

    H = (1 - failures / failure_threshold)   ∈ [0,1]

and a *shape factor*:

    S = sphericity_index * flatness_index

These two scalars are used as multiplicative modifiers in:
1. A privacy‑risk score (`weighted_reconstruction_risk`).
2. A Shapley‑kernel weighted attribution (`weighted_shap_attribution`).

Thus the core topology is a three‑way product `H * S * base_metric`, unifying the
circuit‑breaker state, morphology, and the original domain‑specific metrics. """

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Any, List, Tuple, Dict

import numpy as np

# ----------------------------------------------------------------------
# Shared data structures (identical across parents)
# ----------------------------------------------------------------------
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
        self.last_event_at = "success"

    def record_failure(self) -> None:
        self.failures += 1
        self.last_event_at = "failure"
        if self.failures >= self.failure_threshold:
            self.open = True

    def health_score(self) -> float:
        """Normalized health in [0,1]; 1 means no failures, 0 means at threshold."""
        return max(0.0, 1.0 - self.failures / self.failure_threshold)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "failure_threshold": self.failure_threshold,
            "failures": self.failures,
            "open": self.open,
            "last_event_at": self.last_event_at,
        }

# ----------------------------------------------------------------------
# Morphology‑derived indices (from both parents)
# ----------------------------------------------------------------------
def sphericity_index(length: float, width: float, height: float) -> float:
    """Ratio of the geometric mean of dimensions to the longest dimension."""
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length * width * height) ** (1.0 / 3.0) / max(length, width, height)


def flatness_index(length: float, width: float, height: float) -> float:
    """Inverse of sphericity using the shortest dimension as reference."""
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length * width * height) ** (1.0 / 3.0) / min(length, width, height)


def combined_shape_factor(morph: Morphology) -> float:
    """Product of sphericity and flatness – a dimensionless shape factor."""
    sph = sphericity_index(morph.length, morph.width, morph.height)
    flat = flatness_index(morph.length, morph.width, morph.height)
    return sph * flat

# ----------------------------------------------------------------------
# Privacy‑risk components (from Parent B)
# ----------------------------------------------------------------------
def reconstruction_risk_score(unique_quasi_identifiers: int, total_records: int) -> float:
    """Normalized reconstruction risk in [0,1]."""
    if total_records <= 0:
        raise ValueError("total_records must be positive")
    return max(0.0, min(1.0, unique_quasi_identifiers / total_records))


def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Gaussian kernel used in Fisher score."""
    if width <= 0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)


def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    """Fisher information for a Gaussian‑like observation."""
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity

# ----------------------------------------------------------------------
# SHAP‑like attribution utilities (from Parent A)
# ----------------------------------------------------------------------
def shapley_kernel_weight(subset_size: int, feature_count: int) -> float:
    """
    Classic Shapley kernel: w(S) = (|S|! * (n-|S|-1)!) / n!
    where |S| is the subset size and n is total number of features.
    """
    if not (0 <= subset_size <= feature_count):
        raise ValueError("subset_size must be between 0 and feature_count")
    numerator = math.factorial(subset_size) * math.factorial(feature_count - subset_size - 1)
    denominator = math.factorial(feature_count)
    return numerator / denominator if denominator != 0 else 0.0


def weighted_shap_attribution(
    base_shap_values: np.ndarray,
    cb: EndpointCircuitBreaker,
    morph: Morphology,
) -> np.ndarray:
    """
    Fuse SHAP values with circuit‑breaker health and morphology shape factor.

    For each feature i:
        w_i = shapley_kernel_weight(i, n_features)
        H   = cb.health_score()
        S   = combined_shape_factor(morph)

    Return: H * S * (w_i * base_shap_values_i)
    """
    if base_shap_values.ndim != 1:
        raise ValueError("base_shap_values must be a 1‑D array")
    n = base_shap_values.shape[0]
    weights = np.array([shapley_kernel_weight(i, n) for i in range(n)], dtype=float)
    health = cb.health_score()
    shape_factor = combined_shape_factor(morph)
    fused = health * shape_factor * (weights * base_shap_values)
    return fused

# ----------------------------------------------------------------------
# Hybrid functions demonstrating the unified system
# ----------------------------------------------------------------------
def weighted_reconstruction_risk(
    unique_quasi_identifiers: int,
    total_records: int,
    cb: EndpointCircuitBreaker,
    morph: Morphology,
) -> float:
    """
    Base reconstruction risk scaled by circuit‑breaker health and shape factor.
    """
    base_risk = reconstruction_risk_score(unique_quasi_identifiers, total_records)
    health = cb.health_score()
    shape_factor = combined_shape_factor(morph)
    return health * shape_factor * base_risk


def morphology_enhanced_fisher(
    theta: float,
    center: float,
    width: float,
    morph: Morphology,
    cb: EndpointCircuitBreaker,
) -> float:
    """
    Fisher score modulated by health and shape factor.
    """
    base_fisher = fisher_score(theta, center, width)
    health = cb.health_score()
    shape_factor = combined_shape_factor(morph)
    return health * shape_factor * base_fisher


def simulate_endpoint_operation(cb: EndpointCircuitBreaker, success_prob: float = 0.7) -> None:
    """
    Randomly records a success or failure according to `success_prob`.
    Demonstrates the dynamic evolution of the health score.
    """
    if random.random() < success_prob:
        cb.record_success()
    else:
        cb.record_failure()


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Initialise shared objects
    cb = EndpointCircuitBreaker(failure_threshold=5)
    morph = Morphology(length=2.0, width=1.0, height=0.5, mass=10.0)

    # Simulate a few endpoint interactions
    for _ in range(7):
        simulate_endpoint_operation(cb, success_prob=0.6)

    print("CircuitBreaker state:", cb.as_dict())
    print("Health score:", cb.health_score())

    # Morphology factors
    print("Sphericity:", sphericity_index(morph.length, morph.width, morph.height))
    print("Flatness:", flatness_index(morph.length, morph.width, morph.height))
    print("Combined shape factor:", combined_shape_factor(morph))

    # Privacy risk example
    risk = weighted_reconstruction_risk(
        unique_quasi_identifiers=42,
        total_records=1000,
        cb=cb,
        morph=morph,
    )
    print("Weighted reconstruction risk:", risk)

    # SHAP attribution example
    base_shap = np.array([0.2, -0.1, 0.05, 0.3])
    fused_shap = weighted_shap_attribution(base_shap, cb, morph)
    print("Fused SHAP attribution:", fused_shap)

    # Fisher score example
    fisher = morphology_enhanced_fisher(theta=0.4, center=0.0, width=1.0, morph=morph, cb=cb)
    print("Morphology‑enhanced Fisher score:", fisher)