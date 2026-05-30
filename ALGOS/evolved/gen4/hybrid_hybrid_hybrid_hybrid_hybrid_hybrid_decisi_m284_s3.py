# DARWIN HAMMER — match 284, survivor 3
# gen: 4
# parent_a: hybrid_hybrid_hybrid_endpoi_shap_attribution_m45_s0.py (gen3)
# parent_b: hybrid_hybrid_decision_hygi_decreasing_pruning_m17_s2.py (gen2)
# born: 2026-05-29T23:28:09Z

import math
import random
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

# ----------------------------------------------------------------------
# Core data structures
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Morphology:
    """Geometric description of a physical (or logical) entity."""
    length: float
    width: float
    height: float
    mass: float


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
        self.last_event_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    def record_failure(self) -> None:
        self.failures += 1
        self.open = self.failures >= self.failure_threshold
        self.last_event_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    def allow(self) -> bool:
        """True if the circuit is closed (i.e. endpoint is usable)."""
        return not self.open

    def as_dict(self) -> Dict[str, Any]:
        return {
            "failure_threshold": self.failure_threshold,
            "failures": self.failures,
            "open": self.open,
            "last_event_at": self.last_event_at,
        }


# ----------------------------------------------------------------------
# Mathematical utilities
# ----------------------------------------------------------------------
def sphericity_index(length: float, width: float, height: float) -> float:
    """Ratio of the geometric mean of dimensions to the longest dimension."""
    if min(length, width, height) <= 0:
        raise ValueError("All dimensions must be positive.")
    geo_mean = (length * width * height) ** (1.0 / 3.0)
    return geo_mean / max(length, width, height)


def shapley_kernel_weight(subset_size: int, feature_count: int) -> float:
    """Kernel weight used in exact Shapley value computation."""
    if subset_size < 0 or subset_size > feature_count:
        raise ValueError("Invalid subset size.")
    return 1.0 / (math.comb(feature_count - 1, subset_size) * (feature_count))


def calculate_health_score(morphology: Morphology) -> float:
    """
    Composite health score that blends shape quality with normalized mass.
    The mass term is scaled to keep the score in a comparable range.
    """
    shape_score = sphericity_index(morphology.length, morphology.width, morphology.height)
    # Normalise mass to [0,1] assuming a reasonable upper bound (e.g. 10 kg)
    mass_norm = min(morphology.mass / 10.0, 1.0)
    # Weighted combination – shape is primary, mass is secondary
    return 0.7 * shape_score + 0.3 * mass_norm


def calculate_entropy(feature_vector: List[float]) -> float:
    """
    Shannon entropy of a normalized feature vector.
    Handles zero‑sum vectors gracefully by returning zero entropy.
    """
    total = sum(feature_vector)
    if total == 0:
        return 0.0
    probs = [max(f / total, 0.0) for f in feature_vector]
    # Guard against log(0) by filtering zero probabilities
    return -sum(p * math.log2(p) for p in probs if p > 0)


def calculate_pruning_probability(
    entropy: float,
    t: float,
    alpha: float = 0.1,
    lambda_value: float = 1.0,
) -> float:
    """
    Time‑decaying pruning probability modulated by entropy.
    The denominator uses log2 of the feature count to keep scaling sensible.
    """
    # Avoid division by zero; if feature count is 1, entropy is 0 anyway.
    denominator = 1.0 + entropy
    decay = lambda_value * math.exp(-alpha * t)
    prob = min(1.0, decay) / denominator
    return prob


def shapley_attribution(
    feature_vector: List[float],
    health_score: float,
    circuit_weight: float,
) -> List[float]:
    """
    Approximate Shapley attribution where each feature contribution is
    weighted by the Shapley kernel and modulated by external health and
    circuit‑breaker information.
    """
    n = len(feature_vector)
    if n == 0:
        return []

    # Normalise features to obtain a baseline contribution scale
    total = sum(feature_vector) or 1.0
    normalized = [f / total for f in feature_vector]

    # Compute kernel weights for each possible subset size
    kernel_weights = [shapley_kernel_weight(k, n) for k in range(n)]

    # Aggregate contribution per feature
    attributions = []
    for i, f in enumerate(normalized):
        # Simple linear combination of kernel weights (average) as proxy
        weight = sum(kernel_weights) / n
        contribution = f * weight * health_score * circuit_weight
        attributions.append(contribution)

    # Normalise attributions to sum to health_score * circuit_weight
    scale = (health_score * circuit_weight) / (sum(attributions) or 1.0)
    attributions = [a * scale for a in attributions]
    return attributions


def calculate_hybrid_score(
    feature_vector: List[float],
    morphology: Morphology,
    circuit_breaker: EndpointCircuitBreaker,
    t: float,
) -> Tuple[float, Dict[str, Any]]:
    """
    Deeply integrated hybrid score:
      • health_score from morphology,
      • circuit_weight (1 if closed, 0.5 if open) to penalise failures,
      • entropy‑driven pruning probability,
      • Shapley‑style attribution of features.
    Returns the scalar hybrid score and a diagnostic dictionary.
    """
    health = calculate_health_score(morphology)

    # Circuit breaker penalty: half score when open, full otherwise
    circuit_weight = 1.0 if circuit_breaker.allow() else 0.5

    entropy = calculate_entropy(feature_vector)
    pruning_prob = calculate_pruning_probability(entropy, t)

    # Core hybrid metric
    raw_score = health * circuit_weight * (1.0 - pruning_prob)

    # Detailed attributions for interpretability
    attributions = shapley_attribution(feature_vector, health, circuit_weight)

    diagnostics = {
        "health_score": health,
        "circuit_weight": circuit_weight,
        "entropy": entropy,
        "pruning_probability": pruning_prob,
        "raw_score": raw_score,
        "shapley_attributions": attributions,
        "circuit_breaker_state": circuit_breaker.as_dict(),
    }

    return raw_score, diagnostics


# ----------------------------------------------------------------------
# Example usage
# ----------------------------------------------------------------------
def main() -> None:
    # Example morphology
    morphology = Morphology(length=1.2, width=0.9, height=1.0, mass=2.5)

    # Example feature vector (e.g., sensor readings, model inputs)
    feature_vector = [0.15, 0.35, 0.5]

    # Initialise circuit breaker
    circuit_breaker = EndpointCircuitBreaker(failure_threshold=3)

    # Simulate a failure to demonstrate penalty handling
    circuit_breaker.record_failure()

    # Compute hybrid score at time t = 2.0
    score, details = calculate_hybrid_score(
        feature_vector=feature_vector,
        morphology=morphology,
        circuit_breaker=circuit_breaker,
        t=2.0,
    )

    print("Hybrid Score:", score)
    for key, value in details.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()