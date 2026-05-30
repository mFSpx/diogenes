# DARWIN HAMMER — match 3889, survivor 2
# gen: 5
# parent_a: hybrid_hybrid_endpoint_circ_hybrid_voronoi_parti_m745_s0.py (gen2)
# parent_b: hybrid_hybrid_hybrid_fisher_hybrid_hybrid_gliner_m1234_s0.py (gen4)
# born: 2026-05-29T23:52:25Z

"""Hybrid Endpoint‑Voronoi‑Fisher‑Gini (HEVFG) algorithm.

Parents
-------
* **Parent A** – `hybrid_hybrid_endpoint_circ_hybrid_voronoi_parti_m745_s0.py`  
  Provides a health score derived from a circuit‑breaker model and a
  weighted Voronoi partition where the health score modulates the effective
  distance to a seed.

* **Parent B** – `hybrid_hybrid_hybrid_fisher_hybrid_hybrid_gliner_m1234_s0.py`  
  Supplies Fisher information for a Gaussian “beam”, a ternary evidence
  vector, entropy of a weighted histogram, a Gini‑coefficient evaluator and
  a simple structural‑similarity (SSIM‑like) measure.

Mathematical bridge
-------------------
The health score from Parent A is used as a *spatial weight* in the Voronoi
partition.  The resulting region‑wise aggregated health forms a discrete
distribution on which the Parent B statistics (Fisher‑weighted histogram,
entropy, Gini, SSIM) are computed.  The final decision combines these
statistics linearly:


Decision = α·Gini + β·SSIM + γ·Entropy


Thus the spatial reliability (health) influences the statistical decision
through the Voronoi‑derived distribution, achieving a true fusion of the
two topologies.
"""

import math
import random
import sys
from pathlib import Path
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Parent A building blocks
# ----------------------------------------------------------------------
def now_z() -> str:
    """Current UTC timestamp in ISO‑8601 Zulu format."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


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
        self.last_event_at = now_z()

    def record_failure(self) -> None:
        self.failures += 1
        self.open = self.failures >= self.failure_threshold
        self.last_event_at = now_z()

    def allow(self) -> bool:
        """True if the circuit is closed (i.e. endpoint is usable)."""
        return not self.open

    def as_dict(self) -> dict[str, Any]:
        return {
            "failure_threshold": self.failure_threshold,
            "failures": self.failures,
            "open": self.open,
        }


def euclidean(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Standard Euclidean distance."""
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def health_score(cb: EndpointCircuitBreaker, morphology_factor: float) -> float:
    """
    Compute a normalized health score.

    health = (1 - failure_rate) * (1 - morphology_factor)

    * failure_rate  ∈ [0,1]  (failures / threshold)
    * morphology_factor ∈ [0,1]  (higher = less healthy)
    """
    failure_rate = min(cb.failures / max(cb.failure_threshold, 1), 1.0)
    reliability = 1.0 - failure_rate
    morphology = max(0.0, min(morphology_factor, 1.0))
    return reliability * (1.0 - morphology)


def weighted_voronoi_assign(
    points: List[Tuple[float, float]],
    seeds: List[Tuple[float, float]],
    healths: List[float],
) -> List[int]:
    """
    Assign each point to the nearest seed using a *health‑weighted* distance:

        d_eff = euclidean(p, seed) / health

    Higher health reduces the effective distance, expanding the seed's
    region.
    """
    if len(seeds) != len(healths):
        raise ValueError("seeds and healths must have the same length")
    assignments = []
    for pt in points:
        weighted_distances = [
            euclidean(pt, seed) / (h if h > 0 else 1e-9) for seed, h in zip(seeds, healths)
        ]
        assignments.append(int(np.argmin(weighted_distances)))
    return assignments


# ----------------------------------------------------------------------
# Parent B building blocks
# ----------------------------------------------------------------------
def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Gaussian intensity profile."""
    if width <= 0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)


def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    """
    Fisher information for a Gaussian beam.
    I(θ) = ( (∂/∂θ) ln p(θ) )² * p(θ)
    For a Gaussian, this simplifies to ( (θ‑center) / width² )² * intensity.
    """
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = (theta - center) / (width * width) * intensity
    return derivative * derivative / intensity


def ternary_vector(value: float, thresholds: Tuple[float, float] = (-0.33, 0.33)) -> Tuple[int, int, int]:
    """
    Map a scalar to a 3‑element ternary vector.
    Elements are -1, 0, or +1 depending on where `value` lies relative to thresholds.
    """
    low, high = thresholds
    if value < low:
        return (-1, 0, 0)
    elif value > high:
        return (1, 0, 0)
    else:
        return (0, 1, 0)


def gini_coefficient(weights: np.ndarray) -> float:
    """Gini coefficient of a non‑negative weight vector."""
    if weights.size == 0:
        return 0.0
    sorted_w = np.sort(weights)
    n = weights.size
    cumulative = np.cumsum(sorted_w, dtype=float)
    sum_w = cumulative[-1]
    if sum_w == 0:
        return 0.0
    gini = 1.0 - (2.0 / (n - 1)) * (np.sum((n - np.arange(1, n + 1) + 1) * sorted_w) / sum_w)
    return gini


def entropy(weights: np.ndarray, eps: float = 1e-12) -> float:
    """Shannon entropy (base‑2) of a weight distribution."""
    total = np.sum(weights)
    if total == 0:
        return 0.0
    p = weights / total
    p = np.clip(p, eps, 1.0)
    return -np.sum(p * np.log2(p))


def simple_ssim(a: str, b: str) -> float:
    """
    Very lightweight SSIM‑like similarity for strings.
    Returns a value in [0,1] based on the ratio of common characters.
    """
    set_a, set_b = set(a), set(b)
    if not set_a and not set_b:
        return 1.0
    intersect = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersect / union


# ----------------------------------------------------------------------
# Hybrid core functions
# ----------------------------------------------------------------------
def compute_region_weights(
    assignments: List[int],
    healths: List[float],
    num_seeds: int,
) -> np.ndarray:
    """
    Aggregate health scores per Voronoi region.
    Returns an array `w` where w[i] = Σ health of points assigned to seed i.
    """
    w = np.zeros(num_seeds, dtype=float)
    for idx, seed_idx in enumerate(assignments):
        w[seed_idx] += healths[idx]
    return w


def hybrid_fisher_weights(
    theta: float,
    seed_coords: List[Tuple[float, float]],
    width: float = 1.0,
) -> np.ndarray:
    """
    Compute Fisher scores for each seed using its angular coordinate.
    The angle θ_i is derived from the seed position via atan2.
    Returns an array of Fisher scores.
    """
    scores = []
    for x, y in seed_coords:
        angle = math.atan2(y, x)  # range (‑π, π)
        scores.append(fisher_score(theta, angle, width))
    return np.array(scores, dtype=float)


def hybrid_decision(
    region_weights: np.ndarray,
    fisher_weights: np.ndarray,
    packet: str,
    reference: str,
    alpha: float = 0.4,
    beta: float = 0.3,
    gamma: float = 0.3,
) -> float:
    """
    Combine statistics into a single decision value.

    * `region_weights` – health‑aggregated Voronoi histogram.
    * `fisher_weights` – Fisher information per region.
    * The two are multiplied element‑wise to obtain a final weighted histogram.
    * Gini, entropy and a simple SSIM are computed and linearly combined.
    """
    if region_weights.shape != fisher_weights.shape:
        raise ValueError("Shapes of region_weights and fisher_weights must match")
    weighted_hist = region_weights * fisher_weights

    g = gini_coefficient(weighted_hist)
    h = entropy(weighted_hist)
    s = simple_ssim(packet, reference)

    return alpha * g + beta * s + gamma * h


def hybrid_route(
    points: List[Tuple[float, float]],
    seed_coords: List[Tuple[float, float]],
    circuit_breakers: List[EndpointCircuitBreaker],
    morphology_factors: List[float],
    theta: float,
    packet: str,
    reference: str,
) -> float:
    """
    Full hybrid pipeline:

    1. Compute health scores for each endpoint (circuit breaker + morphology).
    2. Perform health‑weighted Voronoi assignment of `points` to `seed_coords`.
    3. Aggregate health per region → `region_weights`.
    4. Compute Fisher scores for each seed → `fisher_weights`.
    5. Produce final decision using Gini, entropy and SSIM.
    """
    if not (len(circuit_breakers) == len(morphology_factors) == len(points)):
        raise ValueError("Length mismatch among points, circuit_breakers, morphology_factors")

    healths = [
        health_score(cb, mf) for cb, mf in zip(circuit_breakers, morphology_factors)
    ]

    assignments = weighted_voronoi_assign(points, seed_coords, healths)
    region_weights = compute_region_weights(assignments, healths, len(seed_coords))
    fisher_weights = hybrid_fisher_weights(theta, seed_coords)

    decision = hybrid_decision(
        region_weights,
        fisher_weights,
        packet,
        reference,
    )
    return decision


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Generate synthetic endpoints
    num_endpoints = 30
    points = [(random.uniform(-10, 10), random.uniform(-10, 10)) for _ in range(num_endpoints)]
    circuit_breakers = [EndpointCircuitBreaker(failure_threshold=3) for _ in range(num_endpoints)]

    # Randomly inject failures to create variance in health
    for cb in circuit_breakers:
        for _ in range(random.randint(0, 4)):
            if random.random() < 0.3:
                cb.record_failure()
            else:
                cb.record_success()

    morphology_factors = [random.random() for _ in range(num_endpoints)]

    # Seed coordinates (e.g., server locations)
    seed_coords = [(-5, -5), (5, -5), (0, 5)]

    # Parameters for the Fisher component
    theta = random.uniform(-math.pi, math.pi)

    # Dummy packet and reference strings
    packet = "payload_xyz_123"
    reference = "payload_xyz_124"

    decision_value = hybrid_route(
        points,
        seed_coords,
        circuit_breakers,
        morphology_factors,
        theta,
        packet,
        reference,
    )

    print(f"Hybrid decision score: {decision_value:.4f}")