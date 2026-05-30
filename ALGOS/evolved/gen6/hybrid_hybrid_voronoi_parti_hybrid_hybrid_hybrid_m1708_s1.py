# DARWIN HAMMER — match 1708, survivor 1
# gen: 6
# parent_a: hybrid_voronoi_partition_hybrid_endpoint_circ_m47_s1.py (gen2)
# parent_b: hybrid_hybrid_hybrid_fisher_hybrid_hybrid_hybrid_m594_s0.py (gen5)
# born: 2026-05-29T23:38:22Z

"""Hybrid Voronoi‑Fisher‑Circuit Algorithm
========================================

This module fuses the two parent algorithms:

* **Parent A** – Voronoi partitioning of 2‑D engine endpoints together with a
  simple circuit‑breaker that counts failures.
* **Parent B** – Gaussian‑beam modelling, Fisher information (the *fisher_score*
  function) and a structural‑similarity (SSIM) based similarity router.

**Mathematical bridge**

For every endpoint we treat its *morphology* as a 2‑D point *p* ∈ ℝ².
Endpoints that are close in morphology belong to the same Voronoi region *Rₖ*.
Inside a region we have a set of scalar observations (e.g. angles θ) that are
modelled by a Gaussian beam *g(θ; μ, σ)*.  The Fisher information of this model,
`fisher_score(θ)`, quantifies how much each observation contributes to the
identifiability of the underlying parameters.  By aggregating Fisher scores
within a Voronoi region we obtain a region‑wise “information weight”.  This
weight is fed to the circuit‑breaker: a region whose accumulated Fisher
information exceeds a configurable threshold is considered “over‑stressed” and
its breaker is opened.  The same aggregated information is also used to bias
the SSIM‑based similarity routing – packets are routed to the region whose
information weight best matches the packet’s similarity score.

The three core functions below demonstrate this hybrid workflow:
`region_fisher_scores`, `region_circuit_breakers`, and
`similarity_based_routing_hybrid`.  All calculations rely only on ``numpy``,
the Python standard library and ``math`` as required.

"""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any

import numpy as np

# ----------------------------------------------------------------------
# Voronoi utilities (from Parent A)
# ----------------------------------------------------------------------

Point = Tuple[float, float]


def distance(a: Point, b: Point) -> float:
    """Euclidean distance between two points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def nearest(point: Point, seeds: List[Point]) -> int:
    """Index of the seed closest to *point* (ties broken by index)."""
    if not seeds:
        raise ValueError("seeds required")
    return min(range(len(seeds)), key=lambda i: (distance(point, seeds[i]), i))


def assign(points: List[Point], seeds: List[Point]) -> Dict[int, List[Point]]:
    """Assign each point to the nearest seed, returning region → points."""
    regions = {i: [] for i in range(len(seeds))}
    for p in points:
        regions[nearest(p, seeds)].append(p)
    return regions


# ----------------------------------------------------------------------
# Gaussian beam, Fisher information and SSIM (from Parent B)
# ----------------------------------------------------------------------


def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Gaussian beam intensity g(θ; μ, σ)."""
    if width <= 0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)


def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    """
    Fisher information for the Gaussian beam model.
    I(θ) = (∂g/∂θ)² / g   (regularised with *eps* to avoid division by zero).
    """
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity


def ssim(x: np.ndarray, y: np.ndarray,
         dynamic_range: float = 255.0,
         k1: float = 0.01,
         k2: float = 0.03) -> float:
    """Structural Similarity Index Measure for 1‑D signals."""
    if x.shape != y.shape:
        raise ValueError("samples must have equal shape")
    if x.size == 0:
        raise ValueError("samples must not be empty")
    mx = np.mean(x)
    my = np.mean(y)
    vx = np.var(x)
    vy = np.var(y)
    cov = np.mean((x - mx) * (y - my))
    c1 = (k1 * dynamic_range) ** 2
    c2 = (k2 * dynamic_range) ** 2
    numerator = (2 * mx * my + c1) * (2 * cov + c2)
    denominator = (mx * mx + my * my + c1) * (vx + vy + c2)
    return numerator / denominator


# ----------------------------------------------------------------------
# Circuit breaker (completion of Parent A)
# ----------------------------------------------------------------------


class EndpointCircuitBreaker:
    """
    Simple failure counter.  The breaker opens when the cumulative
    Fisher information of a region exceeds *failure_threshold*.
    """

    def __init__(self, failure_threshold: float = 1.0):
        if failure_threshold <= 0:
            raise ValueError("failure_threshold must be positive")
        self.failure_threshold = failure_threshold
        self.accumulated_score: float = 0.0
        self.open: bool = False

    def record(self, score: float) -> None:
        """Add a Fisher score; open the breaker if the threshold is crossed."""
        if self.open:
            return
        self.accumulated_score += score
        if self.accumulated_score >= self.failure_threshold:
            self.open = True

    def reset(self) -> None:
        """Close the breaker and clear the accumulated score."""
        self.accumulated_score = 0.0
        self.open = False

    def is_open(self) -> bool:
        return self.open


# ----------------------------------------------------------------------
# Hybrid core functions
# ----------------------------------------------------------------------


def region_fisher_scores(
    points: List[Point],
    seeds: List[Point],
    theta_map: Dict[Point, float],
    center: float,
    width: float,
) -> Dict[int, float]:
    """
    Compute the *average* Fisher information for each Voronoi region.

    Parameters
    ----------
    points
        List of endpoint coordinates.
    seeds
        Voronoi seed coordinates (defining the regions).
    theta_map
        Mapping ``point → observed θ``.  If a point is missing, it is ignored.
    center, width
        Parameters of the Gaussian beam model.

    Returns
    -------
    dict
        ``region_index → average_fisher_score`` (0.0 for empty regions).
    """
    regions = assign(points, seeds)
    region_scores: Dict[int, float] = {}
    for idx, pts in regions.items():
        scores = []
        for p in pts:
            theta = theta_map.get(p)
            if theta is not None:
                scores.append(fisher_score(theta, center, width))
        region_scores[idx] = sum(scores) / len(scores) if scores else 0.0
    return region_scores


def region_circuit_breakers(
    region_fisher: Dict[int, float],
    base_threshold: float = 1.0,
) -> Dict[int, EndpointCircuitBreaker]:
    """
    Initialise a circuit breaker per region and feed it the region's Fisher
    score.  The breaker opens immediately if the region's average score exceeds
    ``base_threshold``; otherwise it stays closed.

    Returns a dictionary ``region_index → breaker``.
    """
    breakers: Dict[int, EndpointCircuitBreaker] = {}
    for idx, avg_score in region_fisher.items():
        breaker = EndpointCircuitBreaker(failure_threshold=base_threshold)
        breaker.record(avg_score)  # one‑shot record; in a real system this would be incremental
        breakers[idx] = breaker
    return breakers


def similarity_based_routing_hybrid(
    packet: Dict[str, Any],
    reference_text: str,
    seeds: List[Point],
    points: List[Point],
    center: float,
    width: float,
) -> Dict[str, Any]:
    """
    Route a packet to a Voronoi region using SSIM similarity as a proxy for a
    2‑D coordinate.  The similarity value is interpreted as the *x* coordinate;
    the *y* coordinate is fixed at 0.  The nearest seed determines the target
    region.  The function also returns the breaker state for that region.

    The returned dictionary contains:
        - ``region``: index of the selected region
        - ``similarity``: computed SSIM value
        - ``breaker_open``: bool indicating whether the region's breaker is open
    """
    # Extract textual payload; fall back to empty string.
    text = str(
        packet.get("text_surface")
        or packet.get("raw_command")
        or packet.get("text")
        or ""
    )
    # Convert each character to its Unicode code point (as float) for SSIM.
    # If the string is shorter than the reference, pad with zeros.
    txt_vec = np.array([float(ord(c)) for c in text], dtype=float)
    ref_vec = np.array([float(ord(c)) for c in reference_text], dtype=float)
    min_len = min(txt_vec.size, ref_vec.size)
    if min_len == 0:
        similarity = 0.0
    else:
        similarity = ssim(txt_vec[:min_len], ref_vec[:min_len],
                          dynamic_range=255.0, k1=0.01, k2=0.03)

    # Map similarity (in [0,1]) to a point on the x‑axis.
    routing_point: Point = (similarity, 0.0)

    # Determine the region.
    region_idx = nearest(routing_point, seeds)

    # For demonstration we recompute the region's breaker state on‑the‑fly.
    # In a production system the breakers would be stored globally.
    # Here we reuse the Fisher‑based scores for consistency.
    # Build a dummy theta_map where each point gets a random theta.
    dummy_theta_map = {p: random.uniform(center - 3 * width, center + 3 * width)
                       for p in points}
    region_fisher = region_fisher_scores(points, seeds, dummy_theta_map,
                                         center, width)
    breakers = region_circuit_breakers(region_fisher, base_threshold=0.5)

    return {
        "region": region_idx,
        "similarity": similarity,
        "breaker_open": breakers[region_idx].is_open(),
    }


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------


if __name__ == "__main__":
    # Create a small synthetic scenario.
    random.seed(42)
    np.random.seed(42)

    # 5 Voronoi seeds spread in the unit square.
    seeds = [(random.random(), random.random()) for _ in range(5)]

    # 30 endpoint points.
    points = [(random.random(), random.random()) for _ in range(30)]

    # Assign a random observation θ to each point.
    center_theta = 0.0
    width_theta = 1.0
    theta_map = {p: random.gauss(center_theta, width_theta) for p in points}

    # Compute region Fisher scores.
    region_fisher = region_fisher_scores(
        points, seeds, theta_map, center=center_theta, width=width_theta
    )
    print("Region average Fisher scores:")
    for idx, val in region_fisher.items():
        print(f"  Region {idx}: {val:.4f}")

    # Initialise circuit breakers.
    breakers = region_circuit_breakers(region_fisher, base_threshold=0.2)
    for idx, br in breakers.items():
        print(f"  Region {idx} breaker open? {br.is_open()}")

    # Prepare a dummy packet.
    packet = {
        "text_surface": "HelloWorld",
        "raw_command": None,
        "text": None,
        "normalized_intent": "greet",
    }
    reference = "HelloWorld"

    routing = similarity_based_routing_hybrid(
        packet,
        reference,
        seeds,
        points,
        center=center_theta,
        width=width_theta,
    )
    print("\nRouting result:")
    print(routing)

    # Ensure the script exits cleanly.
    sys.exit(0)