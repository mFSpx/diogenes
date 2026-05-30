# DARWIN HAMMER — match 1224, survivor 5
# gen: 5
# parent_a: hybrid_fisher_localization_hybrid_hybrid_path_s_m20_s0.py (gen3)
# parent_b: hybrid_hybrid_voronoi_parti_hybrid_hybrid_hybrid_m104_s5.py (gen4)
# born: 2026-05-29T23:34:33Z

"""Hybrid Fisher‑Voronoi Algorithm
=================================

This module fuses the two parent algorithms:

* **Parent A** – ``fisher_localization`` provides a Gaussian beam model,
  a Fisher‑information score and a lead‑lag transform for time‑series paths.
* **Parent B** – ``hybrid_voronoi_parti`` supplies Euclidean Voronoi region
  construction together with a lightweight circuit‑breaker for failure
  handling.

**Mathematical bridge**

The lead‑lag transform turns an input path ``p(t) ∈ ℝᵈ`` into a
``(2T‑1) × 2d`` point cloud.  For each transformed point we compute a
Fisher‑information score ``I(θ)`` where the angle ``θ`` is the polar angle
of the point’s first two coordinates.  The score is used as an *inverse
weight* in a *weighted Euclidean distance* :

``d_w(p, s) = ‖p – s‖₂ / I_s``

where ``I_s`` is the Fisher score of the Voronoi site ``s``.  This yields a
Voronoi partition that favours sites with high Fisher information (i.e.
highly informative directions).  The circuit‑breaker monitors the number
of points that produce non‑finite scores and aborts the partition if a
configurable threshold is exceeded.

The three public functions below illustrate the hybrid workflow:
``gaussian_beam``, ``lead_lag_transform`` and ``weighted_voronoi_partition``.
"""

import math
import random
import sys
import pathlib
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Tuple, FrozenSet

import numpy as np

# ----------------------------------------------------------------------
# Types shared between the parents
# ----------------------------------------------------------------------
Point = Tuple[float, float]
Blade = FrozenSet[int]          # basis blade represented by a set of indices
Multivector = Dict[Blade, float]  # mapping blade → coefficient

# ----------------------------------------------------------------------
# Parent A – Fisher information utilities
# ----------------------------------------------------------------------
def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Gaussian intensity profile evaluated at angle ``theta``."""
    if width <= 0:
        raise ValueError('width must be positive')
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)


def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    """
    Fisher‑information score for a Gaussian beam.
    It is the squared derivative of the log‑likelihood divided by the intensity.
    """
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity


def lead_lag_transform(path: np.ndarray) -> np.ndarray:
    """
    Convert a (T, d) path into a lead‑lag representation of shape
    (2T‑1, 2d).  The transform interleaves the original point and the next
    point, duplicating coordinates to expose directional information.
    """
    path = np.asarray(path, dtype=float)
    if path.ndim != 2:
        raise ValueError("path must be a 2‑dimensional array (T, d)")
    T, d = path.shape
    out = np.empty((2 * T - 1, 2 * d), dtype=float)
    for t in range(T - 1):
        out[2 * t]     = np.concatenate([path[t],     path[t]])
        out[2 * t + 1] = np.concatenate([path[t + 1], path[t]])
    out[2 * (T - 1)] = np.concatenate([path[T - 1], path[T - 1]])
    return out

# ----------------------------------------------------------------------
# Parent B – Voronoi utilities
# ----------------------------------------------------------------------
def euclidean_distance(a: Point, b: Point) -> float:
    """Standard Euclidean distance in ℝ²."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def compute_voronoi_regions(points: List[Point],
                            sites: List[Point]) -> Dict[int, List[Point]]:
    """
    Assign each point to the nearest site (ordinary Euclidean Voronoi).
    Returns a dict ``site_index → list[points]``.
    """
    regions: Dict[int, List[Point]] = {i: [] for i in range(len(sites))}
    for pt in points:
        distances = [euclidean_distance(pt, s) for s in sites]
        nearest = int(np.argmin(distances))
        regions[nearest].append(pt)
    return regions


class EndpointCircuitBreaker:
    """Simple failure counter that opens after a configurable threshold."""

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
        self.last_event_at = _now_z()

    def record_failure(self) -> None:
        self.failures += 1
        self.open = self.failures >= self.failure_threshold
        self.last_event_at = _now_z()

    def allow(self) -> bool:
        """Return ``True`` if the breaker is closed (i.e. operations are allowed)."""
        return not self.open


def _now_z() -> str:
    """ISO‑8601 timestamp with UTC ``Z`` suffix."""
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')


# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------
def _polar_angle(x: float, y: float) -> float:
    """Return angle θ ∈ [−π, π] of the vector (x, y)."""
    return math.atan2(y, x)


def weighted_voronoi_partition(
    transformed_points: np.ndarray,
    center_theta: float = 0.0,
    width_theta: float = 1.0,
    failure_threshold: int = 5
) -> Dict[int, List[Point]]:
    """
    Build a *Fisher‑weighted* Voronoi diagram.

    Parameters
    ----------
    transformed_points : np.ndarray
        Lead‑lag transformed points of shape (N, 2d).  Only the first two
        coordinates are used for spatial location; the remaining coordinates
        are ignored for the Voronoi geometry.
    center_theta, width_theta : float
        Parameters of the Gaussian beam that defines the Fisher score.
    failure_threshold : int
        Maximum number of points that may produce a non‑finite Fisher score
        before the circuit‑breaker aborts the operation.

    Returns
    -------
    Dict[int, List[Point]]
        Mapping from site index to the list of original points assigned to
        that site under the weighted distance ``‖p‑s‖ / I_s``.
    """
    if transformed_points.ndim != 2 or transformed_points.shape[1] < 2:
        raise ValueError("transformed_points must have at least two columns")

    # Extract 2‑D spatial coordinates for Voronoi geometry
    spatial = transformed_points[:, :2]   # shape (N, 2)

    # Choose a subset of points as Voronoi sites – here we simply take every
    # 5th point to keep the example lightweight.
    site_indices = list(range(0, len(spatial), max(1, len(spatial) // 10)))
    sites = [tuple(spatial[i]) for i in site_indices]

    # Compute Fisher scores for each site based on its polar angle
    site_scores: List[float] = []
    breaker = EndpointCircuitBreaker(failure_threshold)

    for s in sites:
        theta = _polar_angle(s[0], s[1])
        try:
            score = fisher_score(theta, center_theta, width_theta)
            if not math.isfinite(score) or score <= 0:
                raise ValueError("non‑finite or non‑positive score")
            site_scores.append(score)
            breaker.record_success()
        except Exception:
            breaker.record_failure()
            site_scores.append(1.0)  # fallback neutral weight
            if not breaker.allow():
                raise RuntimeError("Circuit breaker opened: too many invalid scores")

    # Weighted assignment: distance divided by site score
    regions: Dict[int, List[Point]] = {i: [] for i in range(len(sites))}
    for pt in spatial:
        weighted_distances = [
            euclidean_distance(pt, sites[i]) / site_scores[i] for i in range(len(sites))
        ]
        nearest = int(np.argmin(weighted_distances))
        regions[nearest].append(tuple(pt))

    return regions


def hybrid_fisher_voronoi(path: np.ndarray,
                          center_theta: float = 0.0,
                          width_theta: float = 1.0,
                          failure_threshold: int = 5) -> Dict[int, List[Point]]:
    """
    End‑to‑end hybrid routine:

    1. Apply the lead‑lag transform to the input path.
    2. Build a Fisher‑weighted Voronoi partition of the transformed points.
    3. Return the region mapping.

    Parameters
    ----------
    path : np.ndarray
        Original trajectory of shape (T, d).
    center_theta, width_theta : float
        Parameters for the Gaussian beam used in Fisher scoring.
    failure_threshold : int
        Threshold for the circuit‑breaker.

    Returns
    -------
    Dict[int, List[Point]]
        Voronoi region assignment.
    """
    transformed = lead_lag_transform(path)
    regions = weighted_voronoi_partition(
        transformed,
        center_theta=center_theta,
        width_theta=width_theta,
        failure_threshold=failure_threshold,
    )
    return regions


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Generate a synthetic 2‑D random walk as a test path
    rng = np.random.default_rng(seed=42)
    steps = rng.normal(loc=0.0, scale=1.0, size=(100, 2))
    path = np.cumsum(steps, axis=0)  # shape (100, 2)

    # Run the hybrid algorithm
    try:
        voronoi_regions = hybrid_fisher_voronoi(
            path,
            center_theta=0.0,
            width_theta=0.8,
            failure_threshold=10,
        )
        print(f"Computed {len(voronoi_regions)} Voronoi regions.")
        for idx, pts in voronoi_regions.items():
            print(f"Region {idx}: {len(pts)} points")
    except Exception as exc:
        print(f"Hybrid routine failed: {exc}", file=sys.stderr)
        sys.exit(1)