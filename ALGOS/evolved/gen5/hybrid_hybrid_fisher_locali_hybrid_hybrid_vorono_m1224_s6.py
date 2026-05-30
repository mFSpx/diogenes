# DARWIN HAMMER — match 1224, survivor 6
# gen: 5
# parent_a: hybrid_fisher_localization_hybrid_hybrid_path_s_m20_s0.py (gen3)
# parent_b: hybrid_hybrid_voronoi_parti_hybrid_hybrid_hybrid_m104_s5.py (gen4)
# born: 2026-05-29T23:34:33Z

"""Hybrid Fisher‑Voronoi‑Path Signature

This module fuses the two parent algorithms:

* **Parent A** – provides a Gaussian beam, Fisher‑information scoring and a
  lead‑lag transform for time‑series paths.
* **Parent B** – supplies Euclidean Voronoi region construction, a circuit‑breaker
  utility and a geometric multivector representation.

**Mathematical bridge** – each Voronoi site is interpreted as a direction
``θ = atan2(y, x)`` in the plane.  The Gaussian beam evaluated at this angle
produces an intensity that, via the Fisher‑information formula,
yields a scalar *weight* for the site.  These weights are used to scale the
centroids of the Voronoi cells, producing a weighted path of region centres.
The lead‑lag transform is then applied to that path and the resulting
trajectory is encoded as a Clifford‑algebra‑like multivector (a mapping
``blade → coefficient``).  The circuit‑breaker guards the computation of the
weighted centroids against pathological inputs.

The three public functions demonstrate the hybrid workflow:


gaussian_beam(...)          → intensity of a directional beam
weighted_voronoi(...)       → Voronoi partition with Fisher‑derived weights
path_to_multivector(...)    → lead‑lag transformed path → multivector


A small smoke test in ``__main__`` exercises the full pipeline."""

import math
import random
import sys
import pathlib
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Tuple, FrozenSet

import numpy as np

# ----------------------------------------------------------------------
# Core Types (from Parent B)
# ----------------------------------------------------------------------
Point = Tuple[float, float]
Blade = FrozenSet[int]          # basis blade represented by a set of indices
Multivector = Dict[Blade, float]  # mapping blade → coefficient

# ----------------------------------------------------------------------
# Parent A – Gaussian beam and Fisher score
# ----------------------------------------------------------------------
def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Gaussian intensity as a function of angle ``theta``."""
    if width <= 0:
        raise ValueError('width must be positive')
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)

def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    """Fisher‑information score for the Gaussian beam."""
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity

# ----------------------------------------------------------------------
# Parent B – Voronoi helpers
# ----------------------------------------------------------------------
def euclidean_distance(a: Point, b: Point) -> float:
    """Standard Euclidean distance in ℝ²."""
    return math.hypot(a[0] - b[0], a[1] - b[1])

def compute_voronoi_regions(points: List[Point],
                            sites: List[Point]) -> Dict[int, List[Point]]:
    """
    Assign each point to the nearest site.
    Returns a dict ``site_index → list[points]``.
    """
    regions: Dict[int, List[Point]] = {i: [] for i in range(len(sites))}
    for pt in points:
        distances = [euclidean_distance(pt, s) for s in sites]
        nearest = int(np.argmin(distances))
        regions[nearest].append(pt)
    return regions

# ----------------------------------------------------------------------
# Parent B – Circuit breaker (used to protect centroid computation)
# ----------------------------------------------------------------------
class EndpointCircuitBreaker:
    """Failure counter that opens after a configurable threshold."""
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
        """True if the circuit is closed (i.e. computation may proceed)."""
        return not self.open

def _now_z() -> str:
    """Simple timestamp placeholder."""
    return "now"

# ----------------------------------------------------------------------
# Lead‑lag transform (Parent A)
# ----------------------------------------------------------------------
def lead_lag_transform(path: np.ndarray) -> np.ndarray:
    """
    Convert a path ``(T, d)`` into a lead‑lag representation of shape
    ``(2*T‑1, 2*d)``.
    """
    path = np.asarray(path, dtype=float)
    if path.ndim != 2:
        raise ValueError("path must be a 2‑D array (time × dimension)")
    T, d = path.shape
    out = np.empty((2 * T - 1, 2 * d), dtype=float)
    for t in range(T - 1):
        out[2 * t]     = np.concatenate([path[t],     path[t]])
        out[2 * t + 1] = np.concatenate([path[t + 1], path[t]])
    out[2 * (T - 1)] = np.concatenate([path[T - 1], path[T - 1]])
    return out

# ----------------------------------------------------------------------
# Hybrid Functions
# ----------------------------------------------------------------------
def weighted_voronoi(points: List[Point],
                    sites: List[Point],
                    beam_center: float = 0.0,
                    beam_width: float = math.pi / 4) -> Tuple[Dict[int, List[Point]], np.ndarray]:
    """
    Compute a Voronoi partition and attach a Fisher‑derived weight to each site.

    For every site we compute the polar angle ``θ = atan2(y, x)`` and evaluate
    ``fisher_score(θ, beam_center, beam_width)``.  The weight is used to scale
    the centroid of the corresponding region.  The ordered list of scaled
    centroids (sorted by site index) is returned as a ``(N, 2)`` NumPy array.

    Returns
    -------
    regions : dict[int, list[Point]]
        Mapping from site index to the points belonging to that region.
    weighted_centroids : np.ndarray, shape (N, 2)
        Scaled centroids; ``N`` is the number of sites.
    """
    # 1️⃣ Voronoi assignment
    regions = compute_voronoi_regions(points, sites)

    # 2️⃣ Circuit‑breaker protects centroid averaging
    breaker = EndpointCircuitBreaker(failure_threshold=5)
    centroids = []
    for idx, site in enumerate(sites):
        pts = regions[idx]
        if not pts:
            # No points – treat centroid as the site itself
            centroid = np.array(site, dtype=float)
        else:
            try:
                centroid = np.mean(np.asarray(pts, dtype=float), axis=0)
                breaker.record_success()
            except Exception:          # pragma: no cover
                breaker.record_failure()
                centroid = np.array(site, dtype=float)

        # 3️⃣ Fisher weight based on angular direction of the site
        theta = math.atan2(site[1], site[0])          # range (‑π, π]
        weight = fisher_score(theta, beam_center, beam_width)
        centroids.append(weight * centroid)

    weighted_centroids = np.vstack(centroids)        # shape (N, 2)
    return regions, weighted_centroids

def path_to_multivector(path: np.ndarray) -> Multivector:
    """
    Encode a lead‑lag transformed path as a multivector.

    Each coordinate of the transformed path contributes a blade consisting of
    the singleton index ``{i}``.  Coefficients are summed over the whole path.
    """
    transformed = lead_lag_transform(path)           # (2T‑1, 2d)
    flat = transformed.reshape(-1)                  # 1‑D view
    mv: Multivector = {}
    for i, coeff in enumerate(flat):
        blade: Blade = frozenset({i})
        mv[blade] = mv.get(blade, 0.0) + float(coeff)
    return mv

def hybrid_fisher_voronoi_signature(text: str,
                                    points: List[Point],
                                    sites: List[Point],
                                    beam_center: float = 0.0,
                                    beam_width: float = math.pi / 4) -> Multivector:
    """
    Full hybrid pipeline:

    1. Extract a dummy numeric feature vector from ``text`` (placeholder).
    2. Compute a weighted Voronoi centroid path.
    3. Concatenate the feature vector to the centroid path to obtain a single
       trajectory.
    4. Return the multivector representation of the lead‑lag transformed trajectory.

    The function demonstrates interaction between the Fisher‑derived weighting,
    the Voronoi geometry, and the algebraic signature.
    """
    # ---- Step 1: dummy feature extraction (scalar list) -----------------
    # In a real scenario this would be a sophisticated NLP pipeline.
    # Here we map each character to its ordinal value and normalise.
    feats = np.fromiter((ord(c) for c in text), dtype=float)
    if feats.size == 0:
        feats = np.array([0.0])
    feats = feats / np.max(feats)                     # scale to [0, 1]

    # ---- Step 2: weighted Voronoi centroids ----------------------------
    _, weighted_centroids = weighted_voronoi(points, sites,
                                            beam_center, beam_width)

    # ---- Step 3: build a single path -----------------------------------
    # Align dimensions: centroids are (N,2); we tile the feature vector to
    # match the number of centroid rows, then concatenate horizontally.
    N = weighted_centroids.shape[0]
    tiled_feats = np.tile(feats[:2], (N, 1))          # ensure at most 2 dims
    trajectory = np.hstack([weighted_centroids, tiled_feats])  # (N, 4)

    # ---- Step 4: multivector signature ---------------------------------
    return path_to_multivector(trajectory)

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Generate random points and sites in the unit square
    rng = np.random.default_rng(42)
    points = [tuple(p) for p in rng.random((200, 2))]
    sites = [tuple(p) for p in rng.random((5, 2))]

    sample_text = "HybridFusion42"

    mv = hybrid_fisher_voronoi_signature(sample_text, points, sites)

    # Print a few blades to verify execution
    print("Multivector sample (first 10 blades):")
    for i, (blade, coeff) in enumerate(mv.items()):
        if i >= 10:
            break
        print(f"  blade {sorted(blade)} : {coeff:.6f}")

    # Simple sanity check: the multivector must contain at least one entry
    assert len(mv) > 0, "Multivector is empty – something went wrong"
    print("Smoke test completed successfully.")