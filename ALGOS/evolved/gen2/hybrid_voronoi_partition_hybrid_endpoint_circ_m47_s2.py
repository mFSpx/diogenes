# DARWIN HAMMER — match 47, survivor 2
# gen: 2
# parent_a: voronoi_partition.py (gen0)
# parent_b: hybrid_endpoint_circuit_bre_serpentina_self_righ_m18_s5.py (gen1)
# born: 2026-05-29T23:23:58Z

"""Hybrid Voronoi‑Circuit‑Morphology Engine.

This module fuses the two parent algorithms:

* **Parent A – Voronoi partition** supplies a Euclidean distance metric and the
  “nearest‑seed” assignment logic.
* **Parent B – EndpointCircuitBreaker & Morphology** supplies a reliability
  counter, a right‑ing (recovery) priority, and shape descriptors
  (sphericity, flatness).

The mathematical bridge is a *hybrid health‑distance score* S defined for an
endpoint *e* and a point *p*:


S(e, p) = (R(e))^w_r   ·   (1 – d(p, c_e)/D_max)^w_d
          · (P(e))^w_p   ·   (σ(e))^w_s   ·   (1/φ(e))^w_f


where  

* `R(e)` – reliability factor (1 if the circuit is closed, 0 otherwise).  
* `d(p, c_e)` – Euclidean distance from *p* to the endpoint’s seed coordinate
  `c_e`.  
* `D_max` – maximal distance among all (point, seed) pairs, used to normalise
  the distance term into `[0, 1]`.  
* `P(e)` – recovery priority from the morphology (higher ⇒ more urgent).  
* `σ(e)` – sphericity index (compact shapes get higher values).  
* `φ(e)` – flatness index (flatter shapes get higher values, we use its inverse).  
* `w_*` – configurable weights that sum to 1.

The score is a weighted geometric mean guaranteeing a value in `[0, 1]`.  
Points are assigned to the endpoint that maximises `S(e, p)`, thus blending
geometric proximity with reliability and morphological fitness.

The module provides three public functions demonstrating the hybrid operation:

* `hybrid_score(endpoint, point, max_dist)` – compute *S* for a single pair.
* `hybrid_assign(points, pool)` – assign a list of points to the best endpoints.
* `hybrid_score_matrix(endpoints, points)` – return a NumPy matrix of all scores.

"""

from __future__ import annotations

import math
import random
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Utility
# ----------------------------------------------------------------------


def now_z() -> str:
    """Current UTC timestamp in ISO‑8601 Zulu format."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


# ----------------------------------------------------------------------
# Parent A – Voronoi helpers
# ----------------------------------------------------------------------


Point = Tuple[float, float]


def euclidean_distance(a: Point, b: Point) -> float:
    """Standard Euclidean distance in ℝ²."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


# ----------------------------------------------------------------------
# Parent B – Circuit‑breaker and Morphology
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
            "last_event_at": self.last_event_at,
        }


@dataclass(frozen=True)
class Morphology:
    """Geometric description of a physical (or logical) entity."""

    length: float
    width: float
    height: float
    mass: float

    def __post_init__(self) -> None:
        for name, value in (
            ("length", self.length),
            ("width", self.width),
            ("height", self.height),
            ("mass", self.mass),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")

    @property
    def dimensions(self) -> Tuple[float, float, float]:
        return self.length, self.width, self.height


def sphericity_index(length: float, width: float, height: float) -> float:
    """Ratio of the geometric mean of dimensions to the longest dimension."""
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    longest = max(length, width, height)
    return (length * width * height) ** (1.0 / 3.0) / longest


def flatness_index(length: float, width: float, height: float) -> float:
    """Flatness = (length+width) / (2·height); larger ⇒ flatter."""
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length + width) / (2.0 * height)


def righting_time_index(m: Morphology, b: float = 1.0 / 3.0,
                        k: float = 0.35, neck_lever: float = 1.0) -> float:
    """
    Exponential model of self‑righting time.
    Larger mass and flatter shapes increase the index.
    """
    if m.mass <= 0 or neck_lever <= 0:
        raise ValueError("mass and neck_lever must be positive")
    fi = flatness_index(m.length, m.width, m.height)
    exp_part = math.exp(min(k * fi, 50.0))      # guard overflow
    return (m.mass ** b) * exp_part / neck_lever


def recovery_priority(m: Morphology, max_index: float = 10.0) -> float:
    """
    Normalised priority (0 → 1) where a larger righting‑time yields a higher
    need for external assistance.
    """
    if max_index <= 0:
        raise ValueError("max_index must be positive")
    raw = righting_time_index(m) / max_index
    return max(0.0, min(1.0, raw))


# ----------------------------------------------------------------------
# Hybrid data structures
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class EngineEndpoint:
    """Endpoint enriched with a Morphology for hybrid scoring."""
    engine_id: str
    channel: str
    residency: str
    runtime: str
    resource_class: str
    always_on: bool
    endpoint: str
    capabilities: List[str]
    morphology: Morphology
    seed: Point                     # geometric seed used for Voronoi‑like distance
    outbound_state: str = "draft_only"

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["morphology"] = asdict(self.morphology)
        return d


class HybridEngineEndpointPool:
    """
    Container that couples each EngineEndpoint with its own circuit breaker.
    Provides the hybrid health‑distance scoring and assignment utilities.
    """

    # ------------------------------------------------------------------
    # Weight configuration (must sum to 1.0)
    # ------------------------------------------------------------------
    _W_RELIABILITY = 0.35
    _W_DISTANCE = 0.25
    _W_RECOVERY = 0.20
    _W_SPHERICITY = 0.15
    _W_FLATNESS = 0.05

    def __init__(self, failure_threshold: int = 3):
        self._breakers: Dict[str, EndpointCircuitBreaker] = {}
        self.endpoints: Dict[str, EngineEndpoint] = {}
        self._failure_threshold = failure_threshold

    # ------------------------------------------------------------------
    # Management
    # ------------------------------------------------------------------
    def add_endpoint(self, ep: EngineEndpoint) -> None:
        if ep.engine_id in self.endpoints:
            raise ValueError(f"Endpoint {ep.engine_id} already exists")
        self.endpoints[ep.engine_id] = ep
        self._breakers[ep.engine_id] = EndpointCircuitBreaker(self._failure_threshold)

    def record_success(self, engine_id: str) -> None:
        self._breakers[engine_id].record_success()

    def record_failure(self, engine_id: str) -> None:
        self._breakers[engine_id].record_failure()

    # ------------------------------------------------------------------
    # Core hybrid metric
    # ------------------------------------------------------------------
    def _reliability_factor(self, engine_id: str) -> float:
        """1.0 if circuit closed, 0.0 if open."""
        return 1.0 if self._breakers[engine_id].allow() else 0.0

    def _sphericity(self, ep: EngineEndpoint) -> float:
        l, w, h = ep.morphology.dimensions
        return sphericity_index(l, w, h)

    def _flatness(self, ep: EngineEndpoint) -> float:
        l, w, h = ep.morphology.dimensions
        return flatness_index(l, w, h)

    def hybrid_score(self, ep: EngineEndpoint, point: Point,
                     max_dist: float) -> float:
        """
        Compute the weighted geometric mean health‑distance score for a single
        endpoint–point pair. Returns a value in [0, 1].
        """
        # reliability (binary but can be extended to a continuous failure rate)
        r = self._reliability_factor(ep.engine_id)

        # normalised proximity term (1 for zero distance, 0 for max_dist)
        d = euclidean_distance(point, ep.seed)
        proximity = 1.0 - (d / max_dist) if max_dist > 0 else 0.0
        proximity = max(0.0, min(1.0, proximity))

        # morphology‑derived components
        p = recovery_priority(ep.morphology)          # already in [0,1]
        sigma = self._sphericity(ep)                  # ≤1, but may exceed 1 for elongated shapes
        sigma = max(0.0, min(1.0, sigma))
        phi = self._flatness(ep)                      # >0
        inv_flat = 1.0 / phi if phi != 0 else 0.0
        inv_flat = max(0.0, min(1.0, inv_flat))

        # Weighted geometric mean
        components = [
            (r, self._W_RELIABILITY),
            (proximity, self._W_DISTANCE),
            (p, self._W_RECOVERY),
            (sigma, self._W_SPHERICITY),
            (inv_flat, self._W_FLATNESS),
        ]
        # Guard against zero components when weight >0 (they zero the whole product)
        product = 1.0
        for value, weight in components:
            # clamp to a tiny epsilon to avoid math domain errors
            val = max(value, 1e-12)
            product *= val ** weight
        return product

    # ------------------------------------------------------------------
    # Public helper APIs
    # ------------------------------------------------------------------
    def assign_points(self, points: List[Point]) -> Dict[str, List[Point]]:
        """
        Assign each point to the endpoint that maximises the hybrid score.
        Returns a dict mapping engine_id → list of assigned points.
        """
        if not self.endpoints:
            raise RuntimeError("No endpoints in the pool")
        # Pre‑compute maximal distance for normalisation
        all_seeds = [ep.seed for ep in self.endpoints.values()]
        max_dist = 0.0
        for p in points:
            for s in all_seeds:
                d = euclidean_distance(p, s)
                if d > max_dist:
                    max_dist = d

        assignments: Dict[str, List[Point]] = {eid: [] for eid in self.endpoints}
        for pt in points:
            best_eid = None
            best_score = -1.0
            for ep in self.endpoints.values():
                sc = self.hybrid_score(ep, pt, max_dist)
                if sc > best_score:
                    best_score = sc
                    best_eid = ep.engine_id
            assignments[best_eid].append(pt)  # type: ignore[arg-type]
        return assignments

    def score_matrix(self, points: List[Point]) -> np.ndarray:
        """
        Return a (n_endpoints × n_points) NumPy array where entry (i, j) is the
        hybrid score of endpoint i for point j.
        """
        if not self.endpoints:
            raise RuntimeError("No endpoints in the pool")
        ep_list = list(self.endpoints.values())
        # Normalise distances once
        max_dist = 0.0
        for p in points:
            for ep in ep_list:
                d = euclidean_distance(p, ep.seed)
                if d > max_dist:
                    max_dist = d

        mat = np.empty((len(ep_list), len(points)), dtype=float)
        for i, ep in enumerate(ep_list):
            for j, pt in enumerate(points):
                mat[i, j] = self.hybrid_score(ep, pt, max_dist)
        return mat


# ----------------------------------------------------------------------
# Convenience top‑level functions (demonstrate hybrid operation)
# ----------------------------------------------------------------------


def hybrid_score(endpoint: EngineEndpoint, point: Point,
                 pool: HybridEngineEndpointPool, max_dist: float) -> float:
    """Thin wrapper around ``HybridEngineEndpointPool.hybrid_score``."""
    return pool.hybrid_score(endpoint, point, max_dist)


def hybrid_assign(points: List[Point], pool: HybridEngineEndpointPool) -> Dict[str, List[Point]]:
    """Assign points using the pool's internal logic."""
    return pool.assign_points(points)


def hybrid_score_matrix(endpoints: List[EngineEndpoint], points: List[Point],
                       pool: HybridEngineEndpointPool) -> np.ndarray:
    """
    Build a temporary pool from the supplied endpoints (no circuit‑breaker state)
    and compute the score matrix. Useful for pure‑mathematical inspection.
    """
    tmp = HybridEngineEndpointPool()
    for ep in endpoints:
        tmp.add_endpoint(ep)
    return tmp.score_matrix(points)


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------


if __name__ == "__main__":
    # Create a few synthetic endpoints with varied morphologies and seeds
    random.seed(42)
    endpoints = [
        EngineEndpoint(
            engine_id=f"engine_{i}",
            channel="http",
            residency="us-east",
            runtime="python3.11",
            resource_class="standard",
            always_on=True,
            endpoint=f"https://api.example.com/{i}",
            capabilities=["compute", "store"],
            morphology=Morphology(
                length=random.uniform(0.5, 2.0),
                width=random.uniform(0.5, 2.0),
                height=random.uniform(0.2, 1.0),
                mass=random.uniform(1.0, 5.0),
            ),
            seed=(random.uniform(0, 10), random.uniform(0, 10)),
        )
        for i in range(4)
    ]

    # Initialise the hybrid pool and add endpoints
    pool = HybridEngineEndpointPool(failure_threshold=2)
    for ep in endpoints:
        pool.add_endpoint(ep)

    # Simulate a few failures on engine_2 to demonstrate circuit‑breaker impact
    pool.record_failure("engine_2")
    pool.record_failure("engine_2")  # opens the circuit (threshold=2)

    # Generate random points in the same 2‑D space
    points = [(random.uniform(0, 10), random.uniform(0, 10)) for _ in range(15)]

    # Perform hybrid assignment
    assignments = hybrid_assign(points, pool)

    # Print a concise report
    for eid, pts in assignments.items():
        print(f"{eid} ← {len(pts)} point(s)")

    # Compute and display the score matrix for inspection
    mat = pool.score_matrix(points)
    print("\nScore matrix (endpoints × points):")
    np.set_printoptions(precision=3, suppress=True)
    print(mat)