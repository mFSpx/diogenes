# DARWIN HAMMER — match 47, survivor 5
# gen: 2
# parent_a: voronoi_partition.py (gen0)
# parent_b: hybrid_endpoint_circuit_bre_serpentina_self_righ_m18_s5.py (gen1)
# born: 2026-05-29T23:23:58Z

from __future__ import annotations

import math
import sys
from dataclasses import asdict, dataclass, field
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

    def reliability(self) -> float:
        """Return a smooth reliability score in (0, 1]."""
        # Linear decay with a floor to avoid exact zero.
        return max(0.01, 1.0 - self.failures / (self.failure_threshold * 2))

    def allow(self) -> bool:
        """True if the circuit is closed (i.e. endpoint is usable)."""
        return not self.open

    def as_dict(self) -> dict[str, Any]:
        return {
            "failure_threshold": self.failure_threshold,
            "failures": self.failures,
            "open": self.open,
            "last_event_at": self.last_event_at,
            "reliability": self.reliability(),
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
    """Ratio of the geometric mean of dimensions to the longest dimension (≤ 1)."""
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    longest = max(length, width, height)
    return (length * width * height) ** (1.0 / 3.0) / longest


def flatness_index(length: float, width: float, height: float) -> float:
    """Flatness = (length+width) / (2·height); larger ⇒ flatter."""
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length + width) / (2.0 * height)


def righting_time_index(
    m: Morphology,
    b: float = 1.0 / 3.0,
    k: float = 0.35,
    neck_lever: float = 1.0,
) -> float:
    """
    Exponential model of self‑righting time.
    Larger mass and flatter shapes increase the index.
    """
    if m.mass <= 0 or neck_lever <= 0:
        raise ValueError("mass and neck_lever must be positive")
    fi = flatness_index(m.length, m.width, m.height)
    exp_part = math.exp(min(k * fi, 50.0))  # guard overflow
    return (m.mass ** b) * exp_part / neck_lever


def recovery_priority(
    m: Morphology,
    max_index: float = 10.0,
) -> float:
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
    Provides a deeper, numerically stable hybrid health‑distance scoring
    and assignment utilities.
    """

    # ------------------------------------------------------------------
    # Default weight configuration (must sum to 1.0)
    # ------------------------------------------------------------------
    _DEFAULT_WEIGHTS: Tuple[float, float, float, float, float] = (
        0.30,  # reliability
        0.20,  # distance
        0.20,  # recovery priority
        0.20,  # sphericity
        0.10,  # flatness (inverse)
    )

    def __init__(
        self,
        failure_threshold: int = 3,
        *,
        weights: Tuple[float, float, float, float, float] | None = None,
    ) -> None:
        """
        Parameters
        ----------
        failure_threshold:
            Threshold for each endpoint's circuit breaker.
        weights:
            Optional custom weights (reliability, distance, recovery, sphericity,
            flatness). Must be non‑negative and sum to 1.0.
        """
        self._breakers: Dict[str, EndpointCircuitBreaker] = {}
        self.endpoints: Dict[str, EngineEndpoint] = {}
        self._failure_threshold = failure_threshold

        if weights is None:
            self._weights = self._DEFAULT_WEIGHTS
        else:
            if any(w < 0 for w in weights):
                raise ValueError("All weights must be non‑negative")
            if not math.isclose(sum(weights), 1.0, rel_tol=1e-9):
                raise ValueError("Weights must sum to 1.0")
            self._weights = weights

        # Pre‑computed per‑endpoint max distance cache – filled lazily.
        self._max_dist_cache: Dict[str, float] = {}

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
    # Core hybrid scoring
    # ------------------------------------------------------------------
    def _max_distance_for_endpoint(self, ep: EngineEndpoint, points: List[Point]) -> float:
        """Return the maximal Euclidean distance from any point to the endpoint seed."""
        cache_key = ep.engine_id
        if cache_key not in self._max_dist_cache:
            if not points:
                self._max_dist_cache[cache_key] = 0.0
            else:
                dists = [euclidean_distance(p, ep.seed) for p in points]
                self._max_dist_cache[cache_key] = max(dists)
        return self._max_dist_cache[cache_key]

    def hybrid_score(
        self,
        endpoint_id: str,
        point: Point,
        *,
        points_for_normalisation: List[Point] | None = None,
        epsilon: float = 1e-12,
    ) -> float:
        """
        Compute a numerically stable hybrid health‑distance score for a single
        (endpoint, point) pair.

        The score is defined as a weighted *log‑geometric* mean:

        S = exp( Σ w_i · log(term_i + ε) )

        where each term_i ∈ (0, 1] and ε prevents log(0).
        """
        if endpoint_id not in self.endpoints:
            raise KeyError(f"Unknown endpoint {endpoint_id}")

        ep = self.endpoints[endpoint_id]
        breaker = self._breakers[endpoint_id]

        # 1️⃣ Reliability (smooth, never zero)
        rel = breaker.reliability()

        # 2️⃣ Normalised distance term
        max_dist = (
            self._max_distance_for_endpoint(ep, points_for_normalisation or [])
            if points_for_normalisation is not None
            else euclidean_distance(point, ep.seed)  # fallback: self‑normalise
        )
        if max_dist <= 0:
            dist_term = 1.0
        else:
            raw = 1.0 - euclidean_distance(point, ep.seed) / max_dist
            dist_term = max(0.0, min(1.0, raw))

        # 3️⃣ Recovery priority (already in [0,1])
        rec = recovery_priority(ep.morphology)

        # 4️⃣ Sphericity (≤ 1, ≥ 0)
        sph = sphericity_index(*ep.morphology.dimensions)

        # 5️⃣ Flatness inverse – we map flatness ∈ (0, ∞) to a bounded term.
        flat_raw = flatness_index(*ep.morphology.dimensions)
        flat_term = 1.0 / (1.0 + flat_raw)  # maps larger flatness → smaller term

        terms = (rel, dist_term, rec, sph, flat_term)

        # Weighted log‑geometric mean (stable for zero‑ish terms)
        w_rel, w_dist, w_rec, w_sph, w_flat = self._weights
        weights = (w_rel, w_dist, w_rec, w_sph, w_flat)

        log_sum = 0.0
        for term, w in zip(terms, weights):
            log_sum += w * math.log(term + epsilon)

        score = math.exp(log_sum)
        # Clamp final score to [0,1] for safety
        return max(0.0, min(1.0, score))

    # ------------------------------------------------------------------
    # Public API wrappers
    # ------------------------------------------------------------------
    def hybrid_assign(
        self,
        points: List[Point],
        *,
        points_for_normalisation: List[Point] | None = None,
    ) -> Dict[Point, str]:
        """
        Assign each point to the endpoint that maximises the hybrid score.
        Returns a mapping ``point → endpoint_id``.
        """
        if not self.endpoints:
            raise RuntimeError("No endpoints have been added to the pool")

        # Pre‑compute per‑endpoint max distances if a normalisation set is supplied.
        if points_for_normalisation is not None:
            for ep in self.endpoints.values():
                self._max_distance_for_endpoint(ep, points_for_normalisation)

        assignment: Dict[Point, str] = {}
        for pt in points:
            best_id = None
            best_score = -1.0
            for eid in self.endpoints:
                sc = self.hybrid_score(
                    eid,
                    pt,
                    points_for_normalisation=points_for_normalisation,
                )
                if sc > best_score:
                    best_score = sc
                    best_id = eid
            assignment[pt] = best_id  # type: ignore[arg-type]  # point is hashable tuple
        return assignment

    def hybrid_score_matrix(
        self,
        points: List[Point],
        *,
        points_for_normalisation: List[Point] | None = None,
    ) -> np.ndarray:
        """
        Return a ``(len(endpoints), len(points))`` NumPy array where entry
        ``[i, j]`` is the hybrid score of endpoint *i* for point *j*.
        Endpoint order follows ``list(self.endpoints)``.
        """
        ep_ids = list(self.endpoints)
        n_ep = len(ep_ids)
        n_pt = len(points)
        mat = np.empty((n_ep, n_pt), dtype=float)

        # Ensure distance normalisation cache is populated if needed.
        if points_for_normalisation is not None:
            for ep in self.endpoints.values():
                self._max_distance_for_endpoint(ep, points_for_normalisation)

        for i, eid in enumerate(ep_ids):
            for j, pt in enumerate(points):
                mat[i, j] = self.hybrid_score(
                    eid,
                    pt,
                    points_for_normalisation=points_for_normalisation,
                )
        return mat

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------
    def endpoint_status(self) -> Dict[str, Dict[str, Any]]:
        """Return a snapshot of each endpoint's breaker state and morphology."""
        status: Dict[str, Dict[str, Any]] = {}
        for eid, ep in self.endpoints.items():
            status[eid] = {
                "breaker": self._breakers[eid].as_dict(),
                "morphology": asdict(ep.morphology),
                "seed": ep.seed,
            }
        return status

    # ------------------------------------------------------------------
    # Weight manipulation
    # ------------------------------------------------------------------
    @property
    def weights(self) -> Tuple[float, float, float, float, float]:
        """Current weight tuple (reliability, distance, recovery, sphericity, flatness)."""
        return self._weights

    @weights.setter
    def weights(self, new_weights: Tuple[float, float, float, float, float]) -> None:
        if any(w < 0 for w in new_weights):
            raise ValueError("All weights must be non‑negative")
        if not math.isclose(sum(new_weights), 1.0, rel_tol=1e-9):
            raise ValueError("Weights must sum to 1.0")
        self._weights = new_weights
        # Invalidate any cached max‑distance values because the scoring
        # semantics have changed (distance term weight may now dominate).
        self._max_dist_cache.clear()