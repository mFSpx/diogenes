# DARWIN HAMMER — match 47, survivor 3
# gen: 2
# parent_a: voronoi_partition.py (gen0)
# parent_b: hybrid_endpoint_circuit_bre_serpentina_self_righ_m18_s5.py (gen1)
# born: 2026-05-29T23:23:58Z

from __future__ import annotations

import math
import numpy as np
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Tuple

# ----------------------------------------------------------------------
# Utility
# ----------------------------------------------------------------------
def now_z() -> str:
    """Current UTC timestamp in ISO‑8601 Zulu format."""
    from datetime import datetime, timezone
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
    _W_RELIABILITY = 0.30
    _W_DISTANCE = 0.30
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
        if engine_id not in self._breakers:
            raise ValueError(f"Endpoint {engine_id} does not exist")
        self._breakers[engine_id].record_success()

    def record_failure(self, engine_id: str) -> None:
        if engine_id not in self._breakers:
            raise ValueError(f"Endpoint {engine_id} does not exist")
        self._breakers[engine_id].record_failure()

    def hybrid_score(self, endpoint: EngineEndpoint, point: Point, max_dist: float) -> float:
        R = 1.0 if self._breakers[endpoint.engine_id].allow() else 0.0  
        d = euclidean_distance(point, endpoint.seed)
        D_max = max_dist
        P = recovery_priority(endpoint.morphology)
        σ = sphericity_index(*endpoint.morphology.dimensions)
        φ = 1.0 / flatness_index(*endpoint.morphology.dimensions)

        S = (R ** self._W_RELIABILITY) * ((1 - d/D_max) ** self._W_DISTANCE) * \
            (P ** self._W_RECOVERY) * (σ ** self._W_SPHERICITY) * (φ ** self._W_FLATNESS)
        return S

    def hybrid_assign(self, points: List[Point]) -> Dict[Point, str]:
        assignments = {}
        for point in points:
            best_endpoint = None
            best_score = -np.inf
            for endpoint in self.endpoints.values():
                score = self.hybrid_score(endpoint, point, max([euclidean_distance(p, e.seed) for p, e in [(point, endpoint)]] + 
                                                [euclidean_distance(p, e.seed) for p, e in self.endpoints.items()]))
                if score > best_score:
                    best_score = score
                    best_endpoint = endpoint
            assignments[point] = best_endpoint.engine_id if best_endpoint else None
        return assignments

    def hybrid_score_matrix(self, points: List[Point]) -> np.ndarray:
        score_matrix = np.zeros((len(points), len(self.endpoints)))
        for i, point in enumerate(points):
            for j, endpoint in enumerate(self.endpoints.values()):
                score_matrix[i, j] = self.hybrid_score(endpoint, point, 
                                                     max([euclidean_distance(p, e.seed) for p, e in [(point, endpoint)]] + 
                                                         [euclidean_distance(p, e.seed) for p, e in self.endpoints.items()]))
        return score_matrix