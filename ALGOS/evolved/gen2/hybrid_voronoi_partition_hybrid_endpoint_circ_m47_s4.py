# DARWIN HAMMER — match 47, survivor 4
# gen: 2
# parent_a: voronoi_partition.py (gen0)
# parent_b: hybrid_endpoint_circuit_bre_serpentina_self_righ_m18_s5.py (gen1)
# born: 2026-05-29T23:23:58Z

import math
import random
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np

def now_z() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

Point = Tuple[float, float]

def euclidean_distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])

class EndpointCircuitBreaker:
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
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    longest = max(length, width, height)
    return (length * width * height) ** (1.0 / 3.0) / longest

def flatness_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length + width) / (2.0 * height)

def righting_time_index(m: Morphology, b: float = 1.0 / 3.0, k: float = 0.35, neck_lever: float = 1.0) -> float:
    if m.mass <= 0 or neck_lever <= 0:
        raise ValueError("mass and neck_lever must be positive")
    fi = flatness_index(m.length, m.width, m.height)
    exp_part = math.exp(min(k * fi, 50.0))      
    return (m.mass ** b) * exp_part / neck_lever

def recovery_priority(m: Morphology, max_index: float = 10.0) -> float:
    if max_index <= 0:
        raise ValueError("max_index must be positive")
    raw = righting_time_index(m) / max_index
    return max(0.0, min(1.0, raw))

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
    seed: Point                     
    outbound_state: str = "draft_only"

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["morphology"] = asdict(self.morphology)
        return d

class HybridEngineEndpointPool:
    _W_RELIABILITY = 0.35
    _W_DISTANCE = 0.25
    _W_RECOVERY = 0.20
    _W_SPHERICITY = 0.15
    _W_FLATNESS = 0.05

    def __init__(self, failure_threshold: int = 3):
        self._breakers: Dict[str, EndpointCircuitBreaker] = {}
        self.endpoints: Dict[str, EngineEndpoint] = {}
        self._failure_threshold = failure_threshold

    def add_endpoint(self, ep: EngineEndpoint) -> None:
        if ep.engine_id in self.endpoints:
            raise ValueError(f"Endpoint {ep.engine_id} already exists")
        self.endpoints[ep.engine_id] = ep
        self._breakers[ep.engine_id] = EndpointCircuitBreaker(self._failure_threshold)

    def record_success(self, engine_id: str) -> None:
        if engine_id in self._breakers:
            self._breakers[engine_id].record_success()

    def hybrid_score(self, endpoint_id: str, point: Point, max_dist: float) -> float:
        if endpoint_id not in self.endpoints:
            raise ValueError(f"Endpoint {endpoint_id} does not exist")
        ep = self.endpoints[endpoint_id]
        d = euclidean_distance(point, ep.seed)
        R = self._breakers[endpoint_id].allow()
        P = recovery_priority(ep.morphology)
        sigma = sphericity_index(*ep.morphology.dimensions)
        phi = flatness_index(*ep.morphology.dimensions)
        return (R ** self._W_RELIABILITY) * ((1 - d / max_dist) ** self._W_DISTANCE) * (P ** self._W_RECOVERY) * (sigma ** self._W_SPHERICITY) * ((1 / phi) ** self._W_FLATNESS)

    def hybrid_assign(self, points: List[Point], endpoint_ids: List[str]) -> Dict[str, List[Point]]:
        assignments = {endpoint_id: [] for endpoint_id in endpoint_ids}
        max_dists = {endpoint_id: max(euclidean_distance(point, self.endpoints[endpoint_id].seed) for point in points) for endpoint_id in endpoint_ids}
        for point in points:
            scores = {endpoint_id: self.hybrid_score(endpoint_id, point, max_dists[endpoint_id]) for endpoint_id in endpoint_ids}
            best_endpoint_id = max(scores, key=scores.get)
            assignments[best_endpoint_id].append(point)
        return assignments

    def hybrid_score_matrix(self, endpoint_ids: List[str], points: List[Point]) -> np.ndarray:
        scores = np.zeros((len(endpoint_ids), len(points)))
        max_dists = {endpoint_id: max(euclidean_distance(point, self.endpoints[endpoint_id].seed) for point in points) for endpoint_id in endpoint_ids}
        for i, endpoint_id in enumerate(endpoint_ids):
            for j, point in enumerate(points):
                scores[i, j] = self.hybrid_score(endpoint_id, point, max_dists[endpoint_id])
        return scores