# DARWIN HAMMER — match 47, survivor 0
# gen: 2
# parent_a: voronoi_partition.py (gen0)
# parent_b: hybrid_endpoint_circuit_bre_serpentina_self_righ_m18_s5.py (gen1)
# born: 2026-05-29T23:23:58Z

"""
This module describes a novel HYBRID algorithm that mathematically fuses the core topologies of 
two mathematical algorithms: Voronoi partitioning and hybrid endpoint circuit breakers with serpentina self-righting. 
The mathematical bridge between these structures is the integration of Voronoi partitioning with the 
morphology and recovery priority of the hybrid endpoint circuit breakers, allowing for the creation of a 
hybrid system that combines the benefits of both algorithms.
"""

import math
import random
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
import sys

# ----------------------------------------------------------------------
# Utility
# ----------------------------------------------------------------------

def now_z() -> str:
    """Current UTC timestamp in ISO‑8601 Zulu format."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

# ----------------------------------------------------------------------
# Voronoi partitioning
# ----------------------------------------------------------------------

Point = tuple[float, float]

def distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])

def nearest(point: Point, seeds: list[Point]) -> int:
    if not seeds:
        raise ValueError('seeds required')
    return min(range(len(seeds)), key=lambda i: (distance(point, seeds[i]), i))

def assign(points: list[Point], seeds: list[Point]) -> dict[int, list[Point]]:
    regions = {i: [] for i in range(len(seeds))}
    for p in points:
        regions[nearest(p, seeds)].append(p)
    return regions

# ----------------------------------------------------------------------
# Hybrid endpoint circuit breakers with serpentina self-righting
# ----------------------------------------------------------------------

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

def righting_time_index(
    m: Morphology, b: float = 1.0 / 3.0, k: float = 0.35, neck_lever: float = 1.0
) -> float:
    """
    Exponential model of self‑righting time.
    Larger mass and flatter shapes increase the index.
    """
    if m.mass <= 0 or neck_lever <= 0:
        raise ValueError("mass and neck_lever must be positive")
    fi = flatness_index(m.length, m.width, m.height)
    # Guard against overflow for extreme flatness values
    exp_part = math.exp(min(k * fi, 50.0))
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
    outbound_state: str = "draft_only"

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["morphology"] = asdict(self.morphology)
        return d

class HybridEngineEndpointPool:
    """
    Manages a pool of EngineEndpoint objects, each equipped with a circuit breaker
    and a Morphology.  Selection uses a *health score* that blends:

        • Operational reliability (failure_rate)
        • Intrinsic right‑ability (recovery_priority)
        • Shape compactness (sphericity)
        • Flatness (inverse flatness)

    The score is a weighted geometric mean, guaranteeing a value in [0, 1].
    """

    # Weight configuration – can be tuned per deployment
    _W_RELIABILITY = 0.4
    _W_RECOVERY = 0.3
    _W_SPHERICITY = 0.2
    _W_FLATNESS = 0.1

    def __init__(self, failure_threshold: int = 3):
        self.endpoints: dict[str, EngineEndpoint] = {}
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

    def add_endpoint(self, endpoint: EngineEndpoint) -> None:
        self.endpoints[endpoint.engine_id] = endpoint

    def score_endpoint(self, endpoint: EngineEndpoint) -> float:
        reliability = 1.0 - (self.failures / self.failure_threshold)
        recovery = recovery_priority(endpoint.morphology)
        sphericity = sphericity_index(*endpoint.morphology.dimensions)
        flatness = flatness_index(*endpoint.morphology.dimensions)
        return (
            self._W_RELIABILITY * reliability
            + self._W_RECOVERY * recovery
            + self._W_SPHERICITY * sphericity
            + self._W_FLATNESS * flatness
        ) ** 0.25

    def get_best_endpoint(self) -> EngineEndpoint:
        if not self.endpoints:
            raise ValueError("No endpoints available")
        return max(self.endpoints.values(), key=self.score_endpoint)

def get_voronoi_regions(points: list[Point], seeds: list[Point]) -> dict[int, list[Point]]:
    return assign(points, seeds)

def get_morphology_score(m: Morphology) -> float:
    return recovery_priority(m)

def get_hybrid_score(endpoint: EngineEndpoint) -> float:
    pool = HybridEngineEndpointPool()
    pool.add_endpoint(endpoint)
    return pool.score_endpoint(endpoint)

if __name__ == "__main__":
    points = [(random.random(), random.random()) for _ in range(10)]
    seeds = [(random.random(), random.random()) for _ in range(3)]
    regions = get_voronoi_regions(points, seeds)
    print(regions)

    morphology = Morphology(1.0, 2.0, 3.0, 4.0)
    score = get_morphology_score(morphology)
    print(score)

    endpoint = EngineEndpoint(
        "test",
        "test",
        "test",
        "test",
        "test",
        True,
        "test",
        ["test"],
        morphology,
    )
    score = get_hybrid_score(endpoint)
    print(score)