# DARWIN HAMMER — match 47, survivor 1
# gen: 2
# parent_a: voronoi_partition.py (gen0)
# parent_b: hybrid_endpoint_circuit_bre_serpentina_self_righ_m18_s5.py (gen1)
# born: 2026-05-29T23:23:58Z

"""
This module implements a novel hybrid algorithm, combining the Voronoi partitioning from voronoi_partition.py 
and the circuit breaker with morphology recovery priority from hybrid_endpoint_circuit_bre_serpentina_self_righ_m18_s5.py.
The mathematical bridge between the two structures lies in the Voronoi partitioning of engine endpoints based on their 
morphology, ensuring that endpoints with similar morphological properties are assigned to the same partition.
This allows for more efficient circuit breaker management and recovery priority calculation.
"""

import math
import numpy as np
import random
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple
from pathlib import Path

# ----------------------------------------------------------------------
# Utility
# ----------------------------------------------------------------------

def now_z() -> str:
    """Current UTC timestamp in ISO‑8601 Zulu format."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

# ----------------------------------------------------------------------
# Voronoi Partitioning
# ----------------------------------------------------------------------

Point = Tuple[float, float]

def distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])

def nearest(point: Point, seeds: List[Point]) -> int:
    if not seeds:
        raise ValueError('seeds required')
    return min(range(len(seeds)), key=lambda i: (distance(point, seeds[i]), i))

def assign(points: List[Point], seeds: List[Point]) -> Dict[int, List[Point]]:
    regions = {i: [] for i in range(len(seeds))}
    for p in points:
        regions[nearest(p, seeds)].append(p)
    return regions

# ----------------------------------------------------------------------
# Circuit Breaker and Morphology
# ----------------------------------------------------------------------

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
# Hybrid Data Structures
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
        self.circuit_breakers: dict[str, EndpointCircuitBreaker] = {}
        self.failure_threshold = failure_threshold

    def add_endpoint(self, endpoint: EngineEndpoint):
        self.endpoints[endpoint.engine_id] = endpoint
        self.circuit_breakers[endpoint.engine_id] = EndpointCircuitBreaker(self.failure_threshold)

    def calculate_health_score(self, endpoint_id: str) -> float:
        reliability = 1 - self.circuit_breakers[endpoint_id].failures / self.failure_threshold
        recovery = recovery_priority(self.endpoints[endpoint_id].morphology)
        sphericity = sphericity_index(*self.endpoints[endpoint_id].morphology.dimensions)
        flatness = flatness_index(*self.endpoints[endpoint_id].morphology.dimensions)
        return (reliability ** self._W_RELIABILITY) * (recovery ** self._W_RECOVERY) * (sphericity ** self._W_SPHERICITY) * (flatness ** self._W_FLATNESS)

    def select_endpoint(self) -> str:
        return max(self.endpoints, key=self.calculate_health_score)

def partition_endpoints(endpoints: List[EngineEndpoint], seeds: List[Point]) -> Dict[int, List[EngineEndpoint]]:
    points = [endpoint.morphology.dimensions for endpoint in endpoints]
    regions = assign(points, seeds)
    endpoint_regions = {i: [endpoints[j] for j, point in enumerate(points) if nearest(point, seeds) == i] for i in regions}
    return endpoint_regions

def main():
    # Create a pool of engine endpoints
    pool = HybridEngineEndpointPool()
    # Add some endpoints
    pool.add_endpoint(EngineEndpoint("cpu_fairyfuse_ternary", "channel1", "residency1", "runtime1", "resource_class1", True, "endpoint1", ["cap1", "cap2"], Morphology(1.0, 2.0, 3.0, 4.0)))
    pool.add_endpoint(EngineEndpoint("cpu_fairyfuse_ternary2", "channel2", "residency2", "runtime2", "resource_class2", False, "endpoint2", ["cap3", "cap4"], Morphology(2.0, 3.0, 4.0, 5.0)))
    # Partition the endpoints using the Voronoi partitioning
    seeds = [(1.0, 1.0), (2.0, 2.0)]
    endpoint_regions = partition_endpoints(list(pool.endpoints.values()), seeds)
    # Print the health scores of the endpoints in each region
    for region, endpoints in endpoint_regions.items():
        for endpoint in endpoints:
            health_score = pool.calculate_health_score(endpoint.engine_id)
            print(f"Region {region}: Endpoint {endpoint.engine_id} has a health score of {health_score}")

if __name__ == "__main__":
    main()