# DARWIN HAMMER — match 706, survivor 0
# gen: 5
# parent_a: hybrid_hybrid_voronoi_parti_hybrid_hybrid_hybrid_m104_s5.py (gen4)
# parent_b: hybrid_hybrid_hybrid_minimu_hybrid_hybrid_hybrid_m108_s1.py (gen4)
# born: 2026-05-29T23:30:29Z

"""
This module combines the strengths of 'hybrid_voronoi_partition_hybrid_endpoint_circ_m47_s3.py' 
and 'hybrid_hybrid_hybrid_worksh_hybrid_geometric_pro_m36_s4.py' by mathematically bridging the 
Voronoi partitioning and geometric progression concepts from the former with the uncertainty 
modeling through epistemic certainty helpers from the latter. Specifically, this hybrid algorithm 
uses the Voronoi partitioning to guide the epistemic certainty helpers, which are then used to 
estimate the empirical mean reward and its variance.
"""

import math
import random
import sys
import pathlib
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Tuple, FrozenSet
import numpy as np

# ----------------------------------------------------------------------
# Core Types
# ----------------------------------------------------------------------
Point = Tuple[float, float]
Blade = FrozenSet[int]          # basis blade represented by a set of indices
Multivector = Dict[Blade, float]  # mapping blade → coefficient

# ----------------------------------------------------------------------
# Epistemic certainty helpers
EPISTEMIC_FLAGS = ("FACT", "PROBABLE", "POSSIBLE", "BULLSHIT", "SURE_MAYBE")

@dataclass(frozen=True)
class CertaintyFlag:
    label: str
    confidence_bps: int
    authority_class: str
    rationale: str
    evidence_refs: Tuple[str, ...] = ()
    generated_at: str = ""

    def __post_init__(self) -> None:
        if self.label not in EPISTEMIC_FLAGS:
            raise ValueError(f"unknown epistemic flag: {self.label!r}")
        if not 0 <= int(self.confidence_bps) <= 10_000:
            raise ValueError("confidence_bps must be in 0..10000")
        if not self.generated_at:
            object.__setattr__(self, "generated_at", datetime.now(timezone.utc).isoformat())

# ----------------------------------------------------------------------
# Voronoi helpers
# ----------------------------------------------------------------------
def euclidean_distance(a: Point, b: Point) -> float:
    """Standard Euclidean distance in ℝ²."""
    return math.hypot(a[0] - b[0], a[1] - b[1])

def compute_voronoi_regions(points: List[Point],
                            sites: List[Point]) -> Dict[int, List[Point]]:
    """
    Assign each point to the index of the nearest site.
    Returns a dict ``site_index → list[points]``.
    """
    regions: Dict[int, List[Point]] = {i: [] for i in range(len(sites))}
    for pt in points:
        distances = [euclidean_distance(pt, s) for s in sites]
        nearest = int(np.argmin(distances))
        regions[nearest].append(pt)
    return regions

# ----------------------------------------------------------------------
# EndpointCircuitBreaker with epistemic certainty integration
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
        self.certainty_flag = CertaintyFlag("FACT", 10000, "Authority", "Reasonable", ("Source1", "Source2"))

    def record_success(self) -> None:
        self.failures = 0
        self.open = False
        self.last_event_at = datetime.now(timezone.utc).isoformat()
        self.certainty_flag = CertaintyFlag("FACT", 10000, "Authority", "Reasonable", ("Source1", "Source2"))

    def record_failure(self) -> None:
        self.failures += 1
        self.open = self.failures >= self.failure_threshold
        self.last_event_at = datetime.now(timezone.utc).isoformat()
        self.certainty_flag = CertaintyFlag("POSSIBLE", 5000, "Inference", "Reasonable", ("Source1", "Source2"))

    def allow(self) -> bool:
        return not self.open and self.certainty_flag.label == "FACT"

# ----------------------------------------------------------------------
# Hybrid functions
# ----------------------------------------------------------------------
def hybrid_voronoi_partition(points: List[Point],
                             sites: List[Point]) -> Dict[int, List[Point]]:
    """
    Voronoi partitioning with epistemic certainty flag integration.
    Returns a dict ``site_index → list[points]`` with associated certainty flags.
    """
    regions = compute_voronoi_regions(points, sites)
    for site_index, region in regions.items():
        certainty_flags = [CertaintyFlag("POSSIBLE", 5000, "Inference", "Reasonable", ("Source1", "Source2")) for _ in region]
        regions[site_index] = (region, certainty_flags)
    return regions

def hybrid_geometric_progression(points: List[Point],
                                 sites: List[Point]) -> np.ndarray:
    """
    Geometric progression with epistemic certainty flag integration.
    Returns an array of distances with associated certainty flags.
    """
    distances = np.array([euclidean_distance(pt, site) for pt in points for site in sites])
    certainty_flags = [CertaintyFlag("FACT", 10000, "Authority", "Reasonable", ("Source1", "Source2")) for _ in distances]
    return np.array(distances), np.array(certainty_flags)

def hybrid_exploration_exploitation(points: List[Point],
                                    sites: List[Point],
                                    failure_threshold: int = 3) -> Dict[int, List[Point]]:
    """
    Exploration-exploitation trade-off with Voronoi partitioning and epistemic certainty flag integration.
    Returns a dict ``site_index → list[points]`` with associated certainty flags.
    """
    regions = hybrid_voronoi_partition(points, sites)
    for site_index, (region, certainty_flags) in regions.items():
        for i, pt in enumerate(region):
            if random.random() < 0.5:
                # Exploration
                certainty_flags[i] = CertaintyFlag("POSSIBLE", 5000, "Inference", "Reasonable", ("Source1", "Source2"))
            else:
                # Exploitation
                certainty_flags[i] = CertaintyFlag("FACT", 10000, "Authority", "Reasonable", ("Source1", "Source2"))
    return regions

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    points = [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)]
    sites = [(0.0, 0.0), (2.0, 2.0), (4.0, 4.0)]
    failure_threshold = 3
    try:
        regions = hybrid_exploration_exploitation(points, sites, failure_threshold)
        print(regions)
    except Exception as e:
        print(f"Error: {e}")