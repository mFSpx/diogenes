# DARWIN HAMMER — match 706, survivor 1
# gen: 5
# parent_a: hybrid_hybrid_voronoi_parti_hybrid_hybrid_hybrid_m104_s5.py (gen4)
# parent_b: hybrid_hybrid_hybrid_minimu_hybrid_hybrid_hybrid_m108_s1.py (gen4)
# born: 2026-05-29T23:30:29Z

"""
This module fuses 'hybrid_hybrid_voronoi_parti_hybrid_hybrid_hybrid_m104_s5.py' 
and 'hybrid_hybrid_hybrid_minimu_hybrid_hybrid_hybrid_m108_s1.py'. The mathematical 
bridge lies in the integration of Voronoi region assignments from the former with 
epistemic certainty helpers from the latter, allowing for a unified system that 
leverages both geometric partitioning and statistical uncertainty modeling.

The mathematical interface between the two parents is the concept of uncertainty 
in region assignments. In the 'hybrid_hybrid_voronoi_parti_hybrid_hybrid_hybrid_m104_s5.py' 
parent, uncertainty is implicitly modeled through the Voronoi region assignments, 
which can be seen as a form of geometric uncertainty. In the 
'hybrid_hybrid_hybrid_minimu_hybrid_hybrid_hybrid_m108_s1.py' parent, uncertainty 
is modeled through epistemic certainty helpers. By combining these two approaches, 
we can create a unified system that leverages both geometric partitioning and 
statistical uncertainty modeling.

Specifically, this hybrid algorithm uses the Voronoi region assignments to guide 
the epistemic certainty helpers, which are then used to estimate the uncertainty 
in the region assignments.
"""

import math
import random
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Iterable, Set
import numpy as np
from pathlib import Path

# ----------------------------------------------------------------------
# Core Types
# ----------------------------------------------------------------------
Point = Tuple[float, float]
Blade = frozenset[int]          # basis blade represented by a set of indices
Multivector = Dict[Blade, float]  # mapping blade → coefficient

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
# Epistemic certainty helpers
# ----------------------------------------------------------------------
def compute_epistemic_certainty(region: List[Point], 
                                sites: List[Point]) -> CertaintyFlag:
    """
    Compute epistemic certainty for a given region.
    """
    num_points = len(region)
    num_sites = len(sites)
    confidence_bps = int((num_points / num_sites) * 10000)
    return CertaintyFlag("POSSIBLE", confidence_bps, "REGION", f"Region with {num_points} points")

# ----------------------------------------------------------------------
# Hybrid functions
# ----------------------------------------------------------------------
def hybrid_voronoi_epistemic(points: List[Point], 
                              sites: List[Point]) -> Dict[int, CertaintyFlag]:
    """
    Compute Voronoi regions and their corresponding epistemic certainties.
    """
    regions = compute_voronoi_regions(points, sites)
    certainties = {i: compute_epistemic_certainty(region, sites) for i, region in regions.items()}
    return certainties

def hybrid_epistemic_voronoi(points: List[Point], 
                              sites: List[Point]) -> List[Tuple[Point, CertaintyFlag]]:
    """
    Compute epistemic certainties for each point in the Voronoi regions.
    """
    regions = compute_voronoi_regions(points, sites)
    certainties = []
    for site_index, region in regions.items():
        certainty = compute_epistemic_certainty(region, sites)
        for point in region:
            certainties.append((point, certainty))
    return certainties

def hybrid_uncertainty_voronoi(points: List[Point], 
                               sites: List[Point]) -> Dict[int, float]:
    """
    Compute uncertainty in Voronoi region assignments.
    """
    regions = compute_voronoi_regions(points, sites)
    uncertainties = {i: 1 - (len(region) / len(points)) for i, region in regions.items()}
    return uncertainties

if __name__ == "__main__":
    points = [(random.uniform(0, 10), random.uniform(0, 10)) for _ in range(100)]
    sites = [(random.uniform(0, 10), random.uniform(0, 10)) for _ in range(5)]
    certainties = hybrid_voronoi_epistemic(points, sites)
    print(certainties)