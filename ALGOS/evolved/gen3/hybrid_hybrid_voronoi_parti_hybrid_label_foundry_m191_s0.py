# DARWIN HAMMER — match 191, survivor 0
# gen: 3
# parent_a: hybrid_voronoi_partition_hybrid_endpoint_circ_m47_s0.py (gen2)
# parent_b: hybrid_label_foundry_hybrid_endpoint_circ_m5_s0.py (gen2)
# born: 2026-05-29T23:25:57Z

"""
This module describes a novel HYBRID algorithm that mathematically fuses the core topologies of 
two mathematical algorithms: Voronoi partitioning and hybrid endpoint circuit breakers with serpentina self-righting 
and weak supervision labeling primitives. 
The mathematical bridge between these structures is the concept of "recovery priority," 
which is used to determine the likelihood of an endpoint recovering from a failure. 
This recovery priority is calculated based on the morphology of the endpoint, 
and this value is then used to adjust the Voronoi partitioning's distance calculation to ensure robust labeling and endpoint management.
"""

import math
import random
import sys
from pathlib import Path
from time import time

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

import numpy as np

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
    return min(range(len(seeds)), key=lambda i: (distance(point, seeds[i]) * (1 + m.recovery_priority()), i))

def assign(points: list[Point], seeds: list[Morphology]) -> dict[int, list[Point]]:
    regions = {i: [] for i in range(len(seeds))}
    for p in points:
        regions[nearest(p, seeds)].append(p)
    return regions

# ----------------------------------------------------------------------
# Morphology and recovery priority
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class Morphology:
    """Geometric description of a physical (or logical) entity."""
    length: float
    width: float
    height: float
    mass: float

    def recovery_priority(self) -> float:
        """Calculates the recovery priority of the endpoint based on its morphology."""
        fi = flatness_index(self.length, self.width, self.height)
        si = sphericity_index(self.length, self.width, self.height)
        return exp(-fi * si)

# ----------------------------------------------------------------------
# Hybrid endpoint circuit breakers with serpentina self-righting and labeling primitives
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class LabelingFunctionResult:
    lf_name: str
    doc_id: str
    label: int


def labeling_function(name: str | None = None):
    def deco(fn: Callable[[Dict[str, Any]], int]):
        fn.lf_name = name or fn.__name__
        return fn

    return deco


def sphericity_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length * width * height) ** (1.0 / 3.0) / length


def flatness_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length + width) / (2.0 * height)


def righting_time_index(m: Morphology, b: float = 1.0 / 3.0, k: float = 0.35, neck_lever: float = 1.0) -> float:
    if m.mass <= 0 or neck_lever <= 0:
        raise ValueError("mass and neck_lever must be positive")
    fi = flatness_index(m.length, m.width, m.height)
    return 1 / (fi + 1)


# ----------------------------------------------------------------------
# Hybrid functions
# ----------------------------------------------------------------------

def hybrid_assign(points: list[Point], seeds: list[Morphology]) -> dict[int, list[Point]]:
    regions = assign(points, seeds)
    for region in regions.values():
        for point in region:
            point_morph = next((m for m in seeds if m.recovery_priority() == 1), Morphology(1, 1, 1, 1))
            point['recovery_priority'] = point_morph.recovery_priority()
    return regions


def hybrid_label(points: list[Point], seeds: list[Morphology]) -> dict[int, List[Tuple[float, float]]]:
    regions = hybrid_assign(points, seeds)
    labeled_regions = {}
    for region in regions.values():
        labels = [point['recovery_priority'] for point in region]
        labeled_regions[len(region)] = labels
    return labeled_regions


def hybrid_break(points: list[Point], seeds: list[Morphology]) -> dict[int, List[Tuple[float, float]]]:
    regions = assign(points, seeds)
    breaking_regions = {}
    for region in regions.values():
        distances = [distance(seeds[i], point) for i, point in enumerate(region)]
        breaking_regions[len(region)] = distances
    return breaking_regions


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------

if __name__ == "__main__":
    morphology = Morphology(5, 5, 5, 10)
    seeds = [morphology] * 10
    points = [(i, 0) for i in range(-10, 10)]
    assigned_regions = hybrid_assign(points, seeds)
    labeled_regions = hybrid_label(points, seeds)
    breaking_regions = hybrid_break(points, seeds)
    for i in assigned_regions.values():
        print(i)
    for i in labeled_regions.values():
        print(i)
    for i in breaking_regions.values():
        print(i)