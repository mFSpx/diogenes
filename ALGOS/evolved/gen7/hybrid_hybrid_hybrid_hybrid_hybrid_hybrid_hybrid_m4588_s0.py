# DARWIN HAMMER — match 4588, survivor 0
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_cockpit_metri_m95_s2.py (gen4)
# parent_b: hybrid_hybrid_hybrid_vorono_hybrid_hybrid_hybrid_m1529_s4.py (gen6)
# born: 2026-05-29T23:56:39Z

"""
Module docstring:
This module implements a novel HYBRID algorithm that fuses the core topologies of two parent algorithms:
- PARENT ALGORITHM A (hybrid_hybrid_hybrid_hybrid_hybrid_cockpit_metri_m95_s2.py): a labeling algorithm with anti-slop ratio and cockpit honesty metrics.
- PARENT ALGORITHM B (hybrid_hybrid_hybrid_vorono_hybrid_hybrid_hybrid_m1529_s4.py): a Voronoi diagram-based algorithm for computing adjacency matrices.
The mathematical bridge between these two algorithms lies in the use of geometric distances and proximity metrics.
In the hybrid algorithm, we use the Voronoi diagram to assign labels to points based on their proximity to site points, while incorporating the anti-slop ratio and cockpit honesty metrics to refine the labeling process.
"""

import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Tuple, Callable
import math
import random
import sys
from pathlib import Path

@dataclass(frozen=True)
class LabelingFunctionResult: 
    lf_name: str 
    doc_id: str 
    label: int 

@dataclass(frozen=True)
class ProbabilisticLabel: 
    doc_id: str 
    label: int 
    confidence: float 

Point = Tuple[float, float]
Blade = frozenset[int]          # basis blade represented by a set of indices
Multivector = Dict[Blade, float]  # mapping blade → coefficient

def euclidean_distance(a: Point, b: Point) -> float:
    """Euclidean distance in ℝ²."""
    return math.hypot(a[0] - b[0], a[1] - b[1])

def compute_voronoi_regions(points: List[Point], sites: List[Point]) -> Dict[int, List[Point]]:
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

def build_adjacency_from_regions(regions: Dict[int, List[Point]], sites: List[Point], radius_factor: float = 1.5) -> np.ndarray:
    """
    Construct a simple weighted adjacency matrix for the Voronoi graph.
    Two sites are connected if the Euclidean distance between them is
    smaller than ``radius_factor`` times the median inter‑site distance.
    Edge weight = 1 / distance (higher weight for closer neighbours).
    """
    n = len(sites)
    adj = np.zeros((n, n), dtype=float)

    # median inter‑site distance as a scale
    dists = []
    for i in range(n):
        for j in range(i + 1, n):
            dists.append(euclidean_distance(sites[i], sites[j]))
    median_dist = np.median(dists) if dists else 1.0

    for i in range(n):
        for j in range(i + 1, n):
            dist = euclidean_distance(sites[i], sites[j])
            if dist < radius_factor * median_dist:
                adj[i, j] = 1.0 / dist
                adj[j, i] = 1.0 / dist

    return adj

def aggregate_labels(batches: list[list[LabelingFunctionResult]]) -> list[ProbabilisticLabel]:
    votes = defaultdict(list)
    for batch in batches:
        for r in batch:
            if r.label in (0,1): votes[r.doc_id].append(r.label)
    out = []
    for d, vs in votes.items():
        if not vs: out.append(ProbabilisticLabel(d, 0, 0.5)); continue
        from collections import Counter
        c = Counter(vs); label = 1 if c[1] >= c[0] else 0; out.append(ProbabilisticLabel(d, label, c[label]/len(vs)))
    return out

def anti_slop_ratio(claims_with_evidence: int, total_claims_emitted: int) -> float:
    return 1.0 if total_claims_emitted <= 0 else max(0.0, min(1.0, claims_with_evidence / total_claims_emitted))

def cockpit_honesty(displayed_ok: int, unknown_displayed_as_ok: int) -> float:
    total = displayed_ok + unknown_displayed_as_ok
    return 1.0 if total <= 0 else max(0.0, min(1.0, displayed_ok / total))

def hybrid_labeling(points: List[Point], sites: List[Point], claims_with_evidence: int, total_claims_emitted: int) -> list[ProbabilisticLabel]:
    regions = compute_voronoi_regions(points, sites)
    labels = []
    for site_index, region_points in regions.items():
        label = 1 if len(region_points) > len(points) / 2 else 0
        confidence = anti_slop_ratio(claims_with_evidence, total_claims_emitted) * cockpit_honesty(len(region_points), len(points) - len(region_points))
        labels.append(ProbabilisticLabel(str(site_index), label, confidence))
    return labels

def hybrid_voronoi_labeling(points: List[Point], sites: List[Point], claims_with_evidence: int, total_claims_emitted: int) -> np.ndarray:
    regions = compute_voronoi_regions(points, sites)
    adj = build_adjacency_from_regions(regions, sites)
    labels = hybrid_labeling(points, sites, claims_with_evidence, total_claims_emitted)
    return adj, labels

if __name__ == "__main__":
    points = [(random.random(), random.random()) for _ in range(10)]
    sites = [(random.random(), random.random()) for _ in range(5)]
    claims_with_evidence = 5
    total_claims_emitted = 10
    adj, labels = hybrid_voronoi_labeling(points, sites, claims_with_evidence, total_claims_emitted)
    print(adj)
    print(labels)