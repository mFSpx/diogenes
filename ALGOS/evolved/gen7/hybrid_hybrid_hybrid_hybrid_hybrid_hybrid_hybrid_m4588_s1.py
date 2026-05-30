# DARWIN HAMMER — match 4588, survivor 1
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_cockpit_metri_m95_s2.py (gen4)
# parent_b: hybrid_hybrid_hybrid_vorono_hybrid_hybrid_hybrid_m1529_s4.py (gen6)
# born: 2026-05-29T23:56:39Z

"""
This module fuses the core topologies of hybrid_hybrid_hybrid_hybrid_hybrid_cockpit_metri_m95_s2.py (PARENT ALGORITHM A)
and hybrid_hybrid_hybrid_vorono_hybrid_hybrid_hybrid_m1529_s4.py (PARENT ALGORITHM B) into a unified system.
The mathematical bridge between the two parents lies in the application of Voronoi diagrams to the labeling problem.
By using Voronoi regions to represent the confidence of labeling functions, we can create a more robust and accurate labeling system.

The governing equations of PARENT ALGORITHM A, specifically the interpolant and flow_target functions,
are integrated with the Voronoi diagram construction and adjacency matrix calculation of PARENT ALGORITHM B.
This integration enables the creation of a hybrid algorithm that leverages the strengths of both parents.

The hybrid algorithm uses Voronoi regions to represent the confidence of labeling functions and calculates
the adjacency matrix of the Voronoi graph to determine the relationships between labeling functions.

Imports:
    numpy
    standard library
    math
    random
    sys
    pathlib
"""

import numpy as np
from collections import defaultdict, Counter
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

def labeling_function(name: str|None=None): 
    def deco(fn: Callable[[dict],int]): 
        fn.lf_name=name or fn.__name__ 
        return fn 
    return deco 

def aggregate_labels(batches: list[list[LabelingFunctionResult]]) -> list[ProbabilisticLabel]: 
    votes=defaultdict(list) 
    for batch in batches: 
        for r in batch: 
            if r.label in (0,1): votes[r.doc_id].append(r.label) 
    out=[] 
    for d,vs in votes.items(): 
        if not vs: out.append(ProbabilisticLabel(d,0,0.5)); continue 
        c=Counter(vs); label=1 if c[1]>=c[0] else 0; out.append(ProbabilisticLabel(d,label,c[label]/len(vs))) 
    return out 

def anti_slop_ratio(claims_with_evidence: int, total_claims_emitted: int) -> float:
    return 1.0 if total_claims_emitted <= 0 else max(0.0, min(1.0, claims_with_evidence / total_claims_emitted))

def cockpit_honesty(displayed_ok: int, unknown_displayed_as_ok: int) -> float:
    total = displayed_ok + unknown_displayed_as_ok
    return 1.0 if total <= 0 else max(0.0, min(1.0, displayed_ok / total))

def interpolant(x0, x1, t):
    x0 = np.asarray(x0, dtype=np.float64)
    x1 = np.asarray(x1, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    if x0.ndim > t.ndim:
        t = t[..., np.newaxis]
    return t * x1 + (1.0 - t) * x0

def flow_target(x0, x1):
    x0 = np.asarray(x0, dtype=np.float64)
    x1 = np.asarray(x1, dtype=np.float64)
    return x1 - x0

def euclidean_distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])

def compute_voronoi_regions(points: List[Tuple[float, float]],
                            sites: List[Tuple[float, float]]) -> dict:
    regions = {i: [] for i in range(len(sites))}
    for pt in points:
        distances = [euclidean_distance(pt, s) for s in sites]
        nearest = int(np.argmin(distances))
        regions[nearest].append(pt)
    return regions

def build_adjacency_from_regions(regions: dict,
                                 sites: List[Tuple[float, float]],
                                 radius_factor: float = 1.5) -> np.ndarray:
    n = len(sites)
    adj = np.zeros((n, n), dtype=float)

    dists = []
    for i in range(n):
        for j in range(i + 1, n):
            dists.append(euclidean_distance(sites[i], sites[j]))
    median_dist = np.median(dists) if dists else 0.0

    for i in range(n):
        for j in range(i + 1, n):
            dist = euclidean_distance(sites[i], sites[j])
            if dist < radius_factor * median_dist:
                weight = 1.0 / dist
                adj[i, j] = weight
                adj[j, i] = weight
    return adj

def hybrid_labeling(batch: list[LabelingFunctionResult], 
                    claims_with_evidence: int, 
                    total_claims_emitted: int,
                    points: List[Tuple[float, float]],
                    sites: List[Tuple[float, float]]) -> list[ProbabilisticLabel]:
    labels = aggregate_labels([batch])
    slop_ratio = anti_slop_ratio(claims_with_evidence, total_claims_emitted)
    regions = compute_voronoi_regions(points, sites)
    adj = build_adjacency_from_regions(regions, sites)

    honest_labels = []
    for label in labels:
        confidence = label.confidence * slop_ratio
        # Use Voronoi regions to adjust confidence
        for region, pts in regions.items():
            if (label.doc_id in [p[0] for p in pts]):
                confidence *= (1.0 + adj[region, region])
        honest_labels.append(ProbabilisticLabel(label.doc_id, label.label, confidence))
    return honest_labels

def example_usage():
    batch = [LabelingFunctionResult("lf1", "doc1", 1), 
             LabelingFunctionResult("lf2", "doc2", 0), 
             LabelingFunctionResult("lf3", "doc3", 1)]
    claims_with_evidence = 10
    total_claims_emitted = 20
    points = [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)]
    sites = [(0.5, 0.5), (1.5, 1.5)]

    labels = hybrid_labeling(batch, claims_with_evidence, total_claims_emitted, points, sites)
    for label in labels:
        print(label)

if __name__ == "__main__":
    example_usage()